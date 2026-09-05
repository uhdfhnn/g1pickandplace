"""Reset-time semantic program and complete pre-rollout IK compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .geometry import Pose
from .offline_ik import IKSolveReport
from .trajectory import JointTarget, JointTrajectory, compile_joint_targets


class FrameIK(Protocol):
    active_joint_names: tuple[str, ...]

    def q_from_named_positions(self, joint_names: tuple[str, ...], positions: np.ndarray) -> np.ndarray: ...

    def named_positions_from_q(self, q: np.ndarray, joint_names: tuple[str, ...]) -> dict[str, float]: ...

    def frame_pose(self, q: np.ndarray) -> Pose: ...

    def solve(self, waypoint_name: str, target: Pose, seed_q: np.ndarray) -> IKSolveReport: ...


@dataclass(frozen=True)
class ResetSnapshot:
    """Privileged state captured once after reset and before planning."""

    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    default_joint_positions: np.ndarray
    robot_base_world: Pose
    object_world: Pose
    target_world: Pose

    def __post_init__(self) -> None:
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        positions = np.asarray(self.joint_positions, dtype=np.float64)
        defaults = np.asarray(self.default_joint_positions, dtype=np.float64)
        if positions.shape != (len(self.joint_names),) or defaults.shape != positions.shape:
            raise ValueError("joint vectors must match joint_names")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(defaults)):
            raise ValueError("joint vectors contain non-finite values")
        positions = positions.copy()
        defaults = defaults.copy()
        positions.setflags(write=False)
        defaults.setflags(write=False)
        object.__setattr__(self, "joint_positions", positions)
        object.__setattr__(self, "default_joint_positions", defaults)


@dataclass(frozen=True)
class PickPlaceConfig:
    fps: int = 50
    grasp_wrist_offset_world: tuple[float, float, float] = (0.0, 0.0, 0.08)
    approach_height: float = 0.12
    lift_height: float = 0.16
    target_approach_height: float = 0.12
    grasp_quaternion_base_xyzw: tuple[float, float, float, float] | None = None

    gripper_joint_names: tuple[str, ...] = ("right_hand_Joint1_1", "right_hand_Joint2_1")
    gripper_open_positions: tuple[float, ...] = (0.03, 0.03)
    gripper_closed_positions: tuple[float, ...] = (-0.02, -0.02)

    pregrasp_duration_s: float = 1.4
    descend_duration_s: float = 0.8
    gripper_duration_s: float = 0.5
    settle_duration_s: float = 0.3
    lift_duration_s: float = 1.0
    transport_duration_s: float = 1.5
    return_duration_s: float = 1.2
    max_joint_step: float = 0.02

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if len(self.gripper_joint_names) != len(self.gripper_open_positions):
            raise ValueError("open gripper positions must match gripper joints")
        if len(self.gripper_joint_names) != len(self.gripper_closed_positions):
            raise ValueError("closed gripper positions must match gripper joints")
        if self.max_joint_step <= 0.0:
            raise ValueError("max_joint_step must be positive")
        scalars = (
            self.approach_height,
            self.lift_height,
            self.target_approach_height,
            self.pregrasp_duration_s,
            self.descend_duration_s,
            self.gripper_duration_s,
            self.settle_duration_s,
            self.lift_duration_s,
            self.transport_duration_s,
            self.return_duration_s,
        )
        if any(value <= 0.0 for value in scalars):
            raise ValueError("heights and durations must be positive")


@dataclass(frozen=True)
class PlanDiagnostics:
    waypoint_iterations: dict[str, int]
    waypoint_residuals: dict[str, float]


class ResetTimePickPlacePlanner:
    """Compile all IK and all joint actions before the first rollout action."""

    def __init__(self, ik: FrameIK, config: PickPlaceConfig | None = None):
        self.ik = ik
        self.config = config or PickPlaceConfig()

    def build(self, snapshot: ResetSnapshot) -> tuple[JointTrajectory, PlanDiagnostics]:
        cfg = self.config
        missing = [name for name in cfg.gripper_joint_names if name not in snapshot.joint_names]
        if missing:
            raise ValueError(f"simulator is missing configured gripper joints: {missing}")

        base_from_world = snapshot.robot_base_world.inverse()
        q_seed = self.ik.q_from_named_positions(snapshot.joint_names, snapshot.joint_positions)
        current_ee = self.ik.frame_pose(q_seed)
        orientation = (
            np.asarray(cfg.grasp_quaternion_base_xyzw, dtype=np.float64)
            if cfg.grasp_quaternion_base_xyzw is not None
            else current_ee.quaternion_xyzw
        )

        offset = np.asarray(cfg.grasp_wrist_offset_world, dtype=np.float64)
        object_wrist_world = snapshot.object_world.position + offset
        target_wrist_world = snapshot.target_world.position + offset

        def target_from_world_position(position_world: np.ndarray) -> Pose:
            position_base = base_from_world.transform_point(position_world)
            return Pose(position_base, orientation)

        grasp = target_from_world_position(object_wrist_world)
        pregrasp = target_from_world_position(object_wrist_world + np.asarray([0.0, 0.0, cfg.approach_height]))
        lifted = target_from_world_position(object_wrist_world + np.asarray([0.0, 0.0, cfg.lift_height]))
        place = target_from_world_position(target_wrist_world)
        preplace = target_from_world_position(target_wrist_world + np.asarray([0.0, 0.0, cfg.target_approach_height]))

        reports: dict[str, IKSolveReport] = {}
        for name, target in (
            ("pregrasp", pregrasp),
            ("grasp", grasp),
            ("lift", lifted),
            ("preplace", preplace),
            ("place", place),
        ):
            report = self.ik.solve(name, target, q_seed)
            reports[name] = report
            q_seed = report.q

        named_solutions = {
            name: self.ik.named_positions_from_q(report.q, snapshot.joint_names)
            for name, report in reports.items()
        }

        def full_target(solution_name: str, *, gripper_closed: bool) -> np.ndarray:
            positions = snapshot.joint_positions.copy()
            name_to_index = {name: index for index, name in enumerate(snapshot.joint_names)}
            for name, value in named_solutions[solution_name].items():
                positions[name_to_index[name]] = value
            grip_values = cfg.gripper_closed_positions if gripper_closed else cfg.gripper_open_positions
            for name, value in zip(cfg.gripper_joint_names, grip_values, strict=True):
                positions[name_to_index[name]] = value
            return positions

        open_home = snapshot.joint_positions.copy()
        index_by_name = {name: index for index, name in enumerate(snapshot.joint_names)}
        for name, value in zip(cfg.gripper_joint_names, cfg.gripper_open_positions, strict=True):
            open_home[index_by_name[name]] = value

        pregrasp_q = full_target("pregrasp", gripper_closed=False)
        grasp_open_q = full_target("grasp", gripper_closed=False)
        grasp_closed_q = full_target("grasp", gripper_closed=True)
        lift_q = full_target("lift", gripper_closed=True)
        preplace_closed_q = full_target("preplace", gripper_closed=True)
        place_closed_q = full_target("place", gripper_closed=True)
        place_open_q = full_target("place", gripper_closed=False)
        preplace_open_q = full_target("preplace", gripper_closed=False)

        program = (
            JointTarget("open_at_home", open_home, cfg.settle_duration_s),
            JointTarget("move_to_pregrasp", pregrasp_q, cfg.pregrasp_duration_s),
            JointTarget("descend_to_grasp", grasp_open_q, cfg.descend_duration_s),
            JointTarget("close_gripper", grasp_closed_q, cfg.gripper_duration_s),
            JointTarget("grasp_settle", grasp_closed_q, cfg.settle_duration_s),
            JointTarget("lift", lift_q, cfg.lift_duration_s),
            JointTarget("transport", preplace_closed_q, cfg.transport_duration_s),
            JointTarget("descend_to_place", place_closed_q, cfg.descend_duration_s),
            JointTarget("open_gripper", place_open_q, cfg.gripper_duration_s),
            JointTarget("release_settle", place_open_q, cfg.settle_duration_s),
            JointTarget("retreat", preplace_open_q, cfg.lift_duration_s),
            JointTarget("return_home", open_home, cfg.return_duration_s),
        )
        trajectory = compile_joint_targets(
            joint_names=snapshot.joint_names,
            initial_absolute_positions=snapshot.joint_positions,
            default_joint_positions=snapshot.default_joint_positions,
            targets=program,
            fps=cfg.fps,
            action_scale=1.0,
            max_joint_step=cfg.max_joint_step,
        )
        diagnostics = PlanDiagnostics(
            waypoint_iterations={name: report.iterations for name, report in reports.items()},
            waypoint_residuals={name: report.residual for name, report in reports.items()},
        )
        return trajectory, diagnostics
