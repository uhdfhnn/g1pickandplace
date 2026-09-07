from __future__ import annotations

import numpy as np
import pytest

from g1pickplace import PickPlaceConfig, Pose, ResetSnapshot, ResetTimePickPlacePlanner
from g1pickplace.offline_ik import IKSolveReport
from g1pickplace.trajectory import OpenLoopPolicy


class CountingIK:
    active_joint_names = ("arm_a", "arm_b")

    def __init__(self, frame_quaternion: np.ndarray | None = None) -> None:
        self.solve_calls: list[str] = []
        self.target_poses: list[tuple[str, Pose]] = []
        self.frame_quaternion = (
            np.asarray([0.0, 0.0, 0.0, 1.0])
            if frame_quaternion is None
            else np.asarray(frame_quaternion, dtype=np.float64)
        )

    def q_from_named_positions(self, joint_names, positions):
        by_name = dict(zip(joint_names, positions, strict=True))
        return np.asarray([by_name["arm_a"], by_name["arm_b"]], dtype=np.float64)

    def named_positions_from_q(self, q, joint_names):
        del joint_names
        return {"arm_a": float(q[0]), "arm_b": float(q[1])}

    def frame_pose(self, q):
        del q
        return Pose(np.asarray([0.2, 0.0, 0.4]), self.frame_quaternion)

    def solve(self, waypoint_name, target, seed_q):
        self.solve_calls.append(waypoint_name)
        self.target_poses.append((waypoint_name, target))
        q = np.asarray(seed_q, dtype=np.float64).copy()
        q[0] = target.position[0]
        q[1] = target.position[2]
        return IKSolveReport(q=q, iterations=2, residual=1.0e-6)


def _snapshot(object_x: float = 0.35) -> ResetSnapshot:
    names = ("arm_a", "arm_b", "right_hand_Joint1_1", "right_hand_Joint2_1")
    return ResetSnapshot(
        joint_names=names,
        joint_positions=np.zeros(4),
        default_joint_positions=np.zeros(4),
        robot_base_world=Pose.identity(),
        object_world=Pose(np.asarray([object_x, 0.0, 0.75]), np.asarray([0.0, 0.0, 0.0, 1.0])),
        target_world=Pose(np.asarray([0.2, 0.2, 0.75]), np.asarray([0.0, 0.0, 0.0, 1.0])),
    )


def test_all_ik_happens_during_build_only() -> None:
    ik = CountingIK()
    trajectory, diagnostics = ResetTimePickPlacePlanner(ik, PickPlaceConfig(fps=20)).build(_snapshot())
    assert ik.solve_calls == ["pregrasp", "grasp", "lift", "preplace", "place"]
    assert set(diagnostics.waypoint_residuals) == set(ik.solve_calls)

    calls_after_build = len(ik.solve_calls)
    policy = OpenLoopPolicy(trajectory)
    for _ in range(10):
        policy.act({"adversarial_observation": np.random.randn(100)})
    assert len(ik.solve_calls) == calls_after_build


def test_gripper_closes_and_reopens_in_frozen_program() -> None:
    trajectory, _ = ResetTimePickPlacePlanner(CountingIK(), PickPlaceConfig(fps=20)).build(_snapshot())
    phase_to_last = {}
    for index, phase in enumerate(trajectory.phases):
        phase_to_last[phase] = index
    right_a = trajectory.joint_names.index("right_hand_Joint1_1")
    right_b = trajectory.joint_names.index("right_hand_Joint2_1")
    np.testing.assert_allclose(
        trajectory.absolute_targets[phase_to_last["close_gripper"], [right_a, right_b]],
        [-0.02, -0.02],
    )
    np.testing.assert_allclose(
        trajectory.absolute_targets[phase_to_last["open_gripper"], [right_a, right_b]],
        [0.03, 0.03],
    )


def test_reset_object_pose_changes_compiled_actions() -> None:
    planner = ResetTimePickPlacePlanner(CountingIK(), PickPlaceConfig(fps=10))
    first, _ = planner.build(_snapshot(0.30))
    second, _ = planner.build(_snapshot(0.40))
    assert not np.array_equal(first.env_actions, second.env_actions)


def test_optional_staging_is_first_and_reuses_approach_clearance() -> None:
    reset_orientation = np.asarray([0.0, 0.0, 0.38268343, 0.92387953])
    ik = CountingIK(frame_quaternion=reset_orientation)
    config = PickPlaceConfig(fps=20, staging_enabled=True)
    trajectory, diagnostics = ResetTimePickPlacePlanner(ik, config).build(_snapshot())

    assert ik.solve_calls == ["staging", "pregrasp", "grasp", "lift", "preplace", "place"]
    assert set(diagnostics.waypoint_residuals) == set(ik.solve_calls)
    unique_phases = list(dict.fromkeys(trajectory.phases))
    assert unique_phases.index("staging") < unique_phases.index("move_to_pregrasp")

    staging_name, staging_target = ik.target_poses[0]
    assert staging_name == "staging"
    # The fake robot base is identity, so +Z is directly the reset wrist's
    # world-frame z coordinate plus the existing approach-height calibration.
    np.testing.assert_allclose(staging_target.position, [0.2, 0.0, 0.4 + config.approach_height])
    np.testing.assert_allclose(staging_target.quaternion_xyzw, reset_orientation / np.linalg.norm(reset_orientation))
    assert trajectory.phases.count("staging") == round(config.pregrasp_duration_s * config.fps)


def test_staging_routes_open_return_through_frozen_staging_solution() -> None:
    ik = CountingIK()
    config = PickPlaceConfig(fps=20, staging_enabled=True)
    trajectory, _ = ResetTimePickPlacePlanner(ik, config).build(_snapshot())

    unique_phases = list(dict.fromkeys(trajectory.phases))
    assert unique_phases.index("release_settle") < unique_phases.index("retreat")
    assert unique_phases.index("retreat") < unique_phases.index("return_via_staging")
    assert unique_phases.index("return_via_staging") < unique_phases.index("return_home")
    assert trajectory.phases.count("return_via_staging") == round(
        config.pregrasp_duration_s * config.fps
    )

    # The return waypoint must reuse the already solved staging posture.  The
    # six reset-time solves (optional staging plus the five task waypoints)
    # therefore remain unchanged by the extra playback segment.
    assert ik.solve_calls == ["staging", "pregrasp", "grasp", "lift", "preplace", "place"]
    staging_last = max(index for index, phase in enumerate(trajectory.phases) if phase == "staging")
    return_staging_last = max(
        index for index, phase in enumerate(trajectory.phases) if phase == "return_via_staging"
    )
    np.testing.assert_allclose(
        trajectory.absolute_targets[return_staging_last],
        trajectory.absolute_targets[staging_last],
    )


def test_post_release_return_can_be_omitted_at_compile_time() -> None:
    config = PickPlaceConfig(
        fps=20,
        staging_enabled=True,
        post_release_return_enabled=False,
    )
    trajectory, _ = ResetTimePickPlacePlanner(CountingIK(), config).build(_snapshot())
    unique_phases = list(dict.fromkeys(trajectory.phases))
    assert unique_phases[-1] == "retreat"
    assert "return_via_staging" not in unique_phases
    assert "return_home" not in unique_phases


def test_post_retreat_settle_is_a_fixed_compiled_hold() -> None:
    config = PickPlaceConfig(
        fps=20,
        post_release_return_enabled=False,
        post_retreat_settle_enabled=True,
    )
    trajectory, _ = ResetTimePickPlacePlanner(CountingIK(), config).build(_snapshot())
    unique_phases = list(dict.fromkeys(trajectory.phases))
    assert unique_phases[-2:] == ["retreat", "post_retreat_settle"]
    assert trajectory.phases.count("post_retreat_settle") == round(
        config.return_duration_s * config.fps
    )


def test_staging_ik_failure_gates_all_later_waypoints() -> None:
    class FailingStagingIK(CountingIK):
        def solve(self, waypoint_name, target, seed_q):
            if waypoint_name == "staging":
                self.solve_calls.append(waypoint_name)
                raise RuntimeError("synthetic staging IK failure")
            return super().solve(waypoint_name, target, seed_q)

    ik = FailingStagingIK()
    with pytest.raises(RuntimeError, match="synthetic staging IK failure"):
        ResetTimePickPlacePlanner(
            ik,
            PickPlaceConfig(staging_enabled=True),
    ).build(_snapshot())
    assert ik.solve_calls == ["staging"]


def test_place_wrist_offset_is_split_from_grasp_offset_when_configured() -> None:
    ik = CountingIK()
    config = PickPlaceConfig(
        fps=20,
        grasp_wrist_offset_world=(0.10, 0.20, 0.30),
        place_wrist_offset_world=(0.40, 0.50, 0.60),
    )

    ResetTimePickPlacePlanner(ik, config).build(_snapshot(object_x=0.35))
    targets = dict(ik.target_poses)
    np.testing.assert_allclose(
        targets["grasp"].position,
        [0.45, 0.20, 1.05],
    )
    np.testing.assert_allclose(
        targets["pregrasp"].position,
        [0.45, 0.20, 1.05 + config.approach_height],
    )
    np.testing.assert_allclose(
        targets["lift"].position,
        [0.45, 0.20, 1.05 + config.lift_height],
    )
    np.testing.assert_allclose(
        targets["place"].position,
        [0.60, 0.70, 1.35],
    )
    np.testing.assert_allclose(
        targets["preplace"].position,
        [0.60, 0.70, 1.35 + config.target_approach_height],
    )


def test_place_wrist_offset_defaults_to_grasp_offset_for_legacy_callers() -> None:
    ik = CountingIK()
    config = PickPlaceConfig(
        fps=20,
        grasp_wrist_offset_world=(0.10, 0.20, 0.30),
    )

    ResetTimePickPlacePlanner(ik, config).build(_snapshot(object_x=0.35))
    targets = dict(ik.target_poses)
    np.testing.assert_allclose(
        targets["place"].position,
        [0.30, 0.40, 1.05],
    )
    np.testing.assert_allclose(
        targets["preplace"].position,
        [0.30, 0.40, 1.05 + config.target_approach_height],
    )
