from __future__ import annotations

import numpy as np

from g1pickplace import PickPlaceConfig, Pose, ResetSnapshot, ResetTimePickPlacePlanner
from g1pickplace.offline_ik import IKSolveReport
from g1pickplace.trajectory import OpenLoopPolicy


class CountingIK:
    active_joint_names = ("arm_a", "arm_b")

    def __init__(self) -> None:
        self.solve_calls: list[str] = []

    def q_from_named_positions(self, joint_names, positions):
        by_name = dict(zip(joint_names, positions, strict=True))
        return np.asarray([by_name["arm_a"], by_name["arm_b"]], dtype=np.float64)

    def named_positions_from_q(self, q, joint_names):
        del joint_names
        return {"arm_a": float(q[0]), "arm_b": float(q[1])}

    def frame_pose(self, q):
        del q
        return Pose(np.asarray([0.2, 0.0, 0.4]), np.asarray([0.0, 0.0, 0.0, 1.0]))

    def solve(self, waypoint_name, target, seed_q):
        self.solve_calls.append(waypoint_name)
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
