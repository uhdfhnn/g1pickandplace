#!/usr/bin/env python3
"""Compile a deterministic fake-IK plan without Isaac Sim or Pinocchio."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from g1pickplace import PickPlaceConfig, Pose, ResetSnapshot, ResetTimePickPlacePlanner
from g1pickplace.offline_ik import IKSolveReport


class PreviewIK:
    active_joint_names = ("right_shoulder_pitch_joint", "right_elbow_joint")

    def q_from_named_positions(self, joint_names, positions):
        del joint_names
        return np.asarray([positions[0], positions[1]], dtype=np.float64)

    def named_positions_from_q(self, q, joint_names):
        return {joint_names[0]: float(q[0]), joint_names[1]: float(q[1])}

    def frame_pose(self, q):
        del q
        return Pose(np.asarray([0.2, 0.0, 0.4]), np.asarray([0.0, 0.0, 0.0, 1.0]))

    def solve(self, waypoint_name, target, seed_q):
        del waypoint_name
        q = np.asarray(seed_q, dtype=np.float64).copy()
        q[0] = target.position[0]
        q[1] = target.position[2]
        return IKSolveReport(q=q, iterations=3, residual=1.0e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/preview_trajectory.npz"))
    args = parser.parse_args()
    names = (
        "right_shoulder_pitch_joint",
        "right_elbow_joint",
        "right_hand_Joint1_1",
        "right_hand_Joint2_1",
    )
    snapshot = ResetSnapshot(
        joint_names=names,
        joint_positions=np.zeros(4),
        default_joint_positions=np.zeros(4),
        robot_base_world=Pose.identity(),
        object_world=Pose(np.asarray([0.35, 0.0, 0.75]), np.asarray([0.0, 0.0, 0.0, 1.0])),
        target_world=Pose(np.asarray([0.20, 0.20, 0.75]), np.asarray([0.0, 0.0, 0.0, 1.0])),
    )
    trajectory, diagnostics = ResetTimePickPlacePlanner(PreviewIK(), PickPlaceConfig()).build(snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trajectory.save_npz(str(args.output))
    print(f"saved: {args.output}")
    print(f"shape: {trajectory.env_actions.shape}")
    print(f"duration_s: {trajectory.duration_s:.2f}")
    print(f"phases: {', '.join(dict.fromkeys(trajectory.phases))}")
    print(f"IK iterations: {diagnostics.waypoint_iterations}")


if __name__ == "__main__":
    main()
