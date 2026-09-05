import numpy as np

from g1pickplace.trajectory import JointTarget, OpenLoopPolicy, compile_joint_targets


def test_default_offset_action_conversion() -> None:
    trajectory = compile_joint_targets(
        joint_names=("j0", "j1"),
        initial_absolute_positions=np.asarray([1.0, -1.0]),
        default_joint_positions=np.asarray([0.5, -0.25]),
        targets=(JointTarget("move", np.asarray([1.5, 0.75]), 0.2),),
        fps=10,
        max_joint_step=None,
    )
    np.testing.assert_allclose(trajectory.absolute_targets[-1], [1.5, 0.75])
    np.testing.assert_allclose(trajectory.env_actions[-1], [1.0, 1.0])


def test_max_joint_step_increases_interpolation_count() -> None:
    trajectory = compile_joint_targets(
        joint_names=("j0",),
        initial_absolute_positions=np.asarray([0.0]),
        default_joint_positions=np.asarray([0.0]),
        targets=(JointTarget("move", np.asarray([1.0]), 0.1),),
        fps=10,
        max_joint_step=0.2,
    )
    assert trajectory.steps == 5
    assert np.max(np.diff(np.concatenate(([0.0], trajectory.absolute_targets[:, 0])))) <= 0.2 + 1.0e-7


def test_policy_is_observation_invariant() -> None:
    trajectory = compile_joint_targets(
        joint_names=("j0",),
        initial_absolute_positions=np.asarray([0.0]),
        default_joint_positions=np.asarray([0.0]),
        targets=(JointTarget("move", np.asarray([0.5]), 0.5),),
        fps=4,
        max_joint_step=None,
    )
    first = OpenLoopPolicy(trajectory)
    second = OpenLoopPolicy(trajectory)
    np.testing.assert_array_equal(first.act({"pixels": np.ones((2, 2, 3))}), second.act(None))
    np.testing.assert_array_equal(first.act({"object_pose": np.full(7, 99.0)}), second.act(object()))
