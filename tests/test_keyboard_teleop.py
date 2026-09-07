from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from g1pickplace.keyboard_teleop import JointJogTeleop


LEFT_ARM = tuple(f"left_arm_{index}" for index in range(7))
RIGHT_ARM = tuple(f"right_arm_{index}" for index in range(7))
LEFT_GRIPPER = ("left_finger_1", "left_finger_2")
RIGHT_GRIPPER = ("right_finger_1", "right_finger_2")
JOINT_NAMES = LEFT_ARM + RIGHT_ARM + LEFT_GRIPPER + RIGHT_GRIPPER


def _controller(*, initial: float = 0.0, step: float = 0.1) -> JointJogTeleop:
    return JointJogTeleop(
        joint_names=JOINT_NAMES,
        initial_positions=np.full(len(JOINT_NAMES), initial),
        joint_limits=np.tile((-0.2, 0.2), (len(JOINT_NAMES), 1)),
        arm_joints_by_side={"left": LEFT_ARM, "right": RIGHT_ARM},
        gripper_joints_by_side={"left": LEFT_GRIPPER, "right": RIGHT_GRIPPER},
        gripper_open_positions=(-0.1, -0.1),
        gripper_closed_positions=(0.15, 0.15),
        jog_step_rad=step,
    )


def test_right_arm_is_default_and_joint_jog_is_limit_clamped() -> None:
    controller = _controller(initial=0.15)
    assert controller.selected_joint_name == RIGHT_ARM[0]

    event = controller.press("RIGHT")
    assert event is not None and event.kind == "jog"
    index = JOINT_NAMES.index(RIGHT_ARM[0])
    assert controller.absolute_targets[index] == pytest.approx(0.2)

    controller.press("RIGHT")
    assert controller.absolute_targets[index] == pytest.approx(0.2)


def test_tab_number_selection_and_left_jog_change_only_selected_joint() -> None:
    controller = _controller()
    before = controller.absolute_targets
    assert controller.press("TAB").message == "active arm: left"
    assert controller.press("KEY_7").message == f"selected {LEFT_ARM[6]}"
    controller.press("LEFT")

    expected = before.copy()
    expected[JOINT_NAMES.index(LEFT_ARM[6])] = -0.1
    np.testing.assert_allclose(controller.absolute_targets, expected)


def test_gripper_commands_apply_only_to_active_side() -> None:
    controller = _controller()
    controller.press("O")
    targets = controller.absolute_targets
    np.testing.assert_allclose(
        targets[[JOINT_NAMES.index(name) for name in RIGHT_GRIPPER]],
        (-0.1, -0.1),
    )
    np.testing.assert_allclose(
        targets[[JOINT_NAMES.index(name) for name in LEFT_GRIPPER]],
        (0.0, 0.0),
    )

    controller.press("TAB")
    controller.press("C")
    targets = controller.absolute_targets
    np.testing.assert_allclose(
        targets[[JOINT_NAMES.index(name) for name in LEFT_GRIPPER]],
        (0.15, 0.15),
    )


def test_quit_keys_are_explicit_and_unknown_keys_are_ignored() -> None:
    controller = _controller()
    assert controller.press("W") is None
    assert not controller.quit_requested
    assert controller.press("ESCAPE").kind == "quit"
    assert controller.quit_requested


def test_constructor_rejects_missing_control_joints() -> None:
    with pytest.raises(ValueError, match="absent from action order"):
        JointJogTeleop(
            joint_names=JOINT_NAMES[:-1],
            initial_positions=np.zeros(len(JOINT_NAMES) - 1),
            joint_limits=np.tile((-1.0, 1.0), (len(JOINT_NAMES) - 1, 1)),
            arm_joints_by_side={"left": LEFT_ARM, "right": RIGHT_ARM},
            gripper_joints_by_side={"left": LEFT_GRIPPER, "right": RIGHT_GRIPPER},
            gripper_open_positions=(-0.1, -0.1),
            gripper_closed_positions=(0.1, 0.1),
        )


def test_runner_exits_to_teleop_before_snapshot_planning_or_policy() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    source = ast.get_source_segment(script.read_text(encoding="utf-8"), main)
    assert source is not None
    teleop = source.index("if args.keyboard_teleop:")
    snapshot = source.index("_capture_demo_reset_snapshot(")
    policy = source.index("policy = OpenLoopPolicy(trajectory)")
    assert teleop < snapshot < policy
