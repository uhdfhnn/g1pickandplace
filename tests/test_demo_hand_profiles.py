from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"


def _load_profile_helpers():
    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    assignment_names = {
        "VALIDATED_FIXED_SEED_PRESET",
        "DEMO_RELATION_VECTORS_WORLD",
        "DEMO_DEFAULT_APPROACH_HEIGHT_M",
        "DEMO_DEFAULT_LIFT_HEIGHT_M",
        "DEMO_DEFAULT_TARGET_APPROACH_HEIGHT_M",
        "OBJECT_RESET_ZERO_OFFSET",
        "DEMO_BASELINE_GREEN_RESET_OFFSET",
        "DEMO_STACK_YELLOW_RESET_OFFSET",
        "DEMO_STACK_RED_RESET_OFFSET",
        "DEMO_TYPE_GREEN_YELLOW_RESET_OFFSET",
        "DEMO_TYPE_YELLOW_GREEN_RESET_OFFSET",
        "DEMO_TYPE_YELLOW_RED_RESET_OFFSET",
        "DEX1_LEFT_ACTIVE_JOINT_NAMES",
        "DEX1_RIGHT_ACTIVE_JOINT_NAMES",
        "DEX1_LEFT_EE_FRAME",
        "DEX1_RIGHT_EE_FRAME",
        "DEX1_LEFT_GRIPPER_JOINT_NAMES",
        "DEX1_RIGHT_GRIPPER_JOINT_NAMES",
        "DEX1_LEFT_GRASP_WRIST_OFFSET_WORLD_M",
        "DEX1_RIGHT_GRASP_WRIST_OFFSET_WORLD_M",
        "DEX1_LEFT_PLACE_WRIST_OFFSET_WORLD_M",
        "DEX1_STACK_PLACE_WRIST_OFFSET_WORLD_M",
        "DEX1_RIGHT_PLACE_WRIST_OFFSET_WORLD_M",
        "DEX1_LEFT_GRASP_QUATERNION_BASE_XYZW",
        "DEX1_RIGHT_GRASP_QUATERNION_BASE_XYZW",
        "DEX1_GRIPPER_OPEN_POSITIONS",
        "DEX1_GRIPPER_CLOSED_POSITIONS",
        "DEX1_HAND_PROFILE_EVIDENCE_SCOPE",
        "DEX1_HAND_PROFILES",
    }
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in assignment_names
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.ClassDef)
            and node.name == "Dex1HandProfile"
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name
            in {
                "_select_demo_hand_profile",
                "_demo_hand_profile_payload",
                "_demo_public_reset_offsets_by_task",
                "_apply_demo_hand_profile",
            }
        )
    ]
    namespace = {
        "Any": Any,
        "Literal": Literal,
        "dataclass": dataclass,
        "replace": replace,
        "math": math,
        "np": np,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace


def test_demo_object_colors_select_the_documented_dex1_hand():
    namespace = _load_profile_helpers()
    select = namespace["_select_demo_hand_profile"]

    assert select(SimpleNamespace(object_color="red")).name == "left"
    assert select(SimpleNamespace(object_color="green")).name == "right"
    assert select(SimpleNamespace(object_color="yellow")).name == "right"


def test_stack_red_uses_specialized_place_profile_without_mutating_base():
    namespace = _load_profile_helpers()
    select = namespace["_select_demo_hand_profile"]
    profiles = namespace["DEX1_HAND_PROFILES"]
    base_left = profiles["left"]

    stack = select(SimpleNamespace(object_color="red", demo_task="stack"))
    relative = select(SimpleNamespace(object_color="red", demo_task="relative-place"))
    typed = select(SimpleNamespace(object_color="red", demo_task="type-place"))

    assert stack is not base_left
    assert stack.grasp_wrist_offset_world == base_left.grasp_wrist_offset_world
    assert stack.place_wrist_offset_world == pytest.approx((-0.020, 0.135, 0.045))
    assert base_left.place_wrist_offset_world == pytest.approx((-0.005, 0.14, -0.03))
    assert relative is base_left
    assert typed is base_left

    payload = namespace["_demo_hand_profile_payload"](stack)
    assert payload["place_wrist_offset_world_m"] == pytest.approx(
        [-0.020, 0.135, 0.045]
    )


def test_left_and_right_profiles_have_exact_public_joint_frame_and_grasp_calibration():
    namespace = _load_profile_helpers()
    profiles = namespace["DEX1_HAND_PROFILES"]

    left = profiles["left"]
    right = profiles["right"]
    assert left.active_joint_names == (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    )
    assert right.active_joint_names == (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    assert left.ee_frame == "left_wrist_yaw_link"
    assert right.ee_frame == "right_wrist_yaw_link"
    assert left.gripper_joint_names == ("left_hand_Joint1_1", "left_hand_Joint2_1")
    assert right.gripper_joint_names == ("right_hand_Joint1_1", "right_hand_Joint2_1")
    assert left.grasp_wrist_offset_world == pytest.approx((-0.025, 0.17, -0.03))
    assert right.grasp_wrist_offset_world == pytest.approx((0.025, 0.16, 0.0))
    assert left.place_wrist_offset_world == pytest.approx((-0.005, 0.14, -0.03))
    assert left.place_wrist_offset_world != left.grasp_wrist_offset_world
    assert right.place_wrist_offset_world == pytest.approx((-0.015, 0.16, 0.0))
    assert right.place_wrist_offset_world != right.grasp_wrist_offset_world
    assert left.grasp_quaternion_base_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert right.grasp_quaternion_base_xyzw == pytest.approx(
        (0.0, 0.17410813759359595, 0.0, 0.9847265389049334)
    )
    assert left.gripper_open_positions == right.gripper_open_positions
    assert left.gripper_closed_positions == right.gripper_closed_positions
    assert left.evidence_scope == right.evidence_scope
    assert "public-URDF" in left.evidence_scope


def test_profile_heights_and_relation_axes_match_approved_conventions():
    namespace = _load_profile_helpers()

    assert namespace["DEMO_DEFAULT_APPROACH_HEIGHT_M"] == pytest.approx(0.16)
    assert namespace["DEMO_DEFAULT_LIFT_HEIGHT_M"] == pytest.approx(0.20)
    assert namespace["DEMO_DEFAULT_TARGET_APPROACH_HEIGHT_M"] == pytest.approx(0.08)
    relation_vectors = namespace["DEMO_RELATION_VECTORS_WORLD"]
    assert relation_vectors["left-of"] == (1.0, 0.0, 0.0)
    assert relation_vectors["right-of"] == (-1.0, 0.0, 0.0)
    assert relation_vectors["in-front-of"] == (0.0, -1.0, 0.0)


def test_public_reset_offsets_are_deterministic_and_non_overlapping():
    namespace = _load_profile_helpers()
    offsets = namespace["_demo_public_reset_offsets_by_task"]
    zero = (0.0, 0.0, 0.0)

    relative = offsets("relative-place", None)
    typed = offsets("type-place", None, "red")
    stack = offsets("stack", None)
    assert relative == typed
    assert relative == {
        "red": zero,
        "yellow": zero,
        "green": namespace["DEMO_BASELINE_GREEN_RESET_OFFSET"],
    }
    assert stack == {
        "red": namespace["DEMO_STACK_RED_RESET_OFFSET"],
        "yellow": namespace["DEMO_STACK_YELLOW_RESET_OFFSET"],
        "green": namespace["DEMO_BASELINE_GREEN_RESET_OFFSET"],
    }
    assert namespace["DEMO_STACK_RED_RESET_OFFSET"] == (0.025, 0.0, 0.0)
    assert namespace["DEMO_STACK_YELLOW_RESET_OFFSET"] == (0.05, 0.05, 0.0)
    assert offsets("type-place", None, "green")["yellow"] == (
        namespace["DEMO_TYPE_GREEN_YELLOW_RESET_OFFSET"]
    )
    assert offsets("type-place", None, "yellow")["green"] == (
        namespace["DEMO_TYPE_YELLOW_GREEN_RESET_OFFSET"]
    )
    assert offsets("type-place", None, "yellow")["red"] == (
        namespace["DEMO_TYPE_YELLOW_RED_RESET_OFFSET"]
    )
    assert namespace["DEMO_TYPE_YELLOW_RED_RESET_OFFSET"] == (0.0, 0.05, 0.0)

    authored = {
        "red": np.asarray((-4.10, -4.08)),
        "yellow": np.asarray((-4.25, -4.05)),
        "green": np.asarray((-4.18, -4.12)),
    }
    half_size_m = 0.025
    for task_offsets in (
        relative,
        stack,
        offsets("type-place", None, "green"),
        offsets("type-place", None, "yellow"),
    ):
        centers = {
            color: authored[color] + np.asarray(delta[:2])
            for color, delta in task_offsets.items()
        }
        for first, second in (("red", "yellow"), ("red", "green"), ("yellow", "green")):
            separation = np.abs(centers[first] - centers[second])
            assert np.any(separation >= 2.0 * half_size_m), (task_offsets, first, second)

    red_variant = (0.04, -0.02, 0.0)
    assert offsets("relative-place", red_variant)["red"] == red_variant
    assert offsets("type-place", red_variant)["red"] == red_variant

    with pytest.raises(ValueError, match="stack demo reserves its red reset"):
        offsets("stack", red_variant)
    with pytest.raises(ValueError, match="reserves the red reset offset"):
        offsets("type-place", red_variant, "yellow")


def _profile_args(profile):
    return SimpleNamespace(
        active_joints=",".join(profile.active_joint_names),
        ee_frame=profile.ee_frame,
        grasp_wrist_offset_world=profile.grasp_wrist_offset_world,
        grasp_quaternion_base_xyzw=profile.grasp_quaternion_base_xyzw,
        gripper_open=profile.gripper_open_positions,
        gripper_closed=profile.gripper_closed_positions,
        approach_height_m=0.16,
        lift_height_m=0.20,
        target_approach_height_m=0.08,
    )


@pytest.mark.parametrize(
    ("option", "field", "bad_value"),
    [
        ("--active-joints", "active_joints", "right_shoulder_pitch_joint"),
        ("--ee-frame", "ee_frame", "right_wrist_yaw_link"),
        ("--grasp-wrist-offset-world", "grasp_wrist_offset_world", (0.025, 0.16, 0.0)),
        ("--target-approach-height-m", "target_approach_height_m", 0.16),
    ],
)
def test_conflicting_explicit_legacy_profile_flags_fail_closed(option, field, bad_value):
    namespace = _load_profile_helpers()
    profile = namespace["DEX1_HAND_PROFILES"]["left"]
    args = _profile_args(profile)
    setattr(args, field, bad_value)

    with pytest.raises(ValueError, match="conflicts with explicit"):
        namespace["_apply_demo_hand_profile"](args, profile, {option})
