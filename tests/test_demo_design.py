from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from g1pickplace import PickPlaceConfig, Pose, ResetSnapshot, ResetTimePickPlacePlanner
from g1pickplace.offline_ik import IKSolveReport
from g1pickplace.trajectory import OpenLoopPolicy


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"


def _load_demo_helpers():
    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    wanted_assignments = {
        "PUBLIC_STACK_TASK_ID",
        "DEMO_OBJECT_SCENE_NAMES",
        "DEMO_DEFAULT_CLEARANCE_M",
        "DEMO_RELATION_VECTORS_WORLD",
        "DEMO_MARKER_SIZE_M",
        "DEMO_PUBLIC_BLOCK_SIZE_M",
        "DEMO_STACK_POSITION_TOLERANCE_M",
        "DEMO_STACK_HEIGHT_TOLERANCE_M",
        "DEMO_STABLE_SPEED_MPS",
        "DEMO_GEOMETRY_EPSILON_M",
        "DEMO_HAND_ENVELOPE_HALF_WIDTH_M",
        "DEMO_AIRBORNE_OBJECT_CLEARANCE_M",
        "DEMO_TABLE_CLEARANCE_PHASES",
        "DEMO_TABLE_CONTACT_EXIT_PHASES",
        "DEMO_AIRBORNE_PHASES",
        "DEMO_PRELOAD_PHASES",
        "DEMO_LOADED_PHASES",
        "DEMO_RELEASE_SETTLE_PHASES",
        "DEMO_POST_RELEASE_PHASES",
        "DEMO_KNOWN_PHASES",
        "SHOVEL_BLOCK_POSITION_TOLERANCE_M",
        "SHOVEL_BLOCK_STABLE_SPEED_MPS",
        "SHOVEL_REQUIRED_LIFT_M",
        "SHOVEL_CONTACT_FORCE_MIN_N",
        "SHOVEL_FINGER_CONTACT_SPECS",
        "SHOVEL_GRASP_LIFT_PHASES",
        "SHOVEL_PHASE_ORDER",
    }
    wanted_functions = {
        "_normalize_demo_instruction",
        "_parsed_demo_instruction",
        "_resolve_demo_task",
        "_demo_target_position_world",
        "_validate_demo_reset_geometry",
        "_aabbs_overlap",
        "_validate_demo_swept_clearance",
        "_evaluate_stack_result",
        "_evaluate_shovel_result",
        "_compose_demo_pick_place_success",
    }
    wanted_classes = {"DemoTaskSpec", "DemoResetSnapshot"}
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in wanted_assignments
                for target in node.targets
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name in wanted_functions)
        or (isinstance(node, ast.ClassDef) and node.name in wanted_classes)
    ]
    namespace = {
        "Any": object,
        "Literal": __import__("typing").Literal,
        "Mapping": dict,
        "MappingProxyType": MappingProxyType,
        "dataclass": dataclass,
        "field": field,
        "math": __import__("math"),
        "json": __import__("json"),
        "np": np,
        "re": __import__("re"),
        "Pose": Pose,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace


def test_demo_resolver_accepts_documented_variants_and_infers_task():
    namespace = _load_demo_helpers()
    resolve = namespace["_resolve_demo_task"]

    spec = resolve(
        instruction="PICK UP THE RED BRICK, AND PLACE IT LEFT OF THE YELLOW SQUARE MARKER!"
    )
    assert spec.demo_task == "relative-place"
    assert spec.object_color == "red"
    assert spec.relation == "left-of"
    assert spec.reference == "yellow-marker"
    assert spec.clearance_m == 0.05

    stack = resolve(demo_task="stack")
    assert stack.demo_task == "stack"
    assert stack.object_color == "red"
    assert stack.reference == "yellow"

    typed_red = resolve(
        demo_task="type-place",
        instruction="Pick up the red block and place it left of the yellow square marker.",
    )
    assert typed_red.demo_task == "type-place"
    assert typed_red.object_color == "red"
    assert typed_red.relation == "left-of"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"demo_task": "relative-place", "object_color": "green"},
        {"instruction": "Pick up an orange object and place it somewhere."},
        {
            "instruction": "Pick up the red block and stack it on the yellow block.",
            "demo_task": "relative-place",
        },
        {
            "demo_task": "type-place",
            "object_color": "green",
            "relation": "left-of",
        },
        {"demo_task": "relative-place", "clearance_m": -1.0e-3},
    ],
)
def test_demo_resolver_rejects_unsupported_or_conflicting_inputs(kwargs):
    with pytest.raises(ValueError):
        _load_demo_helpers()["_resolve_demo_task"](**kwargs)


def test_relative_and_stack_target_geometry_uses_edge_clearance_and_half_heights():
    namespace = _load_demo_helpers()
    resolve = namespace["_resolve_demo_task"]
    target = namespace["_demo_target_position_world"]
    relative = resolve(demo_task="relative-place", clearance_m=0.05)
    position = target(
        relative,
        object_position_world=(-4.10, -4.08, 0.814),
        object_size_world_m=(0.05, 0.05, 0.05),
        reference_position_world=(-4.05, -4.03, 0.84),
        reference_size_world_m=(0.10, 0.10, 0.004),
    )
    # marker half-width + object half-width + 5 cm, along visual-left world +X
    assert position == pytest.approx((-3.925, -4.03, 0.814))

    stack = resolve(demo_task="stack")
    stack_position = target(
        stack,
        object_position_world=(-4.10, -4.08, 0.814),
        object_size_world_m=(0.05, 0.05, 0.05),
        reference_position_world=(-4.25, -4.05, 0.814),
        reference_size_world_m=(0.05, 0.05, 0.05),
    )
    assert stack_position == pytest.approx((-4.25, -4.05, 0.864))


def test_stack_and_shovel_evaluators_are_read_only_and_honest():
    namespace = _load_demo_helpers()
    evaluate_stack = namespace["_evaluate_stack_result"]
    passed = evaluate_stack(
        top_position_world=(0.0, 0.0, 0.10),
        bottom_position_world=(0.0, 0.0, 0.05),
        top_velocity_world_mps=(0.0, 0.0, 0.0),
        bottom_velocity_world_mps=(0.0, 0.0, 0.0),
        target_position_world=(0.0, 0.0, 0.10),
        bottom_reset_position_world=(0.0, 0.0, 0.05),
    )
    assert passed["success"] is True
    assert passed["inside_target_xy"] is True
    failed = evaluate_stack(
        top_position_world=(0.02, 0.0, 0.10),
        bottom_position_world=(0.0, 0.0, 0.05),
        top_velocity_world_mps=(0.0, 0.0, 0.0),
        bottom_velocity_world_mps=(0.0, 0.0, 0.0),
        target_position_world=(0.0, 0.0, 0.10),
        bottom_reset_position_world=(0.0, 0.0, 0.05),
    )
    assert failed["success"] is False

    evaluate_shovel = namespace["_evaluate_shovel_result"]

    def evidence(
        *,
        loaded_blade_contact: bool,
        lifted: bool,
        both_finger_contacts: bool = True,
        tool_lifted: bool = True,
    ):
        contacts = [
            {
                "source": "public_isaac_contact_api",
                "phase": "close_tool_gripper",
                "body_a": "left_hand_Link1_1",
                "body_b": "shovel_tool",
                "normal_force_n": 1.0,
                "sensor_name": "shovel_left_finger_1_contact",
            },
            {
                "source": "public_isaac_contact_api",
                "phase": "insert_blade",
                "body_a": "shovel_tool/blade",
                "body_b": "red_block",
                "normal_force_n": 1.0,
            },
            {
                "source": "public_isaac_contact_api",
                "phase": "unload_settle",
                "body_a": "red_block",
                "body_b": "target_tray/floor",
                "normal_force_n": 1.0,
            },
        ]
        if both_finger_contacts:
            contacts.append(
                {
                    "source": "public_isaac_contact_api",
                    "phase": "close_tool_gripper",
                    "body_a": "shovel_tool",
                    "body_b": "left_hand_Link2_1",
                    "normal_force_n": 1.0,
                    "sensor_name": "shovel_left_finger_2_contact",
                }
            )
        if loaded_blade_contact:
            contacts.append(
                {
                    "source": "public_isaac_contact_api",
                    "phase": "lift_loaded_shovel",
                    "body_a": "shovel_tool/blade",
                    "body_b": "red_block",
                    "normal_force_n": 1.0,
                }
            )
        return {
            "contact_samples": [
                {
                    "source": "public_isaac_contact_api",
                    "api_available": True,
                }
            ],
            "contact_history": contacts,
            "pose_history": [
                {
                    "source": "public_isaac_pose_api",
                    "phase": "lift_tool",
                    "tool_position_world_m": [
                        0.0,
                        0.0,
                        0.90 if tool_lifted else 0.84,
                    ],
                    "red_position_world_m": [0.0, 0.0, 0.90 if lifted else 0.825],
                    "red_speed_mps": 0.0,
                },
                {
                    "source": "public_isaac_pose_api",
                    "phase": "unload_settle",
                    "tool_position_world_m": [
                        0.0,
                        0.0,
                        0.90 if tool_lifted else 0.84,
                    ],
                    "red_position_world_m": [0.0, 0.0, 0.84],
                    "red_speed_mps": 0.0,
                },
            ],
            "reset_tool_position_world_m": [0.0, 0.0, 0.84],
            "reset_red_position_world_m": [0.0, 0.0, 0.825],
            "table_support_plane_z_world_m": 0.80,
            "tray_interior_min_world_m": [-0.10, -0.10, 0.80],
            "tray_interior_max_world_m": [0.10, 0.10, 0.90],
            "block_half_extent_world_m": [0.025, 0.025, 0.025],
            "distractor_displacements_m": {"yellow": 0.0, "green": 0.0},
        }

    partial = evaluate_shovel(
        evidence=evidence(loaded_blade_contact=False, lifted=False)
    )
    assert partial["result"] == "PARTIAL"
    assert partial["block_left_table"] is False
    assert partial["shovel_grasped"] is True
    passed_tool = evaluate_shovel(
        evidence=evidence(loaded_blade_contact=True, lifted=True)
    )
    assert passed_tool["result"] == "PASS"
    assert passed_tool["tool_lift_verified"] is True
    assert passed_tool["finger_contact_verified"] == {
        "shovel_left_finger_1_contact": True,
        "shovel_left_finger_2_contact": True,
    }
    assert evaluate_shovel(
        evidence=evidence(
            loaded_blade_contact=True,
            lifted=True,
            both_finger_contacts=False,
        )
    )["shovel_grasped"] is False
    assert evaluate_shovel(
        evidence=evidence(loaded_blade_contact=True, lifted=True, tool_lifted=False)
    )["shovel_grasped"] is False
    assert evaluate_shovel(evidence={})["result"] == "NOT VERIFIED"


def test_stack_caller_reads_the_evaluator_return_key():
    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    stack_result_subscripts = {
        node.slice.value
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "stack_result"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        )
    }
    assert "inside_target_xy" in stack_result_subscripts
    assert "inside_xy" not in stack_result_subscripts


def test_demo_pick_place_acceptance_uses_relation_not_orthogonal_center_box():
    compose = _load_demo_helpers()["_compose_demo_pick_place_success"]
    # The public evaluator's aggregate success/inside-target fields are false
    # for this 0.01185 m orthogonal center error, while the approved relation
    # and edge-clearance gate remains valid.  The final acceptance must follow
    # the explicit task gates and retain the evaluator object as diagnostics.
    selected_result = SimpleNamespace(
        success=False,
        inside_target_xy=False,
        center_error_m=0.01185,
        height_ok=True,
        stable=True,
    )
    common = {
        "selected_result": selected_result,
        "lift_verified": True,
        "transport_verified": True,
        "table_support_inferred": True,
        "movement_ok": True,
        "distractors_ok": True,
    }
    assert compose(**common, relation_and_clearance_ok=True) is True
    # A wrong relation or edge clearance still gates the same acceptance.
    assert compose(**common, relation_and_clearance_ok=False) is False


def test_demo_snapshot_contains_all_three_objects_and_is_immutable():
    namespace = _load_demo_helpers()
    task = namespace["_resolve_demo_task"](demo_task="stack")
    snapshot = namespace["DemoResetSnapshot"](
        joint_names=("arm",),
        joint_positions=np.asarray([0.0]),
        default_joint_positions=np.asarray([0.0]),
        robot_base_world=Pose.identity(),
        objects_world={
            color: Pose(np.asarray([index, 0.0, 0.8]), np.asarray([0.0, 0.0, 0.0, 1.0]))
            for index, color in enumerate(("red", "yellow", "green"))
        },
        object_sizes_world_m={
            color: namespace["DEMO_PUBLIC_BLOCK_SIZE_M"]
            for color in ("red", "yellow", "green")
        },
        marker_world=Pose.identity(),
        marker_size_world_m=namespace["DEMO_MARKER_SIZE_M"],
        target_world=Pose.identity(),
        selected_object="red",
        task=task,
    )
    assert isinstance(snapshot.objects_world, MappingProxyType)
    assert set(snapshot.objects_world) == {"red", "yellow", "green"}
    with pytest.raises(TypeError):
        snapshot.objects_world["red"] = snapshot.objects_world["yellow"]
    with pytest.raises(ValueError):
        snapshot.joint_positions[0] = 1.0

    report = namespace["_validate_demo_reset_geometry"](snapshot)
    assert report["status"] == "PASS"

    overlapping = namespace["DemoResetSnapshot"](
        joint_names=("arm",),
        joint_positions=np.asarray([0.0]),
        default_joint_positions=np.asarray([0.0]),
        robot_base_world=Pose.identity(),
        objects_world={
            "red": Pose(np.asarray([0.0, 0.0, 0.8]), np.asarray([0.0, 0.0, 0.0, 1.0])),
            "yellow": Pose(np.asarray([0.01, 0.0, 0.8]), np.asarray([0.0, 0.0, 0.0, 1.0])),
            "green": Pose(np.asarray([0.2, 0.0, 0.8]), np.asarray([0.0, 0.0, 0.0, 1.0])),
        },
        object_sizes_world_m={
            color: namespace["DEMO_PUBLIC_BLOCK_SIZE_M"]
            for color in ("red", "yellow", "green")
        },
        marker_world=Pose(np.asarray([0.5, 0.0, 0.8]), np.asarray([0.0, 0.0, 0.0, 1.0])),
        marker_size_world_m=namespace["DEMO_MARKER_SIZE_M"],
        target_world=Pose(np.asarray([0.4, 0.0, 0.8]), np.asarray([0.0, 0.0, 0.0, 1.0])),
        selected_object="red",
        task=namespace["_resolve_demo_task"](demo_task="relative-place"),
    )
    with pytest.raises(RuntimeError, match="unsafe public demo reset geometry"):
        namespace["_validate_demo_reset_geometry"](overlapping)


def _clearance_snapshot(namespace, *, green_position=(0.60, 0.60, 0.825)):
    return namespace["DemoResetSnapshot"](
        joint_names=("x", "y", "z"),
        joint_positions=np.zeros(3),
        default_joint_positions=np.zeros(3),
        robot_base_world=Pose.identity(),
        objects_world={
            "red": Pose(np.asarray((0.0, 0.0, 0.825)), np.asarray((0.0, 0.0, 0.0, 1.0))),
            "yellow": Pose(np.asarray((-0.60, 0.60, 0.825)), np.asarray((0.0, 0.0, 0.0, 1.0))),
            "green": Pose(np.asarray(green_position), np.asarray((0.0, 0.0, 0.0, 1.0))),
        },
        object_sizes_world_m={
            color: namespace["DEMO_PUBLIC_BLOCK_SIZE_M"]
            for color in ("red", "yellow", "green")
        },
        marker_world=Pose(np.asarray((0.3, 0.0, 0.802)), np.asarray((0.0, 0.0, 0.0, 1.0))),
        marker_size_world_m=namespace["DEMO_MARKER_SIZE_M"],
        target_world=Pose(np.asarray((0.3, 0.0, 0.825)), np.asarray((0.0, 0.0, 0.0, 1.0))),
        selected_object="red",
        task=namespace["_resolve_demo_task"](demo_task="relative-place"),
    )


class _ClearanceIK:
    def q_from_named_positions(self, names, positions):
        del names
        return np.asarray(positions, dtype=np.float64)

    def frame_pose(self, q):
        return Pose(np.asarray(q[:3]), np.asarray((0.0, 0.0, 0.0, 1.0)))


def _clearance_trajectory(phases, positions):
    return type(
        "Trajectory",
        (),
        {
            "joint_names": ("x", "y", "z"),
            "absolute_targets": np.asarray(positions, dtype=np.float64),
            "phases": tuple(phases),
        },
    )()


def _clearance_profile(grasp_offset, place_offset):
    return type(
        "Profile",
        (),
        {
            "grasp_wrist_offset_world": grasp_offset,
            "place_wrist_offset_world": place_offset,
        },
    )()


def test_clearance_phase_sets_partition_known_compiler_labels():
    namespace = _load_demo_helpers()
    phase_sets = (
        namespace["DEMO_PRELOAD_PHASES"],
        namespace["DEMO_LOADED_PHASES"],
        namespace["DEMO_RELEASE_SETTLE_PHASES"],
        namespace["DEMO_POST_RELEASE_PHASES"],
    )
    assert all(first.isdisjoint(second) for index, first in enumerate(phase_sets) for second in phase_sets[index + 1 :])
    assert set().union(*phase_sets) == namespace["DEMO_KNOWN_PHASES"]


def test_preload_phase_ignores_future_place_offset_ghost():
    namespace = _load_demo_helpers()
    snapshot = _clearance_snapshot(namespace)
    trajectory = _clearance_trajectory(
        ("staging", "lift_settle", "transport"),
        ((0.0, 0.0, 0.825), (0.0, 0.0, 1.20), (0.10, 0.0, 1.20)),
    )
    profile = _clearance_profile((0.0, 0.0, 0.0), (0.0, 0.0, 0.075))

    def table_if_envelope_drops_below_future_ghost(minimum, maximum):
        del maximum
        return ("/World/packingTable",) if minimum[2] < 0.75 else ()

    report = namespace["_validate_demo_swept_clearance"](
        _ClearanceIK(),
        trajectory,
        snapshot,
        profile,
        table_if_envelope_drops_below_future_ghost,
    )
    assert report["status"] == "PASS"


def test_loaded_phase_keeps_conservative_dual_offset_and_fails_same_geometry():
    namespace = _load_demo_helpers()
    snapshot = _clearance_snapshot(namespace)
    trajectory = _clearance_trajectory(
        ("staging", "lift_settle", "transport"),
        ((0.0, 0.0, 1.20), (0.0, 0.0, 0.825), (0.10, 0.0, 1.20)),
    )
    profile = _clearance_profile((0.0, 0.0, 0.0), (0.0, 0.0, 0.075))

    def table_if_envelope_drops_below_loaded_ghost(minimum, maximum):
        del maximum
        return ("/World/packingTable",) if minimum[2] < 0.75 else ()

    with pytest.raises(RuntimeError, match="unsafe public demo swept clearance"):
        namespace["_validate_demo_swept_clearance"](
            _ClearanceIK(),
            trajectory,
            snapshot,
            profile,
            table_if_envelope_drops_below_loaded_ghost,
        )


def test_open_gripper_uses_loaded_dual_offset_union():
    namespace = _load_demo_helpers()
    snapshot = _clearance_snapshot(namespace)
    trajectory = _clearance_trajectory(
        ("staging", "lift_settle", "transport", "open_gripper"),
        ((0.0, 0.0, 1.20), (0.0, 0.0, 1.20), (0.10, 0.0, 1.20), (0.0, 0.0, 1.0)),
    )
    profile = _clearance_profile((0.0, 0.0, -0.075), (0.0, 0.0, 0.075))
    envelopes = []

    report = namespace["_validate_demo_swept_clearance"](
        _ClearanceIK(),
        trajectory,
        snapshot,
        profile,
        lambda minimum, maximum: (envelopes.append((minimum.copy(), maximum.copy())) or ()),
    )

    assert report["status"] == "PASS"
    open_minimum, open_maximum = envelopes[-1]
    np.testing.assert_allclose(open_minimum, (-0.025, -0.025, 0.90))
    np.testing.assert_allclose(open_maximum, (0.025, 0.025, 1.10))


def test_release_settle_uses_place_offset_only():
    namespace = _load_demo_helpers()
    snapshot = _clearance_snapshot(namespace)
    trajectory = _clearance_trajectory(
        ("staging", "lift_settle", "transport", "release_settle"),
        ((0.0, 0.0, 1.20), (0.0, 0.0, 1.20), (0.10, 0.0, 1.20), (0.0, 0.0, 1.0)),
    )
    profile = _clearance_profile((0.0, 0.0, -0.075), (0.0, 0.0, 0.075))
    envelopes = []

    report = namespace["_validate_demo_swept_clearance"](
        _ClearanceIK(),
        trajectory,
        snapshot,
        profile,
        lambda minimum, maximum: (envelopes.append((minimum.copy(), maximum.copy())) or ()),
    )

    assert report["status"] == "PASS"
    release_minimum, release_maximum = envelopes[-1]
    np.testing.assert_allclose(release_minimum, (-0.025, -0.025, 0.90))
    np.testing.assert_allclose(release_maximum, (0.025, 0.025, 1.025))


def test_post_release_phase_uses_hand_only_envelope():
    namespace = _load_demo_helpers()
    snapshot = _clearance_snapshot(namespace)
    trajectory = _clearance_trajectory(
        ("staging", "lift_settle", "transport", "retreat"),
        ((0.0, 0.0, 1.20), (0.0, 0.0, 1.20), (0.10, 0.0, 1.20), (0.0, 0.0, 1.0)),
    )
    profile = _clearance_profile((0.0, 0.0, -0.075), (0.0, 0.0, 0.075))
    envelopes = []

    report = namespace["_validate_demo_swept_clearance"](
        _ClearanceIK(),
        trajectory,
        snapshot,
        profile,
        lambda minimum, maximum: (envelopes.append((minimum.copy(), maximum.copy())) or ()),
    )

    assert report["status"] == "PASS"
    retreat_minimum, retreat_maximum = envelopes[-1]
    np.testing.assert_allclose(retreat_minimum, (-0.025, -0.025, 0.975))
    np.testing.assert_allclose(retreat_maximum, (0.025, 0.025, 1.025))


def test_retreat_accepts_only_a_contiguous_table_contact_exit_prefix():
    namespace = _load_demo_helpers()
    snapshot = _clearance_snapshot(namespace)
    trajectory = _clearance_trajectory(
        ("staging", "lift_settle", "transport", "retreat", "retreat"),
        (
            (0.0, 0.0, 1.20),
            (0.0, 0.0, 1.20),
            (0.10, 0.0, 1.20),
            (0.0, 0.0, 0.90),
            (0.0, 0.0, 1.10),
        ),
    )
    profile = _clearance_profile((0.0, 0.0, 0.0), (0.0, 0.0, 0.075))

    def table_at_retreat_height(minimum, maximum):
        del maximum
        return ("/World/packingTable",) if minimum[2] < 1.0 else ()

    report = namespace["_validate_demo_swept_clearance"](
        _ClearanceIK(), trajectory, snapshot, profile, table_at_retreat_height
    )
    assert report["status"] == "PASS"


def test_retreat_table_reentry_after_clearance_fails_closed():
    namespace = _load_demo_helpers()
    snapshot = _clearance_snapshot(namespace)
    trajectory = _clearance_trajectory(
        ("staging", "lift_settle", "transport", "retreat", "retreat", "retreat"),
        (
            (0.0, 0.0, 1.20),
            (0.0, 0.0, 1.20),
            (0.10, 0.0, 1.20),
            (0.0, 0.0, 0.90),
            (0.0, 0.0, 1.10),
            (0.0, 0.0, 0.90),
        ),
    )
    profile = _clearance_profile((0.0, 0.0, 0.0), (0.0, 0.0, 0.075))

    def table_at_retreat_height(minimum, maximum):
        del maximum
        return ("/World/packingTable",) if minimum[2] < 1.0 else ()

    with pytest.raises(RuntimeError, match="unsafe public demo swept clearance"):
        namespace["_validate_demo_swept_clearance"](
            _ClearanceIK(), trajectory, snapshot, profile, table_at_retreat_height
        )


def test_unknown_clearance_phase_fails_before_iteration():
    namespace = _load_demo_helpers()
    snapshot = _clearance_snapshot(namespace)
    trajectory = _clearance_trajectory(
        ("new_phase", "lift_settle", "transport"),
        ((0.0, 0.0, 1.20), (0.0, 0.0, 1.20), (0.10, 0.0, 1.20)),
    )
    profile = _clearance_profile((0.0, 0.0, 0.0), (0.0, 0.0, 0.075))

    with pytest.raises(RuntimeError, match="unknown phases"):
        namespace["_validate_demo_swept_clearance"](
            _ClearanceIK(), trajectory, snapshot, profile, lambda minimum, maximum: ()
        )


def test_reset_time_swept_clearance_checks_all_samples_before_policy():
    namespace = _load_demo_helpers()
    snapshot = _clearance_snapshot(namespace)
    trajectory = type(
        "Trajectory",
        (),
        {
            "joint_names": ("x", "y", "z"),
            "absolute_targets": np.asarray(((0.0, 0.0, 1.025), (0.3, 0.0, 0.905))),
            "phases": ("lift_settle", "transport"),
        },
    )()
    profile = type("Profile", (), {"grasp_wrist_offset_world": (0.0, 0.0, 0.0)})()

    report = namespace["_validate_demo_swept_clearance"](
        _ClearanceIK(), trajectory, snapshot, profile, lambda minimum, maximum: ()
    )

    assert report["status"] == "PASS"
    assert report["trajectory_samples"] == 2
    assert report["minimum_airborne_object_clearance_m"] >= 0.08 - 1.0e-12
    assert report["checks_performed"]["physx_scene_queries"] == 2


def test_reset_time_swept_clearance_fails_on_distractor_or_missing_static_meshes():
    namespace = _load_demo_helpers()
    snapshot = _clearance_snapshot(namespace, green_position=(0.3, 0.0, 0.905))
    trajectory = type(
        "Trajectory",
        (),
        {
            "joint_names": ("x", "y", "z"),
            "absolute_targets": np.asarray(((0.0, 0.0, 1.025), (0.3, 0.0, 0.905))),
            "phases": ("lift_settle", "transport"),
        },
    )()
    profile = type("Profile", (), {"grasp_wrist_offset_world": (0.0, 0.0, 0.0)})()

    with pytest.raises(RuntimeError, match="unsafe public demo swept clearance"):
        namespace["_validate_demo_swept_clearance"](
            _ClearanceIK(), trajectory, snapshot, profile, lambda minimum, maximum: ()
        )

    with pytest.raises(RuntimeError, match="physx_scene_overlap"):
        namespace["_validate_demo_swept_clearance"](
            _ClearanceIK(),
            trajectory,
            _clearance_snapshot(namespace),
            profile,
            lambda minimum, maximum: ("/World/envs/env_0/Room/wall",),
        )


@dataclass
class _IK:
    active_joint_names = ("arm",)

    def __post_init__(self):
        self.solve_calls = []

    def q_from_named_positions(self, names, positions):
        del names
        return np.asarray([positions[0]])

    def named_positions_from_q(self, q, names):
        del names
        return {"arm": float(q[0])}

    def frame_pose(self, q):
        del q
        return Pose(np.asarray([0.0, 0.0, 0.4]), np.asarray([0.0, 0.0, 0.0, 1.0]))

    def solve(self, name, target, seed):
        self.solve_calls.append(name)
        result = np.asarray(seed, dtype=np.float64).copy()
        result[0] = target.position[0]
        return IKSolveReport(q=result, iterations=1, residual=1.0e-6)


def test_demo_holds_are_fixed_phases_and_observation_invariant():
    names = ("arm", "right_hand_Joint1_1", "right_hand_Joint2_1")
    snapshot = ResetSnapshot(
        joint_names=names,
        joint_positions=np.zeros(3),
        default_joint_positions=np.zeros(3),
        robot_base_world=Pose.identity(),
        object_world=Pose(np.asarray([0.3, 0.0, 0.75]), np.asarray([0.0, 0.0, 0.0, 1.0])),
        target_world=Pose(np.asarray([0.2, 0.2, 0.75]), np.asarray([0.0, 0.0, 0.0, 1.0])),
    )
    ik = _IK()
    config = PickPlaceConfig(
        fps=10,
        grasp_wrist_offset_world=(0.0, 0.0, 0.0),
        preclose_settle_enabled=True,
        lift_settle_enabled=True,
    )
    trajectory, _ = ResetTimePickPlacePlanner(ik, config).build(snapshot)
    assert "preclose_settle" in trajectory.phases
    assert "lift_settle" in trajectory.phases
    calls = len(ik.solve_calls)
    policy = OpenLoopPolicy(trajectory)
    first = [policy.act({"random": np.random.randn(4)}) for _ in range(trajectory.steps)]
    assert len(ik.solve_calls) == calls
    assert policy.done
    policy.reset()
    second = [policy.act({"different": np.random.randn(8)}) for _ in range(trajectory.steps)]
    np.testing.assert_array_equal(np.stack(first), np.stack(second))


def test_default_script_targets_public_stack_and_never_supports_headless_demo():
    source = SCRIPT.read_text()
    tree = ast.parse(source, filename=str(SCRIPT))
    task_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--task"
    )
    default = next(keyword.value for keyword in task_call.keywords if keyword.arg == "default")
    assert isinstance(default, ast.Name)
    namespace = _load_demo_helpers()
    assert namespace[default.id] == "Isaac-Stack-RgyBlock-G129-Dex1-Joint"
    assert "headless mode is disabled" in source
    assert "sensor_extrinsics_modified" in source


def test_rollout_end_flags_are_diagnostic_and_do_not_break_frozen_playback():
    source = SCRIPT.read_text()
    tree = ast.parse(source, filename=str(SCRIPT))
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    rollout_loops = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.While)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Attribute)
        and node.test.operand.attr == "done"
    ]
    assert len(rollout_loops) == 1
    assert not any(isinstance(node, ast.Break) for node in ast.walk(rollout_loops[0]))
    assert "rollout_end_flags" in source
    assert '"success"] = False' in source
