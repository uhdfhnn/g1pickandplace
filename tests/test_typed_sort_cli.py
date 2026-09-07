from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from g1pickplace import PickPlaceConfig, Pose
from g1pickplace.offline_ik import IKSolveReport
from g1pickplace.trajectory import OpenLoopPolicy


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"


def _load_typed_cli_helpers():
    """Load selector/snapshot helpers without importing Isaac Sim."""

    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    wanted_assignments = {
        "VALIDATED_FIXED_SEED_PRESET",
        "MINIMAL_DEMO_PRESET",
        "VALIDATED_FIXED_SEED_PRESET_OPTION_FIELDS",
        "TYPED_SORT_DEMO_OPTION_FIELDS",
        "TYPED_SORT_OBJECT_TYPES",
        "TYPED_SORT_OBJECT_SOURCE_POSITIONS_WORLD_M",
        "TYPED_SORT_TARGET_POSITIONS_WORLD_M",
        "TYPED_SORT_TARGET_LABEL_BY_TYPE",
        "TYPED_SORT_OBJECT_LABEL_BY_TYPE",
        "TYPED_SORT_BASE_OBJECT_MASS_KG",
        "TYPED_SORT_BASE_STATIC_FRICTION",
        "TYPED_SORT_BASE_DYNAMIC_FRICTION",
        "TYPED_SORT_PACKAGE_SIZE_M",
        "TYPED_SORT_PACKAGE_SOURCE_POSITION_WORLD_M",
        "TYPED_SORT_PACKAGE_TARGET_POSITION_WORLD_M",
        "TYPED_SORT_PACKAGE_MASS_KG",
        "TYPED_SORT_PACKAGE_STATIC_FRICTION",
        "TYPED_SORT_PACKAGE_DYNAMIC_FRICTION",
        "TYPED_SORT_PACKAGE_COLOR_RGB",
        "TYPED_SORT_CUBOID_TRANSPORT_DURATION_S",
        "TYPED_SORT_TARGET_SUPPORT_SIZE_M_BY_TYPE",
        "MINIMAL_DEMO_GOAL_SUPPORT_SIZE_M",
        "TYPED_SORT_EVALUATION_TARGET_HALF_EXTENT_XY_BY_TYPE",
        "TYPED_SORT_EVALUATION_HEIGHT_TOLERANCE_M",
        "TYPED_SORT_EVALUATION_MAXIMUM_SPEED_MPS",
    }
    wanted_functions = {
        "_preset_values_equal",
        "_typed_sort_demo_explicit_fields",
        "_apply_typed_sort_demo",
        "_build_typed_sort_trajectory",
        "_typed_sort_material_calibration",
        "_typed_sort_descend_duration",
        "_typed_sort_transport_duration",
        "_typed_sort_target_support_size",
    }
    wanted_classes = {"TypedSortResetSnapshot"}
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
        "dataclass": dataclass,
        "replace": replace,
        "Mapping": dict,
        "MappingProxyType": MappingProxyType,
        "Pose": Pose,
        "JointTrajectory": __import__(
            "g1pickplace.trajectory", fromlist=["JointTrajectory"]
        ).JointTrajectory,
        "ResetSnapshot": __import__(
            "g1pickplace.planner", fromlist=["ResetSnapshot"]
        ).ResetSnapshot,
        "ResetTimePickPlacePlanner": __import__(
            "g1pickplace.planner", fromlist=["ResetTimePickPlacePlanner"]
        ).ResetTimePickPlacePlanner,
        "np": np,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace


def _call_with_argument(tree: ast.AST, option: str) -> ast.Call:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == option
    )


def test_typed_cli_exposes_only_cuboid_package_and_all() -> None:
    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    call = _call_with_argument(tree, "--typed-sort-demo")
    choices = next(keyword.value for keyword in call.keywords if keyword.arg == "choices")
    assert ast.literal_eval(choices) == ("cuboid", "package", "all")
    # The replacement is complete: no stale type, primitive, label, or
    # compensation name may remain in functional script/config text.
    script_text = SCRIPT.read_text().lower()
    assert "cylinder" not in script_text
    assert "place_program_offset_world_m" not in script_text


def test_typed_lanes_are_distinct_and_package_type_mapped() -> None:
    namespace = _load_typed_cli_helpers()
    source = namespace["TYPED_SORT_OBJECT_SOURCE_POSITIONS_WORLD_M"]
    targets = namespace["TYPED_SORT_TARGET_POSITIONS_WORLD_M"]

    assert namespace["TYPED_SORT_OBJECT_TYPES"] == ("cuboid", "package")
    assert source == {
        "cuboid": (-4.25, -4.03, 0.814),
        "package": (-4.25, -4.10, 0.819),
    }
    assert targets == {
        "cuboid": (-4.35, -4.03, 0.824),
        "package": (-4.35, -4.10, 0.829),
    }
    assert namespace["TYPED_SORT_PACKAGE_SOURCE_POSITION_WORLD_M"] == (
        -4.25,
        -4.10,
        0.819,
    )
    assert namespace["TYPED_SORT_PACKAGE_TARGET_POSITION_WORLD_M"] == (
        -4.35,
        -4.10,
        0.829,
    )
    assert namespace["TYPED_SORT_TARGET_LABEL_BY_TYPE"] == {
        "cuboid": "green",
        "package": "yellow",
    }
    assert namespace["TYPED_SORT_OBJECT_LABEL_BY_TYPE"] == {
        "cuboid": "red_cuboid",
        "package": "blue_package",
    }


def test_package_uses_public_analytic_cuboid_cfg_and_exact_material() -> None:
    namespace = _load_typed_cli_helpers()
    assert namespace["TYPED_SORT_PACKAGE_SIZE_M"] == (0.04, 0.03, 0.05)
    assert namespace["TYPED_SORT_PACKAGE_MASS_KG"] == 0.10
    assert namespace["TYPED_SORT_PACKAGE_STATIC_FRICTION"] == 2.0
    assert namespace["TYPED_SORT_PACKAGE_DYNAMIC_FRICTION"] == 1.5

    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_typed_sort_package_cfg"
    )
    cuboid_calls = [
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "CuboidCfg"
    ]
    assert len(cuboid_calls) == 1
    call = cuboid_calls[0]
    size_keyword = next(keyword for keyword in call.keywords if keyword.arg == "size")
    assert isinstance(size_keyword.value, ast.Name)
    assert size_keyword.value.id == "TYPED_SORT_PACKAGE_SIZE_M"

    rigid_calls = [
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RigidObjectCfg"
    ]
    assert len(rigid_calls) == 1
    init_state = next(
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "InitialStateCfg"
    )
    rotation = next(keyword.value for keyword in init_state.keywords if keyword.arg == "rot")
    assert ast.literal_eval(rotation) == (1.0, 0.0, 0.0, 0.0)

    mass_call = next(
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "MassPropertiesCfg"
    )
    mass = next(keyword.value for keyword in mass_call.keywords if keyword.arg == "mass")
    assert isinstance(mass, ast.Name)
    assert mass.id == "TYPED_SORT_PACKAGE_MASS_KG"
    visual_call = next(
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "PreviewSurfaceCfg"
    )
    diffuse_color = next(
        keyword.value for keyword in visual_call.keywords if keyword.arg == "diffuse_color"
    )
    assert isinstance(diffuse_color, ast.Name)
    assert diffuse_color.id == "TYPED_SORT_PACKAGE_COLOR_RGB"
    material_call = next(
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "RigidBodyMaterialCfg"
    )
    material_keywords = {keyword.arg: keyword.value for keyword in material_call.keywords}
    assert isinstance(material_keywords["static_friction"], ast.Name)
    assert material_keywords["static_friction"].id == "TYPED_SORT_PACKAGE_STATIC_FRICTION"
    assert isinstance(material_keywords["dynamic_friction"], ast.Name)
    assert material_keywords["dynamic_friction"].id == "TYPED_SORT_PACKAGE_DYNAMIC_FRICTION"
    assert namespace["TYPED_SORT_PACKAGE_COLOR_RGB"] == (0.05, 0.25, 0.95)


def test_typed_material_transport_and_support_calibration() -> None:
    namespace = _load_typed_cli_helpers()
    assert namespace["_typed_sort_material_calibration"]("cuboid") == (0.15, 2.0, 1.5)
    assert namespace["_typed_sort_material_calibration"]("package") == (0.10, 2.0, 1.5)
    assert namespace["_typed_sort_descend_duration"]("cuboid", 5.0) == 5.0
    assert namespace["_typed_sort_descend_duration"]("package", 5.0) == 5.0
    assert namespace["_typed_sort_transport_duration"]("cuboid") == 2.0
    assert namespace["_typed_sort_transport_duration"]("package") == 2.0
    assert namespace["TYPED_SORT_TARGET_SUPPORT_SIZE_M_BY_TYPE"] == {
        "cuboid": (0.08, 0.06, 0.01),
        "package": (0.08, 0.06, 0.01),
    }
    assert namespace["_typed_sort_target_support_size"]("cuboid") == (
        0.08,
        0.06,
        0.01,
    )
    assert namespace["_typed_sort_target_support_size"]("package") == (
        0.08,
        0.06,
        0.01,
    )
    assert namespace["TYPED_SORT_EVALUATION_TARGET_HALF_EXTENT_XY_BY_TYPE"] == {
        "cuboid": (0.04, 0.03),
        "package": (0.04, 0.03),
    }
    assert namespace["TYPED_SORT_EVALUATION_HEIGHT_TOLERANCE_M"] == 0.01
    assert namespace["TYPED_SORT_EVALUATION_MAXIMUM_SPEED_MPS"] == 0.01
    for helper_name, helper_args in (
        ("_typed_sort_material_calibration", ("sphere",)),
        ("_typed_sort_descend_duration", ("sphere", 5.0)),
        ("_typed_sort_transport_duration", ("sphere",)),
        ("_typed_sort_target_support_size", ("sphere",)),
    ):
        with pytest.raises(ValueError):
            namespace[helper_name](*helper_args)


def test_typed_evaluation_wires_strict_physical_support_thresholds() -> None:
    namespace = _load_typed_cli_helpers()
    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    typed_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "evaluate_pick_place"
        and any(keyword.arg == "target_half_extent_xy" for keyword in node.keywords)
    )
    keywords = {keyword.arg: keyword.value for keyword in typed_call.keywords}
    extent_expression = keywords["target_half_extent_xy"]
    assert isinstance(extent_expression, ast.Subscript)
    assert isinstance(extent_expression.value, ast.Name)
    assert extent_expression.value.id == "TYPED_SORT_EVALUATION_TARGET_HALF_EXTENT_XY_BY_TYPE"
    assert isinstance(extent_expression.slice, ast.Name)
    assert extent_expression.slice.id == "object_type"
    assert isinstance(keywords["height_tolerance_m"], ast.Name)
    assert keywords["height_tolerance_m"].id == "TYPED_SORT_EVALUATION_HEIGHT_TOLERANCE_M"
    assert isinstance(keywords["maximum_speed_mps"], ast.Name)
    assert keywords["maximum_speed_mps"].id == "TYPED_SORT_EVALUATION_MAXIMUM_SPEED_MPS"
    assert namespace["TYPED_SORT_EVALUATION_TARGET_HALF_EXTENT_XY_BY_TYPE"] == {
        "cuboid": (0.04, 0.03),
        "package": (0.04, 0.03),
    }


def _typed_args(namespace, **overrides):
    preset = namespace["MINIMAL_DEMO_PRESET"]
    values = {
        "typed_sort_demo": "all",
        **{
            field: value
            for field, value in preset.items()
            if field != "minimal_demo_scene"
        },
        "minimal_demo_scene": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_typed_selector_applies_compact_desk_preset() -> None:
    namespace = _load_typed_cli_helpers()
    args = _typed_args(namespace)
    namespace["_apply_typed_sort_demo"](
        args,
        namespace["_typed_sort_demo_explicit_fields"]([]),
    )

    assert args.minimal_demo_scene is True
    assert args.fixed_object_reset is True
    assert args.target_position == (-4.35, -4.03, 0.824)
    assert args.grasp_wrist_offset_world == (0.025, 0.165, 0.0)
    assert args.descend_duration_s == 5.0
    assert args.transport_duration_s == 2.0


def test_typed_selector_rejects_conflicting_calibration() -> None:
    namespace = _load_typed_cli_helpers()
    args = _typed_args(namespace, target_position=(-4.15, -4.03, 0.84))

    with pytest.raises(ValueError, match="conflicts"):
        namespace["_apply_typed_sort_demo"](
            args,
            namespace["_typed_sort_demo_explicit_fields"](
                ["--target-position=-4.15,-4.03,0.84"]
            ),
        )


def test_typed_snapshot_freezes_all_selected_objects_and_targets() -> None:
    namespace = _load_typed_cli_helpers()
    snapshot = namespace["TypedSortResetSnapshot"](
        joint_names=("arm",),
        joint_positions=np.asarray([0.0]),
        default_joint_positions=np.asarray([0.0]),
        robot_base_world=Pose.identity(),
        objects_world={
            "cuboid": Pose(
                np.asarray([1.0, 2.0, 3.0]),
                np.asarray([0.0, 0.0, 0.0, 1.0]),
            ),
            "package": Pose(
                np.asarray([4.0, 5.0, 6.0]),
                np.asarray([0.0, 0.0, 0.0, 1.0]),
            ),
        },
        targets_world={
            "cuboid": Pose(
                np.asarray([7.0, 8.0, 9.0]),
                np.asarray([0.0, 0.0, 0.0, 1.0]),
            ),
            "package": Pose(
                np.asarray([10.0, 11.0, 12.0]),
                np.asarray([0.0, 0.0, 0.0, 1.0]),
            ),
        },
    )

    assert isinstance(snapshot.objects_world, MappingProxyType)
    assert isinstance(snapshot.targets_world, MappingProxyType)
    assert set(snapshot.objects_world) == {"cuboid", "package"}
    assert set(snapshot.targets_world) == {"cuboid", "package"}
    with pytest.raises(TypeError):
        snapshot.objects_world["cuboid"] = snapshot.objects_world["package"]
    with pytest.raises(TypeError):
        snapshot.targets_world["cuboid"] = snapshot.targets_world["package"]


@dataclass
class _CountingIK:
    active_joint_names = ("arm_a", "arm_b")

    def __post_init__(self):
        self.solve_calls = []
        self.solve_target_positions = []

    def q_from_named_positions(self, joint_names, positions):
        by_name = dict(zip(joint_names, positions, strict=True))
        return np.asarray([by_name["arm_a"], by_name["arm_b"]], dtype=np.float64)

    def named_positions_from_q(self, q, joint_names):
        del joint_names
        return {"arm_a": float(q[0]), "arm_b": float(q[1])}

    def frame_pose(self, q):
        del q
        return Pose(
            np.asarray([0.2, 0.0, 0.4]),
            np.asarray([0.0, 0.0, 0.0, 1.0]),
        )

    def solve(self, waypoint_name, target, seed_q):
        self.solve_calls.append(waypoint_name)
        self.solve_target_positions.append((waypoint_name, target.position.copy()))
        q = np.asarray(seed_q, dtype=np.float64).copy()
        q[0] = target.position[0]
        q[1] = target.position[2]
        return IKSolveReport(q=q, iterations=2, residual=1.0e-6)


def test_all_builds_cuboid_then_package_before_open_loop_playback() -> None:
    namespace = _load_typed_cli_helpers()
    args = SimpleNamespace(
        typed_sort_configuration={
            "selected_types": ("cuboid", "package"),
            "scene_names_by_type": {"cuboid": "object", "package": "package"},
        }
    )
    namespace["args"] = args
    snapshot = namespace["TypedSortResetSnapshot"](
        joint_names=("arm_a", "arm_b", "right_hand_Joint1_1", "right_hand_Joint2_1"),
        joint_positions=np.zeros(4),
        default_joint_positions=np.zeros(4),
        robot_base_world=Pose.identity(),
        objects_world={
            "cuboid": Pose(
                np.asarray([0.35, 0.0, 0.75]),
                np.asarray([0.0, 0.0, 0.0, 1.0]),
            ),
            "package": Pose(
                np.asarray([0.40, 0.0, 0.75]),
                np.asarray([0.0, 0.0, 0.0, 1.0]),
            ),
        },
        targets_world={
            "cuboid": Pose(
                np.asarray([0.2, 0.2, 0.75]),
                np.asarray([0.0, 0.0, 0.0, 1.0]),
            ),
            "package": Pose(
                np.asarray([0.3, 0.2, 0.75]),
                np.asarray([0.0, 0.0, 0.0, 1.0]),
            ),
        },
    )
    ik = _CountingIK()
    trajectory, diagnostics = namespace["_build_typed_sort_trajectory"](
        ik,
        snapshot,
        PickPlaceConfig(
            fps=10,
            grasp_wrist_offset_world=(0.0, 0.0, 0.0),
            descend_duration_s=5.0,
            transport_duration_s=2.0,
        ),
    )

    assert ik.solve_calls == [
        "pregrasp",
        "grasp",
        "lift",
        "preplace",
        "place",
    ] * 2
    assert set(diagnostics.waypoint_residuals) == {
        f"{object_type}/{waypoint}"
        for object_type in ("cuboid", "package")
        for waypoint in ("pregrasp", "grasp", "lift", "preplace", "place")
    }
    assert trajectory.phases[0].startswith("typed_sort/cuboid/")
    assert any(phase.startswith("typed_sort/package/") for phase in trajectory.phases)
    assert trajectory.phases.count("typed_sort/cuboid/descend_to_grasp") == 50
    assert trajectory.phases.count("typed_sort/cuboid/descend_to_place") == 50
    assert trajectory.phases.count("typed_sort/package/descend_to_grasp") == 50
    assert trajectory.phases.count("typed_sort/package/descend_to_place") == 50
    assert trajectory.phases.count("typed_sort/cuboid/transport") == 20
    assert trajectory.phases.count("typed_sort/package/transport") == 20
    assert "typed_sort/package/lift_settle" not in trajectory.phases
    assert "typed_sort/package/preplace_settle" not in trajectory.phases

    # Rebuild the cuboid-only program from the same reset snapshot and compare
    # the complete segment.  The second typed program must not perturb the
    # first frozen segment or its physical target.
    combined_cuboid_mask = np.asarray(
        [phase.startswith("typed_sort/cuboid/") for phase in trajectory.phases],
        dtype=bool,
    )
    combined_cuboid_targets = trajectory.absolute_targets[combined_cuboid_mask].copy()
    combined_cuboid_actions = trajectory.env_actions[combined_cuboid_mask].copy()
    namespace["args"].typed_sort_configuration = {
        "selected_types": ("cuboid",),
        "scene_names_by_type": {"cuboid": "object"},
    }
    cuboid_only, _ = namespace["_build_typed_sort_trajectory"](
        _CountingIK(),
        snapshot,
        PickPlaceConfig(
            fps=10,
            grasp_wrist_offset_world=(0.0, 0.0, 0.0),
            descend_duration_s=5.0,
            transport_duration_s=2.0,
        ),
    )
    assert np.array_equal(cuboid_only.absolute_targets, combined_cuboid_targets)
    assert np.array_equal(cuboid_only.env_actions, combined_cuboid_actions)

    physical_cuboid_target = snapshot.targets_world["cuboid"].position.copy()
    physical_package_target = snapshot.targets_world["package"].position.copy()
    cuboid_program_targets = [
        position
        for waypoint, position in ik.solve_target_positions[:5]
        if waypoint in {"preplace", "place"}
    ]
    package_program_targets = [
        position
        for waypoint, position in ik.solve_target_positions[5:]
        if waypoint in {"preplace", "place"}
    ]
    assert all(
        np.array_equal(position[:2], physical_cuboid_target[:2])
        for position in cuboid_program_targets
    )
    assert all(
        np.array_equal(position[:2], physical_package_target[:2])
        for position in package_program_targets
    )
    cuboid_place_target = next(
        position
        for waypoint, position in ik.solve_target_positions[:5]
        if waypoint == "place"
    )
    package_place_target = next(
        position
        for waypoint, position in ik.solve_target_positions[5:]
        if waypoint == "place"
    )
    assert np.array_equal(cuboid_place_target, physical_cuboid_target)
    assert np.array_equal(package_place_target, physical_package_target)
    assert np.array_equal(snapshot.targets_world["cuboid"].position, physical_cuboid_target)
    assert np.array_equal(snapshot.targets_world["package"].position, physical_package_target)

    # The planner has completed all reset-time IK before policy construction;
    # adversarial observations cannot trigger more solves or alter actions.
    calls_after_build = len(ik.solve_calls)
    policy = OpenLoopPolicy(trajectory)
    actions = []
    for _ in range(trajectory.steps):
        actions.append(policy.act({"adversarial_object_pose": np.random.randn(3)}))
    assert len(ik.solve_calls) == calls_after_build
    assert policy.done
    assert len(actions) == trajectory.steps


def test_typed_planner_uses_physical_target_without_offset() -> None:
    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_typed_sort_trajectory"
    )
    planner_target_assignment = next(
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "planner_target"
            for target in node.targets
        )
    )
    assert isinstance(planner_target_assignment.value, ast.Name)
    assert planner_target_assignment.value.id == "physical_target"
