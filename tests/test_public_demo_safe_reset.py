from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"


def _load_safe_reset_helper():
    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "MINIMAL_DEMO_ROBOT_JOINT_SEED_RAD"
            for target in node.targets
        )
    ]
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_safe_reset_posture_config"
    )
    namespace = {"Any": object, "Mapping": Mapping}
    exec(
        compile(ast.Module(body=assignments + [helper], type_ignores=[]), str(SCRIPT), "exec"),
        namespace,
    )
    return namespace


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_public_stack_config_routes_through_the_shared_safe_reset_seed():
    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    public_config = _function(tree, "_configure_public_stack_scene")
    calls = [
        node
        for node in ast.walk(public_config)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_apply_safe_reset_posture_config"
    ]
    assert len(calls) == 1
    context = next(keyword.value for keyword in calls[0].keywords if keyword.arg == "context")
    assert isinstance(context, ast.Constant)
    assert context.value == "public Stack-RgyBlock demo"


def test_public_stack_safe_reset_helper_writes_documented_seed_and_preserves_other_joints():
    namespace = _load_safe_reset_helper()
    seed = namespace["MINIMAL_DEMO_ROBOT_JOINT_SEED_RAD"]
    joint_positions = {name: -0.7 for name in seed}
    joint_positions["waist_yaw_joint"] = 0.3
    scene = SimpleNamespace(
        robot=SimpleNamespace(
            init_state=SimpleNamespace(joint_pos=joint_positions),
        ),
    )

    result = namespace["_apply_safe_reset_posture_config"](
        scene,
        context="public Stack-RgyBlock demo",
    )

    assert result == dict(seed)
    for name, value in seed.items():
        assert scene.robot.init_state.joint_pos[name] == value
    assert scene.robot.init_state.joint_pos["waist_yaw_joint"] == pytest.approx(0.3)


def test_public_stack_safe_reset_helper_replaces_immutable_joint_mapping():
    namespace = _load_safe_reset_helper()
    seed = namespace["MINIMAL_DEMO_ROBOT_JOINT_SEED_RAD"]
    initial = {name: -0.7 for name in seed}
    initial["waist_yaw_joint"] = 0.3
    scene = SimpleNamespace(
        robot=SimpleNamespace(
            init_state=SimpleNamespace(joint_pos=MappingProxyType(initial)),
        ),
    )

    namespace["_apply_safe_reset_posture_config"](
        scene,
        context="public Stack-RgyBlock demo",
    )

    assert isinstance(scene.robot.init_state.joint_pos, dict)
    assert scene.robot.init_state.joint_pos["waist_yaw_joint"] == pytest.approx(0.3)
    assert {
        name: scene.robot.init_state.joint_pos[name]
        for name in seed
    } == dict(seed)


def test_main_applies_and_validates_reset_posture_for_both_demo_paths():
    source = SCRIPT.read_text()
    assert source.count("if args.minimal_demo_scene or args.demo_spec is not None:") >= 2
    write_index = source.rindex("_write_minimal_demo_reset_posture(env)")
    validate_index = source.rindex("_validate_minimal_demo_reset_posture(")
    plan_index = source.index(
        'print("[plan] constructing the public Pinocchio URDF model"'
    )
    assert write_index < validate_index < plan_index


def test_swept_clearance_gate_precedes_plan_save_and_policy_construction():
    source = SCRIPT.read_text()
    validate_index = source.rindex("_validate_demo_swept_clearance(")
    save_index = source.rindex("trajectory.save_npz(")
    policy_index = source.rindex("policy = OpenLoopPolicy(trajectory)")

    assert validate_index < save_index < policy_index
