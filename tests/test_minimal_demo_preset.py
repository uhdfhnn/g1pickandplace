from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"


def _load_preset_helpers():
    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    wanted_assignments = {
        "VALIDATED_FIXED_SEED_PRESET",
        "MINIMAL_DEMO_PRESET",
        "VALIDATED_FIXED_SEED_PRESET_OPTION_FIELDS",
        "MINIMAL_DEMO_PRESET_OPTION_FIELDS",
    }
    wanted_functions = {
        "_preset_values_equal",
        "_minimal_demo_preset_explicit_fields",
        "_apply_minimal_demo_preset",
    }
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
    ]
    namespace = {"argparse": __import__("argparse"), "Any": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace


def _args(**overrides):
    namespace = _load_preset_helpers()
    preset = namespace["MINIMAL_DEMO_PRESET"]
    values = dict(
        minimal_demo_preset=True,
        minimal_demo_scene=False,
        **{
            field: value
            for field, value in preset.items()
            if field != "minimal_demo_scene"
        },
    )
    values.update(overrides)
    return namespace, SimpleNamespace(**values)


def test_minimal_demo_preset_sets_requested_target_and_scene_mode():
    namespace, args = _args()
    apply = namespace["_apply_minimal_demo_preset"]
    explicit_fields = namespace["_minimal_demo_preset_explicit_fields"]

    apply(args, explicit_fields([]))

    assert args.minimal_demo_scene is True
    assert args.target_position == (-4.35, -4.03, 0.824)
    assert args.grasp_wrist_offset_world == (0.025, 0.165, 0.0)


def test_minimal_demo_preset_rejects_conflicting_explicit_target():
    namespace, args = _args(target_position=(-4.15, -4.03, 0.84))
    explicit_fields = namespace["_minimal_demo_preset_explicit_fields"]
    apply = namespace["_apply_minimal_demo_preset"]

    with pytest.raises(ValueError, match="conflicts"):
        apply(args, explicit_fields(["--target-position=-4.15,-4.03,0.84"]))


def test_minimal_preset_option_aliases_are_normalized():
    namespace = _load_preset_helpers()
    explicit_fields = namespace["_minimal_demo_preset_explicit_fields"]
    assert explicit_fields(
        ["--object-reset-offset-x-m", "--minimal-demo-scene"]
    ) == {"object_reset_x_m", "minimal_demo_scene"}


def test_minimal_and_validated_presets_are_declared_mutually_exclusive():
    source = SCRIPT.read_text()
    assert "--validated-fixed-seed-preset cannot be combined with" in source
    assert "args.minimal_demo_preset or args.minimal_demo_scene" in source
