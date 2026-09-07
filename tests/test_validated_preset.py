from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"


def _load_preset_helpers():
    """Load preset constants/helpers without importing Isaac Sim."""

    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    wanted_assignments = {
        "VALIDATED_FIXED_SEED_PRESET",
        "VALIDATED_FIXED_SEED_PRESET_OPTION_FIELDS",
    }
    wanted_functions = {
        "_validated_preset_explicit_fields",
        "_preset_values_equal",
        "_apply_validated_fixed_seed_preset",
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
    return (
        namespace["VALIDATED_FIXED_SEED_PRESET"],
        namespace["_validated_preset_explicit_fields"],
        namespace["_apply_validated_fixed_seed_preset"],
    )


def _args(**overrides):
    preset, _, _ = _load_preset_helpers()
    values = dict(
        validated_fixed_seed_preset=True,
        fixed_object_reset=False,
        reset_seed=None,
        enable_staging=False,
        object_reset_x_m=None,
        object_reset_y_m=None,
        object_reset_yaw_rad=None,
        target_position=(1.0, 2.0, 3.0),
        grasp_wrist_offset_world=(0.0, 0.0, 0.08),
        grasp_quaternion_base_xyzw=None,
        gripper_open=(0.03, 0.03),
        gripper_closed=(-0.02, -0.02),
        approach_height_m=0.12,
        lift_height_m=0.16,
        target_approach_height_m=0.12,
        pregrasp_duration_s=1.4,
        descend_duration_s=0.8,
        gripper_duration_s=0.5,
        settle_duration_s=0.3,
        lift_duration_s=1.0,
        transport_duration_s=1.5,
        return_duration_s=1.2,
    )
    values.update(overrides)
    # Keep this helper aligned with the preset's field set if a future field is
    # added; the test should fail loudly rather than silently miss coverage.
    missing = set(preset) - set(values)
    assert not missing, f"test namespace missing preset fields: {sorted(missing)}"
    return SimpleNamespace(**values)


def test_validated_fixed_seed_preset_applies_exact_phase2_values() -> None:
    preset, explicit_fields, apply = _load_preset_helpers()
    args = _args()
    apply(args, explicit_fields([]))

    for field, expected in preset.items():
        assert getattr(args, field) == expected


def test_matching_explicit_values_are_allowed_but_conflicts_are_rejected() -> None:
    preset, explicit_fields, apply = _load_preset_helpers()
    matching = _args(
        target_position=preset["target_position"],
        reset_seed=preset["reset_seed"],
    )
    matching_fields = explicit_fields(
        ["--target-position", "--reset-seed", "0"]
    )
    apply(matching, matching_fields)
    assert matching.target_position == preset["target_position"]
    assert matching.reset_seed == 0

    conflicting = _args(target_position=(-4.14, -4.03, 0.84))
    with pytest.raises(ValueError, match="conflicts"):
        apply(conflicting, explicit_fields(["--target-position=-4.14,-4.03,0.84"]))


def test_preset_flag_absent_preserves_existing_arguments() -> None:
    _, explicit_fields, apply = _load_preset_helpers()
    args = _args(
        validated_fixed_seed_preset=False,
        target_position=(-4.14, -4.03, 0.84),
        reset_seed=7,
    )
    before = vars(args).copy()
    apply(args, explicit_fields(["--target-position", "--reset-seed", "7"]))
    assert vars(args) == before
