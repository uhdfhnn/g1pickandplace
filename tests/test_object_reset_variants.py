from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest


def _load_validator():
    """Load the dependency-light validator without importing Isaac Sim."""

    script = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"
    tree = ast.parse(script.read_text(), filename=str(script))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validated_object_reset_offsets"
    )
    namespace = {
        "math": math,
        # These mirror the public Unitree EventCfg ranges used by the script;
        # the test exercises validation behavior without importing simulator
        # packages or constructing an environment.
        "OBJECT_RESET_X_BOUNDS_M": (-0.1, 0.1),
        "OBJECT_RESET_Y_BOUNDS_M": (-0.05, 0.05),
        "OBJECT_RESET_YAW_BOUNDS_RAD": (-math.pi, math.pi),
        "OBJECT_RESET_ZERO_OFFSET": 0.0,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(script), "exec"), namespace)
    return namespace["_validated_object_reset_offsets"]


def _load_duration_validator():
    """Load duration validation without importing Isaac Sim."""

    script = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"
    tree = ast.parse(script.read_text(), filename=str(script))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validated_segment_durations"
    )
    namespace = {"math": math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(script), "exec"), namespace)
    return namespace["_validated_segment_durations"]


def _load_height_validator():
    """Load planner-height validation without importing Isaac Sim."""

    script = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"
    tree = ast.parse(script.read_text(), filename=str(script))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validated_planner_heights"
    )
    namespace = {"math": math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(script), "exec"), namespace)
    return namespace["_validated_planner_heights"]


def test_unspecified_offsets_preserve_public_random_reset() -> None:
    validate = _load_validator()
    assert validate(None, None, None, fixed_object_reset=False) is None


def test_exact_variant_defaults_omitted_axes_to_zero() -> None:
    validate = _load_validator()
    assert validate(0.025, None, None, fixed_object_reset=False) == (0.025, 0.0, 0.0)
    assert validate(None, -0.04, 0.5, fixed_object_reset=False) == (0.0, -0.04, 0.5)


def test_fixed_reset_is_the_zero_exact_variant() -> None:
    validate = _load_validator()
    assert validate(None, None, None, fixed_object_reset=True) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="cannot be combined"):
        validate(0.01, None, None, fixed_object_reset=True)


@pytest.mark.parametrize(
    "values",
    [
        (0.100001, 0.0, 0.0),
        (0.0, -0.050001, 0.0),
        (0.0, 0.0, math.pi + 1.0e-6),
        (math.inf, 0.0, 0.0),
    ],
)
def test_variant_bounds_and_finiteness_are_rejected(values: tuple[float, float, float]) -> None:
    validate = _load_validator()
    with pytest.raises(ValueError):
        validate(*values, fixed_object_reset=False)


def test_all_segment_durations_accept_positive_finite_seconds() -> None:
    validate = _load_duration_validator()
    values = {
        "pregrasp_duration_s": 1.4,
        "descend_duration_s": 0.8,
        "gripper_duration_s": 0.5,
        "settle_duration_s": 0.3,
        "lift_duration_s": 1.0,
        "transport_duration_s": 1.5,
        "return_duration_s": 1.2,
    }
    assert validate(values) == values


@pytest.mark.parametrize("bad_value", [0.0, -0.1, math.inf, math.nan])
def test_nonpositive_or_nonfinite_segment_duration_is_rejected(bad_value: float) -> None:
    validate = _load_duration_validator()
    with pytest.raises(ValueError, match="finite and positive seconds"):
        validate({"settle_duration_s": bad_value})


def test_all_planner_heights_accept_positive_finite_metres() -> None:
    validate = _load_height_validator()
    values = {
        "approach_height_m": 0.12,
        "lift_height_m": 0.16,
        "target_approach_height_m": 0.12,
    }
    assert validate(values) == values


@pytest.mark.parametrize("bad_value", [0.0, -0.01, math.inf, math.nan])
def test_nonpositive_or_nonfinite_planner_height_is_rejected(bad_value: float) -> None:
    validate = _load_height_validator()
    with pytest.raises(ValueError, match="finite and positive metres"):
        validate({"lift_height_m": bad_value})
