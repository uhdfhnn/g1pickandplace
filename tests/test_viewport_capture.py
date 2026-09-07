from __future__ import annotations

import ast
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"


def _tree() -> ast.Module:
    return ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _viewport_flag_call(tree: ast.AST) -> ast.Call:
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "parser"
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--viewport-frame"
    ]
    assert len(calls) == 1
    return calls[0]


def test_viewport_frame_flag_defaults_to_none_and_is_a_path():
    call = _viewport_flag_call(_tree())
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert isinstance(keywords["default"], ast.Constant)
    assert keywords["default"].value is None
    assert isinstance(keywords["type"], ast.Name)
    assert keywords["type"].id == "Path"


def test_viewport_capture_uses_public_active_viewport_and_waitable_helper():
    function = _function(_tree(), "_capture_viewport_frame")
    imports = [
        alias.name
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom)
        and node.module == "omni.kit.viewport.utility"
        for alias in node.names
    ]
    assert imports == ["capture_viewport_to_file", "get_active_viewport"]

    attributes = {
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert "wait_for_result" in attributes
    assert "run_coroutine" in attributes
    assert "update" in attributes
    run_call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_coroutine"
    )
    run_keywords = {keyword.arg: keyword.value for keyword in run_call.keywords}
    assert isinstance(run_keywords["run_until_complete"], ast.Constant)
    assert run_keywords["run_until_complete"].value is True

    # The capture helper is diagnostic-only: no environment action, policy,
    # trajectory, or observation-dependent transition may be introduced here.
    source = ast.unparse(function)
    assert "OpenLoopPolicy" not in source
    assert ".step(" not in source
    assert "policy" not in source


def test_viewport_capture_is_rejected_outside_inspect_and_before_expert_setup():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "--viewport-frame is diagnostic-only and requires --inspect-only" in source
    gate = source.index(
        'if args.viewport_frame is not None and not args.inspect_only:'
    )
    gate_error = source.index(
        'parser.error("--viewport-frame is diagnostic-only and requires --inspect-only")',
        gate,
    )
    assert gate < gate_error

    inspect = source.rindex("if args.inspect_only:")
    capture = source.index("_capture_viewport_frame(", inspect)
    expert_import = source.index("from g1pickplace.offline_ik", inspect)
    policy = source.index("policy = OpenLoopPolicy", inspect)
    assert inspect < capture < expert_import < policy


def test_viewport_capture_waits_for_completion_before_file_validation():
    function = _function(_tree(), "_capture_viewport_frame")
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "capture_viewport_to_file"
    ]
    assert len(calls) == 1
    wait_call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "wait_for_result"
    )
    assert any(keyword.arg == "completion_frames" for keyword in wait_call.keywords)

    file_checks = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "output_path"
        and node.func.attr == "is_file"
    ]
    assert any(line > wait_call.lineno for line in file_checks)
