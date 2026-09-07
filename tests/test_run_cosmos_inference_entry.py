from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_cosmos_inference.py"


def _load_wrapper():
    scripts_dir = str(SCRIPT.parent)
    sys.path.insert(0, scripts_dir)
    try:
        spec = importlib.util.spec_from_file_location("run_cosmos_inference_entry", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts_dir)


def test_command_runs_visible_inspect_only_cosmos_inference(tmp_path: Path) -> None:
    wrapper = _load_wrapper()
    arguments = wrapper._parser().parse_args(
        [
            "--instruction",
            "Pick up the red block and stack it on the yellow block.",
            "--prompt",
            "Stack the blocks.",
            "--base-url",
            "http://localhost:8080",
        ]
    )
    command = wrapper._command(arguments, tmp_path)
    assert "--inspect-only" in command
    assert "--enable_cameras" in command
    assert "--cosmos-policy-output-dir" in command
    assert command[command.index("--cosmos-policy-output-dir") + 1] == str(tmp_path)
    assert command[command.index("--cosmos-prompt") + 1] == "Stack the blocks."
    assert command[command.index("--cosmos-duration-s") + 1] == "16.0"
    assert "--headless" not in command
    assert "--rollout" not in command


def test_prompt_defaults_to_instruction_without_duplicate_override(tmp_path: Path) -> None:
    wrapper = _load_wrapper()
    arguments = wrapper._parser().parse_args(["--instruction", "Stack it."])
    command = wrapper._command(arguments, tmp_path)
    assert "--cosmos-prompt" not in command
    assert command[command.index("--instruction") + 1] == "Stack it."
