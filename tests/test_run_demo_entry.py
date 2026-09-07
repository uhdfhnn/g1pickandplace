from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_demo.py"


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("run_demo_entry", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeProcess:
    def __init__(self, status: int, output: str) -> None:
        self.stdout = iter((output,))
        self.returncode = status

    def wait(self) -> int:
        return self.returncode


def _successful_stage_output(stage: str) -> str:
    """Return the runner's minimal structured evidence for one fake gate."""

    if stage == "inspect":
        return "\n".join(
            (
                "[preflight] reset geometry: {\"status\": \"PASS\"}",
                "[inspect] viewport frame: {\"status\": \"PASS\"}",
                "[inspect] complete: expert and trajectory were not constructed",
            )
        ) + "\n"
    if stage == "plan":
        return "\n".join(
            (
                "[preflight] swept clearance: {\"status\": \"PASS\"}",
                "[plan] solved all IK before rollout: {\"grasp\": 1}",
                "[plan] complete: rollout step zero was not started",
            )
        ) + "\n"
    if stage == "rollout":
        return "\n".join(
            (
                "[record-validation] {\"status\": \"PASS\", \"action_hashes_match\": true}",
                "[result] {\"success\": true}",
            )
        ) + "\n"
    raise AssertionError(stage)


def test_instruction_is_mandatory() -> None:
    wrapper = _load_wrapper()
    with pytest.raises(SystemExit):
        wrapper._parser().parse_args([])


def test_keyboard_teleop_is_an_alternative_to_instruction() -> None:
    wrapper = _load_wrapper()
    arguments = wrapper._parser().parse_args(["--keyboard-teleop"])
    assert arguments.keyboard_teleop
    assert arguments.instruction is None


def test_keyboard_teleop_runs_one_visible_non_recording_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _load_wrapper()
    calls: list[list[str]] = []

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        calls.append(command)
        return _FakeProcess(
            0,
            '[teleop] complete: {"mode": "manual_joint_jog", "status": "PASS"}\n',
        )

    monkeypatch.setattr(wrapper.subprocess, "Popen", fake_popen)
    output_dir = tmp_path / "keyboard"
    assert wrapper.main(["--keyboard-teleop", "--output-dir", str(output_dir)]) == 0
    assert len(calls) == 1
    assert "--keyboard-teleop" in calls[0]
    assert "--instruction" not in calls[0]
    assert "--inspect-only" not in calls[0]
    assert "--plan-only" not in calls[0]
    assert "--record-root" not in calls[0]
    assert "--headless" not in calls[0]
    assert (output_dir / "teleop.log").is_file()


def test_keyboard_teleop_rejects_rollout() -> None:
    wrapper = _load_wrapper()
    with pytest.raises(SystemExit):
        wrapper.main(["--keyboard-teleop", "--rollout"])


def test_default_gates_order_instruction_and_recording_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    wrapper = _load_wrapper()
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        calls.append((command, kwargs))
        stage = "inspect" if "--inspect-only" in command else "plan" if "--plan-only" in command else "rollout"
        return _FakeProcess(0, f"child-{stage}-line\n" + _successful_stage_output(stage))

    monkeypatch.setattr(wrapper.subprocess, "Popen", fake_popen)
    instruction = "Pick up the red block; put it left of the yellow marker!"
    output_dir = tmp_path / "demo"
    # This empty stand-in exercises only portable path propagation; the dynamic
    # loader never reads it because every child process is replaced by the fake.
    # A real run requires the ABI-major-5 library documented in SETUP.md.
    assimp_preload = tmp_path / "libassimp.so.5"
    assimp_preload.touch()
    assert wrapper.main(
        [
            "--instruction",
            instruction,
            "--rollout",
            "--output-dir",
            str(output_dir),
            "--assimp-preload",
            str(assimp_preload),
        ]
    ) == 0

    assert len(calls) == 3
    commands = [command for command, _ in calls]
    assert "--inspect-only" in commands[0]
    assert "--plan-only" in commands[1]
    assert "--inspect-only" not in commands[1]
    assert "--plan-only" not in commands[2]
    assert "--headless" not in [token for command in commands for token in command]
    for command, kwargs in calls:
        assert command[command.index("--instruction") + 1] == instruction
        assert kwargs["shell"] is False
        assert kwargs["text"] is True
        assert kwargs["stdout"] is wrapper.subprocess.PIPE
        assert kwargs["stderr"] is wrapper.subprocess.STDOUT
        child_env = kwargs["env"]
        assert isinstance(child_env, dict)
        assert "CYCLONEDDS_HOME" not in child_env
        assert child_env["PYTHONUNBUFFERED"] == "1"
        assert child_env["LD_PRELOAD"].startswith(str(assimp_preload.resolve()))

    assert "--viewport-frame" in commands[0]
    assert "--phase-boundary-frame-root" in commands[0]
    assert str(output_dir / "plan.npz") in commands[1]
    assert "--phase-boundary-frame-root" in commands[2]
    assert "--record-root" in commands[2]
    assert "--dataset-repo-id" in commands[2]
    assert str(output_dir / "rollout.npz") in commands[2]
    assert (output_dir / "inspect.log").is_file()
    assert (output_dir / "plan.log").is_file()
    assert (output_dir / "rollout.log").is_file()
    captured = capsys.readouterr().out
    assert f"[wrapper] output_dir={output_dir.resolve()}" in captured
    assert "[wrapper] inspect start" in captured
    assert "child-inspect-line" in captured
    assert "[wrapper] inspect result=0" in captured
    assert "[wrapper] plan start" in captured
    assert "child-plan-line" in captured
    assert "[wrapper] rollout start" in captured
    assert "child-rollout-line" in captured
    assert "[wrapper] rollout result=0" in captured
    assert "child-inspect-line" in (output_dir / "inspect.log").read_text()


def test_default_stops_after_plan_without_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _load_wrapper()
    calls: list[list[str]] = []

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        calls.append(command)
        stage = "inspect" if "--inspect-only" in command else "plan"
        return _FakeProcess(0, _successful_stage_output(stage))

    monkeypatch.setattr(wrapper.subprocess, "Popen", fake_popen)
    assert wrapper.main(["--instruction", "stack it", "--output-dir", str(tmp_path / "demo")]) == 0
    assert len(calls) == 2
    assert "--inspect-only" in calls[0]
    assert "--plan-only" in calls[1]


@pytest.mark.parametrize("failed_stage", ["inspect", "plan"])
def test_failed_gate_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_stage: str
) -> None:
    wrapper = _load_wrapper()
    calls: list[list[str]] = []

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        calls.append(command)
        is_failed_stage = (
            failed_stage == "inspect" and "--inspect-only" in command
        ) or (failed_stage == "plan" and "--plan-only" in command)
        stage = "inspect" if "--inspect-only" in command else "plan" if "--plan-only" in command else "rollout"
        output = "failed-stage-line\n" if is_failed_stage else _successful_stage_output(stage)
        return _FakeProcess(9 if is_failed_stage else 0, output)

    monkeypatch.setattr(wrapper.subprocess, "Popen", fake_popen)
    status = wrapper.main(
        ["--instruction", "Pick up red and stack it", "--rollout", "--output-dir", str(tmp_path / failed_stage)]
    )
    assert status == 9
    assert len(calls) == (1 if failed_stage == "inspect" else 2)
    assert all("--record-root" not in command for command in calls)


def test_commands_use_public_defaults_and_no_shell_headless() -> None:
    wrapper = _load_wrapper()
    arguments = wrapper._parser().parse_args(["--instruction", "demo"])
    inspect = wrapper._stage_command(arguments, wrapper.INSPECT_STAGE, Path("/tmp/demo"))
    plan = wrapper._stage_command(arguments, wrapper.PLAN_STAGE, Path("/tmp/demo"))
    rollout = wrapper._stage_command(arguments, wrapper.ROLLOUT_STAGE, Path("/tmp/demo"))

    for command in (inspect, plan, rollout):
        assert wrapper.PUBLIC_STACK_TASK_ID in command
        assert "--headless" not in command
        assert "--enable_cameras" in command
        assert "--enable-staging" in command
    assert arguments.unitree_root == wrapper.DEFAULT_UNITREE_ROOT
    assert arguments.urdf == wrapper.DEFAULT_URDF
    assert arguments.package_dir == wrapper.DEFAULT_PACKAGE_DIR
    assert arguments.conda_env == wrapper.DEFAULT_CONDA_ENVIRONMENT
    assert arguments.device == wrapper.DEFAULT_DEVICE
    assert "--viewport-frame" in inspect
    assert "--trajectory-out" in plan
    assert "--record-root" in rollout


def test_missing_explicit_assimp_preload_fails_before_launch(tmp_path: Path) -> None:
    wrapper = _load_wrapper()
    arguments = wrapper._parser().parse_args(
        [
            "--instruction",
            "demo",
            "--assimp-preload",
            str(tmp_path / "missing-libassimp.so.5"),
        ]
    )
    with pytest.raises(FileNotFoundError, match="Assimp preload does not exist"):
        wrapper._runtime_environment(arguments)


def test_shovel_instruction_uses_fixed_three_second_reset_settle() -> None:
    wrapper = _load_wrapper()
    arguments = wrapper._parser().parse_args(
        [
            "--instruction",
            "Pick up the shovel, scoop the red block, and place it in the target tray.",
        ]
    )
    command = wrapper._stage_command(arguments, wrapper.INSPECT_STAGE, Path("/tmp/demo"))
    index = command.index(wrapper.SETTLE_STEPS_FLAG)
    assert command[index + 1] == wrapper.SHOVEL_RESET_SETTLE_STEPS


@pytest.mark.parametrize(
    ("failed_stage", "failure_line", "expected_calls"),
    [
        ("inspect", "[error] RuntimeError: reset preflight failed\n", 1),
        (
            "plan",
            '[preflight] swept clearance: {"status": "FAIL"}\n',
            2,
        ),
    ],
)
def test_zero_return_semantic_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str,
    failure_line: str,
    expected_calls: int,
) -> None:
    """A child return code of zero cannot bypass a failed semantic gate."""

    wrapper = _load_wrapper()
    calls: list[list[str]] = []

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        calls.append(command)
        stage = "inspect" if "--inspect-only" in command else "plan" if "--plan-only" in command else "rollout"
        output = _successful_stage_output(stage)
        if stage == failed_stage:
            output += failure_line
        return _FakeProcess(0, output)

    monkeypatch.setattr(wrapper.subprocess, "Popen", fake_popen)
    status = wrapper.main(
        [
            "--instruction",
            "Pick up red",
            "--rollout",
            "--output-dir",
            str(tmp_path / failed_stage),
        ]
    )

    assert status == wrapper.SEMANTIC_GATE_FAILURE_CODE
    assert len(calls) == expected_calls
    assert all("--record-root" not in command for command in calls)


def test_rollout_semantic_result_failure_uses_dedicated_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _load_wrapper()
    calls: list[list[str]] = []

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        calls.append(command)
        stage = "inspect" if "--inspect-only" in command else "plan" if "--plan-only" in command else "rollout"
        output = _successful_stage_output(stage)
        if stage == "rollout":
            output += '[result] {"success": false}\n'
        return _FakeProcess(0, output)

    monkeypatch.setattr(wrapper.subprocess, "Popen", fake_popen)
    status = wrapper.main(
        [
            "--instruction",
            "Pick up red",
            "--rollout",
            "--output-dir",
            str(tmp_path / "rollout-fail"),
        ]
    )

    assert status == wrapper.SEMANTIC_GATE_FAILURE_CODE
    assert len(calls) == 3


def test_semantic_status_requires_json_payload(tmp_path: Path) -> None:
    wrapper = _load_wrapper()
    output_dir = tmp_path / "malformed"
    output_dir.mkdir()
    (output_dir / "inspect.log").write_text(
        "\n".join(
            (
                "[preflight] reset geometry: status PASS",
                "[inspect] viewport frame: {\"status\": \"PASS\"}",
                "[inspect] complete: expert and trajectory were not constructed",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    passed, reason = wrapper._validate_stage_log(wrapper.INSPECT_STAGE, output_dir)
    assert not passed
    assert "JSON" in reason
