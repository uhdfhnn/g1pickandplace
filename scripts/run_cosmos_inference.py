#!/usr/bin/env python3
"""Run one visible Unitree-simulation first-frame Cosmos policy inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

from run_demo import (
    CAMERA_FLAG,
    DEFAULT_CONDA_ENVIRONMENT,
    DEFAULT_CONDA_EXECUTABLE,
    DEFAULT_DEVICE,
    DEFAULT_PACKAGE_DIR,
    DEFAULT_UNITREE_ROOT,
    DEFAULT_URDF,
    PUBLIC_STACK_TASK_ID,
    REPOSITORY_ROOT,
    _runtime_environment,
)


# The prefix separates inference-only artifacts from the validated open-loop
# demo gates.  PID plus nanosecond wall-clock entropy makes concurrent local
# invocations non-destructive; it has no simulation time unit and never enters
# the model request or policy.  A caller can use --output-dir for a stable
# experiment path, while the default is intentionally unique.
DEFAULT_OUTPUT_PREFIX = "cosmos_inference"

# Sixteen seconds is 160 transitions at the deployed 10-Hz action rate.  It is
# the requested long-horizon inference duration and remains below the locally
# evidenced 512-step server response.  A shorter value can truncate a full
# pick-and-place intent; a much longer open-loop prediction can drift and costs
# more server time.  This setting changes only saved model output and never
# authorizes G1 execution, so callers may override it for controlled probes.
DEFAULT_ACTION_DURATION_S = 16.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--prompt", default=None, help="defaults to --instruction")
    # localhost:8080 is the documented local endpoint of the explicit SSH
    # tunnel to the current Cosmos service.  It is configurable for another
    # verified endpoint; an unavailable/wrong value fails without executing any
    # action in simulation.
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=DEFAULT_ACTION_DURATION_S,
        metavar="SECONDS",
        help="Cosmos action horizon; default 16 seconds (160 actions at 10 Hz)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--unitree-root", type=Path, default=DEFAULT_UNITREE_ROOT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--conda-executable", type=Path, default=DEFAULT_CONDA_EXECUTABLE)
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENVIRONMENT)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    return parser


def _resolve_output_dir(value: Path | None) -> Path:
    if value is None:
        value = REPOSITORY_ROOT / "outputs" / (
            f"{DEFAULT_OUTPUT_PREFIX}_{os.getpid()}_{time.time_ns()}"
        )
    elif not value.is_absolute():
        value = REPOSITORY_ROOT / value
    return value.expanduser().resolve()


def _command(arguments: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        str(arguments.conda_executable),
        "run",
        "--no-capture-output",
        "-n",
        str(arguments.conda_env),
        "python",
        "scripts/run_unitree_mvp.py",
        "--unitree-root",
        str(arguments.unitree_root),
        "--urdf",
        str(arguments.urdf),
        "--package-dir",
        str(arguments.package_dir),
        "--task",
        PUBLIC_STACK_TASK_ID,
        "--instruction",
        arguments.instruction,
        "--device",
        arguments.device,
        CAMERA_FLAG,
        "--inspect-only",
        "--cosmos-base-url",
        arguments.base_url,
        "--cosmos-duration-s",
        str(arguments.duration_s),
        "--cosmos-policy-output-dir",
        str(output_dir),
    ]
    if arguments.prompt is not None:
        command.extend(("--cosmos-prompt", arguments.prompt))
    return command


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output_dir = _resolve_output_dir(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    command = _command(arguments, output_dir)
    log_path = output_dir / "inference.log"
    print(f"[cosmos-wrapper] output_dir={output_dir}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_stream:
        log_stream.write("$ " + " ".join(command) + "\n")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(REPOSITORY_ROOT),
                env=_runtime_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                shell=False,
            )
        except OSError as exc:
            log_stream.write(f"wrapper subprocess error: {exc}\n")
            return 127
        if process.stdout is None:
            log_stream.write("wrapper subprocess provided no stdout pipe\n")
            return 127
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_stream.write(line)
            log_stream.flush()
        status = int(process.wait())
    if status != 0:
        return status
    required = (
        output_dir / "unitree_concat_view.png",
        output_dir / "cosmos_policy_action.json",
        output_dir / "metadata.json",
        output_dir / "sim_inference_context.npz",
    )
    missing = [path.name for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        print(f"[cosmos-wrapper] missing artifact(s): {missing}", flush=True)
        return 1
    print("[cosmos-wrapper] inference complete; action saved but not executed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
