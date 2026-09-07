#!/usr/bin/env python3
"""Replay a saved Cosmos action in visible Unitree simulation and record MP4."""

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


# This output prefix identifies simulator replays rather than model inference.
# PID plus nanosecond wall time are host-local uniqueness fields with no control
# units; they prevent concurrent runs from overwriting evidence.  The directory
# remains configurable for a named experiment and must not already exist.
DEFAULT_OUTPUT_PREFIX = "cosmos_replay"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--unitree-root", type=Path, default=DEFAULT_UNITREE_ROOT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--conda-executable", type=Path, default=DEFAULT_CONDA_EXECUTABLE)
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENVIRONMENT)
    parser.add_argument(
        "--assimp-preload",
        type=Path,
        default=None,
        help="optional Assimp shared library used by the validated Pinocchio collision stack",
    )
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
    action = arguments.action.expanduser().resolve()
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
        "--cosmos-replay-action",
        str(action),
        "--cosmos-replay-video",
        str(output_dir / "cosmos_replay.mp4"),
        "--trajectory-out",
        str(output_dir / "cosmos_replay_trajectory.npz"),
    ]
    if arguments.stats is not None:
        command.extend(("--cosmos-replay-stats", str(arguments.stats.expanduser().resolve())))
    return command


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output_dir = _resolve_output_dir(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    command = _command(arguments, output_dir)
    log_path = output_dir / "replay.log"
    print(f"[cosmos-replay-wrapper] output_dir={output_dir}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_stream:
        log_stream.write("$ " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=str(REPOSITORY_ROOT),
            env=_runtime_environment(arguments),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=False,
        )
        if process.stdout is None:
            raise RuntimeError("replay subprocess provided no stdout pipe")
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_stream.write(line)
            log_stream.flush()
        status = int(process.wait())
    if status != 0:
        return status
    required = (
        output_dir / "cosmos_replay.mp4",
        output_dir / "cosmos_replay_trajectory.npz",
    )
    missing = [path.name for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        print(f"[cosmos-replay-wrapper] missing artifact(s): {missing}", flush=True)
        return 1
    print(f"[cosmos-replay-wrapper] replay video={required[0]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
