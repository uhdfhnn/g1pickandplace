#!/usr/bin/env python3
"""Report optional simulation/runtime dependencies."""

from __future__ import annotations

import importlib.util
from importlib import metadata
import os
import platform
from pathlib import Path


# These are the exact package versions exercised by the accepted visible gates
# and native LeRobot round-trip. They identify software releases (no physical
# units or coordinate frame). Older/newer versions may change Isaac Lab APIs,
# Pinocchio/cmeel ABI resolution, or LeRobot schema behavior, so mismatches fail
# this reproducibility check. The pins are intentionally fixed and should move
# only with SETUP.md, pyproject.toml, a fresh visible run, and recording replay.
REQUIRED_DISTRIBUTIONS = {
    "isaaclab": "0.44.9",
    "isaaclab_tasks": "0.10.45",
    "pin": "2.7.0",
    "lerobot": "0.4.4",
    "datasets": "4.8.4",
    "pyarrow": "25.0.1",
    "av": "15.0.0",
    "imageio": "2.37.0",
    "pandas": "3.0.5",
    "huggingface-hub": "0.35.2",
}

# PyTorch wheels add a local CUDA suffix (for example +cu128) to the public
# 2.7.0 release. The base version is the behavior/API pin validated with Isaac
# Sim 5.0; accepting the suffix permits the CUDA build selected by NVIDIA's
# resolver. A different base version can alter tensor/runtime compatibility and
# is rejected until the full visible stack is revalidated.
REQUIRED_TORCH_BASE_VERSION = "2.7.0"


def main() -> int:
    print(f"Python: {platform.python_version()}")
    required = ("numpy", "torch", "gymnasium", "isaaclab", "pinocchio", "lerobot")
    missing: list[str] = []
    for module in required:
        present = importlib.util.find_spec(module) is not None
        print(f"{module:16s} {'OK' if present else 'MISSING'}")
        if not present:
            missing.append(module)
    for distribution, expected in REQUIRED_DISTRIBUTIONS.items():
        try:
            actual = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
        status = "OK" if actual == expected else f"MISMATCH (expected {expected})"
        print(f"{distribution + ' version':24s} {actual} {status}")
        if actual != expected:
            missing.append(f"{distribution}=={expected}")
    try:
        torch_version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        torch_version = None
    if torch_version is not None:
        torch_base_version = torch_version.split("+", 1)[0]
        status = (
            "OK"
            if torch_base_version == REQUIRED_TORCH_BASE_VERSION
            else f"MISMATCH (expected base {REQUIRED_TORCH_BASE_VERSION})"
        )
        print(f"{'torch version':24s} {torch_version} {status}")
        if torch_base_version != REQUIRED_TORCH_BASE_VERSION:
            missing.append(f"torch~={REQUIRED_TORCH_BASE_VERSION}")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    except Exception as exc:
        print(f"{'LeRobotDataset import':24s} FAILED ({exc})")
        missing.append("LeRobotDataset import")
    else:
        print(f"{'LeRobotDataset import':24s} OK")

    # The environment variable remains supported for non-sibling layouts. The
    # sibling default is derived from this script's checkout so a normal clone
    # has no workstation-specific path. PROJECT_ROOT is deliberately removed:
    # it is ambiguous and previously allowed an unrelated directory to pass.
    sibling_default = Path(__file__).resolve().parents[2] / "unitree_sim_isaaclab"
    project_root = os.environ.get("UNITREE_SIM_ISAACLAB_ROOT")
    path = Path(project_root).expanduser() if project_root else sibling_default
    print(f"Unitree root: {path} ({'OK' if path.is_dir() else 'MISSING'})")
    if not path.is_dir():
        missing.append("UNITREE_SIM_ISAACLAB_ROOT")
    if missing:
        print("\nMissing runtime requirements: " + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
