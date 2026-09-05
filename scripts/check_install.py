#!/usr/bin/env python3
"""Report optional simulation/runtime dependencies."""

from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path


def main() -> int:
    print(f"Python: {platform.python_version()}")
    required = ("numpy", "torch", "gymnasium", "isaaclab", "pinocchio", "lerobot")
    missing: list[str] = []
    for module in required:
        present = importlib.util.find_spec(module) is not None
        print(f"{module:16s} {'OK' if present else 'MISSING'}")
        if not present:
            missing.append(module)
    project_root = os.environ.get("UNITREE_SIM_ISAACLAB_ROOT") or os.environ.get("PROJECT_ROOT")
    if project_root:
        path = Path(project_root).expanduser()
        print(f"Unitree root: {path} ({'OK' if path.is_dir() else 'MISSING'})")
        if not path.is_dir():
            missing.append("UNITREE_SIM_ISAACLAB_ROOT")
    else:
        print("Unitree root: MISSING (set UNITREE_SIM_ISAACLAB_ROOT)")
        missing.append("UNITREE_SIM_ISAACLAB_ROOT")
    if missing:
        print("\nMissing runtime requirements: " + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
