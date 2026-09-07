"""Dependency-light joint-jog state for the visible G1 keyboard demo.

The Omniverse keyboard subscription lives in the simulator runner.  Keeping
the target update logic here makes key mapping, limit clamping, and gripper
commands testable without importing Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


# One key press changes the selected revolute joint by two degrees.  The value
# is radians per discrete key press (not velocity) and was chosen as a cautious
# manual-jog increment: it is visibly responsive at the public 50 Hz hold rate
# while remaining much smaller than a typical G1 arm-joint range.  Larger
# increments can cause abrupt contacts; substantially smaller increments make
# tabletop positioning tedious.  It is intentionally fixed for this demo and
# should become configurable only after a visible contact-safety validation.
JOINT_JOG_STEP_RAD = float(np.deg2rad(2.0))


@dataclass(frozen=True)
class TeleopEvent:
    """One user-visible result of processing a keyboard press."""

    kind: str
    message: str


class JointJogTeleop:
    """Maintain limit-clamped absolute targets for two seven-joint G1 arms."""

    # Carb reports top-row digits as either the printable digit or ``KEY_N``
    # depending on Kit version.  Both spellings are dimensionless key IDs and
    # map in anatomical shoulder-to-wrist order (indices 0..6) from the public
    # seven-joint arm lists.  Sequential digits make selection visible and
    # keyboard-layout-neutral; reordering them would jog a different joint than
    # README promises.  This interface is intentionally fixed for the demo.
    _NUMBER_KEYS = {
        "1": 0,
        "KEY_1": 0,
        "2": 1,
        "KEY_2": 1,
        "3": 2,
        "KEY_3": 2,
        "4": 3,
        "KEY_4": 3,
        "5": 4,
        "KEY_5": 4,
        "6": 5,
        "KEY_6": 5,
        "7": 6,
        "KEY_7": 6,
    }

    def __init__(
        self,
        *,
        joint_names: Sequence[str],
        initial_positions: Sequence[float],
        joint_limits: Sequence[Sequence[float]],
        arm_joints_by_side: Mapping[str, Sequence[str]],
        gripper_joints_by_side: Mapping[str, Sequence[str]],
        gripper_open_positions: Sequence[float],
        gripper_closed_positions: Sequence[float],
        jog_step_rad: float = JOINT_JOG_STEP_RAD,
    ) -> None:
        self.joint_names = tuple(joint_names)
        self._index = {name: index for index, name in enumerate(self.joint_names)}
        if len(self._index) != len(self.joint_names):
            raise ValueError("joint_names must be unique")

        positions = np.asarray(initial_positions, dtype=np.float64)
        limits = np.asarray(joint_limits, dtype=np.float64)
        if positions.shape != (len(self.joint_names),):
            raise ValueError("initial_positions must match joint_names")
        if limits.shape != (len(self.joint_names), 2):
            raise ValueError("joint_limits must have shape (joint_count, 2)")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(limits)):
            raise ValueError("positions and limits must be finite")
        if np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("each joint lower limit must be <= its upper limit")
        if not np.isfinite(jog_step_rad) or jog_step_rad <= 0.0:
            raise ValueError("jog_step_rad must be finite and positive")

        required_sides = ("left", "right")
        self._arm_joints = {
            side: tuple(arm_joints_by_side[side]) for side in required_sides
        }
        self._gripper_joints = {
            side: tuple(gripper_joints_by_side[side]) for side in required_sides
        }
        for side in required_sides:
            if len(self._arm_joints[side]) != len(self._NUMBER_KEYS) // 2:
                raise ValueError(f"{side} arm must expose exactly seven joints")
            if len(self._gripper_joints[side]) != 2:
                raise ValueError(f"{side} gripper must expose exactly two joints")
            missing = [
                name
                for name in self._arm_joints[side] + self._gripper_joints[side]
                if name not in self._index
            ]
            if missing:
                raise ValueError(f"{side} teleop joints are absent from action order: {missing}")

        open_positions = np.asarray(gripper_open_positions, dtype=np.float64)
        closed_positions = np.asarray(gripper_closed_positions, dtype=np.float64)
        if open_positions.shape != (2,) or closed_positions.shape != (2,):
            raise ValueError("gripper targets must each contain two positions")
        if not np.all(np.isfinite(open_positions)) or not np.all(np.isfinite(closed_positions)):
            raise ValueError("gripper targets must be finite")

        self._limits = limits.copy()
        self._targets = np.clip(positions, limits[:, 0], limits[:, 1])
        self._open_positions = open_positions
        self._closed_positions = closed_positions
        self._jog_step_rad = float(jog_step_rad)
        self.side = "right"
        self.selected_joint = 0
        self.quit_requested = False

    @property
    def absolute_targets(self) -> np.ndarray:
        """Return a copy so callers cannot mutate controller state."""

        return self._targets.copy()

    @property
    def selected_joint_name(self) -> str:
        return self._arm_joints[self.side][self.selected_joint]

    def _set_gripper(self, positions: np.ndarray, label: str) -> TeleopEvent:
        for name, position in zip(self._gripper_joints[self.side], positions, strict=True):
            index = self._index[name]
            self._targets[index] = np.clip(
                position,
                self._limits[index, 0],
                self._limits[index, 1],
            )
        return TeleopEvent("gripper", f"{self.side} gripper {label}")

    def press(self, key_name: str) -> TeleopEvent | None:
        """Apply one normalized Omniverse key press."""

        key = str(key_name).upper()
        if key in ("Q", "ESCAPE"):
            self.quit_requested = True
            return TeleopEvent("quit", "quit requested")
        if key == "TAB":
            self.side = "left" if self.side == "right" else "right"
            return TeleopEvent("side", f"active arm: {self.side}")
        if key in self._NUMBER_KEYS:
            self.selected_joint = self._NUMBER_KEYS[key]
            return TeleopEvent("select", f"selected {self.selected_joint_name}")
        if key in ("LEFT", "RIGHT"):
            direction = -1.0 if key == "LEFT" else 1.0
            joint_name = self.selected_joint_name
            index = self._index[joint_name]
            self._targets[index] = np.clip(
                self._targets[index] + direction * self._jog_step_rad,
                self._limits[index, 0],
                self._limits[index, 1],
            )
            return TeleopEvent(
                "jog",
                f"{joint_name} target={self._targets[index]:+.4f} rad",
            )
        if key == "O":
            return self._set_gripper(self._open_positions, "open")
        if key == "C":
            return self._set_gripper(self._closed_positions, "closed")
        return None
