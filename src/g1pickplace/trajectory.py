"""Immutable joint trajectories and observation-invariant playback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def _matrix(value: object, columns: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != columns or array.shape[0] == 0:
        raise ValueError(f"{name} must have shape (T, {columns}) with T > 0, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array.copy()


@dataclass(frozen=True)
class JointTarget:
    """One absolute target configuration and its transition duration."""

    phase: str
    absolute_positions: np.ndarray
    duration_s: float

    def __post_init__(self) -> None:
        positions = np.asarray(self.absolute_positions, dtype=np.float64)
        if positions.ndim != 1 or positions.size == 0:
            raise ValueError("absolute_positions must be a non-empty vector")
        if not np.all(np.isfinite(positions)):
            raise ValueError("absolute_positions contains non-finite values")
        if not self.phase:
            raise ValueError("phase cannot be empty")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        positions = positions.copy()
        positions.setflags(write=False)
        object.__setattr__(self, "absolute_positions", positions)


@dataclass(frozen=True)
class JointTrajectory:
    """Finite precomputed trajectory.

    ``absolute_targets`` are useful for inspection and learning. ``env_actions``
    are the exact values passed to the Isaac Lab environment action term.
    """

    joint_names: tuple[str, ...]
    absolute_targets: np.ndarray
    env_actions: np.ndarray
    phases: tuple[str, ...]
    fps: int

    def __post_init__(self) -> None:
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        absolute = _matrix(self.absolute_targets, len(self.joint_names), "absolute_targets")
        actions = _matrix(self.env_actions, len(self.joint_names), "env_actions")
        if absolute.shape != actions.shape:
            raise ValueError("absolute_targets and env_actions must have the same shape")
        if len(self.phases) != absolute.shape[0]:
            raise ValueError("phases length must match trajectory length")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        absolute.setflags(write=False)
        actions.setflags(write=False)
        object.__setattr__(self, "absolute_targets", absolute)
        object.__setattr__(self, "env_actions", actions)

    @property
    def steps(self) -> int:
        return int(self.env_actions.shape[0])

    @property
    def duration_s(self) -> float:
        return self.steps / self.fps

    def action_at(self, step: int) -> np.ndarray:
        if step < 0:
            raise IndexError("step cannot be negative")
        return self.env_actions[min(step, self.steps - 1)].copy()

    def phase_at(self, step: int) -> str:
        if step < 0:
            raise IndexError("step cannot be negative")
        return self.phases[min(step, self.steps - 1)]

    def save_npz(self, path: str) -> None:
        np.savez_compressed(
            path,
            joint_names=np.asarray(self.joint_names),
            absolute_targets=self.absolute_targets,
            env_actions=self.env_actions,
            phases=np.asarray(self.phases),
            fps=np.asarray([self.fps], dtype=np.int32),
        )


class OpenLoopPolicy:
    """Playback policy whose action depends only on its internal time index."""

    def __init__(self, trajectory: JointTrajectory):
        self.trajectory = trajectory
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    @property
    def done(self) -> bool:
        return self._step >= self.trajectory.steps

    @property
    def current_phase(self) -> str:
        return self.trajectory.phase_at(self._step)

    def reset(self) -> None:
        self._step = 0

    def peek(self) -> np.ndarray:
        return self.trajectory.action_at(self._step)

    def act(self, observation: Any = None) -> np.ndarray:
        # Deliberately ignore observations. This invariant is covered by tests.
        del observation
        action = self.trajectory.action_at(self._step)
        self._step += 1
        return action


def compile_joint_targets(
    *,
    joint_names: Iterable[str],
    initial_absolute_positions: np.ndarray,
    default_joint_positions: np.ndarray,
    targets: Iterable[JointTarget],
    fps: int,
    action_scale: float = 1.0,
    max_joint_step: float | None = 0.02,
) -> JointTrajectory:
    """Interpolate absolute targets and convert to default-offset Isaac actions.

    The Unitree red-block task config uses ``JointPositionActionCfg`` with
    ``scale=1`` and ``use_default_offset=True``. Therefore the environment input
    is ``absolute_target - default_joint_position``. ``action_scale`` remains an
    explicit argument so a changed task config fails visibly rather than being
    hidden.
    """
    names = tuple(joint_names)
    if fps <= 0:
        raise ValueError("fps must be positive")
    if action_scale == 0.0:
        raise ValueError("action_scale cannot be zero")
    initial = np.asarray(initial_absolute_positions, dtype=np.float64)
    defaults = np.asarray(default_joint_positions, dtype=np.float64)
    if initial.shape != (len(names),) or defaults.shape != initial.shape:
        raise ValueError("initial/default joint vectors must match joint_names")

    absolute_frames: list[np.ndarray] = []
    phases: list[str] = []
    current = initial.copy()
    for target in targets:
        if target.absolute_positions.shape != current.shape:
            raise ValueError(f"phase {target.phase!r} target shape does not match the robot")
        requested_steps = max(1, int(round(target.duration_s * fps)))
        if max_joint_step is not None:
            if max_joint_step <= 0.0:
                raise ValueError("max_joint_step must be positive")
            delta_steps = int(np.ceil(np.max(np.abs(target.absolute_positions - current)) / max_joint_step))
            requested_steps = max(requested_steps, delta_steps)
        for index in range(1, requested_steps + 1):
            fraction = index / requested_steps
            absolute_frames.append(current + fraction * (target.absolute_positions - current))
            phases.append(target.phase)
        current = target.absolute_positions.copy()

    if not absolute_frames:
        raise ValueError("at least one target is required")
    absolute = np.stack(absolute_frames)
    env_actions = (absolute - defaults[None, :]) / action_scale
    return JointTrajectory(names, absolute, env_actions, tuple(phases), fps)
