"""Metrics that observe a rollout without influencing its actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PickPlaceResult:
    success: bool
    inside_target_xy: bool
    height_ok: bool
    stable: bool
    center_error_m: float
    final_speed_mps: float


def evaluate_pick_place(
    final_object_position: object,
    final_linear_velocity: object,
    target_object_center: object,
    *,
    target_half_extent_xy: tuple[float, float] = (0.09, 0.09),
    height_tolerance_m: float = 0.05,
    maximum_speed_mps: float = 0.05,
) -> PickPlaceResult:
    position = np.asarray(final_object_position, dtype=np.float64)
    velocity = np.asarray(final_linear_velocity, dtype=np.float64)
    target = np.asarray(target_object_center, dtype=np.float64)
    if position.shape != (3,) or velocity.shape != (3,) or target.shape != (3,):
        raise ValueError("position, velocity, and target must all have shape (3,)")
    delta = position - target
    inside_xy = bool(abs(delta[0]) <= target_half_extent_xy[0] and abs(delta[1]) <= target_half_extent_xy[1])
    height_ok = bool(abs(delta[2]) <= height_tolerance_m)
    speed = float(np.linalg.norm(velocity))
    stable = speed <= maximum_speed_mps
    return PickPlaceResult(
        success=inside_xy and height_ok and stable,
        inside_target_xy=inside_xy,
        height_ok=height_ok,
        stable=stable,
        center_error_m=float(np.linalg.norm(delta)),
        final_speed_mps=speed,
    )
