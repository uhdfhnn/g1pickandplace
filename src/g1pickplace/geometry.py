"""Dependency-light rigid-pose operations using ``xyzw`` quaternions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

_EPS = 1.0e-12


def _vector(value: object, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array.copy()


def normalize_quaternion_xyzw(value: object) -> np.ndarray:
    quaternion = _vector(value, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < _EPS:
        raise ValueError("quaternion norm is zero")
    return quaternion / norm


def quaternion_to_xyzw(value: object, order: Literal["xyzw", "wxyz"]) -> np.ndarray:
    quaternion = _vector(value, 4, "quaternion")
    if order == "wxyz":
        quaternion = quaternion[[1, 2, 3, 0]]
    elif order != "xyzw":
        raise ValueError(f"unknown quaternion order: {order}")
    return normalize_quaternion_xyzw(quaternion)


def quaternion_matrix_xyzw(value: object) -> np.ndarray:
    """Convert a unit quaternion in ``xyzw`` order to a rotation matrix."""
    x, y, z, w = normalize_quaternion_xyzw(value)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_quaternion_xyzw(matrix: object) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized ``xyzw`` quaternion."""
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation contains non-finite values")

    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif index == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
    return normalize_quaternion_xyzw(np.asarray([x, y, z, w]))


def slerp_xyzw(start: object, end: object, fraction: float) -> np.ndarray:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    q0 = normalize_quaternion_xyzw(start)
    q1 = normalize_quaternion_xyzw(end)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion_xyzw(q0 + fraction * (q1 - q0))
    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    return normalize_quaternion_xyzw(
        np.sin((1.0 - fraction) * theta) / sin_theta * q0
        + np.sin(fraction * theta) / sin_theta * q1
    )


@dataclass(frozen=True)
class Pose:
    """Rigid pose whose translation and orientation are immutable."""

    position: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        position = _vector(self.position, 3, "position")
        quaternion = normalize_quaternion_xyzw(self.quaternion_xyzw)
        position.setflags(write=False)
        quaternion.setflags(write=False)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion_xyzw", quaternion)

    @classmethod
    def identity(cls) -> "Pose":
        return cls(np.zeros(3), np.asarray([0.0, 0.0, 0.0, 1.0]))

    @classmethod
    def from_sim(cls, position: object, quaternion: object, order: Literal["xyzw", "wxyz"]) -> "Pose":
        return cls(_vector(position, 3, "position"), quaternion_to_xyzw(quaternion, order))

    @classmethod
    def from_matrix(cls, matrix: object) -> "Pose":
        transform = np.asarray(matrix, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(f"transform must have shape (4, 4), got {transform.shape}")
        return cls(transform[:3, 3], matrix_quaternion_xyzw(transform[:3, :3]))

    def as_matrix(self) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = quaternion_matrix_xyzw(self.quaternion_xyzw)
        result[:3, 3] = self.position
        return result

    def inverse(self) -> "Pose":
        rotation = quaternion_matrix_xyzw(self.quaternion_xyzw)
        inverse_rotation = rotation.T
        return Pose(-inverse_rotation @ self.position, matrix_quaternion_xyzw(inverse_rotation))

    def compose(self, other: "Pose") -> "Pose":
        return Pose.from_matrix(self.as_matrix() @ other.as_matrix())

    def transform_point(self, point: object) -> np.ndarray:
        return quaternion_matrix_xyzw(self.quaternion_xyzw) @ _vector(point, 3, "point") + self.position

    def translated_world(self, delta_xyz: object) -> "Pose":
        return Pose(self.position + _vector(delta_xyz, 3, "delta_xyz"), self.quaternion_xyzw)

    def interpolate(self, other: "Pose", fraction: float) -> "Pose":
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        position = self.position + fraction * (other.position - self.position)
        return Pose(position, slerp_xyzw(self.quaternion_xyzw, other.quaternion_xyzw, fraction))
