"""Independent reset-time Pinocchio IK backend.

The implementation follows the standard damped least-squares frame-IK pattern
from Pinocchio's public inverse-kinematics example. It is intentionally small,
contains no project-private code, and is imported only on hosts with Pinocchio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .geometry import Pose, matrix_quaternion_xyzw, quaternion_matrix_xyzw


class IKPlanningError(RuntimeError):
    """Raised before rollout when a required waypoint is not solvable."""


@dataclass(frozen=True)
class IKSolveReport:
    q: np.ndarray
    iterations: int
    residual: float


class PinocchioFrameIK:
    """Fixed-base frame IK with an explicit active-joint set."""

    def __init__(
        self,
        *,
        urdf_path: str | Path,
        frame_name: str,
        active_joint_names: Iterable[str],
        package_dirs: Iterable[str | Path] = (),
        tolerance: float = 1.0e-4,
        max_iterations: int = 1000,
        integration_step: float = 0.1,
        damping: float = 1.0e-6,
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError(
                "Pinocchio is required for offline IK. Install the 'pin' package in the Isaac Lab environment."
            ) from exc

        self.pin = pin
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)
        package_paths = [str(Path(path).expanduser().resolve()) for path in package_dirs]
        self.model = pin.buildModelFromUrdf(str(self.urdf_path), package_paths)
        self.data = self.model.createData()
        self.frame_name = frame_name
        self.frame_id = self.model.getFrameId(frame_name)
        if self.frame_id >= len(self.model.frames):
            raise ValueError(f"frame {frame_name!r} is absent from {self.urdf_path}")

        self.active_joint_names = tuple(active_joint_names)
        if not self.active_joint_names:
            raise ValueError("active_joint_names cannot be empty")
        self.active_velocity_indices: list[int] = []
        for name in self.active_joint_names:
            joint_id = self.model.getJointId(name)
            if joint_id == 0:
                raise ValueError(f"joint {name!r} is absent from {self.urdf_path}")
            joint = self.model.joints[joint_id]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(f"joint {name!r} must be scalar, got nq={joint.nq}, nv={joint.nv}")
            self.active_velocity_indices.append(int(joint.idx_v))

        if tolerance <= 0.0 or max_iterations <= 0 or integration_step <= 0.0 or damping <= 0.0:
            raise ValueError("IK numerical parameters must be positive")
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.integration_step = float(integration_step)
        self.damping = float(damping)

    def q_from_named_positions(self, joint_names: Iterable[str], positions: np.ndarray) -> np.ndarray:
        """Populate a fixed-base Pinocchio configuration by joint name."""
        names = tuple(joint_names)
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (len(names),):
            raise ValueError("positions must match joint_names")
        q = np.asarray(self.pin.neutral(self.model), dtype=np.float64)
        for name, value in zip(names, positions, strict=True):
            joint_id = self.model.getJointId(name)
            if joint_id == 0:
                continue  # e.g. simulator-only gripper joints
            joint = self.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(f"joint {name!r} is not scalar in the Pinocchio model")
            q[joint.idx_q] = value
        return q

    def named_positions_from_q(self, q: np.ndarray, joint_names: Iterable[str]) -> dict[str, float]:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.model.nq,):
            raise ValueError(f"q must have shape ({self.model.nq},), got {q.shape}")
        result: dict[str, float] = {}
        for name in joint_names:
            joint_id = self.model.getJointId(name)
            if joint_id == 0:
                continue
            joint = self.model.joints[joint_id]
            if joint.nq == 1:
                result[name] = float(q[joint.idx_q])
        return result

    def frame_pose(self, q: np.ndarray) -> Pose:
        self.pin.framesForwardKinematics(self.model, self.data, np.asarray(q, dtype=np.float64))
        placement = self.data.oMf[self.frame_id]
        return Pose(np.asarray(placement.translation), matrix_quaternion_xyzw(np.asarray(placement.rotation)))

    def solve(self, waypoint_name: str, target: Pose, seed_q: np.ndarray) -> IKSolveReport:
        """Solve a waypoint completely before simulation rollout begins."""
        pin = self.pin
        q = np.asarray(seed_q, dtype=np.float64).copy()
        if q.shape != (self.model.nq,):
            raise ValueError(f"seed_q must have shape ({self.model.nq},), got {q.shape}")
        desired = pin.SE3(quaternion_matrix_xyzw(target.quaternion_xyzw), target.position)
        active = np.asarray(self.active_velocity_indices, dtype=np.int64)
        identity = np.eye(6)
        residual = float("inf")

        for iteration in range(self.max_iterations + 1):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[self.frame_id]
            current_to_desired = current.actInv(desired)
            error = np.asarray(pin.log6(current_to_desired).vector, dtype=np.float64)
            residual = float(np.linalg.norm(error))
            if residual < self.tolerance:
                return IKSolveReport(q=q, iterations=iteration, residual=residual)
            if iteration == self.max_iterations:
                break

            jacobian = np.asarray(
                pin.computeFrameJacobian(self.model, self.data, q, self.frame_id, pin.ReferenceFrame.LOCAL),
                dtype=np.float64,
            )
            jacobian = -np.asarray(pin.Jlog6(current_to_desired.inverse())) @ jacobian
            active_jacobian = jacobian[:, active]
            velocity_active = -active_jacobian.T @ np.linalg.solve(
                active_jacobian @ active_jacobian.T + self.damping * identity,
                error,
            )
            velocity = np.zeros(self.model.nv)
            velocity[active] = velocity_active
            q = np.asarray(pin.integrate(self.model, q, velocity * self.integration_step), dtype=np.float64)
            q = self._clip_configuration(q)

        raise IKPlanningError(
            f"IK failed for {waypoint_name!r} after {self.max_iterations} iterations; residual={residual:.6g}"
        )

    def _clip_configuration(self, q: np.ndarray) -> np.ndarray:
        lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64)
        upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64)
        result = q.copy()
        finite = np.isfinite(lower) & np.isfinite(upper) & (lower < upper)
        result[finite] = np.clip(result[finite], lower[finite], upper[finite])
        return result
