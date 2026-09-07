"""Independent reset-time Pinocchio IK backend.

The implementation follows the standard damped least-squares frame-IK pattern
from Pinocchio's public inverse-kinematics example. It is intentionally small,
contains no project-private code, and is imported only on hosts with Pinocchio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from xml.etree import ElementTree

import numpy as np

from .geometry import Pose, matrix_quaternion_xyzw, quaternion_matrix_xyzw


# The alternate-seed generator is deliberately deterministic per reset-time
# waypoint.  ``92417`` is a dimensionless NumPy Generator seed selected by an
# independent probe of the public G1 29-DoF URDF; resetting the generator for
# each solve makes repeated planning from the same snapshot reproducible.  A
# different seed can miss a collision-free IK branch, while a non-deterministic
# generator would make an open-loop artifact impossible to replay.  This is a
# fixed implementation choice, not a runtime control or feedback signal.
COLLISION_MULTISTART_RNG_SEED = 92417

# The first IK solve uses the supplied seed; at most 32 additional active-joint
# seeds are attempted when its endpoint or path is colliding.  The count is
# dimensionless and comes from the public-URDF probe: the shifted compact path
# needed two alternates, while 32 provided margin without a large reset delay.
# Fewer attempts can reject a reachable branch; more attempts only increase
# reset-time work.  It is intentionally fixed until a new public validation
# sweep justifies changing it.
COLLISION_MULTISTART_MAX_ALTERNATES = 32

# Collision paths are sampled at no more than 0.02 rad of active-joint motion
# per interval.  This exactly matches ``PickPlaceConfig.max_joint_step`` and
# the trajectory compiler's upper step, in each joint's URDF positive-axis
# convention.  A larger increment can skip a thin arm/torso intersection,
# while a smaller one only adds reset-time collision queries; the value is
# fixed to the existing compiler contract and is checked before rollout.
COLLISION_PATH_MAX_ACTIVE_STEP_RAD = 0.02


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
        joint_position_limits: Mapping[str, tuple[float, float]] | None = None,
        tolerance: float = 1.0e-4,
        max_iterations: int = 1000,
        integration_step: float = 0.1,
        damping: float = 1.0e-6,
        enable_collision_checking: bool = False,
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
        self.package_dirs = tuple(Path(path).expanduser().resolve() for path in package_dirs)
        missing_package_dirs = [path for path in self.package_dirs if not path.is_dir()]
        if missing_package_dirs:
            raise FileNotFoundError(missing_package_dirs[0])
        # Frame-only IK does not load URDF visual/collision geometry, so mesh
        # package roots are deliberately not passed here. Pinocchio 2.7's
        # model-only overload accepts just the URDF path; package roots belong
        # to the separate geometry-model builders.
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.frame_name = frame_name
        self.frame_id = self.model.getFrameId(frame_name)
        if self.frame_id >= len(self.model.frames):
            raise ValueError(f"frame {frame_name!r} is absent from {self.urdf_path}")

        self.active_joint_names = tuple(active_joint_names)
        if not self.active_joint_names:
            raise ValueError("active_joint_names cannot be empty")
        self.active_velocity_indices: list[int] = []
        self.active_configuration_indices: list[int] = []
        for name in self.active_joint_names:
            joint_id = self.model.getJointId(name)
            if joint_id == 0 or joint_id >= len(self.model.joints):
                raise ValueError(f"joint {name!r} is absent from {self.urdf_path}")
            joint = self.model.joints[joint_id]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(f"joint {name!r} must be scalar, got nq={joint.nq}, nv={joint.nv}")
            self.active_velocity_indices.append(int(joint.idx_v))
            self.active_configuration_indices.append(int(joint.idx_q))

        # Isaac Lab's soft limits are the authoritative safe command envelope.
        # Intersect them with the public URDF's hard limits so a numerically
        # converged reset-time solve cannot compile an action the simulator will
        # clamp. The limits are expressed in radians in each named joint's own
        # positive-axis convention and are supplied from the live loaded asset.
        self.configuration_lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64).copy()
        self.configuration_upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64).copy()
        for name, (sim_lower, sim_upper) in (joint_position_limits or {}).items():
            joint_id = self.model.getJointId(name)
            if joint_id == 0 or joint_id >= len(self.model.joints):
                raise ValueError(f"joint limit supplied for unknown URDF joint {name!r}")
            joint = self.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(f"joint limit supplied for non-scalar joint {name!r}")
            if not np.isfinite(sim_lower) or not np.isfinite(sim_upper) or sim_lower >= sim_upper:
                raise ValueError(f"invalid joint limits for {name!r}: {(sim_lower, sim_upper)}")
            index = int(joint.idx_q)
            self.configuration_lower[index] = max(self.configuration_lower[index], float(sim_lower))
            self.configuration_upper[index] = min(self.configuration_upper[index], float(sim_upper))
            if self.configuration_lower[index] >= self.configuration_upper[index]:
                raise ValueError(f"simulator and URDF joint limits do not overlap for {name!r}")

        if tolerance <= 0.0 or max_iterations <= 0 or integration_step <= 0.0 or damping <= 0.0:
            raise ValueError("IK numerical parameters must be positive")
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.integration_step = float(integration_step)
        self.damping = float(damping)
        self.enable_collision_checking = bool(enable_collision_checking)
        self.geometry_model = None
        self.geometry_data = None
        self.collision_pair_count = 0
        self.collision_filtered_pair_count = 0
        if self.enable_collision_checking:
            self._build_collision_geometry()

    def _build_collision_geometry(self) -> None:
        """Load public URDF collision meshes and retain non-adjacent pairs."""

        pin = self.pin
        try:
            # The geometry model is built from the same URDF and package roots
            # as frame IK.  Passing package_dirs explicitly keeps ``package://``
            # mesh resolution tied to the public asset supplied by the caller.
            geometry_model = pin.buildGeomFromUrdf(
                self.model,
                str(self.urdf_path),
                pin.COLLISION,
                package_dirs=[str(path) for path in self.package_dirs],
            )
            geometry_model.addAllCollisionPairs()
            all_pairs = list(geometry_model.collisionPairs)
            # Read adjacency from this same public URDF instead of exposing
            # ``model.parents``.  Isaac Sim and the cmeel Pinocchio wheel load
            # separate Boost.Python registries in one process, where converting
            # that C++ ``std::vector<size_t>`` can fail even though scalar frame
            # and geometry bindings work.  URDF parent/child link names encode
            # the identical kinematic relation without weakening the gate.
            urdf_root = ElementTree.parse(self.urdf_path).getroot()
            rigid_parent = {
                link.attrib["name"]: link.attrib["name"]
                for link in urdf_root.findall("link")
            }

            def rigid_component(link: str) -> str:
                while rigid_parent[link] != link:
                    rigid_parent[link] = rigid_parent[rigid_parent[link]]
                    link = rigid_parent[link]
                return link

            urdf_joints: list[tuple[str, str, str]] = []
            for joint in urdf_root.findall("joint"):
                parent = joint.find("parent")
                child = joint.find("child")
                if parent is None or child is None:
                    continue
                parent_link = parent.attrib["link"]
                child_link = child.attrib["link"]
                joint_type = joint.attrib.get("type", "")
                urdf_joints.append((joint_type, parent_link, child_link))
                if joint_type == "fixed":
                    first_root = rigid_component(parent_link)
                    second_root = rigid_component(child_link)
                    rigid_parent[second_root] = first_root

            adjacent_components = {
                frozenset((rigid_component(parent), rigid_component(child)))
                for joint_type, parent, child in urdf_joints
                if joint_type != "fixed"
            }
            filtered_pairs = []
            for pair in all_pairs:
                first_object = geometry_model.geometryObjects[pair.first]
                second_object = geometry_model.geometryObjects[pair.second]
                first_link = self.model.frames[int(first_object.parentFrame)].name
                second_link = self.model.frames[int(second_object.parentFrame)].name
                first_component = rigid_component(first_link)
                second_component = rigid_component(second_link)
                same_link = first_component == second_component
                direct_parent_child = (
                    frozenset((first_component, second_component))
                    in adjacent_components
                )
                if same_link or direct_parent_child:
                    filtered_pairs.append(pair)
            # Pinocchio removes a pair by index; reverse order prevents an
            # earlier removal from shifting the remaining pair indices.  The
            # public robot's adjacent meshes overlap by construction, so these
            # exclusions avoid reporting expected same-link/hinge contacts
            # while retaining arm/torso and other non-adjacent self-collisions.
            for pair in reversed(filtered_pairs):
                geometry_model.removeCollisionPair(pair)
            self.geometry_model = geometry_model
            self.geometry_data = pin.GeometryData(geometry_model)
            self.collision_pair_count = len(geometry_model.collisionPairs)
            self.collision_filtered_pair_count = len(filtered_pairs)
        except Exception as exc:
            raise RuntimeError(
                "collision checking requested but public URDF collision geometry could not load: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @property
    def collision_diagnostics(self) -> dict[str, int | bool | float]:
        """Return fixed collision-planning settings for reset diagnostics."""

        return {
            "enabled": self.enable_collision_checking,
            "collision_pair_count": self.collision_pair_count,
            "filtered_adjacent_pair_count": self.collision_filtered_pair_count,
            "multistart_rng_seed": COLLISION_MULTISTART_RNG_SEED,
            "multistart_max_alternates": COLLISION_MULTISTART_MAX_ALTERNATES,
            "path_max_active_step_rad": COLLISION_PATH_MAX_ACTIVE_STEP_RAD,
        }

    def _collision_pairs(self, q: np.ndarray) -> tuple[tuple[str, str], ...]:
        """Return colliding public geometry-object name pairs at ``q``."""

        if not self.enable_collision_checking:
            return ()
        if self.geometry_model is None or self.geometry_data is None:
            raise RuntimeError("collision checking is enabled without geometry data")
        self.pin.computeCollisions(
            self.model,
            self.data,
            self.geometry_model,
            self.geometry_data,
            np.asarray(q, dtype=np.float64),
            False,
        )
        pairs: list[tuple[str, str]] = []
        for index, result in enumerate(self.geometry_data.collisionResults):
            if not result.isCollision():
                continue
            pair = self.geometry_model.collisionPairs[index]
            pairs.append(
                (
                    self.geometry_model.geometryObjects[pair.first].name,
                    self.geometry_model.geometryObjects[pair.second].name,
                )
            )
        return tuple(pairs)

    def validate_configuration(
        self,
        label: str,
        joint_names: Iterable[str],
        positions: np.ndarray,
    ) -> None:
        """Reject a named reset-time configuration when any retained pair collides."""

        if not self.enable_collision_checking:
            return
        q = self.q_from_named_positions(joint_names, positions)
        pairs = self._collision_pairs(q)
        if pairs:
            first_pair = pairs[0]
            raise IKPlanningError(
                f"configuration {label!r} is in collision: "
                f"{first_pair[0]!r} vs {first_pair[1]!r}"
            )

    def _first_path_collision(
        self,
        start_q: np.ndarray,
        end_q: np.ndarray,
    ) -> tuple[int, tuple[str, str]] | None:
        """Check linear active-joint interpolation and return first collision."""

        if not self.enable_collision_checking:
            return None
        start = np.asarray(start_q, dtype=np.float64)
        end = np.asarray(end_q, dtype=np.float64)
        active_start = start[self.active_configuration_indices]
        active_end = end[self.active_configuration_indices]
        max_active_delta = float(np.max(np.abs(active_end - active_start), initial=0.0))
        interval_count = max(1, int(np.ceil(max_active_delta / COLLISION_PATH_MAX_ACTIVE_STEP_RAD)))
        for interval in range(interval_count + 1):
            fraction = interval / interval_count
            q = start + fraction * (end - start)
            pairs = self._collision_pairs(q)
            if pairs:
                return interval, pairs[0]
        return None

    def _alternate_seed(self, seed_q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Randomize only active scalar joints within intersected limits."""

        alternate = np.asarray(seed_q, dtype=np.float64).copy()
        for index in self.active_configuration_indices:
            lower = self.configuration_lower[index]
            upper = self.configuration_upper[index]
            if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
                alternate[index] = rng.uniform(lower, upper)
        return alternate

    def q_from_named_positions(self, joint_names: Iterable[str], positions: np.ndarray) -> np.ndarray:
        """Populate a fixed-base Pinocchio configuration by joint name."""
        names = tuple(joint_names)
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (len(names),):
            raise ValueError("positions must match joint_names")
        q = np.asarray(self.pin.neutral(self.model), dtype=np.float64)
        for name, value in zip(names, positions, strict=True):
            joint_id = self.model.getJointId(name)
            if joint_id == 0 or joint_id >= len(self.model.joints):
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
            if joint_id == 0 or joint_id >= len(self.model.joints):
                continue
            joint = self.model.joints[joint_id]
            if joint.nq == 1:
                result[name] = float(q[joint.idx_q])
        return result

    def frame_pose(self, q: np.ndarray) -> Pose:
        self.pin.framesForwardKinematics(self.model, self.data, np.asarray(q, dtype=np.float64))
        placement = self.data.oMf[self.frame_id]
        return Pose(np.asarray(placement.translation), matrix_quaternion_xyzw(np.asarray(placement.rotation)))

    def _solve_once(self, waypoint_name: str, target: Pose, seed_q: np.ndarray) -> IKSolveReport:
        """Run one damped least-squares solve from exactly one supplied seed."""
        pin = self.pin
        supplied_seed = np.asarray(seed_q, dtype=np.float64).copy()
        q = supplied_seed.copy()
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
            # IK only controls the active scalar joints.  Restoring the
            # supplied inactive coordinates after clipping also guarantees an
            # alternate seed cannot perturb legs, waist, or any other fixed
            # configuration components.
            if self.active_configuration_indices:
                active_set = set(self.active_configuration_indices)
                inactive = [index for index in range(self.model.nq) if index not in active_set]
                q[inactive] = supplied_seed[inactive]

        raise IKPlanningError(
            f"IK failed for {waypoint_name!r} after {self.max_iterations} iterations; residual={residual:.6g}"
        )

    def solve(self, waypoint_name: str, target: Pose, seed_q: np.ndarray) -> IKSolveReport:
        """Solve and collision-check a waypoint completely before rollout."""

        supplied_seed = np.asarray(seed_q, dtype=np.float64).copy()
        if supplied_seed.shape != (self.model.nq,):
            raise ValueError(f"seed_q must have shape ({self.model.nq},), got {supplied_seed.shape}")
        if not self.enable_collision_checking:
            return self._solve_once(waypoint_name, target, supplied_seed)

        rng = np.random.default_rng(COLLISION_MULTISTART_RNG_SEED)
        last_failure = "no candidate was accepted"
        for attempt in range(COLLISION_MULTISTART_MAX_ALTERNATES + 1):
            attempt_seed = (
                supplied_seed
                if attempt == 0
                else self._alternate_seed(supplied_seed, rng)
            )
            try:
                report = self._solve_once(waypoint_name, target, attempt_seed)
            except IKPlanningError as exc:
                last_failure = str(exc)
                continue

            endpoint_pairs = self._collision_pairs(report.q)
            if endpoint_pairs:
                first_pair = endpoint_pairs[0]
                last_failure = (
                    f"endpoint collision between {first_pair[0]!r} and {first_pair[1]!r}"
                )
                continue
            path_collision = self._first_path_collision(supplied_seed, report.q)
            if path_collision is not None:
                _, first_pair = path_collision
                last_failure = (
                    f"straight path collision between {first_pair[0]!r} and {first_pair[1]!r}"
                )
                continue
            return report

        raise IKPlanningError(
            f"collision-aware IK failed for {waypoint_name!r} after "
            f"{COLLISION_MULTISTART_MAX_ALTERNATES} alternate seeds: {last_failure}"
        )

    def validate_trajectory(
        self,
        trajectory: Any,
        *,
        initial_absolute_positions: np.ndarray | None = None,
    ) -> None:
        """Validate every compiled absolute target and its ordered path.

        This is a reset-time gate.  It must run before a trajectory is saved or
        passed to ``OpenLoopPolicy``; no runtime observation or contact state is
        read by the policy or used to change a transition.
        """

        if not self.enable_collision_checking:
            return
        joint_names = tuple(getattr(trajectory, "joint_names", ()))
        absolute_targets = np.asarray(
            getattr(trajectory, "absolute_targets", ()), dtype=np.float64
        )
        phases = tuple(getattr(trajectory, "phases", ()))
        if absolute_targets.ndim != 2 or absolute_targets.shape[1] != len(joint_names):
            raise ValueError("trajectory absolute_targets must be ordered by trajectory joint_names")
        if len(phases) != absolute_targets.shape[0]:
            raise ValueError("trajectory phases must match absolute_targets")

        previous_q = None
        if initial_absolute_positions is not None:
            previous_q = self.q_from_named_positions(joint_names, initial_absolute_positions)
        for step, (absolute_target, phase) in enumerate(zip(absolute_targets, phases, strict=True)):
            q = self.q_from_named_positions(joint_names, absolute_target)
            if previous_q is not None:
                path_collision = self._first_path_collision(previous_q, q)
                if path_collision is not None:
                    _, first_pair = path_collision
                    raise IKPlanningError(
                        f"trajectory collision at step {step} phase {phase!r}: "
                        f"{first_pair[0]!r} vs {first_pair[1]!r}"
                    )
            endpoint_pairs = self._collision_pairs(q)
            if endpoint_pairs:
                first_pair = endpoint_pairs[0]
                raise IKPlanningError(
                    f"trajectory collision at step {step} phase {phase!r}: "
                    f"{first_pair[0]!r} vs {first_pair[1]!r}"
                )
            previous_q = q

    def _clip_configuration(self, q: np.ndarray) -> np.ndarray:
        lower = self.configuration_lower
        upper = self.configuration_upper
        result = q.copy()
        finite = np.isfinite(lower) & np.isfinite(upper) & (lower < upper)
        result[finite] = np.clip(result[finite], lower[finite], upper[finite])
        return result
