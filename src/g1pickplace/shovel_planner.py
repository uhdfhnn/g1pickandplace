"""Reset-time open-loop planner and analytic safety helpers for Task 4.

The shovel is a dynamic simulator asset, not an attachment owned by the
planner.  This module only derives fixed wrist targets from a reset snapshot,
solves every target before constructing a policy, and exposes conservative
compound-tool envelope checks.  Runtime contact and pose observations never
change the compiled trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .geometry import Pose, quaternion_matrix_xyzw
from .trajectory import JointTarget, JointTrajectory, compile_joint_targets


# The ordered labels are the complete fixed shovel program.  They are
# dimensionless compile-time phase identifiers, sourced from the approved
# Task-4 review.  Omitting a phase would remove a required tool orientation or
# support/contact dwell; adding a runtime-selected phase would violate the
# open-loop boundary.  This tuple is intentionally fixed and tested as the
# planner's coverage contract.
SHOVEL_PHASES: tuple[str, ...] = (
    "open_at_home",
    "staging",
    "move_to_tool_pregrasp",
    "descend_to_tool_grasp",
    "close_tool_gripper",
    "tool_grasp_settle",
    "lift_tool",
    "orient_blade_parallel",
    # This fixed phase is a high-Z, behind-red transit at the same world XY
    # used by the scoop approach.  It is a dimensionless program label, added
    # after Gate-09 showed that moving diagonally toward the low scoop target
    # swept the blade through the red cube.  Keeping this transit separate
    # makes the following move_behind_red transition vertical and keeps every
    # intermediate sample subject to the existing collision gate.
    "move_above_behind_red",
    "move_behind_red",
    "lower_blade",
    "insert_blade",
    "tilt_blade_up",
    "lift_loaded_shovel",
    "loaded_settle",
    "transport_loaded_to_tray",
    "tilt_to_unload",
    "unload_settle",
    "retreat_tool",
    "return_via_staging",
    "return_home",
    "final_evaluation_settle",
)

# ``open_at_home`` is the one fixed phase whose identity is the simulator's
# reset robot configuration, not a wrist pose derived from the shovel root.
# Gate-07 swept-clearance evidence showed that converting the reset shovel pose
# through the grasp transform put the hand/tool envelope into the left wrist at
# the first open phase sample.  This compile-time phase set keeps the reset
# frame explicit while still sending it through IK; all later phases remain
# tool-derived because the shovel is physically held.  The set is intentionally
# fixed and not runtime-configurable: adding another reset-frame phase could
# make a carried tool disappear from its physical transform.
SHOVEL_RESET_WRIST_PHASES: frozenset[str] = frozenset({"open_at_home"})

# A 0.12 m world +Z tool lift is a provisional reset-time waypoint height in
# metres.  It is large enough to clear the public 0.05 m block and table
# contact envelope in the analytic model, while a larger lift can exceed the
# left-arm workspace.  The value is fixed in the initial planner profile and
# must be visible-plan calibrated before physical acceptance.
SHOVEL_LIFT_HEIGHT_M = 0.12

# The low scoop lane is raised by 0.022 m from the public table support plane
# in the tool-root frame.  Gate-12 measured up to 0.941 mm of expanded
# handle/table intrusion with the prior 0.021 m offset; the public-URDF sweep
# still left -0.0418 mm at +0.0009 m, while +0.0010 m and larger cleared all
# 22 fixed endpoints with maximum residual 9.987650206768576e-05.  The extra
# millimetre is along world/tool +Z for the identity-oriented behind/lower/
# insert phases and propagates through tilt, loaded, and tray-root targets;
# move_above_behind_red remains on the independent lifted target.  Lower values
# can retain table intrusion; higher values reduce blade-under-cube engagement
# or arm reach.  This fixed metre offset is provisional and needs visible Gate
# C swept-clearance plus physical scoop validation; it is not runtime feedback.
SHOVEL_BLADE_ROOT_SUPPORT_OFFSET_M = 0.022

# The pregrasp root rises by 0.75 of the fixed lift in dimensionless ratio,
# along tool +Z (world +Z for the identity reset orientation).  Gate-08 swept
# clearance found the previous 0.50 fraction left only 0.04 m of wrist height
# after the local -0.02 m grasp offset, so the handle contacted the left
# wrist/yaw-pitch envelope in move_to_tool_pregrasp.  With 0.75, the root
# offset is 0.09 m and the wrist offset is 0.07 m; this clears the 17.5 mm
# handle half-thickness plus the 3 mm slip margin under the current reset
# geometry.  A lower fraction can repeat the early collision; a higher one
# adds descend distance and arm reach.  This fixed compile-time ratio requires
# a fresh visible swept-clearance check and remains non-runtime/configurable.
SHOVEL_PREGRASP_HEIGHT_FRACTION = 0.75

# The blade is inserted 0.055 m along its local front axis (metres) so its
# shallow 0.006 m thickness passes beneath the 0.05 m public block in the
# provisional geometry.  Too little depth cannot establish blade/red contact;
# too much can drive the backstop or blade through the block/table.  It is a
# fixed analytic waypoint, not a runtime feedback distance.
SHOVEL_INSERT_DEPTH_M = 0.055

# A 0.17 m behind-red offset in world +Y places the tool root behind the
# public 50 mm cube while its local -Y wedge points at the cube. It is the
# 135 mm root-to-leading-edge distance plus the cube's 25 mm half extent and a
# 10 mm air gap. Smaller values can begin in contact; larger values add reach
# and insertion travel. Combined with the fixed 55 mm insertion, the handle's
# forward edge remains 35 mm behind the cube centre (10 mm beyond its rear
# face before the 3 mm margin), fixing the real handle/red overlap exposed by
# visible plan 14. This reset-time value must pass visible swept-clearance and
# contact validation and should be rederived if the public cube size changes.
SHOVEL_BEHIND_RED_OFFSET_M = 0.17

# The scoop-lane root is shifted +0.035 m along public world +X relative to
# the red cube; X is horizontal screen-left in the approved front view, while
# the behind-red Y offset above remains world +0.17 m.  The physical rollout
# with local grasp X=-0.025 m established both exact finger contacts but lifted
# the tool only 0.01551485 m before it slipped; wrist images showed the brown
# handle still lateral/outside the closed finger pads.  The requested grasp
# correction moves the wrist 20 mm farther toward tool-local -X.  During the
# identity-oriented scoop phases, adding the same +20 mm in world +X to this
# tool-root lane preserves the previously safe wrist centerline at
# move_above_behind_red: (old tool X + 0.015) + (-0.025) equals
# (new tool X + 0.035) + (-0.045).  This compensation specifically avoids
# reusing the exact left-elbow/torso collision seen with local X=-0.045 and the
# old +0.015 m lane.  Smaller lane offsets can restore that collision after the
# grasp correction; larger offsets can lose blade/red lateral overlap, exceed
# the public arm workspace, or make reset-time IK fail.  The +0.035 m world
# offset is fixed for this coupled provisional calibration, not a runtime
# adjustment, and requires visible inspect, all-waypoint Gate C swept
# clearance, and physical contact validation before acceptance.
SHOVEL_SCOOP_LANE_X_OFFSET_M = 0.035

# The tray-loaded tool root receives a fixed +0.020 m compensation in public
# world +X (metres; +X is presentation-left).  Gate-16 plan-only at grasp
# X=-0.045 m and the uncompensated tray root failed transport IK after 32
# seeds with residual 0.0109621.  The grasp correction moves the wrist -20 mm
# in tool-local X, so shifting only the loaded/unload tray root +20 mm restores
# the already-solved -0.025 m wrist centerline while leaving tray Y, Z, and
# orientation unchanged.  This derivation assumes the authored identity
# tool orientation through the root-X target and the current public tray
# interior; too little compensation can retain the transport IK failure,
# while too much can put the blade/block outside the tray interior or exceed
# the arm workspace.  The value is fixed and provisional, never a runtime
# adjustment, and must pass visible inspect, all-waypoint Gate C, and physical
# tray-support validation.
SHOVEL_TRAY_ROOT_X_COMPENSATION_M = 0.020

# The tool tilt is a fixed -35 degree rotation about world X, represented as a
# unit quaternion in xyzw order. This lifts the local -Y blade leading edge
# lip after insertion without using measured contact; a smaller angle may not
# retain the block and a larger angle can spill or make IK unreachable.
SHOVEL_TILT_QUATERNION_XYZW = (
    -0.3007057995042731,
    0.0,
    0.0,
    0.9537169507482269,
)

# This fixed profile describes the repository-owned USDA compound tool. Sizes
# are full extents in metres in the tool-root local frame: handle along +Y,
# blade width +X, and blade normal +Z. The child colliders are all fixed to one
# dynamic root in the USD; duplicating them as simulator objects would create
# an unphysical disconnected tool. Gate-11 fixed the centred handle X/Y size
# at 35x150 mm after the 160 mm version overlapped the public torso AABB by
# 2.97 mm along world Y. Aborted visible rollout-04 then measured finger-center
# heights z=0.845119655 m and z=0.844903827 m at tool_grasp_settle step 650,
# only 0.216 mm apart, while the old 35 mm handle top was root z=0.812175 m
# + 0.0175 m = 0.829675 m, about 15.3 mm below those centres. Gate20 rejected
# the resulting 50 mm candidate before rollout: its upper envelope created 66
# exact left-elbow overlaps during tilt_blade_up. The 45 mm candidate lowers
# that upper envelope by 5 mm while retaining the fixed bottom and leaving an
# estimated 5.3 mm gap to the rollout-04 finger centres. Its root-local bottom
# remains -0.0175 m and its top is +0.0275 m, exactly 10 mm above the old top,
# from a +0.005 m translation and 0.0225 m half-height. A lower handle can
# remain below the finger centers and fail to grasp; a taller one can repeat
# the exact elbow collision or enter the table/fingers. This fixed
# asset/profile geometry is provisional pending visible reset/viewport,
# all-waypoint Gate C swept-clearance, and physical grasp validation.
SHOVEL_HANDLE_SIZE_M = (0.035, 0.15, 0.045)
SHOVEL_BLADE_SIZE_M = (0.09, 0.075, 0.006)
SHOVEL_BACKSTOP_SIZE_M = (0.09, 0.012, 0.035)

# The local compound bounds conservatively enclose handle, blade, and backstop
# (metres, tool-root frame). The front blade is toward local -Y; this matches
# the authored scene pose that places its front near the red block. Bounds are
# intentionally wider than each primitive so transformed AABB checks cannot
# under-approximate a child collider. The +0.030 m compound Z maximum encloses
# the updated handle top at +0.0275 m with a fixed 2.5 mm conservative pad;
# reducing it could miss handle/table or robot contact, while increasing it can
# reject a reachable path unnecessarily. This bound is fixed to the authored
# asset and must be rechecked by the same visible swept-clearance gate.
SHOVEL_COMPOUND_BOUNDS_MIN_M = (-0.045, -0.135, -0.018)
SHOVEL_COMPOUND_BOUNDS_MAX_M = (0.045, 0.080, 0.030)

# The component bounds are full local-metre collision envelopes copied from
# the repository USDA: the 35x150x45 mm handle, 90x75 mm wedge, and 90x12x35
# mm backstop. The handle's local Y bounds are centred at [-0.075,+0.075] m;
# its local Z bounds are [-0.0175,+0.0275] m after the authored +0.005 m
# translation and 0.0225 m half-height. The compound max-Y remains +0.080 m to
# retain the existing 5 mm conservative enclosure pad, and its max-Z remains
# +0.030 m for a 2.5 mm pad above the new handle top. These bounds let
# preflight exempt only the named contact surface instead of exempting the
# whole compound tool. If they differ from the asset, preflight can either
# miss a collision or reject a safe plan, so the asset/profile fingerprint and
# dependency-light tests keep them fixed.
SHOVEL_COMPONENT_BOUNDS_LOCAL_M: Mapping[
    str, tuple[tuple[float, float, float], tuple[float, float, float]]
] = MappingProxyType(
    {
        "handle": ((-0.0175, -0.075, -0.0175), (0.0175, 0.075, 0.0275)),
        "blade": ((-0.045, -0.135, -0.017), (0.045, -0.060, -0.009)),
        "backstop": ((-0.045, -0.058, -0.0175), (0.045, -0.046, 0.0175)),
    }
)

# Each component AABB is inflated by 3 mm on every local axis to cover a
# provisional held-tool slip/pose error. Three millimetres is 6% of the public
# cube width and remains below the visible 10 mm initial blade gap. A smaller
# margin can miss grip slip; a larger one rejects the narrow intentional scoop
# corridor. It is a fixed conservative preflight margin pending physical trace
# calibration and never changes the runtime trajectory.
SHOVEL_TOOL_SLIP_MARGIN_M = 0.003

# The local wrist frame is 45 mm toward local -X, 60 mm along local +Y, and
# 20 mm below the tool root in local +Z, all in metres.  Local +Y runs along
# the handle toward its rear end, so +0.060 m is 15 mm inward from the former
# +0.075 m rear-face target.  With X=-0.045 m, the exact Gate-18 reset sweep
# using public Pinocchio/HPP-FCL and 32 deterministic alternates found
# y=+0.075 m failed at move_above_behind_red (torso/left_wrist_roll), while
# y=+0.070 m and +0.060 m passed all waypoints with maximum residuals
# 9.90046948e-05 and 9.98203362e-05; y=+0.050 m failed descend with residual
# 0.0173476, and +0.040, +0.030, +0.020, and 0.000 m also failed descend.
# The selected +0.060 m therefore retains 10 mm above the observed lower
# failure boundary while moving the grasp inward from the handle end.  This
# Y correction is coupled to the physical rollout-02 evidence used for X:
# both exact finger contacts were true at Y=+0.075 m, but tool-root lift was
# only 0.01551485 m before slip and wrist frames showed the brown handle
# lateral/outside the closed pads.  Smaller Y can lose handle enclosure or
# descend-time IK; larger Y can restore the torso/wrist collision or leave the
# handle outside the pads.  The -0.045 m X, -0.020 m local-Z, -30 degree
# orientation, +0.035 m scoop lane, +0.020 m tray compensation, and 3 mm slip
# envelope remain unchanged.  This fixed coupled frame is provisional, not
# runtime-configurable, and requires visible reset/viewport, all-waypoint Gate
# C swept clearance, and physical contact validation before acceptance.
# Raising local +Z from -0.030 to -0.020 m raises the wrist 10 mm along the
# tool's +Z axis (which equals world +Z through the identity-oriented scoop
# phases);
# the bracketing Z=-0.025 m candidate failed the strict 1e-4 residual gate with
# residual 0.000161265, while Z=-0.015 m moved into torso/shoulder collision.
# Lower Z can press the palm/body into the handle; higher Z can lose contact
# or exceed the vertical grasp envelope.  This fixed provisional calibration
# requires visible reset/viewport, all-waypoint Gate C, and physical contact
# validation before acceptance.
SHOVEL_TOOL_LOCAL_GRASP_POSITION_M = (-0.045, 0.060, -0.02)
# This -30 degree pitch is a signed angle in degrees about tool-local +Y (also
# world +Y while the tool is at its identity reset orientation, along the
# handle's long axis).  At rollout-03 ``tool_grasp_settle`` step 650, the two
# named finger links were separated by dx=-43.198 mm and dz=27.409 mm; the
# angle arctan2(27.409, 43.198) is about +32.4 degrees, so a -30 degree
# world-Y correction leaves a small residual vertical mismatch instead of
# requiring the exact diagonal.  The angle is fixed from that measured trace,
# not estimated at runtime.  A smaller magnitude can leave the old flat 35
# mm-high handle diagonal to the closing line; a larger magnitude
# can rotate the fingers into the handle/table, alter the blade approach, or
# make IK and collision gates fail.  This provisional angle remains fixed and
# requires visible inspect, all-waypoint Gate C, and physical contact
# validation.
SHOVEL_GRASP_Y_ROTATION_DEG = -30.0

# This named unit quaternion is the xyzw representation of the fixed -30
# degree tool-local/world-Y rotation above: (0, sin(-15 deg), 0, cos(-15
# deg)).  Naming it keeps the testable pitch operation separate from the final
# composed wrist orientation and avoids hiding a frame/sign convention in
# decimal literals.  Its valid use is the identity-oriented grasp calibration;
# applying it in another tool orientation without re-solving the reset-time
# transform would rotate about the wrong world axis.
SHOVEL_GRASP_Y_ROTATION_QUATERNION_XYZW = (
    0.0,
    -0.25881904510252074,
    0.0,
    0.9659258262890683,
)

# The final grasp orientation is xyzw = q_y(-30 deg) ⊗ q_z(-90 deg), so the
# existing -90 degree Z yaw is applied first and the world/tool-frame -30
# degree Y rotation is applied second.  The resulting unit quaternion is
# (0.1830127019,-0.1830127019,-0.6830127019,0.6830127019).  This composition
# assumes active rigid-pose multiplication in ``Pose.compose`` and preserves
# the prior yaw while changing only the closing-line pitch.  A sign/order
# reversal would tilt the handle the opposite way and retain the measured
# vertical mismatch or create a table/blade collision; the value is fixed and
# provisional pending the visible reset/IK/clearance gates and contact trace.
SHOVEL_TOOL_LOCAL_GRASP_QUATERNION_XYZW = (
    0.1830127018922193,
    -0.1830127018922193,
    -0.6830127018922193,
    0.6830127018922193,
)

# The static tray's open front faces world -Y; these full extents are metres in
# the tray-root frame and are used only for interior/support geometry.  A
# narrower interior rejects a valid 0.05 m cube, while a larger one could
# report a block inside while it is visibly outside the authored walls.
SHOVEL_TRAY_OUTER_SIZE_M = (0.22, 0.16, 0.05)
SHOVEL_TRAY_INTERIOR_MIN_LOCAL_M = (-0.09, -0.065, 0.0)
SHOVEL_TRAY_INTERIOR_MAX_LOCAL_M = (0.09, 0.055, 0.05)


class ShovelFrameIK(Protocol):
    """Subset of the public reset-time frame IK interface used here."""

    def q_from_named_positions(
        self, joint_names: tuple[str, ...], positions: np.ndarray
    ) -> np.ndarray: ...

    def named_positions_from_q(
        self, q: np.ndarray, joint_names: tuple[str, ...]
    ) -> dict[str, float]: ...

    def frame_pose(self, q: np.ndarray) -> Pose: ...

    def solve(self, waypoint_name: str, target: Pose, seed_q: np.ndarray) -> Any: ...


def _finite_vector(value: object, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return array.copy()


def _tuple_vector(value: object, size: int, name: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _finite_vector(value, size, name))


def _freeze_array(value: object, size: int, name: str) -> np.ndarray:
    array = _finite_vector(value, size, name)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ShovelToolProfile:
    """Immutable geometry, grasp-frame, and asset metadata for one shovel."""

    name: str
    asset_path: str
    tool_local_grasp_frame: Pose
    compound_bounds_min_local_m: tuple[float, float, float]
    compound_bounds_max_local_m: tuple[float, float, float]
    blade_contact_bounds_min_local_m: tuple[float, float, float]
    blade_contact_bounds_max_local_m: tuple[float, float, float]
    component_bounds_local_m: Mapping[
        str, tuple[tuple[float, float, float], tuple[float, float, float]]
    ] = field(default_factory=lambda: SHOVEL_COMPONENT_BOUNDS_LOCAL_M)
    slip_margin_m: float = SHOVEL_TOOL_SLIP_MARGIN_M
    tray_interior_min_local_m: tuple[float, float, float] = SHOVEL_TRAY_INTERIOR_MIN_LOCAL_M
    tray_interior_max_local_m: tuple[float, float, float] = SHOVEL_TRAY_INTERIOR_MAX_LOCAL_M

    def __post_init__(self) -> None:
        if not self.name or not self.asset_path:
            raise ValueError("shovel profile name and asset_path must be non-empty")
        for field_name in (
            "compound_bounds_min_local_m",
            "compound_bounds_max_local_m",
            "blade_contact_bounds_min_local_m",
            "blade_contact_bounds_max_local_m",
            "tray_interior_min_local_m",
            "tray_interior_max_local_m",
        ):
            vector = _finite_vector(getattr(self, field_name), 3, field_name)
            object.__setattr__(self, field_name, _tuple_vector(vector, 3, field_name))
        if np.any(
            np.asarray(self.compound_bounds_min_local_m)
            >= np.asarray(self.compound_bounds_max_local_m)
        ):
            raise ValueError("compound bounds must have min < max on every axis")
        if np.any(
            np.asarray(self.blade_contact_bounds_min_local_m)
            >= np.asarray(self.blade_contact_bounds_max_local_m)
        ):
            raise ValueError("blade contact bounds must have min < max on every axis")
        if np.any(
            np.asarray(self.tray_interior_min_local_m)
            >= np.asarray(self.tray_interior_max_local_m)
        ):
            raise ValueError("tray interior bounds must have min < max on every axis")
        components: dict[
            str, tuple[tuple[float, float, float], tuple[float, float, float]]
        ] = {}
        for component, raw_bounds in dict(self.component_bounds_local_m).items():
            if len(raw_bounds) != 2:
                raise ValueError(f"{component} component bounds must contain min and max")
            minimum = _tuple_vector(raw_bounds[0], 3, f"{component}.minimum")
            maximum = _tuple_vector(raw_bounds[1], 3, f"{component}.maximum")
            if np.any(np.asarray(minimum) >= np.asarray(maximum)):
                raise ValueError(f"{component} component bounds must have min < max")
            components[str(component)] = (minimum, maximum)
        if set(components) != {"handle", "blade", "backstop"}:
            raise ValueError("component bounds must define exactly handle, blade, and backstop")
        if not np.isfinite(self.slip_margin_m) or self.slip_margin_m < 0.0:
            raise ValueError("slip_margin_m must be finite and non-negative metres")
        object.__setattr__(self, "component_bounds_local_m", MappingProxyType(components))

    @classmethod
    def default(cls, asset_path: str) -> "ShovelToolProfile":
        """Create the public analytic profile for the repository-owned USDA."""

        # Blade contact bounds deliberately exclude the handle/backstop.  They
        # are local metres and cover the shallow blade plus a 1 mm envelope;
        # this is a conservative contact-provenance mask, not a control rule.
        return cls(
            name="entrance_compound_shovel",
            asset_path=str(asset_path),
            tool_local_grasp_frame=Pose(
                np.asarray(SHOVEL_TOOL_LOCAL_GRASP_POSITION_M),
                np.asarray(SHOVEL_TOOL_LOCAL_GRASP_QUATERNION_XYZW),
            ),
            compound_bounds_min_local_m=SHOVEL_COMPOUND_BOUNDS_MIN_M,
            compound_bounds_max_local_m=SHOVEL_COMPOUND_BOUNDS_MAX_M,
            blade_contact_bounds_min_local_m=(-0.046, -0.142, -0.004),
            blade_contact_bounds_max_local_m=(0.046, -0.055, 0.004),
        )

    @property
    def configuration_fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "asset_path": self.asset_path,
            "tool_local_grasp_frame": {
                "position": self.tool_local_grasp_frame.position.tolist(),
                "quaternion_xyzw": self.tool_local_grasp_frame.quaternion_xyzw.tolist(),
            },
            "compound_min": self.compound_bounds_min_local_m,
            "compound_max": self.compound_bounds_max_local_m,
            "blade_min": self.blade_contact_bounds_min_local_m,
            "blade_max": self.blade_contact_bounds_max_local_m,
            "tray_min": self.tray_interior_min_local_m,
            "tray_max": self.tray_interior_max_local_m,
            "components": dict(self.component_bounds_local_m),
            "slip_margin_m": self.slip_margin_m,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class ShovelResetSnapshot:
    """Immutable reset aggregate used by the shovel planner."""

    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    default_joint_positions: np.ndarray
    robot_base_world: Pose
    tool_world: Pose
    tool_twist_world: tuple[float, ...]
    red_block_world: Pose
    red_block_twist_world: tuple[float, ...]
    tray_world: Pose
    tray_twist_world: tuple[float, ...]
    distractors_world: Mapping[str, Pose]
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        positions = _freeze_array(self.joint_positions, len(self.joint_names), "joint_positions")
        defaults = _freeze_array(
            self.default_joint_positions, len(self.joint_names), "default_joint_positions"
        )
        object.__setattr__(self, "joint_positions", positions)
        object.__setattr__(self, "default_joint_positions", defaults)
        for field_name in ("tool_twist_world", "red_block_twist_world", "tray_twist_world"):
            object.__setattr__(
                self,
                field_name,
                _tuple_vector(getattr(self, field_name), 6, field_name),
            )
        if not self.configuration_fingerprint:
            raise ValueError("configuration_fingerprint must be non-empty")
        object.__setattr__(self, "distractors_world", MappingProxyType(dict(self.distractors_world)))


@dataclass(frozen=True)
class ToolWristTransform:
    """Reset-derived immutable transforms between tool root and wrist frame."""

    wrist_from_tool: Pose
    tool_from_wrist: Pose


def derive_wrist_tool_transform(profile: ShovelToolProfile) -> ToolWristTransform:
    """Derive both directions from the explicit tool-local grasp frame."""

    wrist_from_tool = profile.tool_local_grasp_frame
    return ToolWristTransform(
        wrist_from_tool=wrist_from_tool,
        tool_from_wrist=wrist_from_tool.inverse(),
    )


def wrist_pose_from_tool(tool_world: Pose, profile: ShovelToolProfile) -> Pose:
    """Return the wrist world pose for a tool root pose using fixed geometry."""

    return tool_world.compose(derive_wrist_tool_transform(profile).wrist_from_tool)


def tool_pose_from_wrist(wrist_world: Pose, profile: ShovelToolProfile) -> Pose:
    """Return the tool root world pose using the same reset-derived transform."""

    return wrist_world.compose(derive_wrist_tool_transform(profile).tool_from_wrist)


def _box_corners(minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (x, y, z)
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )


def transformed_compound_tool_aabb(
    tool_world: Pose, profile: ShovelToolProfile
) -> tuple[np.ndarray, np.ndarray]:
    """Transform all compound bounds and return a conservative world AABB."""

    minimum = np.asarray(profile.compound_bounds_min_local_m, dtype=np.float64)
    maximum = np.asarray(profile.compound_bounds_max_local_m, dtype=np.float64)
    world_corners = np.asarray(
        [tool_world.transform_point(corner) for corner in _box_corners(minimum, maximum)]
    )
    return world_corners.min(axis=0), world_corners.max(axis=0)


def transformed_blade_aabb(
    tool_world: Pose, profile: ShovelToolProfile
) -> tuple[np.ndarray, np.ndarray]:
    """Transform only the blade contact sub-envelope for provenance masks."""

    minimum = np.asarray(profile.blade_contact_bounds_min_local_m, dtype=np.float64)
    maximum = np.asarray(profile.blade_contact_bounds_max_local_m, dtype=np.float64)
    world_corners = np.asarray(
        [tool_world.transform_point(corner) for corner in _box_corners(minimum, maximum)]
    )
    return world_corners.min(axis=0), world_corners.max(axis=0)


def transformed_component_aabbs(
    tool_world: Pose, profile: ShovelToolProfile
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return conservative world AABBs for each fixed tool collider."""

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    margin = float(profile.slip_margin_m)
    for name, (raw_minimum, raw_maximum) in profile.component_bounds_local_m.items():
        minimum = np.asarray(raw_minimum, dtype=np.float64) - margin
        maximum = np.asarray(raw_maximum, dtype=np.float64) + margin
        world_corners = np.asarray(
            [tool_world.transform_point(corner) for corner in _box_corners(minimum, maximum)]
        )
        result[name] = (world_corners.min(axis=0), world_corners.max(axis=0))
    return result


def transformed_tray_interior_aabb(
    tray_world: Pose, profile: ShovelToolProfile
) -> tuple[np.ndarray, np.ndarray]:
    """Return the tray interior bounds in world coordinates."""

    minimum = np.asarray(profile.tray_interior_min_local_m, dtype=np.float64)
    maximum = np.asarray(profile.tray_interior_max_local_m, dtype=np.float64)
    world_corners = np.asarray(
        [tray_world.transform_point(corner) for corner in _box_corners(minimum, maximum)]
    )
    return world_corners.min(axis=0), world_corners.max(axis=0)


# Intended contacts are compile-time phase masks.  The lower/insert/initial
# tilt phases may touch the table with the blade; after insertion, only the
# blade/red relationship is intentional.  Unknown phase labels fail closed in
# ``validate_shovel_swept_clearance`` rather than treating an unreviewed
# contact as safe.
SHOVEL_INTENDED_CONTACTS_BY_PHASE: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "open_at_home": frozenset({"packing_table"}),
        "staging": frozenset({"packing_table"}),
        "move_to_tool_pregrasp": frozenset({"packing_table"}),
        "descend_to_tool_grasp": frozenset({"packing_table"}),
        "close_tool_gripper": frozenset({"packing_table"}),
        "tool_grasp_settle": frozenset({"packing_table"}),
        "lift_tool": frozenset({"packing_table"}),
        # The high behind-red transit is deliberately contact-free.  It uses
        # the lifted root at roughly 68 mm above the cube top (derived in
        # ``_phase_tool_targets``), so admitting table/red contact here would
        # hide a regression in the dogleg path.
        "move_above_behind_red": frozenset(),
        "lower_blade": frozenset({"packing_table"}),
        # The insert ends with the public 50 mm cube entering the 90 mm-wide
        # blade mouth.  Red-block contact is the task objective in this fixed
        # phase; omitting it makes the conservative AABBs reject the first
        # genuine scoop contact.  No distractor label is admitted here.
        "insert_blade": frozenset({"packing_table", "red_block"}),
        "tilt_blade_up": frozenset({"packing_table", "red_block"}),
        "lift_loaded_shovel": frozenset({"red_block"}),
        "loaded_settle": frozenset({"red_block"}),
        "transport_loaded_to_tray": frozenset({"red_block", "target_tray"}),
        "tilt_to_unload": frozenset({"red_block", "target_tray"}),
        "unload_settle": frozenset({"red_block", "target_tray"}),
        "retreat_tool": frozenset({"target_tray"}),
        "return_home": frozenset({"packing_table"}),
        "final_evaluation_settle": frozenset({"packing_table"}),
    }
)

# Hand/wrist geometry is the intended rigid-body grasp interface after the
# fixed close phase.  These dimensionless name tokens are only an exception to
# the broad robot AABB check; torso, shoulder, elbow, and all unknown robot
# geometry remain collision-fatal.  A missing/renamed hand link is not silently
# ignored because the full robot envelope still has to be supplied.
SHOVEL_ALLOWED_GRASP_ROBOT_TOKENS = ("hand", "wrist")
SHOVEL_GRASP_CONTACT_START_PHASE = "descend_to_tool_grasp"

# During these fixed phases the compound tool is either physically resting on
# the table, leaving that support, or returning to it. Table contact by any
# component is therefore intentional only here. Scoop phases are deliberately
# absent: ``lower_blade``/``insert_blade`` may exempt the blade alone, so a
# handle or backstop/table overlap still fails. The labels come from the frozen
# program and cannot be selected by runtime contact.
# This per-phase component map is intentionally more precise than a whole-tool
# table exemption. ``handle``/``blade``/``backstop`` are geometry labels, not
# runtime observations. All three may overlap the conservative 3 mm-expanded
# envelope while the 120 g shovel rests on or leaves the table.  During the
# scoop, only the blade and its 18 mm-high rear backstop may skim the support
# plane; the handle remains collision-fatal.  These entries were derived from
# the public compound USD dimensions and the 2026-09-06 visible plan-only
# traces (plan 12: reset support; plan 13: insert samples 19--34).  Removing a
# required entry produces false preflight failures; adding ``handle`` to a
# scoop phase could hide a physically invalid downward wrist pose.  The map is
# fixed for this asset/profile and should become configurable if either changes.
SHOVEL_TABLE_CONTACT_COMPONENTS_BY_PHASE: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "open_at_home": frozenset({"handle", "blade", "backstop"}),
        "staging": frozenset({"handle", "blade", "backstop"}),
        "move_to_tool_pregrasp": frozenset({"handle", "blade", "backstop"}),
        "descend_to_tool_grasp": frozenset({"handle", "blade", "backstop"}),
        "close_tool_gripper": frozenset({"handle", "blade", "backstop"}),
        "tool_grasp_settle": frozenset({"handle", "blade", "backstop"}),
        "lift_tool": frozenset({"handle", "blade", "backstop"}),
        "lower_blade": frozenset({"blade", "backstop"}),
        "insert_blade": frozenset({"blade", "backstop"}),
        "tilt_blade_up": frozenset({"blade", "backstop"}),
        "return_home": frozenset({"handle", "blade", "backstop"}),
        "final_evaluation_settle": frozenset({"handle", "blade", "backstop"}),
    }
)


def _overlap(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> bool:
    # A touching face is an intended contact candidate, not an unintentional
    # overlap.  Strict inequalities preserve the conservative collision gate.
    return bool(np.all(min_a < max_b) and np.all(min_b < max_a))


@dataclass(frozen=True)
class ShovelClearanceReport:
    """Immutable reset-time compound-tool clearance result."""

    status: str
    samples_checked: int
    failure_count: int
    first_failures: tuple[Mapping[str, Any], ...] = ()
    robot_check_complete: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "samples_checked": self.samples_checked,
            "failure_count": self.failure_count,
            "first_failures": [dict(item) for item in self.first_failures],
            "robot_check_complete": self.robot_check_complete,
            "intended_contacts_by_phase": {
                phase: sorted(values)
                for phase, values in SHOVEL_INTENDED_CONTACTS_BY_PHASE.items()
            },
        }


def validate_shovel_swept_clearance(
    *,
    phase_tool_poses: Mapping[str, Pose | tuple[Pose, ...] | list[Pose]],
    profile: ShovelToolProfile,
    scene_aabbs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    robot_aabbs_by_phase: Mapping[
        str,
        Mapping[str, tuple[np.ndarray, np.ndarray]]
        | tuple[Mapping[str, tuple[np.ndarray, np.ndarray]], ...]
        | list[Mapping[str, tuple[np.ndarray, np.ndarray]]],
    ]
    | None,
    robot_exact_collisions_by_phase: Mapping[
        str,
        Mapping[str, Mapping[str, bool]]
        | tuple[Mapping[str, Mapping[str, bool]], ...]
        | list[Mapping[str, Mapping[str, bool]]],
    ]
    | None = None,
) -> ShovelClearanceReport:
    """Check the transformed compound tool against static/exact robot checks.

    ``phase_tool_poses`` is reset-time compiled data, not a runtime history.
    ``robot_exact_collisions_by_phase`` contains one reset-time record per
    compiled sample: ``{robot_geometry_name: {tool_component: bool}}``.  It is
    the narrow-phase result from public HPP-FCL/Pinocchio geometry; the AABB
    mapping is only an optional broad-phase envelope and never overrides a
    false exact result.  Missing exact phase/sample records are ``NOT
    VERIFIED``; label/component/sample-shape mismatches are ``FAIL``.  The
    caller may omit an exact record only to obtain an honest non-rollout result.
    """

    expected = set(SHOVEL_PHASES)
    actual = set(phase_tool_poses)
    unknown = actual - expected
    missing = expected - actual
    if unknown or missing:
        details = {
            "kind": "phase_coverage",
            "unknown": sorted(unknown),
            "missing": sorted(missing),
        }
        return ShovelClearanceReport(
            status="FAIL",
            samples_checked=0,
            failure_count=1,
            first_failures=(details,),
            robot_check_complete=False,
        )
    if robot_exact_collisions_by_phase is None:
        return ShovelClearanceReport(
            status="NOT VERIFIED",
            samples_checked=0,
            failure_count=0,
            first_failures=(
                {
                    "kind": "missing_tool_robot_narrow_phase_check",
                    "reason": "public HPP-FCL tool-vs-URDF records were not supplied",
                },
            ),
            robot_check_complete=False,
        )

    failures: list[Mapping[str, Any]] = []
    samples = 0
    for phase in SHOVEL_PHASES:
        raw_tool_poses = phase_tool_poses[phase]
        tool_poses = (
            (raw_tool_poses,)
            if isinstance(raw_tool_poses, Pose)
            else tuple(raw_tool_poses)
        )
        if not tool_poses or not all(isinstance(pose, Pose) for pose in tool_poses):
            return ShovelClearanceReport(
                status="FAIL",
                samples_checked=samples,
                failure_count=len(failures) + 1,
                first_failures=tuple(
                    [
                        *failures[:20],
                        {"kind": "invalid_tool_pose_samples", "phase": phase},
                    ]
                ),
                robot_check_complete=True,
            )
        intended = SHOVEL_INTENDED_CONTACTS_BY_PHASE.get(phase, frozenset())
        static_items = list(scene_aabbs.items())

        # Exact records are mandatory. A missing phase is not a collision
        # result and therefore remains NOT VERIFIED rather than being inferred
        # from a broad AABB or treated as safe.
        if phase not in robot_exact_collisions_by_phase:
            return ShovelClearanceReport(
                status="NOT VERIFIED",
                samples_checked=samples,
                failure_count=len(failures),
                first_failures=tuple(
                    [
                        *failures[:20],
                        {
                            "kind": "missing_robot_exact_phase",
                            "phase": phase,
                        },
                    ]
                ),
                robot_check_complete=False,
            )
        raw_exact_samples = robot_exact_collisions_by_phase[phase]
        exact_samples = (
            (raw_exact_samples,)
            if isinstance(raw_exact_samples, Mapping)
            else tuple(raw_exact_samples)
        )
        if not exact_samples:
            return ShovelClearanceReport(
                status="NOT VERIFIED",
                samples_checked=samples,
                failure_count=len(failures),
                first_failures=tuple(
                    [
                        *failures[:20],
                        {
                            "kind": "missing_robot_exact_samples",
                            "phase": phase,
                        },
                    ]
                ),
                robot_check_complete=False,
            )
        if len(exact_samples) != len(tool_poses):
            return ShovelClearanceReport(
                status="FAIL",
                samples_checked=samples,
                failure_count=len(failures) + 1,
                first_failures=tuple(
                    [
                        *failures[:20],
                        {
                            "kind": "robot_exact_sample_count_mismatch",
                            "phase": phase,
                            "expected": len(tool_poses),
                            "actual": len(exact_samples),
                        },
                    ]
                ),
                robot_check_complete=True,
            )

        # Broad AABBs are optional. If supplied, they must have the same
        # per-sample geometry labels as the exact records, but their overlap is
        # never used as the robot collision decision.
        robot_samples = None
        if robot_aabbs_by_phase is not None and phase in robot_aabbs_by_phase:
            raw_robot_samples = robot_aabbs_by_phase[phase]
            robot_samples = (
                (raw_robot_samples,)
                if isinstance(raw_robot_samples, Mapping)
                else tuple(raw_robot_samples)
            )
            if len(robot_samples) != len(tool_poses):
                return ShovelClearanceReport(
                    status="FAIL",
                    samples_checked=samples,
                    failure_count=len(failures) + 1,
                    first_failures=tuple(
                        [
                            *failures[:20],
                            {"kind": "robot_tool_sample_count_mismatch", "phase": phase},
                        ]
                    ),
                    robot_check_complete=True,
                )

        expected_components = set(profile.component_bounds_local_m)
        for sample_index, tool_pose in enumerate(tool_poses):
            component_aabbs = transformed_component_aabbs(tool_pose, profile)
            exact_record = exact_samples[sample_index]
            if not isinstance(exact_record, Mapping):
                return ShovelClearanceReport(
                    status="FAIL",
                    samples_checked=samples,
                    failure_count=len(failures) + 1,
                    first_failures=tuple(
                        [
                            *failures[:20],
                            {
                                "kind": "invalid_robot_exact_sample",
                                "phase": phase,
                                "sample": sample_index,
                            },
                        ]
                    ),
                    robot_check_complete=True,
                )
            if not exact_record:
                return ShovelClearanceReport(
                    status="NOT VERIFIED",
                    samples_checked=samples,
                    failure_count=len(failures),
                    first_failures=tuple(
                        [
                            *failures[:20],
                            {
                                "kind": "missing_robot_exact_geometry_records",
                                "phase": phase,
                                "sample": sample_index,
                            },
                        ]
                    ),
                    robot_check_complete=False,
                )
            for label, component_results in exact_record.items():
                if not isinstance(label, str) or not isinstance(component_results, Mapping):
                    return ShovelClearanceReport(
                        status="FAIL",
                        samples_checked=samples,
                        failure_count=len(failures) + 1,
                        first_failures=tuple(
                            [
                                *failures[:20],
                                {
                                    "kind": "invalid_robot_exact_record",
                                    "phase": phase,
                                    "sample": sample_index,
                                    "label": str(label),
                                },
                            ]
                        ),
                        robot_check_complete=True,
                    )
                if set(component_results) != expected_components or any(
                    not isinstance(value, (bool, np.bool_))
                    for value in component_results.values()
                ):
                    return ShovelClearanceReport(
                        status="FAIL",
                        samples_checked=samples,
                        failure_count=len(failures) + 1,
                        first_failures=tuple(
                            [
                                *failures[:20],
                                {
                                    "kind": "robot_exact_component_mismatch",
                                    "phase": phase,
                                    "sample": sample_index,
                                    "label": label,
                                    "expected_components": sorted(expected_components),
                                    "actual_components": sorted(component_results),
                                },
                            ]
                        ),
                        robot_check_complete=True,
                    )

            exact_labels = set(exact_record)
            broad_by_label: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None
            if robot_samples is not None:
                broad_by_label = robot_samples[sample_index]
                if not isinstance(broad_by_label, Mapping):
                    return ShovelClearanceReport(
                        status="FAIL",
                        samples_checked=samples,
                        failure_count=len(failures) + 1,
                        first_failures=tuple(
                            [
                                *failures[:20],
                                {
                                    "kind": "invalid_robot_aabb_sample",
                                    "phase": phase,
                                    "sample": sample_index,
                                },
                            ]
                        ),
                        robot_check_complete=True,
                    )
                if set(broad_by_label) != exact_labels:
                    return ShovelClearanceReport(
                        status="FAIL",
                        samples_checked=samples,
                        failure_count=len(failures) + 1,
                        first_failures=tuple(
                            [
                                *failures[:20],
                                {
                                    "kind": "robot_exact_label_mismatch",
                                    "phase": phase,
                                    "sample": sample_index,
                                    "exact_labels": sorted(exact_labels),
                                    "broad_labels": sorted(broad_by_label),
                                },
                            ]
                        ),
                        robot_check_complete=True,
                    )

            def contact_allowed(label: str, component: str) -> bool:
                return bool(
                    label in intended
                    and (
                        (
                            label == "packing_table"
                            and component
                            in SHOVEL_TABLE_CONTACT_COMPONENTS_BY_PHASE.get(
                                phase, frozenset()
                            )
                        )
                        or (label == "red_block" and component in {"blade", "backstop"})
                        or (label == "target_tray" and component == "blade")
                    )
                )

            def hand_grasp_allowed(label: str, component: str) -> bool:
                return bool(
                    any(
                        token in label.casefold()
                        for token in SHOVEL_ALLOWED_GRASP_ROBOT_TOKENS
                    )
                    and component == "handle"
                    and SHOVEL_PHASES.index(phase)
                    >= SHOVEL_PHASES.index(SHOVEL_GRASP_CONTACT_START_PHASE)
                )

            for component, (tool_min, tool_max) in component_aabbs.items():
                for label, (other_min_raw, other_max_raw) in static_items:
                    other_min = _finite_vector(other_min_raw, 3, f"{label}.minimum")
                    other_max = _finite_vector(other_max_raw, 3, f"{label}.maximum")
                    if np.any(other_min >= other_max):
                        return ShovelClearanceReport(
                            status="FAIL",
                            samples_checked=samples,
                            failure_count=len(failures) + 1,
                            first_failures=tuple(
                                [
                                    *failures[:20],
                                    {"kind": "invalid_scene_aabb", "label": label},
                                ]
                            ),
                            robot_check_complete=True,
                        )
                    samples += 1
                    if contact_allowed(label, component) or hand_grasp_allowed(label, component):
                        continue
                    if _overlap(tool_min, tool_max, other_min, other_max):
                        failures.append(
                            {
                                "kind": "compound_tool_overlap",
                                "phase": phase,
                                "sample": sample_index,
                                "label": label,
                                "component": component,
                                "intended_contacts": sorted(intended),
                            }
                        )

                # Exact HPP-FCL results, not the optional robot AABB overlap,
                # decide robot collisions. A true elbow/torso result remains
                # fatal; only the existing post-grasp handle hand/wrist mask
                # can exempt it.
                for label in sorted(exact_labels):
                    if broad_by_label is not None:
                        other_min_raw, other_max_raw = broad_by_label[label]
                        other_min = _finite_vector(other_min_raw, 3, f"{label}.minimum")
                        other_max = _finite_vector(other_max_raw, 3, f"{label}.maximum")
                        if np.any(other_min >= other_max):
                            return ShovelClearanceReport(
                                status="FAIL",
                                samples_checked=samples,
                                failure_count=len(failures) + 1,
                                first_failures=tuple(
                                    [
                                        *failures[:20],
                                        {"kind": "invalid_scene_aabb", "label": label},
                                    ]
                                ),
                                robot_check_complete=True,
                            )
                    samples += 1
                    if contact_allowed(label, component) or hand_grasp_allowed(label, component):
                        continue
                    if bool(exact_record[label][component]):
                        failures.append(
                            {
                                "kind": "exact_tool_robot_overlap",
                                "phase": phase,
                                "sample": sample_index,
                                "label": label,
                                "component": component,
                                "intended_contacts": sorted(intended),
                            }
                        )

    return ShovelClearanceReport(
        status="FAIL" if failures else "PASS",
        samples_checked=samples,
        failure_count=len(failures),
        first_failures=tuple(failures[:20]),
        robot_check_complete=True,
    )


@dataclass(frozen=True)
class ShovelPlanConfig:
    """Fixed timing and gripper settings for the shovel open-loop program."""

    fps: int = 50
    phase_duration_s: float = 0.5
    max_joint_step: float = 0.02
    lift_height_m: float = SHOVEL_LIFT_HEIGHT_M
    insert_depth_m: float = SHOVEL_INSERT_DEPTH_M
    behind_red_offset_m: float = SHOVEL_BEHIND_RED_OFFSET_M
    gripper_joint_names: tuple[str, ...] = (
        "left_hand_Joint1_1",
        "left_hand_Joint2_1",
    )
    gripper_open_positions: tuple[float, ...] = (0.03, 0.03)
    gripper_closed_positions: tuple[float, ...] = (-0.02, -0.02)
    phase_durations_s: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.phase_duration_s <= 0.0 or self.max_joint_step <= 0.0:
            raise ValueError("fps, phase_duration_s, and max_joint_step must be positive")
        if self.lift_height_m <= 0.0 or self.insert_depth_m <= 0.0 or self.behind_red_offset_m <= 0.0:
            raise ValueError("shovel geometric distances must be positive metres")
        if len(self.gripper_joint_names) != len(self.gripper_open_positions):
            raise ValueError("open gripper positions must match gripper joints")
        if len(self.gripper_joint_names) != len(self.gripper_closed_positions):
            raise ValueError("closed gripper positions must match gripper joints")
        durations = dict(self.phase_durations_s or {})
        unknown = set(durations) - set(SHOVEL_PHASES)
        if unknown or any(float(value) <= 0.0 for value in durations.values()):
            raise ValueError("phase_durations_s must contain positive known phase names")
        object.__setattr__(self, "phase_durations_s", MappingProxyType(durations))

    def duration_for(self, phase: str) -> float:
        return float(self.phase_durations_s.get(phase, self.phase_duration_s))


@dataclass(frozen=True)
class ShovelPlanDiagnostics:
    """Reset-time solve and tool-pose evidence for one compiled plan."""

    waypoint_iterations: Mapping[str, int]
    waypoint_residuals: Mapping[str, float]
    waypoint_targets_base: Mapping[str, Pose]
    tool_targets_world: Mapping[str, Pose]
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "waypoint_iterations", MappingProxyType(dict(self.waypoint_iterations)))
        object.__setattr__(self, "waypoint_residuals", MappingProxyType(dict(self.waypoint_residuals)))
        object.__setattr__(self, "waypoint_targets_base", MappingProxyType(dict(self.waypoint_targets_base)))
        object.__setattr__(self, "tool_targets_world", MappingProxyType(dict(self.tool_targets_world)))


def _rotated_tool_pose(position: np.ndarray, quaternion_xyzw: tuple[float, ...]) -> Pose:
    return Pose(np.asarray(position, dtype=np.float64), np.asarray(quaternion_xyzw, dtype=np.float64))


def _phase_tool_targets(
    snapshot: ShovelResetSnapshot, config: ShovelPlanConfig
) -> dict[str, Pose]:
    """Construct the complete fixed world-frame root-pose program."""

    home = snapshot.tool_world
    red = snapshot.red_block_world.position
    tray = snapshot.tray_world.position
    staging = home.translated_world((0.0, 0.0, config.lift_height_m))
    pregrasp = home.translated_world(
        (0.0, 0.0, SHOVEL_PREGRASP_HEIGHT_FRACTION * config.lift_height_m)
    )
    lifted = home.translated_world((0.0, 0.0, config.lift_height_m))
    # The public cube bottom is the table support plane. With the wedge's
    # 17 mm local bottom plus the 3 mm slip envelope, the named 22 mm root
    # offset puts the conservative leading lip 2 mm above the table without
    # disabling collision and keeps the 17.5 mm backstop clear. Gate-12's
    # measured 0.941 mm expanded handle/table intrusion at 21 mm motivates
    # this fixed 1 mm lift; the 45 mm clear approach is below
    # the cube top but keeps the wedge off the table until ``lower_blade``;
    # both are fixed geometry-derived positions, not observed contact waits.
    table_support_z = float(red[2]) - 0.025
    blade_root_z = table_support_z + SHOVEL_BLADE_ROOT_SUPPORT_OFFSET_M
    scoop_lane_x = float(red[0]) + SHOVEL_SCOOP_LANE_X_OFFSET_M
    behind = np.asarray(
        [scoop_lane_x, red[1] + config.behind_red_offset_m, blade_root_z + 0.045],
        dtype=np.float64,
    )
    # The high transit reuses the fixed lifted root (reset tool +0.12 m world
    # +Z) while adopting the behind-red XY.  On the Gate-09 reset trace the
    # red cube top was 0.8190513 + 0.025 = 0.8440513 m and the lifted tool
    # root was about 0.8122761 + 0.12 = 0.9322761 m.  The conservative blade
    # lower envelope is 0.017 m below the root plus the 0.003 m slip margin,
    # leaving 0.9322761 - 0.020 - 0.8440513 = 0.0682248 m (about 68 mm) of
    # cube-top clearance.  This fixed geometry-derived waypoint is not a
    # runtime lift decision: too low recreates the diagonal blade/red sweep;
    # too high can exceed the public arm workspace.  It remains provisional
    # until the visible reset-time IK and swept-clearance gates pass.
    above_behind = np.asarray(
        [behind[0], behind[1], lifted.position[2]],
        dtype=np.float64,
    )
    lower = behind + np.asarray([0.0, 0.0, -0.045])
    insert = lower + np.asarray([0.0, -config.insert_depth_m, 0.0])
    # Raising the root 60 mm while applying the fixed -X tilt keeps the
    # 150 mm handle's +Y rear end above the 0.794051 m support plane: at 35
    # degrees it swings down about 46 mm, plus its 9 mm half-thickness and the
    # 3 mm preflight margin. The prior 20 mm rise produced a real handle/table
    # overlap in visible plan 15. Less than roughly 58 mm can strike the table;
    # much more consumes arm workspace. This fixed lift is geometry-derived
    # for the current handle and must be recalibrated if its length changes.
    tilted = _rotated_tool_pose(
        insert + np.asarray([0.0, 0.0, 0.06]),
        SHOVEL_TILT_QUATERNION_XYZW,
    )
    loaded = tilted.translated_world((0.0, 0.0, config.lift_height_m))
    tray_center = np.asarray(tray, dtype=np.float64)
    # Local blade centre is 87.5 mm ahead of the root on -Y, so both tray roots
    # shift +87.5 mm in world Y to centre the payload. Loaded transport keeps
    # the already solved lift height: the 2026-09-06 visible trials that also
    # descended 37 mm failed transport and unload IK with 0.0318405 and
    # 0.0397133 residuals. Unload therefore changes only orientation at the
    # same height, leaving about 137 mm above the tray root; this clears the
    # 50 mm walls but increases drop distance. These are fixed reset-time
    # waypoints; too low can hit a wall and too high can spill the cube.
    tray_loaded_root = np.asarray(
        [
            tray_center[0] + SHOVEL_TRAY_ROOT_X_COMPENSATION_M,
            tray_center[1] + 0.0875,
            loaded.position[2],
        ],
        dtype=np.float64,
    )
    tray_unload_root = tray_loaded_root.copy()
    tray_loaded_pose = _rotated_tool_pose(
        tray_loaded_root,
        SHOVEL_TILT_QUATERNION_XYZW,
    )
    tray_unload_pose = _rotated_tool_pose(
        tray_unload_root,
        (0.0, 0.0, 0.0, 1.0),
    )
    return {
        "open_at_home": home,
        "staging": staging,
        "move_to_tool_pregrasp": pregrasp,
        "descend_to_tool_grasp": home,
        "close_tool_gripper": home,
        "tool_grasp_settle": home,
        "lift_tool": lifted,
        "orient_blade_parallel": lifted,
        "move_above_behind_red": _rotated_tool_pose(
            above_behind, (0.0, 0.0, 0.0, 1.0)
        ),
        "move_behind_red": _rotated_tool_pose(behind, (0.0, 0.0, 0.0, 1.0)),
        "lower_blade": _rotated_tool_pose(lower, (0.0, 0.0, 0.0, 1.0)),
        "insert_blade": _rotated_tool_pose(insert, (0.0, 0.0, 0.0, 1.0)),
        "tilt_blade_up": tilted,
        "lift_loaded_shovel": loaded,
        "loaded_settle": loaded,
        "transport_loaded_to_tray": tray_loaded_pose,
        "tilt_to_unload": tray_unload_pose,
        "unload_settle": tray_unload_pose,
        "retreat_tool": tray_unload_pose.translated_world((0.0, 0.0, config.lift_height_m)),
        "return_via_staging": staging,
        "return_home": home,
        "final_evaluation_settle": home,
    }


class ResetTimeShovelPlanner:
    """Compile all shovel IK and joint targets before policy construction."""

    def __init__(self, ik: ShovelFrameIK, profile: ShovelToolProfile, config: ShovelPlanConfig | None = None):
        self.ik = ik
        self.profile = profile
        self.config = config or ShovelPlanConfig()

    def build(
        self, snapshot: ShovelResetSnapshot
    ) -> tuple[JointTrajectory, ShovelPlanDiagnostics]:
        cfg = self.config
        if snapshot.configuration_fingerprint != self.profile.configuration_fingerprint:
            raise ValueError("shovel snapshot/profile configuration fingerprint mismatch")
        missing = [name for name in cfg.gripper_joint_names if name not in snapshot.joint_names]
        if missing:
            raise ValueError(f"simulator is missing shovel gripper joints: {missing}")
        base_from_world = snapshot.robot_base_world.inverse()
        reset_q = np.asarray(
            self.ik.q_from_named_positions(snapshot.joint_names, snapshot.joint_positions),
            dtype=np.float64,
        ).copy()
        reset_q.setflags(write=False)
        q_seed = reset_q.copy()
        targets_world = _phase_tool_targets(snapshot, cfg)
        transform = derive_wrist_tool_transform(self.profile)
        reports: dict[str, Any] = {}
        targets_base: dict[str, Pose] = {}
        for phase in SHOVEL_PHASES:
            if phase in SHOVEL_RESET_WRIST_PHASES:
                # The first waypoint must validate the reset robot wrist
                # frame itself.  It is not a permission to skip IK or a
                # post-reset write: the exact reset q is immutable, passed to
                # the public frame query, and still solved below before policy
                # construction.
                wrist_base = self.ik.frame_pose(reset_q)
            else:
                tool_world = targets_world[phase]
                wrist_world = tool_world.compose(transform.wrist_from_tool)
                wrist_base = Pose(
                    base_from_world.transform_point(wrist_world.position),
                    base_from_world.compose(wrist_world).quaternion_xyzw,
                )
            targets_base[phase] = wrist_base
            # Every solve occurs here, before ``compile_joint_targets`` and
            # before callers can construct ``OpenLoopPolicy``.  Duplicate
            # dwell targets are solved explicitly so a phase cannot silently
            # inherit an unchecked orientation/transform.
            report = self.ik.solve(phase, wrist_base, q_seed)
            reports[phase] = report
            q_seed = report.q

        named_solutions = {
            phase: self.ik.named_positions_from_q(report.q, snapshot.joint_names)
            for phase, report in reports.items()
        }
        index_by_name = {name: index for index, name in enumerate(snapshot.joint_names)}

        def full_target(phase: str, closed: bool) -> np.ndarray:
            positions = snapshot.joint_positions.copy()
            for name, value in named_solutions[phase].items():
                positions[index_by_name[name]] = value
            grip = cfg.gripper_closed_positions if closed else cfg.gripper_open_positions
            for name, value in zip(cfg.gripper_joint_names, grip, strict=True):
                positions[index_by_name[name]] = value
            return positions

        program_targets: list[JointTarget] = []
        closed = False
        for phase in SHOVEL_PHASES:
            if phase == "close_tool_gripper":
                closed = True
            program_targets.append(
                JointTarget(
                    phase,
                    full_target(phase, closed),
                    cfg.duration_for(phase),
                )
            )
        trajectory = compile_joint_targets(
            joint_names=snapshot.joint_names,
            initial_absolute_positions=snapshot.joint_positions,
            default_joint_positions=snapshot.default_joint_positions,
            targets=program_targets,
            fps=cfg.fps,
            action_scale=1.0,
            max_joint_step=cfg.max_joint_step,
        )
        diagnostics = ShovelPlanDiagnostics(
            waypoint_iterations={phase: int(reports[phase].iterations) for phase in SHOVEL_PHASES},
            waypoint_residuals={phase: float(reports[phase].residual) for phase in SHOVEL_PHASES},
            waypoint_targets_base=targets_base,
            tool_targets_world=targets_world,
            configuration_fingerprint=self.profile.configuration_fingerprint,
        )
        return trajectory, diagnostics


__all__ = [
    "SHOVEL_PHASES",
    "SHOVEL_RESET_WRIST_PHASES",
    "SHOVEL_BLADE_ROOT_SUPPORT_OFFSET_M",
    "SHOVEL_PREGRASP_HEIGHT_FRACTION",
    "SHOVEL_SCOOP_LANE_X_OFFSET_M",
    "SHOVEL_TRAY_ROOT_X_COMPENSATION_M",
    "SHOVEL_GRASP_Y_ROTATION_DEG",
    "SHOVEL_GRASP_Y_ROTATION_QUATERNION_XYZW",
    "SHOVEL_TOOL_LOCAL_GRASP_QUATERNION_XYZW",
    "SHOVEL_INTENDED_CONTACTS_BY_PHASE",
    "ShovelToolProfile",
    "ShovelResetSnapshot",
    "ShovelPlanConfig",
    "ShovelPlanDiagnostics",
    "ShovelClearanceReport",
    "ToolWristTransform",
    "ResetTimeShovelPlanner",
    "derive_wrist_tool_transform",
    "wrist_pose_from_tool",
    "tool_pose_from_wrist",
    "transformed_compound_tool_aabb",
    "transformed_blade_aabb",
    "transformed_component_aabbs",
    "transformed_tray_interior_aabb",
    "validate_shovel_swept_clearance",
]
