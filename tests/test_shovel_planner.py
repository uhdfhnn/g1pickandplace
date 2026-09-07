from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from g1pickplace import OpenLoopPolicy, Pose
from g1pickplace.shovel_planner import (
    SHOVEL_BLADE_ROOT_SUPPORT_OFFSET_M,
    SHOVEL_INTENDED_CONTACTS_BY_PHASE,
    SHOVEL_COMPOUND_BOUNDS_MIN_M,
    SHOVEL_COMPOUND_BOUNDS_MAX_M,
    SHOVEL_COMPONENT_BOUNDS_LOCAL_M,
    SHOVEL_GRASP_Y_ROTATION_DEG,
    SHOVEL_GRASP_Y_ROTATION_QUATERNION_XYZW,
    SHOVEL_PREGRASP_HEIGHT_FRACTION,
    SHOVEL_PHASES,
    SHOVEL_SCOOP_LANE_X_OFFSET_M,
    SHOVEL_SOCKET_BOUNDS_MAX_LOCAL_M,
    SHOVEL_SOCKET_BOUNDS_MIN_LOCAL_M,
    SHOVEL_SOCKET_FRONT_RAIL_CENTER_Y_M,
    SHOVEL_SOCKET_HOLE_SIZE_M,
    SHOVEL_SOCKET_LEFT_HOLE_BOUNDS_LOCAL_M,
    SHOVEL_SOCKET_OUTER_RAIL_CENTER_X_M,
    SHOVEL_SOCKET_RAIL_CENTER_Z_M,
    SHOVEL_SOCKET_RAIL_SIZE_M,
    SHOVEL_SOCKET_REAR_RAIL_CENTER_Y_M,
    SHOVEL_SOCKET_RIB_SIZE_M,
    SHOVEL_SOCKET_RIGHT_HOLE_BOUNDS_LOCAL_M,
    SHOVEL_TRAY_ROOT_X_COMPENSATION_M,
    SHOVEL_TOOL_LOCAL_GRASP_POSITION_M,
    SHOVEL_TOOL_LOCAL_GRASP_QUATERNION_XYZW,
    ResetTimeShovelPlanner,
    ShovelPlanConfig,
    ShovelResetSnapshot,
    ShovelToolProfile,
    derive_wrist_tool_transform,
    tool_pose_from_wrist,
    transformed_compound_tool_aabb,
    validate_shovel_swept_clearance,
    wrist_pose_from_tool,
)
from g1pickplace.geometry import quaternion_matrix_xyzw


ASSET = Path(__file__).parents[1] / "assets" / "shovel" / "compound_dynamic_shovel.usda"
TRAY_ASSET = Path(__file__).parents[1] / "assets" / "shovel" / "open_front_static_tray.usda"


def _profile() -> ShovelToolProfile:
    return ShovelToolProfile.default(str(ASSET))


def _snapshot(profile: ShovelToolProfile | None = None) -> ShovelResetSnapshot:
    profile = profile or _profile()
    return ShovelResetSnapshot(
        joint_names=("left_arm", "left_hand_Joint1_1", "left_hand_Joint2_1"),
        joint_positions=np.zeros(3),
        default_joint_positions=np.zeros(3),
        robot_base_world=Pose.identity(),
        tool_world=Pose(np.asarray([0.0, 0.0, 0.84]), np.asarray([0.0, 0.0, 0.0, 1.0])),
        tool_twist_world=(0.0,) * 6,
        red_block_world=Pose(
            np.asarray([0.15, 0.0, 0.825]), np.asarray([0.0, 0.0, 0.0, 1.0])
        ),
        red_block_twist_world=(0.0,) * 6,
        tray_world=Pose(
            np.asarray([0.4, 0.0, 0.814]), np.asarray([0.0, 0.0, 0.0, 1.0])
        ),
        tray_twist_world=(0.0,) * 6,
        distractors_world={
            "yellow": Pose(
                np.asarray([-0.2, 0.0, 0.825]), np.asarray([0.0, 0.0, 0.0, 1.0])
            ),
            "green": Pose(
                np.asarray([-0.35, 0.0, 0.825]), np.asarray([0.0, 0.0, 0.0, 1.0])
            ),
        },
        configuration_fingerprint=profile.configuration_fingerprint,
    )


def test_public_assets_have_one_dynamic_root_and_static_open_tray() -> None:
    tool_text = ASSET.read_text(encoding="utf-8")
    tray_text = TRAY_ASSET.read_text(encoding="utf-8")
    assert tool_text.count("PhysicsRigidBodyAPI") == 1
    for child in (
        "left_outer_rail",
        "right_outer_rail",
        "central_rib",
        "front_rail",
        "rear_rail",
        "backstop",
    ):
        assert f'Cube "{child}"' in tool_text
    assert tool_text.count('rel material:binding:physics = </ShovelTool/GripPhysicsMaterial>') == 5
    assert 'Mesh "shallow_wedge_blade"' in tool_text
    assert 'physics:approximation = "convexHull"' in tool_text
    assert 'PhysicsCollisionAPI' in tool_text
    assert "PhysicsRigidBodyAPI" not in tray_text
    for child in ("floor", "left_side", "right_side", "rear", "front_ramp"):
        assert f'Cube "{child}"' in tray_text


def test_dual_finger_socket_asset_bounds_and_grasp_frame_match() -> None:
    tool_text = ASSET.read_text(encoding="utf-8")
    rail_names = (
        "left_outer_rail",
        "right_outer_rail",
        "central_rib",
        "front_rail",
        "rear_rail",
    )
    # Each rail is a colliding child under the one dynamic root. The central
    # rib and four perimeter rails leave two empty cavities; no solid primitive
    # is allowed to occupy either socket opening.
    assert tool_text.count('def Cube "') == 6
    for rail in rail_names:
        assert f'def Cube "{rail}"' in tool_text
        assert tool_text.count(f'def Cube "{rail}"') == 1
    assert SHOVEL_SOCKET_RAIL_SIZE_M == (0.008, 0.060, 0.040)
    assert SHOVEL_SOCKET_RIB_SIZE_M == (0.010, 0.060, 0.040)
    assert SHOVEL_SOCKET_HOLE_SIZE_M == (0.042, 0.044)
    assert SHOVEL_SOCKET_OUTER_RAIL_CENTER_X_M == pytest.approx(0.051)
    assert SHOVEL_SOCKET_FRONT_RAIL_CENTER_Y_M == pytest.approx(-0.116)
    assert SHOVEL_SOCKET_REAR_RAIL_CENTER_Y_M == pytest.approx(-0.064)
    assert SHOVEL_SOCKET_RAIL_CENTER_Z_M == pytest.approx(0.011)
    assert SHOVEL_SOCKET_BOUNDS_MIN_LOCAL_M == (-0.055, -0.120, -0.009)
    assert SHOVEL_SOCKET_BOUNDS_MAX_LOCAL_M == (0.055, -0.060, 0.031)
    assert SHOVEL_COMPONENT_BOUNDS_LOCAL_M["finger_socket"] == (
        SHOVEL_SOCKET_BOUNDS_MIN_LOCAL_M,
        SHOVEL_SOCKET_BOUNDS_MAX_LOCAL_M,
    )
    assert SHOVEL_SOCKET_LEFT_HOLE_BOUNDS_LOCAL_M == ((-0.047, -0.112), (-0.005, -0.068))
    assert SHOVEL_SOCKET_RIGHT_HOLE_BOUNDS_LOCAL_M == ((0.005, -0.112), (0.047, -0.068))
    for hole_min, hole_max in (
        SHOVEL_SOCKET_LEFT_HOLE_BOUNDS_LOCAL_M,
        SHOVEL_SOCKET_RIGHT_HOLE_BOUNDS_LOCAL_M,
    ):
        assert hole_max[0] - hole_min[0] == pytest.approx(SHOVEL_SOCKET_HOLE_SIZE_M[0])
        assert hole_max[1] - hole_min[1] == pytest.approx(SHOVEL_SOCKET_HOLE_SIZE_M[1])
    assert SHOVEL_COMPOUND_BOUNDS_MIN_M == (-0.058, -0.138, -0.018)
    assert SHOVEL_COMPOUND_BOUNDS_MAX_M == (0.058, -0.043, 0.034)
    profile = _profile()
    np.testing.assert_allclose(
        profile.tool_local_grasp_frame.position,
        SHOVEL_TOOL_LOCAL_GRASP_POSITION_M,
    )
    assert SHOVEL_TOOL_LOCAL_GRASP_POSITION_M[1] == pytest.approx(0.060)
    assert SHOVEL_TOOL_LOCAL_GRASP_POSITION_M[0] == pytest.approx(-0.035)
    np.testing.assert_allclose(
        derive_wrist_tool_transform(profile).wrist_from_tool.position,
        (-0.035, 0.060, -0.02),
    )


def test_tool_wrist_transform_round_trips_without_recalibration() -> None:
    profile = _profile()
    tool = Pose(
        np.asarray([1.0, -2.0, 0.8]),
        np.asarray([0.0, 0.38268343, 0.0, 0.92387953]),
    )
    wrist = wrist_pose_from_tool(tool, profile)
    reconstructed = tool_pose_from_wrist(wrist, profile)
    np.testing.assert_allclose(reconstructed.position, tool.position, atol=1.0e-9)
    np.testing.assert_allclose(reconstructed.quaternion_xyzw, tool.quaternion_xyzw, atol=1.0e-9)
    transform = derive_wrist_tool_transform(profile)
    np.testing.assert_allclose(
        transform.wrist_from_tool.position,
        (-0.035, 0.060, -0.02),
    )


def test_grasp_orientation_matches_measured_diagonal_closing_line() -> None:
    profile = _profile()
    expected_xyzw = np.asarray(
        (0.1830127019, -0.1830127019, -0.6830127019, 0.6830127019),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        SHOVEL_TOOL_LOCAL_GRASP_QUATERNION_XYZW,
        expected_xyzw,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        profile.tool_local_grasp_frame.quaternion_xyzw,
        expected_xyzw,
        atol=1.0e-10,
    )
    # Check the full-precision named constant rather than the rounded display
    # tuple above; the profile also normalizes it when constructing its Pose.
    assert np.linalg.norm(SHOVEL_TOOL_LOCAL_GRASP_QUATERNION_XYZW) == pytest.approx(
        1.0, abs=1.0e-15
    )
    assert SHOVEL_GRASP_Y_ROTATION_DEG == pytest.approx(-30.0)

    # These signed/unsigned components are named measurements from the
    # aborted rollout-03 step-650 wrist frame: x is the observed horizontal
    # Link1_1-minus-Link2_1 separation (-43.198 mm), while z is its 27.409 mm
    # vertical separation magnitude.  The test vector intentionally keeps the
    # measured horizontal sign and positive vertical magnitude because the
    # acceptance claim is reduction of |z| under the fixed -30 degree pitch.
    measured_finger_dx_m = -0.043198
    measured_finger_z_separation_m = 0.027409
    measured_separation = np.asarray(
        [measured_finger_dx_m, 0.0, measured_finger_z_separation_m],
        dtype=np.float64,
    )
    rotated_separation = quaternion_matrix_xyzw(
        SHOVEL_GRASP_Y_ROTATION_QUATERNION_XYZW
    ) @ measured_separation
    np.testing.assert_allclose(
        abs(rotated_separation[2]),
        0.0021378903,
        atol=1.0e-8,
    )
    assert abs(rotated_separation[2]) < abs(measured_separation[2])


def test_snapshot_arrays_and_mapping_are_immutable() -> None:
    snapshot = _snapshot()
    with pytest.raises(ValueError):
        snapshot.joint_positions[0] = 1.0
    with pytest.raises(TypeError):
        snapshot.distractors_world["yellow"] = snapshot.distractors_world["green"]
    with pytest.raises(FrozenInstanceError):
        snapshot.configuration_fingerprint = "changed"


class _FakeIK:
    def __init__(self, fail_phase: str | None = None) -> None:
        self.fail_phase = fail_phase
        self.solve_calls: list[str] = []

    def q_from_named_positions(self, joint_names, positions):
        del joint_names
        return np.asarray(positions, dtype=np.float64).copy()

    def frame_pose(self, q):
        del q
        return Pose.identity()

    def named_positions_from_q(self, q, joint_names):
        return {name: float(q[index]) for index, name in enumerate(joint_names)}

    def solve(self, waypoint_name, target, seed_q):
        self.solve_calls.append(waypoint_name)
        if waypoint_name == self.fail_phase:
            raise RuntimeError(f"synthetic shovel IK failure: {waypoint_name}")
        q = np.asarray(seed_q, dtype=np.float64).copy()
        q[0] = target.position[0]
        return SimpleNamespace(q=q, iterations=2, residual=1.0e-6)


def test_open_at_home_uses_reset_wrist_frame_not_tool_grasp_transform() -> None:
    ik = _FakeIK()
    _, diagnostics = ResetTimeShovelPlanner(
        ik,
        _profile(),
        ShovelPlanConfig(fps=10, phase_duration_s=0.1, max_joint_step=10.0),
    ).build(_snapshot())
    reset_q = ik.q_from_named_positions(
        _snapshot().joint_names,
        _snapshot().joint_positions,
    )
    reset_wrist = ik.frame_pose(reset_q)
    np.testing.assert_allclose(
        diagnostics.waypoint_targets_base["open_at_home"].position,
        reset_wrist.position,
    )
    np.testing.assert_allclose(
        diagnostics.waypoint_targets_base["open_at_home"].quaternion_xyzw,
        reset_wrist.quaternion_xyzw,
    )
    assert not np.allclose(
        diagnostics.waypoint_targets_base["open_at_home"].position,
        diagnostics.waypoint_targets_base["descend_to_tool_grasp"].position,
    )


def test_all_shovel_ik_precedes_frozen_policy() -> None:
    ik = _FakeIK()
    trajectory, diagnostics = ResetTimeShovelPlanner(
        ik,
        _profile(),
        ShovelPlanConfig(fps=10, phase_duration_s=0.1, max_joint_step=10.0),
    ).build(_snapshot())
    assert tuple(ik.solve_calls) == SHOVEL_PHASES
    assert tuple(diagnostics.waypoint_targets_base) == SHOVEL_PHASES
    assert tuple(dict.fromkeys(trajectory.phases)) == SHOVEL_PHASES
    policy = OpenLoopPolicy(trajectory)
    first = policy.act({"unexpected": 1})
    policy.reset()
    np.testing.assert_allclose(first, policy.act({"different": 2}))
    assert tuple(ik.solve_calls) == SHOVEL_PHASES


def test_pregrasp_height_uses_gate08_clearance_fraction() -> None:
    snapshot = _snapshot()
    config = ShovelPlanConfig(fps=10, phase_duration_s=0.1, max_joint_step=10.0)
    _, diagnostics = ResetTimeShovelPlanner(_FakeIK(), _profile(), config).build(snapshot)
    expected_z = (
        snapshot.tool_world.position[2]
        + SHOVEL_PREGRASP_HEIGHT_FRACTION * config.lift_height_m
    )
    np.testing.assert_allclose(
        diagnostics.tool_targets_world["move_to_tool_pregrasp"].position[2],
        expected_z,
    )


def test_above_behind_red_is_high_vertical_transit_before_scoop() -> None:
    snapshot = _snapshot()
    config = ShovelPlanConfig(fps=10, phase_duration_s=0.1, max_joint_step=10.0)
    _, diagnostics = ResetTimeShovelPlanner(_FakeIK(), _profile(), config).build(snapshot)
    above = diagnostics.tool_targets_world["move_above_behind_red"]
    behind = diagnostics.tool_targets_world["move_behind_red"]

    # The new fixed waypoint preserves behind-red world XY and uses the same
    # reset-time lift as ``lift_tool``; the next waypoint therefore changes Z
    # only.  This is the regression contract for Gate-09's diagonal sweep
    # correction, not a runtime contact-driven route.
    np.testing.assert_allclose(above.position[:2], behind.position[:2])
    np.testing.assert_allclose(
        above.position[2],
        snapshot.tool_world.position[2] + config.lift_height_m,
    )
    assert above.position[2] > behind.position[2]


def test_scoop_lane_uses_fixed_world_x_offset_for_all_approach_phases() -> None:
    snapshot = _snapshot()
    config = ShovelPlanConfig(fps=10, phase_duration_s=0.1, max_joint_step=10.0)
    _, diagnostics = ResetTimeShovelPlanner(_FakeIK(), _profile(), config).build(snapshot)
    assert SHOVEL_SCOOP_LANE_X_OFFSET_M == pytest.approx(0.025)
    expected_x = snapshot.red_block_world.position[0] + SHOVEL_SCOOP_LANE_X_OFFSET_M
    for phase in (
        "move_above_behind_red",
        "move_behind_red",
        "lower_blade",
        "insert_blade",
    ):
        np.testing.assert_allclose(
            diagnostics.tool_targets_world[phase].position[0],
            expected_x,
        )
    # Tilt/lift targets are derived from insert_blade and must retain the same
    # fixed lane rather than silently returning to the cube centreline.
    for phase in ("tilt_blade_up", "lift_loaded_shovel", "loaded_settle"):
        np.testing.assert_allclose(
            diagnostics.tool_targets_world[phase].position[0],
            expected_x,
        )


def test_tray_root_compensation_restores_previous_wrist_centerline() -> None:
    snapshot = _snapshot()
    config = ShovelPlanConfig(fps=10, phase_duration_s=0.1, max_joint_step=10.0)
    profile = _profile()
    _, diagnostics = ResetTimeShovelPlanner(_FakeIK(), profile, config).build(snapshot)

    tray_root = diagnostics.tool_targets_world["transport_loaded_to_tray"]
    assert SHOVEL_TRAY_ROOT_X_COMPENSATION_M == pytest.approx(0.010)
    np.testing.assert_allclose(
        tray_root.position[0],
        snapshot.tray_world.position[0] + SHOVEL_TRAY_ROOT_X_COMPENSATION_M,
    )
    # With the identity-X component of the authored tray orientation, the new
    # (+10 mm root, -35 mm grasp) wrist X equals the old uncompensated
    # (-25 mm grasp) wrist centerline. This is a reset-time arithmetic
    # regression check, not a runtime correction or contact exemption.
    new_wrist_x = wrist_pose_from_tool(tray_root, profile).position[0]
    old_wrist_x = snapshot.tray_world.position[0] - 0.025
    np.testing.assert_allclose(new_wrist_x, old_wrist_x)
    np.testing.assert_allclose(
        tray_root.position[1:],
        (
            diagnostics.tool_targets_world["tilt_to_unload"].position[1],
            diagnostics.tool_targets_world["lift_loaded_shovel"].position[2],
        ),
    )


def test_ik_failure_happens_before_trajectory_or_policy_creation() -> None:
    ik = _FakeIK(fail_phase="insert_blade")
    with pytest.raises(RuntimeError, match="insert_blade"):
        ResetTimeShovelPlanner(
            ik,
            _profile(),
            ShovelPlanConfig(fps=10, phase_duration_s=0.1, max_joint_step=10.0),
        ).build(_snapshot())
    assert ik.solve_calls == list(SHOVEL_PHASES[: SHOVEL_PHASES.index("insert_blade") + 1])


def test_transformed_compound_envelope_rotates_and_translates() -> None:
    profile = _profile()
    tool = Pose(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)]),
    )
    minimum, maximum = transformed_compound_tool_aabb(tool, profile)
    # The socket/blade envelope is entirely on local -Y, so a +90-degree yaw
    # moves its world-X interval to the positive side of the root.  Checking
    # the resulting intervals catches stale positive-Y geometry bounds while
    # retaining the transform contract for the new compound geometry.
    np.testing.assert_allclose(minimum, (1.043, 1.942, 2.982), atol=1.0e-12)
    np.testing.assert_allclose(maximum, (1.138, 2.058, 3.034), atol=1.0e-12)
    assert np.all(maximum > minimum)


def _phase_poses() -> dict[str, Pose]:
    return {
        phase: Pose(
            np.asarray([float(index), 2.0, 3.0]),
            np.asarray([0.0, 0.0, 0.0, 1.0]),
        )
        for index, phase in enumerate(SHOVEL_PHASES)
    }


def _empty_robot_aabbs() -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    return {
        phase: {
            "robot_link": (
                np.asarray([-100.0, -100.0, -100.0]),
                np.asarray([-99.0, -99.0, -99.0]),
            )
        }
        for phase in SHOVEL_PHASES
    }


def _empty_robot_exact() -> dict[str, dict[str, dict[str, bool]]]:
    return {
        phase: {
            "robot_link": {
                "finger_socket": False,
                "blade": False,
                "backstop": False,
            }
        }
        for phase in SHOVEL_PHASES
    }


def _exact_robot_records(
    profile: ShovelToolProfile,
    label: str = "elbow_link",
) -> dict[str, dict[str, dict[str, bool]]]:
    components = {component: False for component in profile.component_bounds_local_m}
    return {
        phase: {label: dict(components)}
        for phase in SHOVEL_PHASES
    }


def _robot_aabbs_with_label(
    label: str = "elbow_link",
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    return {
        phase: {
            label: (
                np.asarray([-100.0, -100.0, -100.0]),
                np.asarray([-99.0, -99.0, -99.0]),
            )
        }
        for phase in SHOVEL_PHASES
    }


def test_swept_clearance_requires_robot_envelope_and_rejects_unknown_phase() -> None:
    profile = _profile()
    poses = _phase_poses()
    static = {
        "packing_table": (np.asarray([-100.0, -100.0, -100.0]), np.asarray([-99.0, -99.0, -99.0])),
    }
    report = validate_shovel_swept_clearance(
        phase_tool_poses=poses,
        profile=profile,
        scene_aabbs=static,
        robot_aabbs_by_phase=None,
    )
    assert report.status == "NOT VERIFIED"
    bad_poses = dict(poses)
    bad_poses["unexpected"] = poses["open_at_home"]
    bad_report = validate_shovel_swept_clearance(
        phase_tool_poses=bad_poses,
        profile=profile,
        scene_aabbs=static,
        robot_aabbs_by_phase=_empty_robot_aabbs(),
        robot_exact_collisions_by_phase=_empty_robot_exact(),
    )
    assert bad_report.status == "FAIL"


def test_swept_clearance_allows_scoped_contacts_but_rejects_other_overlap() -> None:
    profile = _profile()
    poses = _phase_poses()
    static = {
        "packing_table": (np.asarray([-10.0, -10.0, -10.0]), np.asarray([-9.0, -9.0, -9.0])),
    }
    pass_report = validate_shovel_swept_clearance(
        phase_tool_poses=poses,
        profile=profile,
        scene_aabbs=static,
        robot_aabbs_by_phase=_empty_robot_aabbs(),
        robot_exact_collisions_by_phase=_empty_robot_exact(),
    )
    assert pass_report.status == "PASS"
    overlapping = dict(static)
    overlapping["yellow_block"] = (
        np.asarray([-0.2, 1.95, 2.97]),
        np.asarray([0.2, 2.05, 3.03]),
    )
    fail_report = validate_shovel_swept_clearance(
        phase_tool_poses=poses,
        profile=profile,
        scene_aabbs=overlapping,
        robot_aabbs_by_phase=_empty_robot_aabbs(),
        robot_exact_collisions_by_phase=_empty_robot_exact(),
    )
    assert fail_report.status == "FAIL"


def test_exact_robot_collision_is_fatal_even_with_nonoverlapping_broad_aabb() -> None:
    profile = _profile()
    poses = _phase_poses()
    static = {
        "packing_table": (
            np.asarray([-100.0, -100.0, -100.0]),
            np.asarray([-99.0, -99.0, -99.0]),
        ),
    }
    exact = _exact_robot_records(profile)
    exact["move_above_behind_red"]["elbow_link"]["finger_socket"] = True
    report = validate_shovel_swept_clearance(
        phase_tool_poses=poses,
        profile=profile,
        scene_aabbs=static,
        robot_aabbs_by_phase=_robot_aabbs_with_label(),
        robot_exact_collisions_by_phase=exact,
    )
    assert report.status == "FAIL"
    assert any(
        failure["kind"] == "exact_tool_robot_overlap"
        and failure["label"] == "elbow_link"
        for failure in report.first_failures
    )


def test_socket_grasp_exemption_is_hand_only_and_phase_gated() -> None:
    profile = _profile()
    poses = _phase_poses()
    static = {
        "packing_table": (
            np.asarray([-100.0, -100.0, -100.0]),
            np.asarray([-99.0, -99.0, -99.0]),
        ),
    }

    # A named hand contact with the socket is the one intended grasp interface,
    # and becomes exempt starting at the fixed descend phase.
    hand_exact = _exact_robot_records(profile, label="left_hand_Link1_1")
    hand_exact["descend_to_tool_grasp"]["left_hand_Link1_1"]["finger_socket"] = True
    hand_report = validate_shovel_swept_clearance(
        phase_tool_poses=poses,
        profile=profile,
        scene_aabbs=static,
        robot_aabbs_by_phase=_robot_aabbs_with_label("left_hand_Link1_1"),
        robot_exact_collisions_by_phase=hand_exact,
    )
    assert hand_report.status == "PASS"

    # The same named contact before descend is not a grasp yet and must remain
    # fatal, so a premature hand/socket overlap cannot hide a bad reset path.
    early_hand_exact = _exact_robot_records(profile, label="left_hand_Link1_1")
    early_hand_exact["move_to_tool_pregrasp"]["left_hand_Link1_1"]["finger_socket"] = True
    early_hand_report = validate_shovel_swept_clearance(
        phase_tool_poses=poses,
        profile=profile,
        scene_aabbs=static,
        robot_aabbs_by_phase=_robot_aabbs_with_label("left_hand_Link1_1"),
        robot_exact_collisions_by_phase=early_hand_exact,
    )
    assert early_hand_report.status == "FAIL"
    assert any(failure["component"] == "finger_socket" for failure in early_hand_report.first_failures)

    # Elbow/body geometry is never part of the grasp mask, even after descend.
    elbow_exact = _exact_robot_records(profile, label="left_elbow_link")
    elbow_exact["descend_to_tool_grasp"]["left_elbow_link"]["finger_socket"] = True
    elbow_report = validate_shovel_swept_clearance(
        phase_tool_poses=poses,
        profile=profile,
        scene_aabbs=static,
        robot_aabbs_by_phase=_robot_aabbs_with_label("left_elbow_link"),
        robot_exact_collisions_by_phase=elbow_exact,
    )
    assert elbow_report.status == "FAIL"
    assert any(failure["label"] == "left_elbow_link" for failure in elbow_report.first_failures)


def test_socket_grasp_exemption_never_applies_to_static_scene_labels() -> None:
    profile = _profile()
    poses = _phase_poses()
    descend_x = float(SHOVEL_PHASES.index("descend_to_tool_grasp"))
    # Deliberately use a misleading static-object label containing ``wrist``.
    # A scene object is never a robot link, so overlap with the socket must fail
    # even after the fixed grasp-contact phase begins.
    static = {
        "wrist_fixture": (
            np.asarray([descend_x - 0.1, 1.9, 2.9]),
            np.asarray([descend_x + 0.1, 2.1, 3.1]),
        ),
    }
    report = validate_shovel_swept_clearance(
        phase_tool_poses=poses,
        profile=profile,
        scene_aabbs=static,
        robot_aabbs_by_phase=_empty_robot_aabbs(),
        robot_exact_collisions_by_phase=_empty_robot_exact(),
    )
    assert report.status == "FAIL"
    assert any(
        failure["kind"] == "compound_tool_overlap"
        and failure["label"] == "wrist_fixture"
        and failure["component"] == "finger_socket"
        for failure in report.first_failures
    )


def test_socket_table_contact_uses_only_the_existing_phase_support_map() -> None:
    profile = _profile()
    poses = _phase_poses()
    exact = _empty_robot_exact()
    broad = _empty_robot_aabbs()

    # At reset/open phase the pre-existing table support map explicitly allows
    # the complete resting compound, including the socket rails.
    open_static = {
        "packing_table": (
            np.asarray([-0.1, 1.9, 2.9]),
            np.asarray([0.1, 2.1, 3.1]),
        ),
    }
    open_report = validate_shovel_swept_clearance(
        phase_tool_poses=poses,
        profile=profile,
        scene_aabbs=open_static,
        robot_aabbs_by_phase=broad,
        robot_exact_collisions_by_phase=exact,
    )
    assert open_report.status == "PASS"

    # At insert phase, only blade/backstop table contact is in the frozen map;
    # a socket/table overlap must therefore fail even though the table is an
    # intended contact label for the overall phase.
    insert_x = float(SHOVEL_PHASES.index("insert_blade"))
    insert_static = {
        "packing_table": (
            np.asarray([insert_x - 0.1, 1.9, 2.9]),
            np.asarray([insert_x + 0.1, 2.1, 3.1]),
        ),
    }
    insert_report = validate_shovel_swept_clearance(
        phase_tool_poses=poses,
        profile=profile,
        scene_aabbs=insert_static,
        robot_aabbs_by_phase=broad,
        robot_exact_collisions_by_phase=exact,
    )
    assert insert_report.status == "FAIL"
    assert any(
        failure["label"] == "packing_table" and failure["component"] == "finger_socket"
        for failure in insert_report.first_failures
    )


def test_false_exact_result_does_not_reject_conservative_robot_aabb() -> None:
    profile = _profile()
    poses = _phase_poses()
    static = {
        "packing_table": (
            np.asarray([-100.0, -100.0, -100.0]),
            np.asarray([-99.0, -99.0, -99.0]),
        ),
    }
    # The broad box covers every synthetic tool pose, but the exact record is
    # the authoritative narrow-phase result and reports no collision.
    report = validate_shovel_swept_clearance(
        phase_tool_poses=poses,
        profile=profile,
        scene_aabbs=static,
        robot_aabbs_by_phase={
            phase: {
                "elbow_link": (
                    np.asarray([-100.0, -100.0, -100.0]),
                    np.asarray([100.0, 100.0, 100.0]),
                )
            }
            for phase in SHOVEL_PHASES
        },
        robot_exact_collisions_by_phase=_exact_robot_records(profile),
    )
    assert report.status == "PASS"


def test_missing_or_mismatched_exact_robot_records_fail_closed() -> None:
    profile = _profile()
    poses = _phase_poses()
    static = {
        "packing_table": (
            np.asarray([-100.0, -100.0, -100.0]),
            np.asarray([-99.0, -99.0, -99.0]),
        ),
    }
    missing_phase = _empty_robot_exact()
    missing_phase.pop("move_behind_red")
    missing_report = validate_shovel_swept_clearance(
        phase_tool_poses=poses,
        profile=profile,
        scene_aabbs=static,
        robot_aabbs_by_phase=None,
        robot_exact_collisions_by_phase=missing_phase,
    )
    assert missing_report.status == "NOT VERIFIED"

    mismatched = _empty_robot_exact()
    mismatched["open_at_home"] = [mismatched["open_at_home"], mismatched["open_at_home"]]
    mismatch_report = validate_shovel_swept_clearance(
        phase_tool_poses=poses,
        profile=profile,
        scene_aabbs=static,
        robot_aabbs_by_phase=None,
        robot_exact_collisions_by_phase=mismatched,
    )
    assert mismatch_report.status == "FAIL"


def test_gate12_low_scoop_root_uses_named_one_millimetre_lift() -> None:
    snapshot = _snapshot()
    config = ShovelPlanConfig(fps=10, phase_duration_s=0.1, max_joint_step=10.0)
    _, diagnostics = ResetTimeShovelPlanner(_FakeIK(), _profile(), config).build(snapshot)
    expected_root_z = snapshot.red_block_world.position[2] - 0.025 + SHOVEL_BLADE_ROOT_SUPPORT_OFFSET_M + 0.045
    np.testing.assert_allclose(
        diagnostics.tool_targets_world["move_behind_red"].position[2],
        expected_root_z,
    )
    np.testing.assert_allclose(
        diagnostics.tool_targets_world["lower_blade"].position[2],
        expected_root_z - 0.045,
    )


def test_intended_contact_mask_is_explicit_and_unknown_phases_are_not_safe() -> None:
    assert "packing_table" in SHOVEL_INTENDED_CONTACTS_BY_PHASE["lower_blade"]
    assert "red_block" in SHOVEL_INTENDED_CONTACTS_BY_PHASE["insert_blade"]
    assert "red_block" in SHOVEL_INTENDED_CONTACTS_BY_PHASE["tilt_blade_up"]
    assert "yellow_block" not in SHOVEL_INTENDED_CONTACTS_BY_PHASE["insert_blade"]
