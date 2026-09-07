from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from g1pickplace.offline_ik import (
    COLLISION_MULTISTART_MAX_ALTERNATES,
    COLLISION_MULTISTART_RNG_SEED,
    COLLISION_PATH_MAX_ACTIVE_STEP_RAD,
    IKPlanningError,
    PinocchioFrameIK,
)


def _fake_ik(*, collision_enabled: bool = True) -> PinocchioFrameIK:
    """Build only the collision helper state; no Pinocchio import is needed."""

    ik = object.__new__(PinocchioFrameIK)
    ik.enable_collision_checking = collision_enabled
    ik.active_configuration_indices = [0]
    ik.configuration_lower = np.asarray([-1.0, -2.0])
    ik.configuration_upper = np.asarray([1.0, 2.0])
    return ik


def test_multistart_constants_match_public_probe_contract():
    assert COLLISION_MULTISTART_RNG_SEED == 92417
    assert COLLISION_MULTISTART_MAX_ALTERNATES == 32
    assert COLLISION_PATH_MAX_ACTIVE_STEP_RAD == 0.02


def test_alternate_seed_only_changes_active_joint_within_limits():
    ik = _fake_ik()
    rng = np.random.default_rng(COLLISION_MULTISTART_RNG_SEED)
    original = np.asarray([0.25, 0.75])

    alternate = ik._alternate_seed(original, rng)

    assert alternate[1] == original[1]
    assert -1.0 <= alternate[0] <= 1.0


def test_path_checker_uses_active_step_limit_and_reports_first_pair():
    ik = _fake_ik()
    checked = []

    def collision_pairs(q):
        checked.append(float(q[0]))
        return (("upper_arm", "torso"),) if q[0] >= 0.5 else ()

    ik._collision_pairs = collision_pairs
    result = ik._first_path_collision(np.asarray([0.0, 0.2]), np.asarray([0.6, 0.2]))

    assert result is not None
    sample, pair = result
    assert sample == 25
    assert pair == ("upper_arm", "torso")
    assert len(checked) == 26
    np.testing.assert_allclose(np.diff(checked), COLLISION_PATH_MAX_ACTIVE_STEP_RAD)


def test_compiled_trajectory_gate_reports_first_step_phase_and_pair():
    ik = _fake_ik()
    ik.q_from_named_positions = lambda names, positions: np.asarray([positions[0], 0.0])
    ik._collision_pairs = lambda q: (("upper_arm", "torso"),) if q[0] >= 0.5 else ()
    trajectory = SimpleNamespace(
        joint_names=("right_shoulder_pitch_joint",),
        absolute_targets=np.asarray([[0.0], [0.6]], dtype=np.float32),
        phases=("home", "move_to_pregrasp"),
    )

    with pytest.raises(
        IKPlanningError,
        match=r"step 1 phase 'move_to_pregrasp'.*upper_arm.*torso",
    ):
        ik.validate_trajectory(
            trajectory,
            initial_absolute_positions=np.asarray([0.0]),
        )


def test_compiled_trajectory_gate_is_noop_when_collision_checking_disabled():
    ik = _fake_ik(collision_enabled=False)
    trajectory = SimpleNamespace(
        joint_names=("joint",),
        absolute_targets=np.asarray([[0.0]], dtype=np.float32),
        phases=("home",),
    )
    ik.validate_trajectory(trajectory)


def test_reset_configuration_gate_reports_first_pair():
    ik = _fake_ik()
    ik.q_from_named_positions = lambda names, positions: np.asarray(positions)
    ik._collision_pairs = lambda q: (("right_hand", "torso"),)

    with pytest.raises(IKPlanningError, match=r"live reset.*right_hand.*torso"):
        ik.validate_configuration("live reset", ("joint",), np.asarray([0.0]))
