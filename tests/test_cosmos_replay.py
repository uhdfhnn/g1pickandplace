import json
from pathlib import Path

import numpy as np
import pytest

from g1pickplace.cosmos_replay import (
    COSMOS_RAW_ACTION_DIM,
    decode_selected_wrist,
    denormalize_quantile,
    gripper_target,
    load_cosmos_action,
    rot6d_to_matrix,
)


def test_denormalize_maps_endpoints_to_quantiles() -> None:
    low = np.arange(COSMOS_RAW_ACTION_DIM, dtype=np.float64)
    high = low + 2.0
    normalized = np.stack((-np.ones(COSMOS_RAW_ACTION_DIM), np.ones(COSMOS_RAW_ACTION_DIM)))
    result = denormalize_quantile(normalized, low, high)
    np.testing.assert_allclose(result[0], low)
    np.testing.assert_allclose(result[1], high)


def test_rot6d_projects_to_proper_rotation() -> None:
    rotation = rot6d_to_matrix(np.asarray([1.0, 0.1, 0.0, 0.0, 0.9, 0.2]))
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_left_decode_integrates_translation_and_preserves_native_identity() -> None:
    low = np.zeros(COSMOS_RAW_ACTION_DIM)
    high = np.full(COSMOS_RAW_ACTION_DIM, 2.0)
    normalized = -np.ones((2, COSMOS_RAW_ACTION_DIM))
    # Denormalized left pose is [0.1,0,0, 1,0,0, 0,1,0] on both frames.
    desired = np.asarray([0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    normalized[:, 19:28] = desired - 1.0
    decoded = decode_selected_wrist(
        normalized,
        stats={"q01": low, "q99": high},
        side="left",
        initial_wrist_base=np.eye(4),
    )
    np.testing.assert_allclose(
        decoded.wrist_base_targets[:, :3, :3],
        np.repeat(np.eye(3)[None, :, :], repeats=2, axis=0),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(decoded.wrist_base_targets[-1, :3, 3], [0.0, -0.2, 0.0])


def test_gripper_fraction_uses_zero_closed_one_open() -> None:
    assert gripper_target((-0.02, -0.02), (0.03, 0.03), 0.0) == pytest.approx((0.03, 0.03))
    assert gripper_target((-0.02, -0.02), (0.03, 0.03), 1.0) == pytest.approx((-0.02, -0.02))


def test_load_action_accepts_inference_directory(tmp_path: Path) -> None:
    action = np.zeros((4, COSMOS_RAW_ACTION_DIM), dtype=np.float32)
    np.savez_compressed(tmp_path / "sim_inference_context.npz", cosmos_action=action)
    np.testing.assert_array_equal(load_cosmos_action(tmp_path), action)


def test_load_action_accepts_json_shape_data(tmp_path: Path) -> None:
    action = np.zeros((2, COSMOS_RAW_ACTION_DIM), dtype=np.float32)
    path = tmp_path / "action.json"
    path.write_text(json.dumps({"shape": list(action.shape), "data": action.ravel().tolist()}))
    np.testing.assert_array_equal(load_cosmos_action(path), action)
