import numpy as np
import pytest

from g1pickplace.lerobot_writer import ACTION_HASH_FORMAT, action_array_sha256, make_lerobot_features


def test_vla_schema_has_core_fields_and_cameras() -> None:
    features = make_lerobot_features(
        ["j0", "j1"],
        camera_shapes={"front": (480, 640, 3), "wrist_right": (240, 320, 4)},
        use_videos=True,
    )
    assert features["observation.state"]["shape"] == (2,)
    assert features["action"]["shape"] == (2,)
    assert features["observation.images.front"]["dtype"] == "video"
    assert features["observation.images.wrist_right"]["shape"] == (240, 320, 3)
    assert features["observation.object_pose"]["shape"] == (7,)


def test_action_hash_is_dtype_and_layout_canonical() -> None:
    actions = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    noncontiguous = np.asfortranarray(actions.astype(np.float32))
    assert ACTION_HASH_FORMAT == "sha256-shape-int64-le-float32-le-v1"
    assert action_array_sha256(actions) == action_array_sha256(noncontiguous)
    assert action_array_sha256(actions) != action_array_sha256(actions.reshape(1, 4))


def test_action_hash_rejects_non_matrix() -> None:
    with pytest.raises(ValueError, match="2-D"):
        action_array_sha256(np.zeros(4, dtype=np.float32))
