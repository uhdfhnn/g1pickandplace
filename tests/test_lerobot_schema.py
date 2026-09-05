from g1pickplace.lerobot_writer import make_lerobot_features


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
