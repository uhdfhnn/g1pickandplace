import numpy as np
import pytest

from g1pickplace.geometry import (
    Pose,
    matrix_quaternion_xyzw,
    quaternion_matrix_xyzw,
    quaternion_to_xyzw,
)


def test_quaternion_order_conversion() -> None:
    np.testing.assert_allclose(
        quaternion_to_xyzw([1.0, 0.0, 0.0, 0.0], "wxyz"),
        [0.0, 0.0, 0.0, 1.0],
    )


def test_rotation_round_trip() -> None:
    quaternion = np.asarray([0.1, -0.2, 0.3, 0.9])
    quaternion /= np.linalg.norm(quaternion)
    recovered = matrix_quaternion_xyzw(quaternion_matrix_xyzw(quaternion))
    assert abs(float(np.dot(quaternion, recovered))) == pytest.approx(1.0, abs=1.0e-7)


def test_pose_inverse_and_compose() -> None:
    pose = Pose(np.asarray([1.0, -2.0, 0.5]), np.asarray([0.0, 0.0, np.sin(0.2), np.cos(0.2)]))
    identity = pose.inverse().compose(pose).as_matrix()
    np.testing.assert_allclose(identity, np.eye(4), atol=1.0e-9)
