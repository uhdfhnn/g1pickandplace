import numpy as np

from g1pickplace.evaluation import evaluate_pick_place


def test_success_requires_location_height_and_stability() -> None:
    target = np.asarray([0.0, 0.0, 0.8])
    assert evaluate_pick_place(target, np.zeros(3), target).success
    assert not evaluate_pick_place(np.asarray([0.2, 0.0, 0.8]), np.zeros(3), target).success
    assert not evaluate_pick_place(target, np.asarray([0.2, 0.0, 0.0]), target).success
    assert not evaluate_pick_place(np.asarray([0.0, 0.0, 0.9]), np.zeros(3), target).success
