from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from g1pickplace import cosmos_inference as cosmos


RUNNER = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"


class _Response:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class _Session:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, object]]] = []
        self.gets: list[tuple[str, dict[str, object]]] = []
        self.polls = [
            {"status": "running", "progress": 0.5},
            {
                "status": "completed",
                "action": {
                    "shape": [16, 29],
                    "dtype": "torch.bfloat16",
                    "data": np.arange(16 * 29, dtype=np.float32).reshape(16, 29).tolist(),
                },
            },
        ]

    def post(self, url: str, **kwargs: object) -> _Response:
        self.posts.append((url, kwargs))
        return _Response({"id": "job-123"})

    def get(self, url: str, **kwargs: object) -> _Response:
        self.gets.append((url, kwargs))
        return _Response(self.polls.pop(0))


def test_normalize_rgb_handles_unit_float_rgba_and_nonfinite_values() -> None:
    source = np.array([[[0.0, 0.5, 1.0, 0.2], [np.nan, np.inf, -np.inf, 0.8]]])
    result = cosmos.normalize_rgb(source)
    assert result.dtype == np.uint8
    assert result.shape == (1, 2, 3)
    assert result.tolist() == [[[0, 127, 255], [0, 255, 0]]]


def test_concat_view_preserves_camera_order(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cv2 = SimpleNamespace(
        INTER_AREA=3,
        resize=lambda image, size, interpolation: np.full(
            (size[1], size[0], 3), image[0, 0], dtype=np.uint8
        ),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    images = {
        "front": np.full((8, 8, 3), 10, dtype=np.uint8),
        "left_wrist": np.full((8, 8, 3), 20, dtype=np.uint8),
        "right_wrist": np.full((8, 8, 3), 30, dtype=np.uint8),
    }
    result = cosmos.make_unitree_concat_view(images)
    assert result.shape == (720, 640, 3)
    assert np.all(result[:480] == 10)
    assert np.all(result[480:, :320] == 20)
    assert np.all(result[480:, 320:] == 30)


def test_request_policy_action_submits_expected_contract_and_polls(tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"png")
    session = _Session()
    sleeps: list[float] = []
    config = cosmos.CosmosPolicyConfig(
        base_url="http://localhost:8080/",
        prompt="Stack the red block on the yellow block.",
    )

    action, action_obj, extra = cosmos.request_policy_action(
        image,
        config,
        session=session,
        sleep=sleeps.append,
    )

    assert action.shape == (16, 29)
    assert action.dtype == np.float32
    assert action_obj["shape"] == [16, 29]
    assert session.posts[0][0] == "http://localhost:8080/v1/videos"
    form = session.posts[0][1]["data"]
    assert isinstance(form, dict)
    assert form["num_frames"] == "17"
    assert form["fps"] == "10"
    assert json.loads(form["extra_params"])["view_point"] == "concat_view"
    assert extra["raw_action_dim"] == 29
    assert [url for url, _ in session.gets] == [
        "http://localhost:8080/v1/videos/job-123",
        "http://localhost:8080/v1/videos/job-123",
    ]
    assert sleeps == [2.0]


def test_policy_action_rejects_wrong_shape_and_nonfinite_values() -> None:
    config = cosmos.CosmosPolicyConfig()
    with pytest.raises(ValueError, match="Expected finite"):
        cosmos.validate_policy_action(np.zeros((16, 28)), config)
    invalid = np.zeros((16, 29), dtype=np.float32)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="Expected finite"):
        cosmos.validate_policy_action(invalid, config)


def test_config_rejects_policy_frame_count_mismatch() -> None:
    with pytest.raises(ValueError, match=r"action_chunk_size \+ 1"):
        cosmos.CosmosPolicyConfig(num_frames=16).validate()


def test_sixteen_second_config_requests_160_actions_and_161_frames() -> None:
    config = cosmos.policy_config_for_duration(
        duration_s=16.0,
        base_url="http://localhost:8080",
        prompt="Stack the blocks.",
    )
    assert config.action_chunk_size == 160
    assert config.num_frames == 161
    assert config.action_chunk_size / config.fps == 16.0


def test_duration_config_rejects_unverified_horizon() -> None:
    with pytest.raises(ValueError, match="largest locally verified"):
        cosmos.policy_config_for_duration(
            duration_s=60.0,
            base_url="http://localhost:8080",
            prompt="Stack the blocks.",
        )


def test_sim_integration_is_inspect_only_and_precedes_expert_construction() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    inspect = source.index("if args.inspect_only:", source.index("_print_runtime_diagnostics"))
    inference = source.index("run_first_frame_inference(", inspect)
    expert_import = source.index("from g1pickplace.offline_ik", inspect)
    policy = source.index("policy = OpenLoopPolicy", expert_import)
    assert inspect < inference < expert_import < policy
    assert "--cosmos-policy-output-dir is inference-only and requires --inspect-only" in source
    assert "cosmos-policy-output-dir" in source


def test_cosmos_module_has_no_simulator_action_path() -> None:
    source = Path(cosmos.__file__).read_text(encoding="utf-8")
    assert "env.step(" not in source
    assert "robot_apply_action" not in source
    assert "import torch" not in source
