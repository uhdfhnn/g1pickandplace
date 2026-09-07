"""Cosmos policy inference from the Unitree simulation's first RGB frame.

This module deliberately stops at validated Cosmos model-space actions.  The
returned 29 values describe AgiBotWorld camera/wrist deltas and grippers; they
are not Unitree G1 joint commands and must never be passed directly to
``env.step``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import mimetypes
from pathlib import Path
import time
from typing import Any, Mapping, Protocol

import numpy as np


# These names are the normalized keys emitted by ``_camera_images`` in the
# public Unitree runner.  Their order reproduces the AgiBotWorld concat-view
# semantics: front on top, then left wrist before right wrist.  Swapping the
# wrist order changes the learned viewpoint meaning; a missing name therefore
# fails instead of silently substituting another camera.  The mapping is fixed
# by the deployed Cosmos ``concat_view`` contract and should change only with a
# checkpoint/server contract update validated using a saved request artifact.
COSMOS_CAMERA_ORDER = ("front", "left_wrist", "right_wrist")

# Pixel dimensions reproduce the deployed AgiBotWorld conditioning canvas.
# Width/height are image axes (+X right, +Y down): the front view is 640x480,
# each wrist is 320x240, and the two wrist views form a 640x240 bottom row.
# These exact values come from the working BrickBench Cosmos client.  Smaller
# images discard manipulation detail; larger or differently arranged images
# shift the model's training distribution.  They are intentionally fixed for
# this checkpoint and must be versioned/configured if the server contract
# changes.
COSMOS_FRONT_SIZE_PX = (640, 480)
COSMOS_WRIST_SIZE_PX = (320, 240)
COSMOS_CONCAT_SHAPE = (720, 640, 3)

# These request values are the deployed Cosmos3/vLLM-Omni policy contract, not
# G1 controller calibration.  A 16-transition chunk at 10 frames/s spans about
# 1.6 seconds; ``num_frames`` is therefore 17 because it includes the single
# conditioning frame.  The sampling values (30 flow steps, guidance 1.0,
# flow-shift 10.0, seed 0) come from the previously successful client request.
# Lower sampling work can reduce quality; changed guidance/flow scheduling can
# materially alter output; changed action width cannot be decoded by the
# current contract.  All remain configurable through ``CosmosPolicyConfig``
# for a deliberately versioned server/checkpoint, while these defaults preserve
# the known deployment.
COSMOS_ACTION_CHUNK_SIZE = 16
COSMOS_RAW_ACTION_DIM = 29
COSMOS_NUM_FRAMES = COSMOS_ACTION_CHUNK_SIZE + 1
COSMOS_TEMPORAL_FPS = 10
COSMOS_INFERENCE_STEPS = 30
COSMOS_GUIDANCE_SCALE = 1.0
COSMOS_FLOW_SHIFT = 10.0
COSMOS_SEED = 0
COSMOS_IMAGE_SIZE = 480
COSMOS_DOMAIN_NAME = "agibotworld"
COSMOS_VIEW_POINT = "concat_view"
COSMOS_ACTION_MODE = "policy"

# Existing successful server artifacts contain 128-step and 512-step policy
# responses, establishing 512 actions as the largest locally evidenced request
# horizon for this deployed API.  The unit is model action rows at 10 Hz (51.2
# seconds at the default temporal rate), not simulator ticks.  A higher value
# could create an unvalidated, expensive request or exhaust response memory; a
# lower cap would reject the user's 160-step/16-second target despite evidence
# that longer chunks work.  This is a client safety cap and should be raised
# only after another finite server response is recorded and validated.
COSMOS_MAX_VERIFIED_ACTION_STEPS = 512

# Sixty seconds is the per-HTTP-operation timeout used by the working client;
# it bounds a stalled submit or poll request but intentionally does not cap the
# total asynchronous job duration.  Two seconds is the same server-friendly
# polling cadence: shorter intervals add needless service load, while much
# longer intervals delay completion reporting.  Values are seconds, have no
# spatial frame, are configurable for network conditions, and are validated as
# finite positive durations before any request is sent.
COSMOS_REQUEST_TIMEOUT_S = 60.0
COSMOS_POLL_INTERVAL_S = 2.0


class HTTPSession(Protocol):
    """Small requests-compatible surface used for dependency-light tests."""

    def post(self, url: str, **kwargs: Any) -> Any: ...

    def get(self, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class CosmosPolicyConfig:
    """Versioned request contract for one first-frame policy inference."""

    base_url: str = "http://localhost:8080"
    prompt: str = "Assemble the blocks."
    domain_name: str = COSMOS_DOMAIN_NAME
    action_chunk_size: int = COSMOS_ACTION_CHUNK_SIZE
    raw_action_dim: int = COSMOS_RAW_ACTION_DIM
    num_frames: int = COSMOS_NUM_FRAMES
    fps: int = COSMOS_TEMPORAL_FPS
    num_inference_steps: int = COSMOS_INFERENCE_STEPS
    guidance_scale: float = COSMOS_GUIDANCE_SCALE
    flow_shift: float = COSMOS_FLOW_SHIFT
    seed: int = COSMOS_SEED
    request_timeout_s: float = COSMOS_REQUEST_TIMEOUT_S
    poll_interval_s: float = COSMOS_POLL_INTERVAL_S

    def validate(self) -> None:
        if not self.base_url.strip():
            raise ValueError("Cosmos base URL must not be empty")
        if not self.prompt.strip():
            raise ValueError("Cosmos prompt must not be empty")
        if self.action_chunk_size <= 0 or self.raw_action_dim <= 0:
            raise ValueError("Cosmos action dimensions must be positive")
        if self.action_chunk_size > COSMOS_MAX_VERIFIED_ACTION_STEPS:
            raise ValueError(
                "Cosmos action_chunk_size exceeds the largest locally verified "
                f"server response ({COSMOS_MAX_VERIFIED_ACTION_STEPS})"
            )
        if self.num_frames != self.action_chunk_size + 1:
            raise ValueError("Cosmos policy num_frames must equal action_chunk_size + 1")
        if self.fps <= 0 or self.num_inference_steps <= 0:
            raise ValueError("Cosmos fps and inference steps must be positive")
        for label, value in (
            ("guidance_scale", self.guidance_scale),
            ("flow_shift", self.flow_shift),
            ("request_timeout_s", self.request_timeout_s),
            ("poll_interval_s", self.poll_interval_s),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"Cosmos {label} must be finite and positive")


def policy_config_for_duration(
    *,
    duration_s: float,
    base_url: str,
    prompt: str,
) -> CosmosPolicyConfig:
    """Build the fixed-10-Hz request whose action horizon best matches seconds."""

    seconds = float(duration_s)
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("Cosmos duration_s must be finite and positive")
    # Nearest-step quantization is required because the endpoint accepts a
    # whole number of 10-Hz transitions.  Half a model period (0.05 s) is the
    # maximum timing difference introduced by rounding; smaller durations can
    # collapse to zero and are rejected.  This follows the endpoint's temporal
    # contract rather than Isaac's 50-Hz controller clock.
    action_steps = int(round(seconds * COSMOS_TEMPORAL_FPS))
    if action_steps <= 0:
        raise ValueError("Cosmos duration_s is shorter than one 10-Hz action step")
    config = CosmosPolicyConfig(
        base_url=base_url,
        prompt=prompt,
        action_chunk_size=action_steps,
        num_frames=action_steps + 1,
    )
    config.validate()
    return config


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an HWC RGB/RGBA observation to uint8 RGB."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4 RGB image, got {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        # The 1.5 cutoff is a unit-range detector inherited from the working
        # client: normal rendered RGB is [0,1], while byte-like float RGB is
        # [0,255].  The margin tolerates minor HDR/numerical overshoot.  A lower
        # cutoff could leave unit RGB nearly black; a much higher one could
        # multiply byte data and saturate it.  It is intentionally fixed to the
        # source preprocessing and covered by unit tests.
        if array.size and float(array.max()) <= 1.5:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def center_crop_resize(frame: np.ndarray, size_px: tuple[int, int]) -> np.ndarray:
    """Center-crop RGB to an aspect ratio, then resize with area sampling."""

    import cv2

    width, height = size_px
    if width <= 0 or height <= 0:
        raise ValueError(f"Target image dimensions must be positive, got {size_px}")
    source = normalize_rgb(frame)
    source_h, source_w = source.shape[:2]
    target_aspect = width / height
    source_aspect = source_w / source_h
    if source_aspect > target_aspect:
        crop_w = int(round(source_h * target_aspect))
        x0 = max(0, (source_w - crop_w) // 2)
        cropped = source[:, x0 : x0 + crop_w]
    else:
        crop_h = int(round(source_w / target_aspect))
        y0 = max(0, (source_h - crop_h) // 2)
        cropped = source[y0 : y0 + crop_h, :]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)


def make_unitree_concat_view(images: Mapping[str, np.ndarray]) -> np.ndarray:
    """Create the Cosmos 640x720 front/left/right conditioning image."""

    missing = [name for name in COSMOS_CAMERA_ORDER if name not in images]
    if missing:
        raise KeyError(f"Missing Cosmos camera image(s): {missing}; available: {sorted(images)}")
    front = center_crop_resize(images["front"], COSMOS_FRONT_SIZE_PX)
    left = center_crop_resize(images["left_wrist"], COSMOS_WRIST_SIZE_PX)
    right = center_crop_resize(images["right_wrist"], COSMOS_WRIST_SIZE_PX)
    result = np.concatenate((front, np.concatenate((left, right), axis=1)), axis=0)
    if result.shape != COSMOS_CONCAT_SHAPE or result.dtype != np.uint8:
        raise RuntimeError(f"Invalid Cosmos concat view: shape={result.shape}, dtype={result.dtype}")
    return result


def write_rgb_image(path: Path, frame_rgb: np.ndarray) -> None:
    """Write RGB through OpenCV's explicitly converted BGR file boundary."""

    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    written = cv2.imwrite(str(path), cv2.cvtColor(normalize_rgb(frame_rgb), cv2.COLOR_RGB2BGR))
    if not written:
        raise RuntimeError(f"Could not write Cosmos input image: {path}")


def action_to_array(obj: Any) -> np.ndarray:
    """Decode either a nested Cosmos action object or a bare numeric array."""

    value = obj.get("action", obj) if isinstance(obj, dict) else obj
    if isinstance(value, dict) and "data" in value:
        action = np.asarray(value["data"], dtype=np.float32)
        shape = value.get("shape")
        if shape is not None:
            action = action.reshape(tuple(int(dimension) for dimension in shape))
    else:
        action = np.asarray(value, dtype=np.float32)
    return action


def validate_policy_action(action_obj: Any, config: CosmosPolicyConfig) -> np.ndarray:
    """Require the exact finite model-space action contract."""

    action = action_to_array(action_obj)
    expected = (config.action_chunk_size, config.raw_action_dim)
    if action.shape != expected or not np.isfinite(action).all():
        raise ValueError(f"Expected finite Cosmos policy action {expected}, got {action.shape}")
    return np.ascontiguousarray(action, dtype=np.float32)


def _request_payload(config: CosmosPolicyConfig) -> tuple[dict[str, str], dict[str, Any]]:
    extra_params = {
        "action_mode": COSMOS_ACTION_MODE,
        "domain_name": config.domain_name,
        "action_chunk_size": config.action_chunk_size,
        "image_size": COSMOS_IMAGE_SIZE,
        "view_point": COSMOS_VIEW_POINT,
        "raw_action_dim": config.raw_action_dim,
        "guardrails": False,
    }
    form = {
        "prompt": config.prompt,
        "num_frames": str(config.num_frames),
        "fps": str(config.fps),
        "num_inference_steps": str(config.num_inference_steps),
        "guidance_scale": str(config.guidance_scale),
        "flow_shift": str(config.flow_shift),
        "seed": str(config.seed),
        "extra_params": json.dumps(extra_params),
    }
    return form, extra_params


def request_policy_action(
    image_path: Path,
    config: CosmosPolicyConfig,
    *,
    session: HTTPSession | None = None,
    sleep: Any = time.sleep,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Submit one conditioning PNG, poll the async job, and validate action."""

    import requests

    config.validate()
    input_path = image_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    http = requests.Session() if session is None else session
    form, extra_params = _request_payload(config)
    mime_type = mimetypes.guess_type(input_path.name)[0] or "application/octet-stream"
    submit_url = f"{config.base_url.rstrip('/')}/v1/videos"
    with input_path.open("rb") as stream:
        response = http.post(
            submit_url,
            data=form,
            files={"input_reference": (input_path.name, stream, mime_type)},
            timeout=config.request_timeout_s,
        )
    response.raise_for_status()
    submitted = response.json()
    if not isinstance(submitted, dict) or not submitted.get("id"):
        raise RuntimeError(f"Cosmos submit response has no job id: {submitted!r}")
    job_id = str(submitted["id"])
    poll_url = f"{submit_url}/{job_id}"
    while True:
        response = http.get(poll_url, timeout=config.request_timeout_s)
        response.raise_for_status()
        final = response.json()
        if not isinstance(final, dict):
            raise RuntimeError(f"Cosmos job response is not an object: {final!r}")
        status = final.get("status")
        if status == "completed":
            break
        if status in {"failed", "cancelled"}:
            raise RuntimeError("Cosmos job did not complete: " + json.dumps(final, sort_keys=True))
        sleep(config.poll_interval_s)
    action_obj = final.get("action", final)
    action = validate_policy_action(action_obj, config)
    return action, dict(action_obj) if isinstance(action_obj, dict) else {"data": action_obj}, extra_params


def run_first_frame_inference(
    *,
    images: Mapping[str, np.ndarray],
    current_qpos: np.ndarray,
    home_qpos: np.ndarray,
    action_joint_names: tuple[str, ...],
    output_dir: Path,
    config: CosmosPolicyConfig,
    session: HTTPSession | None = None,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Capture, infer, and save a dry-run Cosmos policy artifact bundle."""

    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    concat_path = output / "unitree_concat_view.png"
    action_path = output / "cosmos_policy_action.json"
    metadata_path = output / "metadata.json"
    context_path = output / "sim_inference_context.npz"
    concat = make_unitree_concat_view(images)
    write_rgb_image(concat_path, concat)
    action, action_obj, extra_params = request_policy_action(
        concat_path,
        config,
        session=session,
        sleep=sleep,
    )
    action_path.write_text(json.dumps(action_obj, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        context_path,
        current_qpos=np.asarray(current_qpos, dtype=np.float32),
        home_qpos=np.asarray(home_qpos, dtype=np.float32),
        action_joint_names=np.asarray(action_joint_names),
        cosmos_action=action,
    )
    metadata: dict[str, Any] = {
        "mode": "policy",
        "execution": "dry-run",
        "prompt": config.prompt,
        "base_url": config.base_url,
        "camera_order": list(COSMOS_CAMERA_ORDER),
        "source_camera_shapes": {name: list(np.asarray(images[name]).shape) for name in COSMOS_CAMERA_ORDER},
        "concat_shape": list(concat.shape),
        "cosmos_action_shape": list(action.shape),
        "action_duration_s": config.action_chunk_size / config.fps,
        "g1_action_dimension": len(action_joint_names),
        "request": {
            "num_frames": config.num_frames,
            "fps": config.fps,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "flow_shift": config.flow_shift,
            "seed": config.seed,
            "extra_params": extra_params,
        },
        "safety": (
            "Cosmos 29-D output is normalized AgiBotWorld end-effector model space; "
            "it was saved but not applied to the Unitree G1 simulation."
        ),
        "artifacts": {
            "concat_view": concat_path.name,
            "action": action_path.name,
            "context": context_path.name,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
