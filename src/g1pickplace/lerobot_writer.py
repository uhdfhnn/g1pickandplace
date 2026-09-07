"""Native LeRobot v3 episode writer with lazy optional imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# The trajectory identity is SHA-256 over a two-int64 shape header followed by
# contiguous little-endian float32 action bytes.  Float32 matches the public
# LeRobot ``action`` feature and the values actually handed to Isaac Lab; the
# shape header prevents differently shaped arrays with identical flattened
# bytes from colliding under this application-level convention.  SHA-256 was
# selected as a stable cross-process digest, not as a control/security input.
# Changing the dtype or byte order would make plan and dataset hashes differ,
# so this version string is fixed metadata and validation recomputes both sides
# with the same function.
ACTION_HASH_FORMAT = "sha256-shape-int64-le-float32-le-v1"


def action_array_sha256(actions: object) -> str:
    """Return the canonical digest for a two-dimensional action array."""

    array = np.asarray(actions)
    if array.ndim != 2:
        raise ValueError(f"actions must be a 2-D array, got shape {array.shape}")
    shape = np.asarray(array.shape, dtype="<i8")
    payload = np.ascontiguousarray(array, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(shape.tobytes(order="C"))
    digest.update(payload.tobytes(order="C"))
    return digest.hexdigest()


def validate_lerobot_episode(
    *,
    root: str | Path,
    repo_id: str,
    expected_actions: object,
    expected_task: str,
    expected_fps: int,
    expected_joint_names: Sequence[str],
    expected_camera_names: Sequence[str],
) -> dict[str, Any]:
    """Reopen and validate one native public LeRobot episode.

    The decoded first and last frames prove that every declared video stream
    is readable.  Tabular columns are checked across the entire episode.  This
    post-rollout function has no reference to the simulator or policy and
    cannot influence the already-consumed frozen action sequence.
    """

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("LeRobot is not installed in this Python environment") from exc

    expected = np.ascontiguousarray(expected_actions, dtype=np.float32)
    if expected.ndim != 2:
        raise ValueError(f"expected_actions must be 2-D, got shape {expected.shape}")
    dataset = LeRobotDataset(repo_id=repo_id, root=Path(root))
    if len(dataset) != expected.shape[0]:
        raise RuntimeError(
            f"LeRobot frame count {len(dataset)} != trajectory steps {expected.shape[0]}"
        )
    if int(dataset.fps) != int(expected_fps):
        raise RuntimeError(f"LeRobot fps {dataset.fps} != expected {expected_fps}")
    if int(dataset.meta.total_episodes) != 1:
        raise RuntimeError(
            f"LeRobot episode count {dataset.meta.total_episodes} != expected 1"
        )
    if int(dataset.meta.total_frames) != expected.shape[0]:
        raise RuntimeError(
            "LeRobot metadata frame count "
            f"{dataset.meta.total_frames} != expected {expected.shape[0]}"
        )

    joint_names = list(expected_joint_names)
    for key in ("observation.state", "action"):
        feature = dataset.features[key]
        if feature["dtype"] != "float32" or tuple(feature["shape"]) != (len(joint_names),):
            raise RuntimeError(f"LeRobot {key} schema mismatch: {feature}")
        if list(feature["names"]["axes"]) != joint_names:
            raise RuntimeError(f"LeRobot {key} ordered joint names do not match the live action term")

    actions = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
    states = np.asarray(dataset.hf_dataset["observation.state"], dtype=np.float32)
    if actions.shape != expected.shape or states.shape != expected.shape:
        raise RuntimeError(
            f"LeRobot tabular shapes action={actions.shape}, state={states.shape}, "
            f"expected={expected.shape}"
        )
    if not np.array_equal(actions, expected):
        maximum_error = float(np.max(np.abs(actions - expected)))
        raise RuntimeError(
            "LeRobot actions are not byte-equivalent to the frozen trajectory; "
            f"maximum absolute error={maximum_error}"
        )

    frame_indices = np.asarray(dataset.hf_dataset["frame_index"], dtype=np.int64)
    episode_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    timestamps = np.asarray(dataset.hf_dataset["timestamp"], dtype=np.float64)
    expected_indices = np.arange(expected.shape[0], dtype=np.int64)
    if not np.array_equal(frame_indices, expected_indices):
        raise RuntimeError("LeRobot frame indices are not one contiguous episode")
    if not np.array_equal(episode_indices, np.zeros_like(expected_indices)):
        raise RuntimeError("LeRobot episode boundaries do not describe exactly episode zero")
    expected_timestamps = expected_indices.astype(np.float64) / float(expected_fps)
    # Three microseconds is far below the 20 ms public 50 Hz frame period while
    # accommodating the measured 1.84 us worst-case float32 serialization
    # error at a 36 s/1,800-frame episode length.
    # A smaller tolerance can reject representational rounding; a larger one
    # can hide dropped/misaligned frames.  This fixed post-recording threshold
    # is validation-only and must be revised if episode durations become much
    # longer or timestamp storage changes.
    timestamp_tolerance_s = 3.0e-6
    maximum_timestamp_error_s = float(np.max(np.abs(timestamps - expected_timestamps)))
    if maximum_timestamp_error_s > timestamp_tolerance_s:
        raise RuntimeError(
            f"LeRobot timestamp error {maximum_timestamp_error_s} s exceeds "
            f"{timestamp_tolerance_s} s"
        )

    task_indices = np.asarray(dataset.hf_dataset["task_index"], dtype=np.int64)
    task_names = [dataset.meta.tasks.iloc[int(index)].name for index in task_indices]
    if any(task != expected_task for task in task_names):
        raise RuntimeError("LeRobot task string differs from the executed instruction")

    expected_video_keys = {f"observation.images.{name}" for name in expected_camera_names}
    actual_video_keys = set(dataset.meta.video_keys)
    if actual_video_keys != expected_video_keys:
        raise RuntimeError(
            f"LeRobot video keys {sorted(actual_video_keys)} != expected "
            f"{sorted(expected_video_keys)}"
        )
    decoded_indices = sorted({0, expected.shape[0] - 1})
    decoded_shapes: dict[str, list[int]] = {}
    for index in decoded_indices:
        item = dataset[index]
        if item["task"] != expected_task:
            raise RuntimeError(f"decoded LeRobot frame {index} has the wrong task string")
        for key in expected_video_keys:
            image = np.asarray(item[key])
            if image.ndim != 3 or image.shape[0] != 3:
                raise RuntimeError(
                    f"decoded LeRobot video frame {index} for {key} has shape {image.shape}"
                )
            decoded_shapes[key] = list(image.shape)

    expected_hash = action_array_sha256(expected)
    recorded_hash = action_array_sha256(actions)
    report = {
        "status": "PASS",
        "native_api": "lerobot.datasets.lerobot_dataset.LeRobotDataset",
        "episode_count": 1,
        "frame_count": int(expected.shape[0]),
        "fps": int(expected_fps),
        "action_shape": list(actions.shape),
        "action_dtype": str(actions.dtype),
        "state_shape": list(states.shape),
        "state_dtype": str(states.dtype),
        "maximum_timestamp_error_s": maximum_timestamp_error_s,
        "task": expected_task,
        "video_keys": sorted(actual_video_keys),
        "decoded_video_shapes_chw": decoded_shapes,
        "action_hash_format": ACTION_HASH_FORMAT,
        "trajectory_action_sha256": expected_hash,
        "recorded_action_sha256": recorded_hash,
        "action_hashes_match": expected_hash == recorded_hash,
    }
    metadata_path = Path(root) / "meta" / "g1pickandplace_validation.json"
    metadata_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def make_lerobot_features(
    joint_names: Sequence[str],
    *,
    camera_shapes: Mapping[str, tuple[int, int, int]],
    use_videos: bool,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": {"axes": list(joint_names)},
        },
        "action": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": {"axes": list(joint_names)},
        },
        "observation.object_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"axes": ["x", "y", "z", "qx", "qy", "qz", "qw"]},
        },
        "observation.target_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"axes": ["x", "y", "z", "qx", "qy", "qz", "qw"]},
        },
        "observation.phase_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": {"axes": ["phase_index"]},
        },
    }
    for name, shape in camera_shapes.items():
        if len(shape) != 3 or shape[2] not in (3, 4):
            raise ValueError(f"camera {name!r} shape must be HxWx3/4, got {shape}")
        features[f"observation.images.{name}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": (shape[0], shape[1], 3),
            "names": ["height", "width", "channels"],
        }
    return features


class LeRobotEpisodeWriter:
    """Small adapter around public ``LeRobotDataset`` APIs."""

    def __init__(
        self,
        *,
        root: str | Path,
        repo_id: str,
        fps: int,
        joint_names: Sequence[str],
        camera_shapes: Mapping[str, tuple[int, int, int]],
        use_videos: bool,
    ) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError("LeRobot is not installed in this Python environment") from exc
        self.features = make_lerobot_features(
            joint_names,
            camera_shapes=camera_shapes,
            use_videos=use_videos,
        )
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=Path(root),
            robot_type="unitree_g1_sim",
            features=self.features,
            use_videos=use_videos,
        )
        self.camera_names = tuple(camera_shapes)
        self._finalized = False

    def add_frame(
        self,
        *,
        joint_positions: object,
        action: object,
        object_pose_xyzw: object,
        target_pose_xyzw: object,
        phase_index: int,
        task: str,
        images: Mapping[str, np.ndarray],
    ) -> None:
        frame: dict[str, Any] = {
            "observation.state": np.asarray(joint_positions, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "observation.object_pose": np.asarray(object_pose_xyzw, dtype=np.float32),
            "observation.target_pose": np.asarray(target_pose_xyzw, dtype=np.float32),
            "observation.phase_index": np.asarray([phase_index], dtype=np.int64),
            "task": task,
        }
        for name in self.camera_names:
            if name not in images:
                raise ValueError(f"missing camera image {name!r}")
            image = np.asarray(images[name])
            if image.ndim != 3 or image.shape[2] not in (3, 4):
                raise ValueError(f"camera image {name!r} has invalid shape {image.shape}")
            frame[f"observation.images.{name}"] = image[..., :3].astype(np.uint8, copy=False)
        self.dataset.add_frame(frame)

    def save_episode(self) -> None:
        self.dataset.save_episode()

    def clear_episode(self) -> None:
        self.dataset.clear_episode_buffer()

    def finalize(self) -> None:
        if not self._finalized:
            self.dataset.finalize()
            self._finalized = True

    def __enter__(self) -> "LeRobotEpisodeWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is not None:
            self.clear_episode()
        self.finalize()
