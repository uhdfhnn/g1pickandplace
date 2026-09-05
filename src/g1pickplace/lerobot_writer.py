"""Native LeRobot v3 episode writer with lazy optional imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


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
