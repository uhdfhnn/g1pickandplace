"""Decode a saved Cosmos 29-D action into a validated G1 replay trajectory.

Cosmos predicts AgiBotWorld relative wrist poses rather than Unitree joint
commands.  This module keeps that distinction explicit: it denormalizes and
integrates one selected wrist, solves every model-rate pose with the supplied
G1 IK backend, then interpolates those solutions at the simulator control rate.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from .geometry import Pose
from .trajectory import JointTrajectory


# The deployed policy contract contains 29 dimensionless normalized channels:
# head pose 0:9, right wrist 9:18, right gripper 18, left wrist 19:28, and
# left gripper 28.  This exact width comes from the working AgiBotWorld Cosmos
# client and its saved quantile statistics.  A smaller/larger width changes the
# semantic offsets, so it is intentionally fixed until a versioned checkpoint
# contract and matching stats are available.
COSMOS_RAW_ACTION_DIM = 29

# Cosmos actions are time-indexed at 10 Hz, as specified by the successful
# policy request and saved metadata.  The value is model transitions/second,
# not Isaac physics ticks.  A lower value stretches motion and a higher value
# compresses it; it is intentionally fixed for this checkpoint and checked
# against the requested simulator rate before trajectory construction.
COSMOS_ACTION_FPS = 10

# These proper rotations convert the native right/left wrist axes into the
# OpenCV wrist axes used when the AgiBotWorld policy was trained.  They are
# dimensionless basis changes copied from the reference Cosmos decoder.  Their
# signs are side-specific: swapping them mirrors wrist motion.  They remain
# fixed to that model convention and should change only with paired frame-level
# validation for a new robot/checkpoint.
RIGHT_TO_OPENCV = np.asarray(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
LEFT_TO_OPENCV = np.asarray(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

# These q01/q99 values are the AgiBotWorld beta LeRobot action statistics used
# by the reference decoder.  They are stored beside this module so an installed
# replay cannot silently read a different checkout.  Values have mixed units:
# metres for each pose translation, dimensionless rotation-6D components, and
# a dimensionless [0,1] open fraction for each gripper.  Wrong statistics scale
# every relative step and can make a cumulative trajectory unreachable.  The
# file is fixed to this model contract; a caller may explicitly provide another
# versioned stats file for controlled validation.
DEFAULT_STATS_PATH = Path(__file__).with_name("agibotworld_beta_lerobot_stats.json")

# Each 10-Hz IK target is linearly expanded into five 50-Hz commands.  The
# resulting per-command active-joint limit is 0.02 rad, identical to the
# existing trajectory compiler and collision-path sampling contract.  More
# than 0.02 rad can skip a thin self-collision or demand an abrupt servo step;
# much less can reject harmless motion.  The value is intentionally fixed to
# the repository's validated planner contract, while a non-50-Hz simulator is
# still supported when its rate is an integer multiple of 10 Hz.
MAX_SIM_JOINT_STEP_RAD = 0.02


@dataclass(frozen=True)
class DecodedWristTrajectory:
    """One selected wrist's denormalized model-rate targets."""

    raw_action: np.ndarray
    wrist_base_targets: np.ndarray
    gripper_open_fraction: np.ndarray


def load_cosmos_action(path: str | Path) -> np.ndarray:
    """Load ``(T,29)`` model-space actions from an inference artifact or file."""

    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / "sim_inference_context.npz"
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            if "cosmos_action" not in payload:
                raise KeyError(f"{source} does not contain 'cosmos_action'")
            action = np.asarray(payload["cosmos_action"], dtype=np.float64)
    elif source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        action_payload = payload.get("action", payload)
        shape = tuple(int(value) for value in action_payload["shape"])
        action = np.asarray(action_payload["data"], dtype=np.float64).reshape(shape)
    else:
        raise ValueError(f"unsupported Cosmos action artifact: {source}")
    if action.ndim != 2 or action.shape[0] == 0 or action.shape[1] != COSMOS_RAW_ACTION_DIM:
        raise ValueError(
            f"Cosmos action must have shape (T,{COSMOS_RAW_ACTION_DIM}) with T>0, got {action.shape}"
        )
    if not np.all(np.isfinite(action)):
        raise ValueError("Cosmos action contains non-finite values")
    return action


def load_quantile_stats(path: str | Path = DEFAULT_STATS_PATH) -> dict[str, np.ndarray]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    q01 = np.asarray(payload["q01"], dtype=np.float64)
    q99 = np.asarray(payload["q99"], dtype=np.float64)
    expected = (COSMOS_RAW_ACTION_DIM,)
    if q01.shape != expected or q99.shape != expected:
        raise ValueError(f"Cosmos q01/q99 must each have shape {expected}")
    if not np.all(np.isfinite(q01)) or not np.all(np.isfinite(q99)):
        raise ValueError("Cosmos quantile statistics contain non-finite values")
    if np.any(q99 <= q01):
        raise ValueError("every Cosmos q99 value must exceed q01")
    return {"q01": q01, "q99": q99}


def denormalize_quantile(
    normalized_action: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
) -> np.ndarray:
    normalized = np.asarray(normalized_action, dtype=np.float64)
    low = np.asarray(q01, dtype=np.float64)
    high = np.asarray(q99, dtype=np.float64)
    if normalized.ndim != 2 or normalized.shape[1] != COSMOS_RAW_ACTION_DIM:
        raise ValueError(f"normalized action must have shape (T,{COSMOS_RAW_ACTION_DIM})")
    if low.shape != (COSMOS_RAW_ACTION_DIM,) or high.shape != low.shape:
        raise ValueError("quantile statistics do not match the Cosmos action width")
    if not np.all(np.isfinite(normalized)):
        raise ValueError("normalized action contains non-finite values")
    return 0.5 * (normalized + 1.0) * (high - low) + low


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rot6d, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("rot6d must be one finite 6-vector")
    approximate = np.stack(
        (values[:3], values[3:], np.cross(values[:3], values[3:])),
        axis=-1,
    )
    u, _, vt = np.linalg.svd(approximate)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _rotation_transform(rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    return transform


def _pose_delta(pose9: np.ndarray) -> np.ndarray:
    values = np.asarray(pose9, dtype=np.float64)
    if values.shape != (9,):
        raise ValueError("pose delta must be one 9-vector")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot6d_to_matrix(values[3:])
    transform[:3, 3] = values[:3]
    return transform


def decode_selected_wrist(
    normalized_action: np.ndarray,
    *,
    stats: Mapping[str, np.ndarray],
    side: Literal["left", "right"],
    initial_wrist_base: np.ndarray,
) -> DecodedWristTrajectory:
    """Integrate one native wrist from its initial robot-base transform."""

    initial = np.asarray(initial_wrist_base, dtype=np.float64)
    if initial.shape != (4, 4) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_wrist_base must be one finite 4x4 transform")
    raw = denormalize_quantile(normalized_action, stats["q01"], stats["q99"])
    if side == "left":
        pose_slice = slice(19, 28)
        gripper_index = 28
        native_to_opencv = LEFT_TO_OPENCV
    elif side == "right":
        pose_slice = slice(9, 18)
        gripper_index = 18
        native_to_opencv = RIGHT_TO_OPENCV
    else:
        raise ValueError(f"unsupported hand side: {side!r}")

    basis = _rotation_transform(native_to_opencv)
    opencv_wrist = initial @ basis
    opencv_to_native = np.linalg.inv(basis)
    targets: list[np.ndarray] = []
    for row in raw:
        opencv_wrist = opencv_wrist @ _pose_delta(row[pose_slice])
        targets.append(opencv_wrist @ opencv_to_native)
    return DecodedWristTrajectory(
        raw_action=raw,
        wrist_base_targets=np.asarray(targets, dtype=np.float64),
        gripper_open_fraction=np.clip(raw[:, gripper_index], 0.0, 1.0),
    )


def gripper_target(
    open_positions: tuple[float, float],
    closed_positions: tuple[float, float],
    open_fraction: float,
) -> tuple[float, float]:
    """Map the Cosmos convention (0 closed, 1 open) to two G1 joints."""

    fraction = float(np.clip(open_fraction, 0.0, 1.0))
    return tuple(
        float(closed + fraction * (opened - closed))
        for opened, closed in zip(open_positions, closed_positions, strict=True)
    )


def _rotation_error_deg(actual: Pose, target: Pose) -> float:
    relative = actual.as_matrix()[:3, :3].T @ target.as_matrix()[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def compile_cosmos_replay_trajectory(
    *,
    normalized_action: np.ndarray,
    stats: Mapping[str, np.ndarray],
    side: Literal["left", "right"],
    initial_wrist_base: Pose,
    ik: Any,
    action_names: tuple[str, ...],
    current_joint_positions: np.ndarray,
    default_joint_positions: np.ndarray,
    action_limits: np.ndarray,
    gripper_joint_names: tuple[str, str],
    gripper_open_positions: tuple[float, float],
    gripper_closed_positions: tuple[float, float],
    sim_fps: int,
) -> tuple[JointTrajectory, dict[str, Any]]:
    """Solve all model targets, validate increments, and compile 50-Hz replay."""

    if sim_fps <= 0 or sim_fps % COSMOS_ACTION_FPS != 0:
        raise ValueError(
            f"sim_fps must be a positive integer multiple of {COSMOS_ACTION_FPS}"
        )
    repeats = sim_fps // COSMOS_ACTION_FPS
    names = tuple(action_names)
    current = np.asarray(current_joint_positions, dtype=np.float64)
    defaults = np.asarray(default_joint_positions, dtype=np.float64)
    limits = np.asarray(action_limits, dtype=np.float64)
    if current.shape != (len(names),) or defaults.shape != current.shape:
        raise ValueError("current/default joint vectors must match action_names")
    if limits.shape != (len(names), 2):
        raise ValueError("action_limits must have shape (len(action_names),2)")
    index_by_name = {name: index for index, name in enumerate(names)}
    missing = [name for name in gripper_joint_names if name not in index_by_name]
    if missing:
        raise ValueError(f"replay gripper joints are absent from action_names: {missing}")
    controlled_names = tuple(ik.active_joint_names) + tuple(gripper_joint_names)
    missing_controlled = [name for name in controlled_names if name not in index_by_name]
    if missing_controlled:
        raise ValueError(f"replay controlled joints are absent from action_names: {missing_controlled}")
    controlled_indexes = np.asarray(
        [index_by_name[name] for name in controlled_names],
        dtype=np.int64,
    )

    decoded = decode_selected_wrist(
        normalized_action,
        stats=stats,
        side=side,
        initial_wrist_base=initial_wrist_base.as_matrix(),
    )
    seed_q = ik.q_from_named_positions(names, current)
    previous = current.copy()
    absolute_frames: list[np.ndarray] = []
    position_errors_m: list[float] = []
    rotation_errors_deg: list[float] = []
    model_joint_deltas_rad: list[float] = []

    for step, matrix in enumerate(decoded.wrist_base_targets):
        target_pose = Pose.from_matrix(matrix)
        report = ik.solve(f"cosmos-replay-{step:03d}", target_pose, seed_q)
        solved_pose = ik.frame_pose(report.q)
        position_errors_m.append(float(np.linalg.norm(solved_pose.position - target_pose.position)))
        rotation_errors_deg.append(_rotation_error_deg(solved_pose, target_pose))
        named_solution = ik.named_positions_from_q(report.q, names)
        endpoint = previous.copy()
        for name, value in named_solution.items():
            if name in index_by_name:
                endpoint[index_by_name[name]] = float(value)
        for name, value in zip(
            gripper_joint_names,
            gripper_target(
                gripper_open_positions,
                gripper_closed_positions,
                decoded.gripper_open_fraction[step],
            ),
            strict=True,
        ):
            endpoint[index_by_name[name]] = value

        # The live fixed-base reset can leave passive leg joints just outside
        # their soft command envelope even though Cosmos never changes them.
        # Only the seven selected arm joints and two selected gripper joints are
        # controlled here; applying whole-body limit checks would reject an
        # unchanged simulator reset, while omitting a controlled index could
        # command an unsafe target.  The exact controlled set comes from the IK
        # configuration and hand profile and is therefore validated by name.
        below = endpoint[controlled_indexes] < limits[controlled_indexes, 0]
        above = endpoint[controlled_indexes] > limits[controlled_indexes, 1]
        if np.any(below | above):
            bad_controlled_index = int(np.flatnonzero(below | above)[0])
            bad_index = int(controlled_indexes[bad_controlled_index])
            raise RuntimeError(
                f"Cosmos replay step {step} exceeds {names[bad_index]} limits: "
                f"target={endpoint[bad_index]:.6g}, limits={tuple(limits[bad_index])}"
            )
        model_delta = float(
            np.max(np.abs(endpoint[controlled_indexes] - previous[controlled_indexes]))
        )
        model_joint_deltas_rad.append(model_delta)
        if model_delta / repeats > MAX_SIM_JOINT_STEP_RAD + 1.0e-12:
            raise RuntimeError(
                f"Cosmos replay step {step} requires {model_delta / repeats:.6g} rad per "
                f"{sim_fps}-Hz command, above {MAX_SIM_JOINT_STEP_RAD:.6g} rad"
            )
        start = previous.copy()
        for substep in range(1, repeats + 1):
            absolute_frames.append(start + (substep / repeats) * (endpoint - start))
        previous = endpoint
        seed_q = report.q

    absolute = np.asarray(absolute_frames, dtype=np.float64)
    actions = absolute - defaults[None, :]
    trajectory = JointTrajectory(
        joint_names=names,
        absolute_targets=absolute,
        env_actions=actions,
        phases=tuple("cosmos-replay" for _ in range(len(absolute))),
        fps=sim_fps,
    )
    translations = decoded.wrist_base_targets[:, :3, 3]
    initial_position = initial_wrist_base.position
    return trajectory, {
        "status": "PASS",
        "source_steps": int(np.asarray(normalized_action).shape[0]),
        "model_fps": COSMOS_ACTION_FPS,
        "sim_fps": int(sim_fps),
        "sim_steps": trajectory.steps,
        "duration_s": trajectory.duration_s,
        "selected_side": side,
        "maximum_ik_position_error_m": max(position_errors_m, default=0.0),
        "maximum_ik_rotation_error_deg": max(rotation_errors_deg, default=0.0),
        "maximum_model_step_joint_delta_rad": max(model_joint_deltas_rad, default=0.0),
        "maximum_sim_step_joint_delta_rad": (
            max(model_joint_deltas_rad, default=0.0) / repeats
        ),
        "wrist_net_translation_m": float(np.linalg.norm(translations[-1] - initial_position)),
        "gripper_open_fraction_min": float(np.min(decoded.gripper_open_fraction)),
        "gripper_open_fraction_max": float(np.max(decoded.gripper_open_fraction)),
        "coordinate_hypothesis": (
            "AgiBotWorld quantile-denormalized framewise wrist deltas; selected native G1 "
            "wrist basis converted through the reference side-specific OpenCV frame"
        ),
    }


class CosmosReplayVideoWriter:
    """Write the same three-camera concat layout used for Cosmos conditioning."""

    def __init__(self, path: str | Path, *, fps: int = COSMOS_ACTION_FPS) -> None:
        if fps <= 0:
            raise ValueError("video fps must be positive")
        self.path = Path(path).expanduser().resolve()
        self.fps = int(fps)
        self._writer: Any | None = None
        self.frames = 0

    def add(self, images: Mapping[str, np.ndarray]) -> None:
        import cv2

        from .cosmos_inference import make_unitree_concat_view

        rgb = make_unitree_concat_view(images)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # ``mp4v`` is the broadly available OpenCV/FFmpeg MPEG-4 codec in
            # the validated Isaac environment.  The four-character identifier
            # has no units; an unavailable codec fails before a partial replay
            # is claimed.  H.264 would be smaller but is not present in every
            # workstation build, so this fixed portable default is deliberate.
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.path),
                fourcc,
                float(self.fps),
                (int(rgb.shape[1]), int(rgb.shape[0])),
            )
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                raise RuntimeError(f"could not open replay video for writing: {self.path}")
        self._writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
