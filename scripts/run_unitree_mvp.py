#!/usr/bin/env python3
"""Run strict reset-time open-loop IK on Unitree's public G1 red-block task.

All frame IK calls occur while building the trajectory after reset. Once the
rollout starts, ``OpenLoopPolicy`` only indexes a frozen joint-action array.
Object poses, cameras, contacts, and success metrics never alter control.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np


def _floats(text: str, count: int, label: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must contain comma-separated numbers") from exc
    if len(values) != count:
        raise argparse.ArgumentTypeError(f"{label} must contain exactly {count} numbers")
    return values


def _vec3(text: str) -> tuple[float, float, float]:
    return _floats(text, 3, "vector")  # type: ignore[return-value]


def _quat(text: str) -> tuple[float, float, float, float]:
    return _floats(text, 4, "quaternion")  # type: ignore[return-value]


def _vec2(text: str) -> tuple[float, float]:
    return _floats(text, 2, "gripper target")  # type: ignore[return-value]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--unitree-root",
    type=Path,
    default=Path(os.environ.get("UNITREE_SIM_ISAACLAB_ROOT", os.environ.get("PROJECT_ROOT", "."))),
    help="checkout of unitreerobotics/unitree_sim_isaaclab",
)
parser.add_argument(
    "--task",
    default="Isaac-PickPlace-RedBlock-G129-Dex1-Joint",
    help="Unitree direct-joint red-block task",
)
parser.add_argument("--urdf", type=Path, required=True, help="fixed-base public G1 29-DoF URDF")
parser.add_argument("--package-dir", type=Path, action="append", default=[])
parser.add_argument("--ee-frame", default="right_wrist_yaw_link")
parser.add_argument(
    "--active-joints",
    default=(
        "right_shoulder_pitch_joint,right_shoulder_roll_joint,right_shoulder_yaw_joint,"
        "right_elbow_joint,right_wrist_roll_joint,right_wrist_pitch_joint,right_wrist_yaw_joint"
    ),
)
parser.add_argument("--target-position", type=_vec3, default=(-4.05, -4.03, 0.84))
parser.add_argument("--grasp-wrist-offset-world", type=_vec3, default=(0.0, 0.0, 0.08))
parser.add_argument("--grasp-quaternion-base-xyzw", type=_quat, default=None)
parser.add_argument("--gripper-open", type=_vec2, default=(0.03, 0.03))
parser.add_argument("--gripper-closed", type=_vec2, default=(-0.02, -0.02))
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--settle-steps", type=int, default=50)
parser.add_argument("--sim-quaternion-order", choices=("xyzw", "wxyz"), default="wxyz")
parser.add_argument("--trajectory-out", type=Path, default=Path("outputs/g1_pickplace_open_loop.npz"))
parser.add_argument("--record-root", type=Path, default=None)
parser.add_argument("--dataset-repo-id", default="local/g1-pickplace-open-loop")
parser.add_argument("--task-text", default="Pick up the red block and place it in the green target area.")
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--camera", action="append", default=["front_camera", "right_wrist_camera"])

# AppLauncher is imported only after PROJECT_ROOT/sys.path are prepared below.
known, _ = parser.parse_known_args()
unitree_root = known.unitree_root.expanduser().resolve()
if not unitree_root.is_dir():
    parser.error(f"--unitree-root does not exist: {unitree_root}")
os.environ["PROJECT_ROOT"] = str(unitree_root)
sys.path.insert(0, str(unitree_root))

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.record_root is not None and not args.no_video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac/Unitree imports must happen after SimulationApp starts.
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from g1pickplace import OpenLoopPolicy, PickPlaceConfig, Pose, ResetSnapshot, ResetTimePickPlacePlanner  # noqa: E402
from g1pickplace.evaluation import evaluate_pick_place  # noqa: E402
from g1pickplace.lerobot_writer import LeRobotEpisodeWriter  # noqa: E402
from g1pickplace.offline_ik import PinocchioFrameIK  # noqa: E402


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _identity_quaternion(order: Literal["xyzw", "wxyz"]) -> tuple[float, float, float, float]:
    return (0.0, 0.0, 0.0, 1.0) if order == "xyzw" else (1.0, 0.0, 0.0, 0.0)


def _configure_environment(env_cfg: Any) -> None:
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    sim_dt = float(env_cfg.sim.dt)
    env_cfg.decimation = max(1, int(round(1.0 / (sim_dt * args.fps))))
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 60.0

    # Make the public scene's cube easier to grasp without replacing its task code.
    if getattr(env_cfg.scene.object.spawn, "mass_props", None) is not None:
        env_cfg.scene.object.spawn.mass_props.mass = 0.15
    if getattr(env_cfg.scene.object.spawn, "physics_material", None) is not None:
        env_cfg.scene.object.spawn.physics_material.static_friction = 2.0
        env_cfg.scene.object.spawn.physics_material.dynamic_friction = 1.5
        env_cfg.scene.object.spawn.physics_material.restitution = 0.0

    # The original termination detects its own reset condition. Evaluation here
    # is separate and must not truncate the frozen action sequence.
    if getattr(env_cfg, "terminations", None) is not None:
        for name in vars(env_cfg.terminations):
            if not name.startswith("_"):
                try:
                    setattr(env_cfg.terminations, name, None)
                except Exception:
                    pass

    marker_z = float(args.target_position[2]) - 0.032
    marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OpenLoopTarget",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(float(args.target_position[0]), float(args.target_position[1]), marker_z),
            rot=_identity_quaternion(args.sim_quaternion_order),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.18, 0.18, 0.004),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.8, 0.1), metallic=0.0),
        ),
    )
    setattr(env_cfg.scene, "open_loop_target", marker)


def _action_joint_names(env: Any) -> tuple[str, ...]:
    term = getattr(env.action_manager, "_terms", {}).get("joint_pos")
    names = tuple(getattr(term, "_joint_names", ()))
    if names:
        return names
    return tuple(env.scene["robot"].data.joint_names)


def _ordered_joint_state(env: Any, joint_names: tuple[str, ...], field: str) -> np.ndarray:
    robot = env.scene["robot"]
    robot_names = tuple(robot.data.joint_names)
    values = _numpy(getattr(robot.data, field))[0]
    by_name = {name: float(values[index]) for index, name in enumerate(robot_names)}
    missing = [name for name in joint_names if name not in by_name]
    if missing:
        raise RuntimeError(f"robot state is missing action joints: {missing}")
    return np.asarray([by_name[name] for name in joint_names], dtype=np.float64)


def _root_pose(env: Any, asset_name: str) -> Pose:
    data = env.scene[asset_name].data
    position = _numpy(data.root_pos_w)[0]
    quaternion = _numpy(data.root_quat_w)[0]
    return Pose.from_sim(position, quaternion, args.sim_quaternion_order)


def _object_state(env: Any) -> tuple[Pose, np.ndarray]:
    data = env.scene["object"].data
    pose = Pose.from_sim(
        _numpy(data.root_pos_w)[0],
        _numpy(data.root_quat_w)[0],
        args.sim_quaternion_order,
    )
    velocity = np.asarray(_numpy(data.root_lin_vel_w)[0], dtype=np.float64)
    return pose, velocity


def _camera_images(env: Any) -> dict[str, np.ndarray]:
    if args.no_video:
        return {}
    result: dict[str, np.ndarray] = {}
    for configured_name in args.camera:
        if configured_name not in env.scene.keys():
            continue
        sensor = env.scene[configured_name]
        try:
            image = _numpy(sensor.data.output["rgb"])[0]
        except Exception:
            continue
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            continue
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * (255.0 if float(image.max(initial=0.0)) <= 1.0 else 1.0), 0, 255)
        short_name = configured_name.removesuffix("_camera").replace("wrist_", "wrist_")
        result[short_name] = image[..., :3].astype(np.uint8, copy=False)
    return result


def _phase_index_map(phases: tuple[str, ...]) -> dict[str, int]:
    return {name: index for index, name in enumerate(dict.fromkeys(phases))}


def main() -> int:
    env = None
    recorder = None
    try:
        importlib.import_module("tasks.g1_tasks")
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        _configure_environment(env_cfg)
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        observation, _ = env.reset()

        action_names = _action_joint_names(env)
        current = _ordered_joint_state(env, action_names, "joint_pos")
        defaults = _ordered_joint_state(env, action_names, "default_joint_pos")
        hold = (current - defaults).astype(np.float32)
        for _ in range(max(0, args.settle_steps)):
            observation, _, _, _, _ = env.step(
                torch.as_tensor(hold, dtype=torch.float32, device=env.device).unsqueeze(0)
            )

        # Exactly one reset snapshot drives planning.
        current = _ordered_joint_state(env, action_names, "joint_pos")
        defaults = _ordered_joint_state(env, action_names, "default_joint_pos")
        object_pose, _ = _object_state(env)
        target_pose = Pose(
            np.asarray(args.target_position, dtype=np.float64),
            np.asarray([0.0, 0.0, 0.0, 1.0]),
        )
        snapshot = ResetSnapshot(
            joint_names=action_names,
            joint_positions=current,
            default_joint_positions=defaults,
            robot_base_world=_root_pose(env, "robot"),
            object_world=object_pose,
            target_world=target_pose,
        )

        active_joints = tuple(name.strip() for name in args.active_joints.split(",") if name.strip())
        ik = PinocchioFrameIK(
            urdf_path=args.urdf,
            frame_name=args.ee_frame,
            active_joint_names=active_joints,
            package_dirs=args.package_dir,
        )
        planner = ResetTimePickPlacePlanner(
            ik,
            PickPlaceConfig(
                fps=args.fps,
                grasp_wrist_offset_world=args.grasp_wrist_offset_world,
                grasp_quaternion_base_xyzw=args.grasp_quaternion_base_xyzw,
                gripper_open_positions=args.gripper_open,
                gripper_closed_positions=args.gripper_closed,
            ),
        )
        trajectory, diagnostics = planner.build(snapshot)
        policy = OpenLoopPolicy(trajectory)
        args.trajectory_out.parent.mkdir(parents=True, exist_ok=True)
        trajectory.save_npz(str(args.trajectory_out))
        print(f"[plan] solved all IK before rollout: {json.dumps(diagnostics.waypoint_iterations)}")
        print(f"[plan] {trajectory.steps} steps, {trajectory.duration_s:.2f}s -> {args.trajectory_out}")

        initial_images = _camera_images(env)
        if args.record_root is not None:
            recorder = LeRobotEpisodeWriter(
                root=args.record_root,
                repo_id=args.dataset_repo_id,
                fps=args.fps,
                joint_names=action_names,
                camera_shapes={name: tuple(image.shape) for name, image in initial_images.items()},
                use_videos=bool(initial_images),
            )
            if not initial_images and not args.no_video:
                print("[record] cameras unavailable; recording state/action-only LeRobot data")

        phase_indices = _phase_index_map(trajectory.phases)
        target_vector = np.concatenate((target_pose.position, target_pose.quaternion_xyzw))
        while not policy.done:
            phase = policy.current_phase
            action = policy.act(observation)  # ignored by design
            if recorder is not None:
                object_pose, _ = _object_state(env)
                recorder.add_frame(
                    joint_positions=_ordered_joint_state(env, action_names, "joint_pos"),
                    action=action,
                    object_pose_xyzw=np.concatenate((object_pose.position, object_pose.quaternion_xyzw)),
                    target_pose_xyzw=target_vector,
                    phase_index=phase_indices[phase],
                    task=args.task_text,
                    images=_camera_images(env),
                )
            observation, _, terminated, truncated, _ = env.step(
                torch.as_tensor(action, dtype=torch.float32, device=env.device).unsqueeze(0)
            )
            if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                print("[run] environment ended before the frozen trajectory was exhausted")
                break

        if recorder is not None:
            recorder.save_episode()
            recorder.finalize()

        final_pose, final_velocity = _object_state(env)
        result = evaluate_pick_place(final_pose.position, final_velocity, target_pose.position)
        print("[result] " + json.dumps(result.__dict__, indent=2))
        return 0 if result.success else 2
    finally:
        if recorder is not None:
            recorder.finalize()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
