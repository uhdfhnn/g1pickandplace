# Unitree G1 Pick-and-Place MVP

Independent **reset-time open-loop IK** baseline for the public Unitree G1 red-block task in NVIDIA Isaac Lab, plus native LeRobot demonstration recording.

> **Current validation status:** all dependency-light code and 11 unit tests pass. The repository was authored without an Isaac Sim GPU runtime, so the simulator entry point has not yet been executed end to end. The first GPU pass must verify the local Unitree/Isaac Lab versions, URDF-to-USD joint/frame alignment, wrist offset/orientation, and Dex1 close values before this can be reported as a successful physical interaction.

## What this MVP does

The default backend is Unitree's public direct-joint task:

```text
Isaac-PickPlace-RedBlock-G129-Dex1-Joint
```

It uses the easy built-in red cuboid and two-finger Dex1 gripper. At reset it:

1. lets physics settle;
2. snapshots the robot joints, robot base pose, object pose, and fixed target pose once;
3. solves every Cartesian waypoint through an independent Pinocchio damped-least-squares backend;
4. converts all solved configurations into an immutable joint trajectory;
5. starts rollout only after all IK calls have completed.

During rollout:

```python
def act(self, observation=None):
    del observation
    action = self.trajectory.action_at(self.step)
    self.step += 1
    return action
```

There is no online IK, replanning, pose-error transition, contact-triggered branch, recovery controller, or action dependence on the observation. Cameras and object state are used only for recording and final evaluation.

## Manipulation program

```text
open_at_home
→ move_to_pregrasp
→ descend_to_grasp
→ close_gripper
→ grasp_settle
→ lift
→ transport
→ descend_to_place
→ open_gripper
→ release_settle
→ retreat
→ return_home
```

All transitions are compiled into time-indexed joint targets before rollout.

## Independence and provenance

This repository does **not** copy, paste, port, translate, mechanically rewrite, or import code from BrickSim, BrickBench, RoCoBrick, or related private repositories. Generic phases such as approach, grasp, lift, transport, release, and retreat are standard robotics concepts. The implementation uses public Unitree, Isaac Lab, Pinocchio, and LeRobot interfaces and is intentionally organized differently from the private waypoint policy shown during planning.

See [`docs/REFERENCES.md`](docs/REFERENCES.md) for the public sources reviewed.

## Repository layout

```text
src/g1pickplace/geometry.py       rigid transforms and quaternion conversion
src/g1pickplace/offline_ik.py     reset-time Pinocchio frame IK
src/g1pickplace/planner.py        semantic pick/place program and IK compilation
src/g1pickplace/trajectory.py     immutable joint trajectory and open-loop player
src/g1pickplace/evaluation.py     non-controlling success metrics
src/g1pickplace/lerobot_writer.py native LeRobot v3 writer
scripts/run_unitree_mvp.py        Unitree Isaac Lab integration
scripts/preview_plan.py           simulator-free planner smoke test
scripts/check_install.py          runtime dependency check
tests/                            dependency-light correctness tests
docs/                             design, research, and runbook notes
```

## Install

First install the public Unitree simulation repository and its supported Isaac Sim / Isaac Lab environment. Set its checkout path:

```bash
export UNITREE_SIM_ISAACLAB_ROOT=/absolute/path/to/unitree_sim_isaaclab
export PROJECT_ROOT="$UNITREE_SIM_ISAACLAB_ROOT"
```

Install this package in the same Python environment:

```bash
python -m pip install -e '.[dev]'
python scripts/check_install.py
```

The strict offline solver also needs a fixed-base public G1 29-DoF URDF and Pinocchio (`pin`). Do not use the simulator USD as an implicit kinematic model: pass the exact URDF explicitly and confirm that its joint names and `right_wrist_yaw_link` frame match the loaded USD.

## Test without Isaac Sim

```bash
python -m pytest
python scripts/preview_plan.py --output outputs/preview_trajectory.npz
```

The preview uses a fake IK backend only to validate planning, trajectory compilation, action offset conversion, and observation invariance. It is not a manipulation result.

## Run the simulator MVP

Example from an activated Isaac Lab environment:

```bash
python scripts/run_unitree_mvp.py \
  --unitree-root "$UNITREE_SIM_ISAACLAB_ROOT" \
  --urdf /absolute/path/to/g1_29dof.urdf \
  --package-dir /absolute/path/to/g1_description \
  --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint \
  --device cuda:0 \
  --enable_cameras
```

Headless data collection:

```bash
python scripts/run_unitree_mvp.py \
  --unitree-root "$UNITREE_SIM_ISAACLAB_ROOT" \
  --urdf /absolute/path/to/g1_29dof.urdf \
  --package-dir /absolute/path/to/g1_description \
  --device cuda:0 \
  --headless \
  --enable_cameras \
  --record-root datasets/g1_pickplace_mvp \
  --dataset-repo-id local/g1-pickplace-mvp
```

State/action-only fallback:

```bash
python scripts/run_unitree_mvp.py \
  --unitree-root "$UNITREE_SIM_ISAACLAB_ROOT" \
  --urdf /absolute/path/to/g1_29dof.urdf \
  --headless \
  --no-video \
  --record-root datasets/g1_pickplace_state_only
```

## First GPU validation checklist

1. Confirm that `Isaac-PickPlace-RedBlock-G129-Dex1-Joint` appears in the local task list and resets.
2. Confirm the simulator quaternion convention. The Unitree 4.5-style configs use `wxyz`; newer Isaac Lab builds may use `xyzw`. Set `--sim-quaternion-order` explicitly.
3. Confirm that the action term resolves to the same ordered joint names as `robot.data.joint_names`; the runner reads and validates the resolved action order.
4. Confirm the fixed-base URDF matches the USD's 29 body-joint names and base frame.
5. Confirm the configured end-effector frame exists. The default is `right_wrist_yaw_link`.
6. Run only through trajectory compilation first. Reject the episode if any waypoint fails IK.
7. Inspect the pregrasp pose and tune `--grasp-wrist-offset-world` and `--grasp-quaternion-base-xyzw`.
8. Confirm Dex1 open/close signs and limits. Public Unitree code maps roughly `0.03` open to `-0.02` closed, but the loaded asset is authoritative.
9. Verify genuine rigid contact: no teleport, weld, object attachment, or hidden pose reset.
10. Run ten fixed-seed trials, report failure modes, then enable bounded reset variants.
11. Inspect one written LeRobot episode and replay it before collecting a large dataset.

Useful tuning flags:

```bash
--target-position=-4.05,-4.03,0.84
--grasp-wrist-offset-world=0.0,0.0,0.08
--grasp-quaternion-base-xyzw=0.0,0.0,0.0,1.0
--gripper-open=0.03,0.03
--gripper-closed=-0.02,-0.02
--sim-quaternion-order=wxyz
```

Do not assume the identity grasp quaternion is correct; omitting the flag preserves the reset end-effector orientation.

## LeRobot data

Each episode can contain:

| Feature | Meaning |
|---|---|
| `observation.state` | ordered absolute robot joint positions |
| `observation.images.front` | head/front RGB, when available |
| `observation.images.right_wrist` | right wrist RGB, when available |
| `action` | exact default-offset joint action sent to Isaac Lab |
| `observation.object_pose` | privileged object pose for diagnosis/evaluation |
| `observation.target_pose` | target annotation |
| `observation.phase_index` | diagnostic program phase, not required as policy input |
| `task` | natural-language instruction |

The writer uses the public LeRobot lifecycle:

```text
LeRobotDataset.create → add_frame → save_episode → finalize
```

## Success criterion

The final evaluator requires:

- object center within the target's XY half-extents;
- object height near the target surface;
- final linear speed below the stability threshold.

The evaluator is read-only with respect to control.

## Scope after MVP

Only after Task 1 has a reproducible success rate and valid LeRobot episodes:

1. add a second distractor and text-conditioned single-object selection;
2. add bounded setup variants and reset-time reachability rejection;
3. add the pusher-and-puck tool task, with two-object stacking as fallback;
4. optionally integrate an existing policy for inference-only evaluation.
