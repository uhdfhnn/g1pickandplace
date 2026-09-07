# Unitree G1 Manipulation Demos

This repository implements and evaluates six reproducible Unitree G1 demos in
the public `Isaac-Stack-RgyBlock-G129-Dex1-Joint` scene.

| Demo | What we built | Current result |
| --- | --- | --- |
| Task 1 | Relative placement of the red block with respect to a marker | **PASS** |
| Task 2 | Instruction-conditioned selection among red, yellow, and green blocks | **PASS** for the recorded green/right prompt; red and yellow are plan-gated |
| Task 3 | Multi-object interaction: stack the red block on the yellow block | **PASS** |
| Task 4 | Tool use: grasp a shovel, scoop the red block, and unload it into a tray | **PARTIAL**; no successful scoop yet |
| Teleoperation | Visible, limit-clamped keyboard control of both arms and Dex1 grippers | **PASS** as an interactive demo; no autonomous success claim |
| Cosmos | First-frame policy inference, G1 action adaptation, safety preflight, and three-view replay | **PIPELINE PASS / TASK FAIL** on the recorded stack prompt |

> [!IMPORTANT]
> **All large artifacts—RGB videos, trajectories, logs, and LeRobot
> datasets—are in the [Google Drive artifact folder →](https://drive.google.com/drive/folders/1ymrNqLwD-4VxWCrw5eiCdH7-AAMfuNw-?usp=drive_link).**
> Large binaries are intentionally kept outside Git. The paths below identify
> files inside the downloaded artifact bundle or local `outputs/` tree.

## Project brief and coverage

The project brief encouraged completing as much as possible of the following:

1. Use Isaac Lab with `unitree_sim_isaaclab` to create up to three
   increasingly difficult Unitree G1 tasks. Each task should have a text
   instruction and several setup variants to reduce overfitting, following
   ideas from datasets such as LIBERO.
2. Control the simulated robot without a learned model and verify physical
   interaction with the environment.
3. Build a data-collection pipeline and collect the observations and actions
   needed to train a VLA model.
4. If time and hardware permit, integrate an existing policy, run inference on
   one task, and evaluate whether its generated actions complete the task.

Tasks 1–3 fulfill the requested increasing-difficulty task sequence. Task 4 is
an additional experimental tool-use extension. The model-free baseline,
keyboard teleoperation, native LeRobot data pipeline, and Cosmos integration
address goals 2–4.

| Brief item | Progress and result |
| --- | --- |
| Increasing task difficulty | Tasks 1–3 progress from one-object placement to instruction-conditioned selection and two-object stacking; all have a physically recorded PASS. |
| Setup variation | Task 1 includes a base run and one accepted reset variant; a second variant was rejected after a physical stability failure. Task 2 includes three instruction/object plans, with the green case physically recorded. Task 3 retains successful and rejected calibration evidence. |
| Model-free control | A reset-time open-loop IK expert completes all planning before rollout. Keyboard teleoperation provides a second, manual model-free path. |
| VLA data collection | Successful runs are stored as native LeRobot v3 episodes with language, synchronized RGB, robot state, action, object/target pose, and phase labels. |
| Existing-policy integration | Cosmos produced a full action chunk that was adapted, preflighted, replayed on G1, recorded, and evaluated. The pipeline succeeded, but that policy rollout did not complete the stack. |

**Approximate time:** the approved task design, final implementation, and
validation took about **5 hours**. Earlier Phase 0 setup and exploratory
calibration were not separately timed, so 5 hours is not claimed as the full
end-to-end project total.

### Data collected for VLA training

Each recorded autonomous episode stores:

| Field | Shape / rate | Purpose |
| --- | --- | --- |
| Text task | one instruction per episode | Language conditioning |
| `observation.images.front` | 480 × 640 RGB at 50 Hz | Head/front scene context |
| `observation.images.left_wrist` | 480 × 640 RGB at 50 Hz | Left-hand manipulation view |
| `observation.images.right_wrist` | 480 × 640 RGB at 50 Hz | Right-hand manipulation view |
| `observation.state` | 33-dimensional `float32` at 50 Hz | Ordered G1 body and Dex1 joint state |
| `action` | 33-dimensional `float32` at 50 Hz | Executed Isaac Lab joint command |
| `observation.object_pose` | XYZ + XYZW quaternion | Selected-object pose |
| `observation.target_pose` | XYZ + XYZW quaternion | Target pose |
| `observation.phase_index` | scalar | Demonstration phase label |

Task 1 contributes 1,550 synchronized frames; Tasks 2 and 3 contribute 1,650
frames each. Task 4 has a format-valid 2,515-frame negative episode that is
useful for diagnostics but must not be mixed into a success-only training set.
The current dataset is a verified demonstration sample, not a sufficiently
diverse VLA training corpus.

The autonomous expert snapshots the scene once, resolves the text instruction,
solves every IK waypoint, freezes the complete action sequence, and finishes
limits/collision/clearance checks before rollout step 0. Runtime images, state,
contacts, rewards, and errors are recorded for evaluation but never change the
policy.

Detailed evidence is in
[docs/ENTRANCE_TEST_REPORT.md](docs/ENTRANCE_TEST_REPORT.md), and expanded
visible-gate commands are in
[docs/RUN_ENTRANCE_TEST_DEMO.md](docs/RUN_ENTRANCE_TEST_DEMO.md).

## 1. Setup

Clone the project and run the reproducible bootstrap:

```bash
git clone https://github.com/uhdfhnn/g1pickandplace.git
cd g1pickandplace
bash scripts/setup_environment.sh
```

The bootstrap installs the validated public Unitree, Isaac Lab, ROS
description, SDK, CycloneDDS, Pinocchio, and LeRobot dependencies as sibling
checkouts. Hardware requirements, pinned versions, Assimp setup, and recovery
steps are documented in [SETUP.md](SETUP.md).

Validate the dependency-light parts of the checkout:

```bash
python3 -m pytest -q
python3 -m compileall -q src scripts tests
git diff --check
```

Current validation: **186 tests passed**; compilation and whitespace checks
passed.

## 2. Task 1 result — relative placement

**Text prompt**

```text
Pick up the red block and place it left of the yellow square marker.
```

**Result: PASS.** The robot selected only the red block, lifted it by
`0.074059 m`, and placed it left of the marker with `0.047375 m` measured
edge clearance for a requested `0.050000 m`. Yellow and green did not move.

**Data collected:** one native LeRobot episode; 1,550 rows at 50 Hz; 33-D
state/action; text prompt; front, left-wrist, and right-wrist RGB; object and
target poses; phase labels; matching frozen/recorded action hashes.

| RGB evidence | Artifact path |
| --- | --- |
| Head/front video | `deliverables/unitree_g1_manipulation_task123/task1_relative_place/base/lerobot/videos/observation.images.front/chunk-000/file-000.mp4` |
| Left-wrist video | `deliverables/unitree_g1_manipulation_task123/task1_relative_place/base/lerobot/videos/observation.images.left_wrist/chunk-000/file-000.mp4` |
| Right-wrist video | `deliverables/unitree_g1_manipulation_task123/task1_relative_place/base/lerobot/videos/observation.images.right_wrist/chunk-000/file-000.mp4` |
| Quick front-camera copy | `deliverables/unitree_g1_manipulation_task123/recordings/task1_relative_place_front_from_waypoint0.mp4` |

[Open Task 1 artifacts in Google Drive →](https://drive.google.com/drive/folders/1ymrNqLwD-4VxWCrw5eiCdH7-AAMfuNw-?usp=drive_link)

**Problem encountered:** one negative-offset reset variant passed planning but
failed physical stability. It is retained as rejected evidence and is not
counted as a successful setup variant.

## 3. Task 2 result — instruction-conditioned selection

**Recorded text prompt**

```text
Pick up the green block and place it right of the yellow square marker.
```

The other supported Task 2 prompts select red/left and yellow/front. They have
passed reset-time planning but are not claimed as recorded physical successes.

**Result: PASS for the recorded green/right prompt.** All three blocks were
present in the immutable reset snapshot. The instruction selected green, moved
it `0.169501 m`, achieved `0.054717 m` edge clearance, and left red and
yellow stationary.

**Data collected:** one native LeRobot episode; 1,650 rows at 50 Hz; 33-D
state/action; the exact green/right prompt; three synchronized RGB streams;
selected-object and target poses; phase labels; matching action hashes.

| RGB evidence | Artifact path |
| --- | --- |
| Head/front video | `deliverables/unitree_g1_manipulation_task123/task2_type_conditioned_green/lerobot/videos/observation.images.front/chunk-000/file-000.mp4` |
| Left-wrist video | `deliverables/unitree_g1_manipulation_task123/task2_type_conditioned_green/lerobot/videos/observation.images.left_wrist/chunk-000/file-000.mp4` |
| Right-wrist video | `deliverables/unitree_g1_manipulation_task123/task2_type_conditioned_green/lerobot/videos/observation.images.right_wrist/chunk-000/file-000.mp4` |
| Quick front-camera copy | `deliverables/unitree_g1_manipulation_task123/recordings/task2_type_conditioned_front_from_waypoint0.mp4` |

[Open Task 2 artifacts in Google Drive →](https://drive.google.com/drive/folders/1ymrNqLwD-4VxWCrw5eiCdH7-AAMfuNw-?usp=drive_link)

**Problem encountered:** language parsing is deliberately finite. Unsupported
or conflicting paraphrases fail before simulation instead of silently
selecting the wrong object or relation.

## 4. Task 3 result — red-on-yellow stacking

**Text prompt**

```text
Pick up the red block and stack it on the yellow block.
```

**Result: PASS.** The accepted rollout lifted red by `0.071563 m` and
produced a stable stack with `0.003331 m` top-center XY error and
`0.000002 m` vertical error. Yellow remained upright and green did not move.

**Data collected:** one native LeRobot episode; 1,650 rows at 50 Hz; 33-D
state/action; stack prompt; three synchronized RGB streams; top/bottom object
evidence; target pose and phase labels; matching action hashes.

| RGB evidence | Artifact path |
| --- | --- |
| Head/front video | `deliverables/unitree_g1_manipulation_task123/task3_red_on_yellow_stack/lerobot/videos/observation.images.front/chunk-000/file-000.mp4` |
| Left-wrist video | `deliverables/unitree_g1_manipulation_task123/task3_red_on_yellow_stack/lerobot/videos/observation.images.left_wrist/chunk-000/file-000.mp4` |
| Right-wrist video | `deliverables/unitree_g1_manipulation_task123/task3_red_on_yellow_stack/lerobot/videos/observation.images.right_wrist/chunk-000/file-000.mp4` |
| Quick front-camera copy | `deliverables/unitree_g1_manipulation_task123/recordings/task3_stack_front_from_waypoint0.mp4` |

[Open Task 3 artifacts in Google Drive →](https://drive.google.com/drive/folders/1ymrNqLwD-4VxWCrw5eiCdH7-AAMfuNw-?usp=drive_link)

**Problem encountered:** an earlier calibration moved the yellow support block
and failed stability. The accepted calibration was rerun through inspect,
plan, physical rollout, recording, and dataset validation.

## 5. Task 4 result — shovel tool use

**Text prompt**

```text
Pick up the shovel, scoop the red block, and place it in the target tray.
```

**Result: PARTIAL, not PASS.** The reset-time planner solves all 22 waypoints
and completes 345,138 swept checks with zero failures before rollout. In the
best complete physical attempt, both named finger contacts occurred and the
shovel left the table, but lift was only `0.015515 m`, below the required
`0.050000 m`. The red block was not causally scooped into the tray. Later
handle and drive-gain trials also dropped the tool. The current dual-finger
socket design is covered by dependency-light tests but does not yet have an
evaluator-valid physical success recording.

**Data collected:** one format-valid negative LeRobot episode with 2,515 rows
at 50 Hz, 33-D state/action, the shovel prompt, three synchronized RGB streams,
tool/object/target observations, phase labels, and matching action hashes;
additional inspect, plan, contact, and boundary-frame diagnostics from later
partial attempts.

The videos below are **negative/partial evidence** and must not be presented as
a successful scoop.

| RGB evidence | Artifact path |
| --- | --- |
| Head/front video | `outputs/task4_graspcenter_rollout_02/lerobot/videos/observation.images.front/chunk-000/file-000.mp4` |
| Left-wrist video | `outputs/task4_graspcenter_rollout_02/lerobot/videos/observation.images.left_wrist/chunk-000/file-000.mp4` |
| Right-wrist video | `outputs/task4_graspcenter_rollout_02/lerobot/videos/observation.images.right_wrist/chunk-000/file-000.mp4` |
| Latest partial key frames | `outputs/task4_drive2x_rollout_06/rollout_frames/` |

[Open Task 4 artifacts in Google Drive →](https://drive.google.com/drive/folders/1ymrNqLwD-4VxWCrw5eiCdH7-AAMfuNw-?usp=drive_link)

**Problem encountered:** holding the tool through lift and transport remains
the limiting failure. Doubling the left-finger drive gains did not fix it,
which points to grasp/contact geometry rather than configured gain alone.

## 6. Teleoperation demo result

**Text prompt:** not applicable. This is direct manual keyboard control, not a
language-conditioned autonomous task.

**Result: PASS as an interactive demo.** The operator can switch arms, select
any of seven arm joints, jog the target in either direction, and open or close
the selected Dex1 gripper. Every command is clamped to live simulator soft
limits, while unselected joints hold their captured reset positions.
Teleoperation is isolated from the autonomous path: it does not construct
`OpenLoopPolicy`, evaluate task success, or create a LeRobot episode.

**Data collected:** a timestamped `teleop.log` containing mode and clean-exit
evidence. No autonomous trajectory, success label, or training episode is
created, because manual teleoperation is currently a control-verification demo
rather than a recording mode.

There is no dedicated prerecorded teleoperation RGB video in the current local
artifact set. A run is verifiable from the visible viewport and
`teleop.log`; a future head/wrist capture can be published in the same
[Google Drive artifact folder](https://drive.google.com/drive/folders/1ymrNqLwD-4VxWCrw5eiCdH7-AAMfuNw-?usp=drive_link)
without adding large media files to Git.

**Problem encountered:** the simulator viewport must have keyboard focus.
Teleoperation intentionally remains transparent joint-space jogging rather
than online Cartesian IK.

## 7. Cosmos inference result

**Text prompt**

```text
Pick up the red block and stack it on the yellow block.
```

**Result: inference pipeline PASS; physical task FAIL.** The simulator captured
settled 480 × 640 RGB frames from the head/front, left-wrist, and right-wrist
cameras and built a 720 × 640 `concat_view`. Cosmos returned a 16-second
action chunk with shape `160 × 29` at 10 Hz. The adapter
quantile-denormalized the AgiBotWorld action, mapped the selected wrist and
gripper to G1, solved the complete path before rollout, and passed preflight
with 800 simulator commands at 50 Hz.

The replay and three-camera video completed, but the evaluator reported
`success=false`: red was not lifted or stacked, and its final XY error from
yellow was `0.146105 m`. This validates inference, action adaptation,
safety-gating, and replay plumbing—not policy task performance on G1.

**Data collected:** three first-frame RGB inputs; the concatenated model input;
prompt and request metadata; a `160 × 29` Cosmos action; initial G1 context;
an 800-step frozen G1 replay trajectory; preflight metrics; final task metrics;
and a 160-frame, 16-second concatenated RGB replay at 10 fps.

| RGB evidence | Artifact path |
| --- | --- |
| Head + both wrists, concatenated replay | `outputs/cosmos_stack_red_on_yellow_16s_replay_run3/cosmos_replay.mp4` |
| First-frame head/wrist input | `outputs/cosmos_stack_red_on_yellow_16s/unitree_concat_view.png` |
| Replay midpoint key frame | `outputs/cosmos_stack_red_on_yellow_16s_replay_run3/replay_midpoint.png` |
| Model output and metadata | `outputs/cosmos_stack_red_on_yellow_16s/cosmos_policy_action.json` and `metadata.json` |

[Open Cosmos artifacts in Google Drive →](https://drive.google.com/drive/folders/1ymrNqLwD-4VxWCrw5eiCdH7-AAMfuNw-?usp=drive_link)

**Problem encountered:** Cosmos predicts normalized AgiBotWorld end-effector
actions, while the Unitree environment expects 33 ordered joint commands. The
adapter and preflight bridge this representation gap, but the recorded policy
motion still did not grasp the block.

## 8. How to run

### Tasks 1–4

The safe default opens visible inspect and plan gates, then exits before
physical rollout:

```bash
python3 scripts/run_demo.py \
  --instruction "Pick up the red block and stack it on the yellow block."
```

Add `--rollout` only after the visible plan passes:

```bash
python3 scripts/run_demo.py \
  --instruction "Pick up the red block and stack it on the yellow block." \
  --rollout
```

Supported autonomous text prompts are:

```text
Pick up the red block and place it left of the yellow square marker.
Pick up the green block and place it right of the yellow square marker.
Pick up the yellow block and place it in front of the yellow square marker.
Pick up the red block and stack it on the yellow block.
Pick up the shovel, scoop the red block, and place it in the target tray.
```

Task 4 rollout remains experimental. A passing plan is not a physical Task 4
PASS; only a complete recording that satisfies the scoop-and-tray evaluator
can establish success.

### Keyboard teleoperation

```bash
python3 scripts/run_demo.py --keyboard-teleop
```

Click the Isaac Sim viewport once to give it keyboard focus:

| Key | Action |
| --- | --- |
| `Tab` | Switch between the right and left arm |
| `1`–`7` | Select shoulder pitch through wrist yaw |
| `Left` / `Right` | Decrease / increase the selected target by 2 degrees |
| `O` / `C` | Open / close the selected arm's Dex1 gripper |
| `Q` or `Esc` | Exit cleanly |

### Cosmos inference and replay

Open the SSH tunnel to the Cosmos policy server and verify the endpoint:

```bash
ssh -L 8080:localhost:8080 <user>@<cosmos-host>
curl http://localhost:8080/v1/models
```

Run one visible first-frame inference. The returned action is saved but not
executed by this command:

```bash
python3 scripts/run_cosmos_inference.py \
  --base-url http://localhost:8080 \
  --duration-s 16 \
  --instruction "Pick up the red block and stack it on the yellow block." \
  --output-dir outputs/cosmos_stack_red_on_yellow_16s
```

Replay the saved action through the G1 adapter and record the concatenated RGB
video:

```bash
python3 scripts/replay_cosmos_action.py \
  --action outputs/cosmos_stack_red_on_yellow_16s \
  --instruction "Pick up the red block and stack it on the yellow block." \
  --output-dir outputs/cosmos_stack_red_on_yellow_16s_replay
```

The replay fails closed before action 0 if action decoding, IK, joint-limit,
step-size, or URDF self-collision validation fails. A completed replay video
does not imply task success; the final simulator evaluator is authoritative.
