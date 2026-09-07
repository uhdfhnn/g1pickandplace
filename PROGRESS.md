# Progress log

## 2026-09-05 — Phase 0 host verification

Initial repository state:

```text
## main...origin/main
```

The checkout was clean before validation. The prescribed commands were run
verbatim first; both failed because this host has no `python` executable.
`python3 -m pytest` then passed all 11 dependency-light tests. The uninstalled
checkout also made `python3 scripts/preview_plan.py` fail to import
`g1pickplace`; the equivalent `PYTHONPATH=src python3 scripts/preview_plan.py`
succeeded and saved a 480-by-4, 50 Hz, 9.60 s preview trajectory. Its fake IK
backend reported 3 iterations for each of pregrasp, grasp, lift, preplace, and
place. No implementation files were changed before these checks.

Host and public dependency manifest:

```text
unitree_sim_isaaclab commit: e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc
Isaac Sim: 5.0.0-rc.45+release.23960.184afb15.gl (standalone VERSION file)
Isaac Lab: v2.2.0, commit 46dff135f44683f031edf346e544fcfd8456b2bb
Isaac Lab Python packages: isaaclab 0.44.9, isaaclab_tasks 0.10.45
Pinocchio: 2.7.0 (installed distribution: pin 2.7.0)
LeRobot: not installed in the Isaac environment at the time of Phase 0 capture
NVIDIA driver: 580.167.08
NVIDIA-SMI maximum supported CUDA: 13.0
GPU: NVIDIA GeForce RTX 5090, 32607 MiB
```

The Unitree README specifically directs RTX 50-series hosts to Isaac Sim
5.0.0. The simulator environment uses Isaac Sim's bundled PyTorch
2.7.0+cu128 and NumPy 1.26.0.

Registered public G1 task IDs found in the pinned Unitree checkout (runtime
registry confirmation is part of the simulator smoke test):

```text
Isaac-Move-Cylinder-G129-Dex1-Wholebody
Isaac-Move-Cylinder-G129-Dex3-Wholebody
Isaac-Move-Cylinder-G129-Inspire-Wholebody
Isaac-Pick-Redblock-Into-Drawer-G129-Dex1-Joint
Isaac-Pick-Redblock-Into-Drawer-G129-Dex3-Joint
Isaac-PickPlace-Cylinder-G129-Dex1-Joint
Isaac-PickPlace-Cylinder-G129-Dex3-Joint
Isaac-PickPlace-Cylinder-G129-Inspire-Joint
Isaac-PickPlace-RedBlock-G129-Dex1-Joint
Isaac-PickPlace-RedBlock-G129-Dex3-Joint
Isaac-PickPlace-RedBlock-G129-Inspire-Joint
Isaac-Stack-RgyBlock-G129-Dex1-Joint
Isaac-Stack-RgyBlock-G129-Dex3-Joint
Isaac-Stack-RgyBlock-G129-Inspire-Joint
```

The direct-joint red-block USD has a 33-dimensional action: 29 G1 body joints
plus four Dex1 finger joints. Its authored movable-joint order is:

```text
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
waist_yaw_joint
waist_roll_joint
waist_pitch_joint
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_joint
left_wrist_roll_joint
left_wrist_pitch_joint
left_wrist_yaw_joint
left_hand_Joint1_1
left_hand_Joint2_1
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint
right_hand_Joint1_1
right_hand_Joint2_1
```

URDF/USD compatibility was checked against Unitree's public
`unitree_ros/robots/g1_description/g1_29dof.urdf` at commit
`7d6075f7f58588b189b940130e3edab3c839b2df`. Pinocchio loads it as a fixed-base
29-joint model. All 29 ordered body-joint names exactly equal the USD order
after filtering the four simulator-only Dex1 joints. Both models contain the
`right_wrist_yaw_link` frame. At the neutral configuration, the
pelvis-to-wrist translation differs by `1.05e-8 m` and orientation by
`2.0815e-4 rad` (about 0.0119 degrees), consistent with USD rounding. This is
compatible for reset-time IK; the simulator smoke test must still confirm the
runtime-resolved order, limits, poses, and frame.

Phase 0 blockers/uncertainty entering the smoke test:

- LeRobot is not yet installed in the simulator environment; its current
  dependency resolver would replace Isaac Lab's required NumPy/Gymnasium
  versions unless constrained.
- The USD order above is authored stage order, not yet the runtime-resolved
  Isaac Lab action-term order.
- Scene creation, cameras, action limits, and real reset-time IK remain to be
  validated in the running simulator.

## 2026-09-05 — Phase 1 simulator smoke test and planner-only gate

Files changed:

```text
PROGRESS.md
scripts/run_unitree_mvp.py
src/g1pickplace/offline_ik.py
```

The public `Isaac-PickPlace-RedBlock-G129-Dex1-Joint` environment launched in
headless camera mode. `--inspect-only` returned before constructing either the
expert or trajectory. The live scene contained `robot`, `object`,
`packing_table`, `open_loop_target`, `front_camera`, and
`right_wrist_camera`; front and wrist RGB were both `480 x 640 x 3`. The live
registry contained all 14 G1 task IDs listed in Phase 0.

The live action term is 33-dimensional, has scale `1.0`, uses the default
offset, and therefore expects
`environment_action = absolute_joint_target - default_joint_position`. All
authored default positions resolved to `0 rad`. The runtime action order below
is authoritative; it differs from raw USD traversal order recorded in Phase 0:

```text
left_hip_pitch_joint
right_hip_pitch_joint
waist_yaw_joint
left_hip_roll_joint
right_hip_roll_joint
waist_roll_joint
left_hip_yaw_joint
right_hip_yaw_joint
waist_pitch_joint
left_knee_joint
right_knee_joint
left_shoulder_pitch_joint
right_shoulder_pitch_joint
left_ankle_pitch_joint
right_ankle_pitch_joint
left_shoulder_roll_joint
right_shoulder_roll_joint
left_ankle_roll_joint
right_ankle_roll_joint
left_shoulder_yaw_joint
right_shoulder_yaw_joint
left_elbow_joint
right_elbow_joint
left_wrist_roll_joint
right_wrist_roll_joint
left_wrist_pitch_joint
right_wrist_pitch_joint
left_wrist_yaw_joint
right_wrist_yaw_joint
left_hand_Joint1_1
left_hand_Joint2_1
right_hand_Joint1_1
right_hand_Joint2_1
```

On the successful fixed-pose reset, the robot base pose was
`[-4.19999981, -3.70000005, 0.75999999] m` with world quaternion XYZW
`[-1.23e-11, 1.23e-11, 0.70711358, -0.70709999]`; the cube pose was
`[-4.25, -4.03000021, 0.83999997] m` with identity quaternion; and
`right_wrist_yaw_link` was at
`[-4.34865618, -3.89968371, 0.85455197] m` with quaternion XYZW
`[-0.00136197, -0.00129112, 0.70710902, -0.70710205]`.

The runner now has explicit inspection and planning gates. It also intersects
the URDF hard limits with the loaded simulator soft limits for every active IK
joint and refuses to save a trajectory if any controlled arm or gripper target
is outside the live command envelope. This caught the original `0.03/-0.02 rad`
finger presets outside the live Dex1 soft range
`[-0.017775, 0.022275] rad`; the planner-only validation used `0.02 rad` open
and `-0.015 rad` closed. Pinocchio 2.7 integration was corrected to use its
frame-only URDF loader overload and to recognize its one-past-the-array
unknown-joint sentinel for simulator-only finger joints.

The successful planner-only command was:

```text
unset CYCLONEDDS_HOME
PYTHONUNBUFFERED=1 conda run --no-capture-output -n unitree_sim_env python scripts/run_unitree_mvp.py --unitree-root ../unitree_sim_isaaclab --urdf ../unitree_ros/robots/g1_description/g1_29dof.urdf --package-dir ../unitree_ros/robots/g1_description --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint --device cuda:0 --enable_cameras --headless --plan-only --fixed-object-reset --reset-seed 0 --settle-steps 0 --grasp-wrist-offset-world=0.0,0.16,0.08 --grasp-quaternion-base-xyzw=0.0,0.0,0.0,1.0 --gripper-open=0.02,0.02 --gripper-closed=-0.015,-0.015 --trajectory-out outputs/g1_pickplace_open_loop.npz
```

The `0.16 m` positive world-Y wrist offset puts the wrist behind the block
along the G1's approach direction for the authored `-90 degree` base yaw. It
was selected from public URDF reachability checks over the fixed public cube
pose; smaller offsets made the lift unreachable at the live wrist-pitch soft
limit, while larger offsets made pregrasp unreachable. Identity in the base
frame was used instead of the slightly gravity-deflected reset orientation.
These are provisional planning values, not a validated grasp calibration.

All five waypoints converged before trajectory creation:

```text
waypoint   iterations   residual
pregrasp           73   9.9254e-05
grasp              68   9.2161e-05
lift              536   9.8505e-05
preplace           74   9.6831e-05
place              68   9.3833e-05
```

The process then printed `rollout step zero was not started` and exited. No
trajectory action was sent. `outputs/g1_pickplace_open_loop.npz` contains
finite `absolute_targets` and `env_actions` arrays of shape `497 x 33`, all 12
semantic phases in order, `50 Hz`, and `9.94 s` duration. Maximum per-step
joint motion is `0.0199633 rad`; controlled-limit violations are empty; and,
because defaults are zero, saved environment actions exactly equal absolute
targets. The right wrist pitch reaches its live upper soft limit exactly.

Commands run after implementation and their results:

```text
python3 -m pytest
  11 passed in 0.05s
PYTHONPATH=src python3 scripts/preview_plan.py --output outputs/preview_trajectory_after_phase1.npz
  480 x 4, 50 Hz, 9.60 s; all five fake-IK waypoints reported
python3 -m compileall -q src scripts
  passed
git diff --check
  passed
```

Blockers and uncertainty:

- No rollout is authorized yet. The lift solution lands exactly on the live
  wrist-pitch soft limit, so Phase 2 should tune the documented pose parameters
  to create margin before sending action zero.
- Unitree's full random reset range includes poses the default right-arm-only
  solver cannot reach. Those variants were rejected during planning, before
  any rollout, as required.
- Setting `CYCLONEDDS_HOME` to the local install prefix makes Unitree's global
  DDS initializer terminate silently; it must remain unset for this runtime.
- Isaac Sim reports a Warp `cuDeviceGetUuid` driver-entry warning on the RTX
  5090 and missing decorative warehouse MDL materials. Physics, cameras, scene
  reset, and planner-only construction nevertheless completed.
- LeRobot remains uninstalled because its unconstrained resolver conflicts with
  the pinned Isaac Lab NumPy/Gymnasium environment. No dataset work was needed
  or attempted in Phases 0-1.

Next implementation action: before Phase 2 rollout, tune only the documented
wrist offset/orientation, gripper targets, target, and durations against the
fixed pose so every solve has joint-limit margin; then validate real contact,
lift, and release without attachment or teleportation.

## 2026-09-05 — repository MVP scaffold (historical baseline)

Completed:

- independent rigid-pose utilities;
- independent reset-time Pinocchio frame-IK backend;
- semantic pick/place planner that solves all IK before rollout;
- immutable absolute-target and environment-action trajectories;
- observation-invariant policy player;
- read-only placement/stability metrics;
- native LeRobot feature schema and episode writer;
- Unitree red-block integration script;
- public reference review and provenance statement;
- 11 dependency-light tests and GitHub Actions workflow.

Validation performed:

```text
python -m compileall -q src scripts tests
python -m pytest
python scripts/preview_plan.py --output /tmp/g1_preview.npz
```

Observed:

```text
11 tests passed
preview trajectory: 480 x 4 at 50 Hz, 9.60 s
```

Not yet validated when the scaffold was written (superseded where Phase 0/1
above records completed checks):

- Isaac Sim launch;
- Unitree asset availability;
- URDF/USD kinematic equivalence;
- real frame name and quaternion convention;
- actual grasp orientation and wrist offset;
- Dex1 contact/closure values;
- task success rate;
- camera recording and LeRobot video finalization in the target environment.

Hours: fill in actual human time before submission. Do not infer or fabricate it from commit timestamps.

## 2026-09-05 — Phase 2 fixed-seed physical validation complete

### Files changed

- `scripts/run_unitree_mvp.py`: added documented CLI controls for all existing
  segment durations and world-+Z approach/lift heights, fixed/exact public
  object resets, live soft-limit validation, read-only phase-boundary state and
  optional front/right-wrist PNG diagnostics, and the staging switch. These
  diagnostics never change actions or phase transitions.
- `src/g1pickplace/offline_ik.py`: intersects public URDF hard limits with the
  live Isaac Lab soft command limits before reset-time solving.
- `src/g1pickplace/planner.py`: optional staging is solved before the five task
  waypoints and reused on both sides of the task. The post-release
  `return_via_staging` segment reuses the already-frozen staging joint target;
  it performs no additional IK and introduces no feedback.
- `tests/test_planner.py`, `tests/test_object_reset_variants.py`, and
  `tests/test_phase_boundary_frames.py`: dependency-light coverage for the
  staging solve/order/reuse, exact reset bounds, duration/height validation,
  and deterministic diagnostic filenames.
- `PROGRESS.md`: this Phase 2 audit.

### Exact commands run

The calibrated plan-only gate was run before its matching rollout:

```bash
unset CYCLONEDDS_HOME
PYTHONUNBUFFERED=1 conda run --no-capture-output -n unitree_sim_env python scripts/run_unitree_mvp.py \
  --unitree-root ../unitree_sim_isaaclab \
  --urdf ../unitree_ros/robots/g1_description/g1_29dof.urdf \
  --package-dir ../unitree_ros/robots/g1_description \
  --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint --device cuda:0 --enable_cameras --headless \
  --plan-only --fixed-object-reset --reset-seed 0 --enable-staging \
  --target-position=-4.15,-4.03,0.84 --grasp-wrist-offset-world=0.025,0.165,0.0 \
  --grasp-quaternion-base-xyzw=0,0.17410813759359595,0,0.9847265389049334 \
  --gripper-open=-0.0175,-0.0175 --gripper-closed=0.0222,0.0222 \
  --approach-height-m 0.16 --lift-height-m 0.12 --target-approach-height-m 0.12 \
  --pregrasp-duration-s 3 --descend-duration-s 5 --gripper-duration-s 1 --settle-duration-s 1 \
  --lift-duration-s 3 --transport-duration-s 2 --return-duration-s 2 \
  --trajectory-out outputs/phase2_calibrated_safe_return_plan.npz \
  2>&1 | tee outputs/phase2_calibrated_safe_return_plan.log
```

```bash
unset CYCLONEDDS_HOME
PYTHONUNBUFFERED=1 conda run --no-capture-output -n unitree_sim_env python scripts/run_unitree_mvp.py \
  --unitree-root ../unitree_sim_isaaclab \
  --urdf ../unitree_ros/robots/g1_description/g1_29dof.urdf \
  --package-dir ../unitree_ros/robots/g1_description \
  --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint --device cuda:0 --enable_cameras --headless \
  --fixed-object-reset --reset-seed 0 --enable-staging \
  --target-position=-4.15,-4.03,0.84 --grasp-wrist-offset-world=0.025,0.165,0.0 \
  --grasp-quaternion-base-xyzw=0,0.17410813759359595,0,0.9847265389049334 \
  --gripper-open=-0.0175,-0.0175 --gripper-closed=0.0222,0.0222 \
  --approach-height-m 0.16 --lift-height-m 0.12 --target-approach-height-m 0.12 \
  --pregrasp-duration-s 3 --descend-duration-s 5 --gripper-duration-s 1 --settle-duration-s 1 \
  --lift-duration-s 3 --transport-duration-s 2 --return-duration-s 2 \
  --phase-boundary-frame-root outputs/phase2_calibrated_safe_return_frames \
  --trajectory-out outputs/phase2_calibrated_safe_return_rollout.npz \
  2>&1 | tee outputs/phase2_calibrated_safe_return_rollout.log
```

The ten fixed-seed episodes used the same arguments in ten fresh, sequential
processes. The public task contains camera sensors and rejected an initial
attempt which omitted `--enable_cameras`; all ten attempts ended during scene
construction, before reset, planning, or rollout, and were overwritten by the
valid repetition logs below.

```bash
mkdir -p outputs/phase2_fixed10
unset CYCLONEDDS_HOME
for i in $(seq -w 0 9); do
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n unitree_sim_env python scripts/run_unitree_mvp.py \
    --unitree-root ../unitree_sim_isaaclab \
    --urdf ../unitree_ros/robots/g1_description/g1_29dof.urdf \
    --package-dir ../unitree_ros/robots/g1_description \
    --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint --device cuda:0 --enable_cameras --headless \
    --fixed-object-reset --reset-seed 0 --enable-staging \
    --target-position=-4.15,-4.03,0.84 --grasp-wrist-offset-world=0.025,0.165,0.0 \
    --grasp-quaternion-base-xyzw=0,0.17410813759359595,0,0.9847265389049334 \
    --gripper-open=-0.0175,-0.0175 --gripper-closed=0.0222,0.0222 \
    --approach-height-m 0.16 --lift-height-m 0.12 --target-approach-height-m 0.12 \
    --pregrasp-duration-s 3 --descend-duration-s 5 --gripper-duration-s 1 --settle-duration-s 1 \
    --lift-duration-s 3 --transport-duration-s 2 --return-duration-s 2 \
    --trajectory-out "outputs/phase2_fixed10/episode_${i}.npz" \
    > "outputs/phase2_fixed10/episode_${i}.log" 2>&1
  rg '^\[(plan|result|error)' "outputs/phase2_fixed10/episode_${i}.log" | tail -n 4 | sed "s/^/[episode ${i}] /"
done
```

The successful public LeRobot/video episode used:

```bash
unset CYCLONEDDS_HOME
PYTHONUNBUFFERED=1 conda run --no-capture-output -n unitree_sim_env python scripts/run_unitree_mvp.py \
  --unitree-root ../unitree_sim_isaaclab \
  --urdf ../unitree_ros/robots/g1_description/g1_29dof.urdf \
  --package-dir ../unitree_ros/robots/g1_description \
  --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint --device cuda:0 --enable_cameras --headless \
  --fixed-object-reset --reset-seed 0 --enable-staging \
  --target-position=-4.15,-4.03,0.84 --grasp-wrist-offset-world=0.025,0.165,0.0 \
  --grasp-quaternion-base-xyzw=0,0.17410813759359595,0,0.9847265389049334 \
  --gripper-open=-0.0175,-0.0175 --gripper-closed=0.0222,0.0222 \
  --approach-height-m 0.16 --lift-height-m 0.12 --target-approach-height-m 0.12 \
  --pregrasp-duration-s 3 --descend-duration-s 5 --gripper-duration-s 1 --settle-duration-s 1 \
  --lift-duration-s 3 --transport-duration-s 2 --return-duration-s 2 \
  --record-root datasets/phase2_fixed_success --dataset-repo-id local/g1-pickplace-phase2-success \
  --task-text 'Pick up the red block and place it in the green target area.' \
  --trajectory-out outputs/phase2_fixed_success_dataset.npz \
  2>&1 | tee outputs/phase2_fixed_success_dataset.log
```

Dataset tables were read with the installed `pyarrow` in `unitree_sim_env`.
Each video was independently decoded/count-probed with this exact loop:

```bash
for video in datasets/phase2_fixed_success/videos/*/chunk-000/file-000.mp4; do
  ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration \
    -of json "$video"
done
```

### Observed result

- Every calibrated plan reported all six reset-time solves before step zero:
  staging 71, pregrasp 77, grasp 70, lift 68, preplace 66, place 68
  iterations for the fixed pose. The frozen trajectory is 1,700 actions at
  50 Hz (34.00 s).
- The diagnostic episode shows a real two-finger contact grasp in both front
  and wrist frames. The cube center is 0.824051 m after reset and 0.846170 m
  at the start of transport, a retained 22.119 mm lift. It is visibly clamped
  through transport, then returns to table height after the open phase.
- Ten of ten fresh fixed-seed episodes succeeded. All ten traces are
  identical: release-settle position
  `[-4.142960, -4.025752, 0.824076]` m at 0.000430 m/s, and final public result
  `inside_target_xy=true`, `height_ok=true`, `stable=true`, `success=true`.
- Failure counts across ten episodes: grasp 0, lift 0, transport 0, placement
  0. The phase classification is supported by the first episode's camera
  evidence plus identical numeric boundary traces in all ten logs.
- `datasets/phase2_fixed_success` is a native public LeRobot 0.4.4 dataset:
  one episode, 1,700 rows, state/action width 33, object/target pose width 7,
  frame indices 0–1699, timestamps 0.00–33.98 s at 50 Hz, episode interval
  `[0,1700)`, and the requested task string. Front and right-wrist AV1 videos
  are each 640x480, 34.0 s at 50 Hz, and decode to exactly 1,700 frames.

### Blockers and uncertainty

- No Phase 2 blocker remains on this host.
- The public task reports large persistent left-ankle command/state residuals
  unrelated to the active right-arm grasp. Right-arm tracking lag is visible
  during motion, so the successful calibration should not be generalized
  outside the validated reset bounds without repeating the preflight gate.
- A direct post-release diagonal return disturbed the placed cube. The
  staging-enabled return removes that failure mode using an already solved
  waypoint, but it still moves the cube within the target during retreat;
  final placement remains stable and inside the public target tolerance.
- Isaac Sim emits repeated DDS warnings and a shared-memory cleanup warning;
  neither altered exit status, dataset finalization, or decoded video counts.

### Next implementation action

Phase 2 is complete. Add bounded exact XY/yaw reset variants, reject any failed
reset-time plan before rollout, collect one episode per accepted variant, and
validate each dataset and video (Phase 3).

## 2026-09-05 — Phase 3 bounded variants complete

### Files changed

- `scripts/run_unitree_mvp.py`: exact deterministic object reset offsets with
  public bounds X `[-0.10, +0.10]` m, Y `[-0.05, +0.05]` m, and yaw
  `[-pi, +pi]` rad; invalid values fail before Isaac Sim construction and any
  IK failure exits before rollout step zero.
- `tests/test_object_reset_variants.py`: bounds, finiteness, omitted-axis zero
  normalization, and fixed-reset conflict coverage.
- `PROGRESS.md`: this Phase 3 audit.

### Exact commands run

Both candidates were plan-gated in fresh processes before either rollout:

```bash
mkdir -p outputs/phase3_variants
unset CYCLONEDDS_HOME
for spec in 'a 0.01 0.0 0.20' 'b -0.01 0.005 -0.20'; do
  set -- $spec
  name=$1
  x=$2
  y=$3
  yaw=$4
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n unitree_sim_env python scripts/run_unitree_mvp.py \
    --unitree-root ../unitree_sim_isaaclab \
    --urdf ../unitree_ros/robots/g1_description/g1_29dof.urdf \
    --package-dir ../unitree_ros/robots/g1_description \
    --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint --device cuda:0 --enable_cameras --headless --plan-only \
    --object-reset-x-m "$x" --object-reset-y-m "$y" --object-reset-yaw-rad "$yaw" --reset-seed 0 --enable-staging \
    --target-position=-4.15,-4.03,0.84 --grasp-wrist-offset-world=0.025,0.165,0.0 \
    --grasp-quaternion-base-xyzw=0,0.17410813759359595,0,0.9847265389049334 \
    --gripper-open=-0.0175,-0.0175 --gripper-closed=0.0222,0.0222 \
    --approach-height-m 0.16 --lift-height-m 0.12 --target-approach-height-m 0.12 \
    --pregrasp-duration-s 3 --descend-duration-s 5 --gripper-duration-s 1 --settle-duration-s 1 \
    --lift-duration-s 3 --transport-duration-s 2 --return-duration-s 2 \
    --trajectory-out "outputs/phase3_variants/${name}_plan.npz" \
    > "outputs/phase3_variants/${name}_plan.log" 2>&1
  rg '^\[(reset|plan|error)' "outputs/phase3_variants/${name}_plan.log" | tail -n 6 | sed "s/^/[variant ${name}] /"
done
```

One native LeRobot/video episode was then collected for each accepted variant:

```bash
unset CYCLONEDDS_HOME
for spec in 'a 0.01 0.0 0.20' 'b -0.01 0.005 -0.20'; do
  set -- $spec
  name=$1
  x=$2
  y=$3
  yaw=$4
  task_text="Pick up the red block from bounded variant ${name} and place it in the green target area."
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n unitree_sim_env python scripts/run_unitree_mvp.py \
    --unitree-root ../unitree_sim_isaaclab \
    --urdf ../unitree_ros/robots/g1_description/g1_29dof.urdf \
    --package-dir ../unitree_ros/robots/g1_description \
    --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint --device cuda:0 --enable_cameras --headless \
    --object-reset-x-m "$x" --object-reset-y-m "$y" --object-reset-yaw-rad "$yaw" --reset-seed 0 --enable-staging \
    --target-position=-4.15,-4.03,0.84 --grasp-wrist-offset-world=0.025,0.165,0.0 \
    --grasp-quaternion-base-xyzw=0,0.17410813759359595,0,0.9847265389049334 \
    --gripper-open=-0.0175,-0.0175 --gripper-closed=0.0222,0.0222 \
    --approach-height-m 0.16 --lift-height-m 0.12 --target-approach-height-m 0.12 \
    --pregrasp-duration-s 3 --descend-duration-s 5 --gripper-duration-s 1 --settle-duration-s 1 \
    --lift-duration-s 3 --transport-duration-s 2 --return-duration-s 2 \
    --record-root "datasets/phase3_variant_${name}" --dataset-repo-id "local/g1-pickplace-phase3-variant-${name}" \
    --task-text "$task_text" --trajectory-out "outputs/phase3_variants/${name}_rollout.npz" \
    > "outputs/phase3_variants/${name}_rollout.log" 2>&1
  rg '^\[(plan|result|error)' "outputs/phase3_variants/${name}_rollout.log" | tail -n 4 | sed "s/^/[variant ${name}] /"
done
```

Dataset tables were validated with installed `pyarrow`; video decoding used:

```bash
for video in datasets/phase3_variant_a/videos/*/chunk-000/file-000.mp4 datasets/phase3_variant_b/videos/*/chunk-000/file-000.mp4; do
  ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration \
    -of json "$video"
done
```

### Observed result

- Accepted variant A: authored reset plus X `+0.010` m, Y `0.000` m,
  yaw `+0.20` rad. All six reset-time solves succeeded in plan-only mode;
  rollout final result was `inside_target_xy=true`, `height_ok=true`,
  `stable=true`, `success=true`, center error 0.072165 m, speed
  0.000552 m/s.
- Accepted variant B: authored reset plus X `-0.010` m, Y `+0.005` m,
  yaw `-0.20` rad. All six reset-time solves succeeded in plan-only mode;
  rollout final result was `inside_target_xy=true`, `height_ok=true`,
  `stable=true`, `success=true`, center error 0.061740 m, speed
  0.000711 m/s.
- The recorded initial XY values were `(-4.24000, -4.03002)` m for A and
  `(-4.26000, -4.02500)` m for B, matching the requested authored-pose
  offsets within simulator integration precision.
- Each variant has one native LeRobot episode with 1,700 rows, state/action
  width 33, object/target pose width 7, exact 50 Hz timestamp cadence within
  float32 tolerance, frame indices 0–1699, episode interval `[0,1700)`, and
  its exact variant task string. Each front and wrist AV1 video is 640x480,
  34.0 s at 50 Hz and decodes to 1,700 frames.
- The variant bounds are enforced before Isaac Sim starts. Any in-bounds pose
  whose reset-time staging or task waypoint IK fails raises and exits before
  trajectory playback; no rejected candidate can reach rollout step zero.

### Blockers and uncertainty

- No Phase 3 blocker remains for the two accepted bounded variants.
- Only the two explicitly listed offsets were physically validated. The full
  legal reset box is intentionally not claimed as successful; its extremes
  can be rejected by the reset-time IK gate.
- Cube yaw is not used after the single reset snapshot to alter phase timing or
  contact behavior. The expert remains a fixed action array during rollout.

### Next implementation action

All CODEX.md phases are complete. Keep the calibrated command and validated
variant set as the evidence-backed operating envelope; broaden it only by
adding new plan-only candidates first, then independently validating their
contact-based rollouts and native LeRobot artifacts.

### Final dependency-light verification

```bash
python3 -m pytest -q
PYTHONPATH=src python3 scripts/preview_plan.py --output outputs/preview_trajectory_final.npz
python3 -m compileall -q src scripts tests
git diff --check
rg -n "class OpenLoopPolicy|def act\\(|\\.solve\\(" src/g1pickplace scripts/run_unitree_mvp.py
git status --short
```

Observed: 34 tests passed; the independent preview saved a `(480, 4)` action
array covering 9.60 s; compilation and `git diff --check` were clean. The
source audit finds the only runtime policy action method in
`src/g1pickplace/trajectory.py`, where its observation argument is explicitly
discarded, and the only planner IK call in `src/g1pickplace/planner.py` inside
`build()` before the policy and rollout loop are constructed.

## 2026-09-05 — Post-phase minimal desk demonstration refinement

### Requested scene and object changes

- `--minimal-demo-preset` is the short, opt-in visible-demo configuration.
- The public warehouse room USD is removed while the public packing desk,
  robot, object, lights, and cameras remain.
- The red cube is 0.04 m per side. Its center is authored at world Z=0.814 m,
  based on the observed desk collision top at about Z=0.794 m.
- The green goal is a physical static 0.08 x 0.06 x 0.01 m rectangular support,
  centered at world `(-4.35, -4.03, 0.799)` m. The desired cube center on its
  top is world `(-4.35, -4.03, 0.824)` m.
- The table-mounted reset posture uses shoulder pitches 0, shoulder rolls
  `+0.15/-0.15` rad, and elbows `+0.10` rad. It was selected by a public-URDF
  collision-checked grid in the actual mounted base frame. The runner writes
  it as reset state (not as trajectory motion), settles, and fails before IK
  if any of those six live joints differs from default by more than 0.05 rad.
- The original validated warehouse preset cannot be combined with either
  minimal-scene option.

### Strict pre-rollout collision gates

- The live reset configuration is checked before waypoint construction.
- Deterministic multistart IK rejects colliding endpoints and sampled paths.
- Every compiled trajectory step is checked before `OpenLoopPolicy` is
  constructed. No observation or contact changes the frozen rollout.
- Coverage is explicitly limited to the exact-compatible public 29-DoF
  arm/body URDF. The public Dex1 URDF contains finger meshes, but its validated
  wrist frame differs by 0.005 m from the public USD/29-DoF pair, so it was not
  substituted silently. Dex1 finger clearance was checked in saved visible
  front/right-wrist phase-boundary images.
- No BrickSim, BrickBench, RoCoBrick, or related private implementation was
  opened, copied, or imitated. Public Unitree/Isaac assets were used.

### Dependency-light validation

```bash
python3 -m pytest -q
python3 -m compileall -q scripts src tests
git diff --check
```

Observed: 54 tests passed; compilation and whitespace checks were clean.

### Final visible plan-only gate (no `--headless`)

```bash
unset CYCLONEDDS_HOME
export LD_PRELOAD="$CONDA_PREFIX/lib/python3.11/site-packages/cmeel.prefix/lib/libassimp.so.5"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n unitree_sim_env \
  python scripts/run_unitree_mvp.py \
  --unitree-root ../unitree_sim_isaaclab \
  --urdf ../unitree_ros/robots/g1_description/g1_29dof.urdf \
  --package-dir ../unitree_ros/robots/g1_description \
  --minimal-demo-preset --device cuda:0 --enable_cameras --plan-only \
  --trajectory-out outputs/minimal_demo_visible_plan_mounted.npz \
  2>&1 | tee outputs/minimal_demo_visible_plan_mounted.log
```

Observed: the live wrist was at world Z=0.844506 m after reset; all six IK
waypoints succeeded (`staging`, `pregrasp`, `grasp`, `lift`, `preplace`, and
`place`); all 1,700 compiled steps passed the 610 retained public-URDF
collision-pair checks; rollout step zero was not started.

### Final visible physical rollout (no `--headless`)

The same command was run without `--plan-only`, with
`--phase-boundary-frame-root outputs/minimal_demo_mounted_frames` and
`--trajectory-out outputs/minimal_demo_visible_rollout_mounted.npz`.

Observed final result: `inside_target_xy=true`, `height_ok=true`,
`stable=true`, `success=true`, center error 0.021301 m. The cube settled at
world `(-4.329, -4.030, 0.824)` m on the physical support. Twenty-eight
phase-boundary PNGs cover 14 phases from both front and right-wrist cameras;
the reset view shows separated arms, and the grasp/lift/release views show the
0.04 m cube between the Dex1 fingers without a visible body intersection.

One rejected intermediate rollout used the standing whole-body seed from the
public config. It put the wrist below this mounted desk, produced large joint
tracking errors, missed the cube, and was superseded. Its failure is retained
in `outputs/minimal_demo_visible_rollout_corrected.log`; it is not an accepted
calibration.

### Public toy/food and multi-object follow-up

Isaac Sim's installed public examples reference the YCB asset collection,
including `/Isaac/Props/YCB/Axis_Aligned/011_banana.usd` plus food packages,
a bowl, and a mug. These are public asset-server references, not guaranteed
local files, and require scale/contact validation before replacing the cube.

The public Unitree registration includes
`Isaac-Stack-RgyBlock-G129-Dex1-Joint`. A strict open-loop multi-object demo
should use two or three small colored rigid objects, take exactly one reset
snapshot of all poses, solve every waypoint for every object up front, compile
one immutable sequential action array, and execute only after the entire array
passes limits/collision validation. This is a separate task/preset; the current
single-object expert does not claim multi-object support.

## 2026-09-06 — Approved entrance-test design implementation and validation

The implementation source of truth is
`spec/entrance_test_demo_design.md`. The current baseline restores the public
`Isaac-Stack-RgyBlock-G129-Dex1-Joint` warehouse/table/three-block scene and
keeps Dex3 and Inspire out of scope. The experimental minimal scene above is
retained as historical work only and is not evidence for this design.

### Implementation boundary

- One immutable aggregate reset snapshot records the three object poses,
  marker, robot base, ordered joint state/default state, and Task 4 assets.
- A finite instruction grammar selects red, green, or yellow and one supported
  relation. Unsupported or conflicting inputs fail before planning.
- The Dex1 hand profile, safe reset, public cameras, semantic labels, object
  material settings, and screenshot-like GUI view are applied before reset.
- All optional staging plus five task IK waypoints are solved at reset time.
  The resulting action array is checked against 610 public-URDF collision pairs
  and swept world/distractor clearance before `OpenLoopPolicy` is constructed.
- The post-review table rule permits only a contiguous retreat contact-exit
  prefix. The hand envelope must clear the table before the phase ends and may
  not re-enter afterward. This accommodates continuous upward separation while
  still rejecting a retreat that never clears or later intersects the table.
- `OpenLoopPolicy.act(observation)` explicitly discards `observation` and only
  advances the frozen action index. Evaluation and LeRobot recording are
  read-only consumers. No attachment, weld, teleport, kinematic object write,
  rollout-time IK, replanning, or feedback-driven phase transition was added.
- Only public Unitree, Isaac Lab, Pinocchio, and native LeRobot APIs/assets were
  used. No BrickSim, BrickBench, RoCoBrick, or related private code/assets were
  accessed, copied, imitated, translated, or ported.

### Task evidence

- Task 1 relative placement: PASS. The accepted red run is
  `outputs/design_task1_rollout_accept_recorded.log`; lift was 0.074059 m,
  transport 0.133814 m, edge clearance 0.047375 m for a 0.050000 m request,
  final speed 0.000267 m/s, and both distractors had zero displacement. Native
  LeRobot validation is under `outputs/design_task1_lerobot_accept/`.
- Task 1 approved visible Stack-scene variant (+0.010 m X, 0 Y, +0.15 rad yaw)
  passed. The (-0.010 m X, +0.010 m Y, -0.15 rad yaw) candidate passed Gate C
  but failed physical stability at 0.052611 m/s; an extended-settle attempt
  still measured 0.018343 m/s. It is rejected. Older PickPlace-scene Phase 3
  variants are not evidence for this restored Stack baseline.
- Task 2 type-conditioned selection: red, green, and yellow each passed visible
  plan-only. The accepted physical green/right run is
  `outputs/design_task2_green_recorded_rollout.log`: lift 0.058911 m,
  transport 0.197043 m, clearance 0.054717 m, final speed 0.000307 m/s, and
  red/yellow displacement zero. Native validation is under
  `outputs/design_task2_green_lerobot_accept/`.
- Task 3 red-on-yellow stack: PASS. The accepted physical run is
  `outputs/design_task3_stack_calibrated8_rollout.log`; lift was 0.071563 m,
  top XY error 0.003331 m, vertical error 0.000002 m, bottom displacement
  0.000156 m, top/bottom speeds 0.001141/0.000471 m/s, yellow remained upright,
  and green displacement was zero. The final native episode has 1650 rows at
  50 Hz and three 640x480 AV1 streams under
  `outputs/design_task3_stack_lerobot_accept2/`.
- Task 4 shovel (historical checkpoint, superseded below): NOT RUN / BLOCKED.
  Visible inspect confirms the original
  analytic shovel/tray entities in `outputs/design_shovel_inspect_visible.log`.
  `outputs/design_shovel_visible_plan_blocked.log` fails closed before IK and
  policy because there is no validated held-shovel transform or conservative
  swept handle/blade envelope. No rollout was attempted and no push is called a
  scoop or PARTIAL result.

### Final visible gates

Gate B used `--inspect-only` and the diagnostic-only `--viewport-frame` option.
It saved the actual active GUI viewport to
`outputs/design_task3_stack_gui_viewport.png`, three public sensor images to
`outputs/design_task3_stack_final_inspect_frames/`, and the log to
`outputs/design_task3_stack_final_inspect.log`. The expert and trajectory were
not constructed.

The final visible Gate C command was:

```bash
unset CYCLONEDDS_HOME
export LD_PRELOAD="$CONDA_PREFIX/lib/python3.11/site-packages/cmeel.prefix/lib/libassimp.so.5"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n unitree_sim_env \
  python scripts/run_unitree_mvp.py \
  --unitree-root ../unitree_sim_isaaclab \
  --urdf ../unitree_ros/robots/g1_description/g1_29dof.urdf \
  --package-dir ../unitree_ros/robots/g1_description \
  --instruction 'Pick up the red block and stack it on the yellow block.' \
  --device cuda:0 --enable_cameras --plan-only \
  --trajectory-out outputs/design_task3_stack_final_gate_c.npz \
  2>&1 | tee outputs/design_task3_stack_final_gate_c.log
```

Observed: all six waypoints solved (71/94/71/73/68/64 iterations), 1,650
trajectory samples, 2,800 distractor checks, 1,650 PhysX scene queries, zero
failures, and minimum airborne-object clearance 0.121780 m versus the 0.040 m
requirement. Rollout step zero was not started. Frozen action SHA-256:
`a82eee52269fb756850ab70e45bc780e52d9fee197cbb64a92befcff1efefdf2`,
identical to the accepted physical recording.

### Final dependency-light and source audit

```bash
PYTHONPATH=src python3 -m pytest -q -o addopts='' -rA
PYTHONPATH=src python3 scripts/preview_plan.py --output outputs/preview_trajectory_final_design.npz
PYTHONPATH=src python3 -m compileall -q src scripts tests
git diff --check
rg -n "class OpenLoopPolicy|def act\(|\.solve\(" src scripts tests
```

Observed at that checkpoint: 113 tests passed in 1.61 s; preview generated a `(480, 4)` frozen
action array; compileall and whitespace validation passed. The only production
planner IK call remains in `SemanticPickPlacePlanner.build()`, and the runtime
policy action method discards its observation. No stale simulator process
remained after the visible plan-only run.

The approved-design implementation and evidence collection occupied
approximately five wall-clock hours, based on artifacts from 03:58 through
08:57 local time. Earlier Phase 0 and historical calibration time was not
separately tracked.

### Simplified visible entry point

`scripts/run_demo.py` is now the single operator-facing entry. It requires an
explicit `--instruction`, supplies the validated public paths/runtime defaults,
and launches visible inspect followed by visible plan. Physical execution and
native LeRobot recording are opt-in together through `--rollout`.

The wrapper forwards the instruction unchanged to every subprocess, streams
each log to the terminal and artifact file, and creates a unique output folder
by default. It does not trust process status alone: structured reset geometry,
viewport, swept-clearance, solved-IK, plan-complete, rollout result, recording
status, and action-hash fields are parsed. A missing/malformed/failed semantic
marker returns wrapper status 86 and prevents every later gate.

A real visible Task 1 smoke test without `--rollout` is recorded under
`outputs/single_entry_task1_smoke2/`. Inspect passed with an actual GUI capture;
plan passed all six reset-time IK solves and swept checks, saved 1,650 actions,
and exited before rollout step zero. No physical action was executed by this
wrapper smoke test.

Final dependency-light validation after adding the wrapper and semantic gate
tests: 124 tests passed in 1.30 s; compileall and `git diff --check` passed. No
simulator process remained.

## 2026-09-06 — Task 4 reachable hand-side shovel follow-up

The earlier `NOT RUN / BLOCKED` Task 4 entry above is historical and is
superseded by this section. The repository now has a reset-time held-tool
transform, exact public-URDF/HPP-FCL swept checks, named public Isaac contact
sensors, and read-only pose/contact evaluation. The shovel remains at the
hand-side reset `(-3.90, -3.97, 0.84)` m; moving it to X=-4.00 m was rejected
in visible inspect because it initially overlapped the left hand.

- Reachability is verified, but physical task success is not. The current
  45 mm-high rectangular handle candidate passed visible Gate 21: all 22 IK
  waypoints solved, all 345,138 swept checks passed with zero failures, and
  the 2,501-step plan with SHA-256
  `87b8d7c777dcf53e45cb4ed5b790e27a2ac7bb6251d5d519e4ac60d6838fb322`
  stopped before rollout step zero. Evidence is under
  `outputs/task4_handle45_gate_21/`.
- The rejected 50 mm handle candidate failed closed before rollout at Gate 20
  with 66 exact handle/left-elbow overlaps during `tilt_blade_up`; no collision
  exemption was added.
- The completed earlier rollout-02 produced a valid native LeRobot episode:
  2,515 frames at 50 Hz, three 640x480 RGB streams, and matching action hash
  `3b73273a7084e87e6e13efdb1dceba66998084b48df3babf0fb0e79c90166c97`.
  Its semantic result was FAIL: both named finger contacts occurred and table
  contact was lost, but tool lift was only 0.0155148506 m versus the 0.05 m
  requirement, with no causal blade/red contact or tray placement.
- The current 45 mm rollout-05 retry was intentionally stopped after visible
  phase frames showed the shovel dropped/remained on the left table before the
  hand moved behind the red block. It is physical PARTIAL evidence only and is
  not a completed recording or evaluator PASS.

Task 4 is therefore **PARTIAL**, not PASS. Tasks 1-3 retain their accepted
physical evidence. The strict open-loop boundary remained intact throughout:
all IK and collision work occurred before rollout step zero; contacts, poses,
images, rewards, and observations never changed an action or phase; no
attachment, teleport, kinematic object write, or private project code/assets
were used. Final dependency-light validation at this checkpoint is 154 tests
passed; compileall and `git diff --check` passed, and no simulator process
remained.

## 2026-09-06 — Task 4 left-gripper drive-only experiment

The shovel handle already uses deliberately high static/dynamic friction
(`20.0/15.0`), so this experiment changed no material, geometry, trajectory,
effort limit, velocity limit, joint friction, or armature. Only the resolved
Task 4 configuration doubles the two left Dex1 finger drives from the public
`800.0 N*m/rad` stiffness and `3.0 N*m*s/rad` damping to `1600.0` and `6.0`.
The two right-hand joints remain at `800.0/3.0`; Tasks 1-3 never enter this
configuration branch. A new fail-closed runtime gate reads the live Isaac Lab
joint stiffness/damping tensors rather than trusting the configuration object.

- The visible inspect in `outputs/task4_drive2x_gate_22/` reports live left
  gains `1600.0/6.0`, live right gains `800.0/3.0`, and status `PASS`.
- The following visible plan solved all 22 reset-time IK waypoints and passed
  all 345,138 swept checks with zero failures. It froze 2,501 actions with
  SHA-256 `644d3ed63409da174a94bbeb14f8fda6528be6f0e79a6cce3d3c8cba74b9b253`
  and explicitly exited before rollout step zero.
- A separately gated visible physical attempt is under
  `outputs/task4_drive2x_rollout_06/`. Its inspect and plan stages passed before
  execution. At the `lift_tool` boundary the handle was still adjacent to the
  left gripper, but by `move_above_behind_red` the handle was visibly resting
  on the left side of the table while the hand continued toward the red block;
  `move_behind_red` confirmed the separation. The attempt was intentionally
  stopped, so it is neither a completed recording nor a Task 4 semantic PASS.

This isolated 2x drive increase therefore did **not** fix the held-tool
failure. Task 4 remains **PARTIAL**. The result points to grasp geometry/contact
placement rather than insufficient configured drive gain alone; raising the
gain further without changing that geometry is not supported by this evidence
and risks larger contact impulses or solver instability. The strict open-loop
boundary remained unchanged, all simulator runs were visible, and no simulator
process remained afterward. Final dependency-light validation is 168 tests
passed; compileall and `git diff --check` passed.
