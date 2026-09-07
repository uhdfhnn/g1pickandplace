# Run the entrance-test demo

## Recommended single entry

Run from the repository root. The instruction is mandatory; there is no
implicit task prompt. This command opens visible inspect-only and plan-only
processes in order, saves the GUI/sensor evidence and frozen plan, and stops
before physical rollout:

~~~bash
cd g1pickandplace

python3 scripts/run_demo.py \
  --instruction "Pick up the red block and place it left of the yellow square marker."
~~~

To request physical execution and native LeRobot recording, add `--rollout`:

~~~bash
python3 scripts/run_demo.py \
  --instruction "Pick up the green block and place it right of the yellow square marker." \
  --rollout
~~~

The wrapper creates and prints a unique directory under `outputs/`. It never
passes `--headless`; it configures the validated public paths, Assimp preload,
DDS environment, camera evidence, seed, staging, and recording internally. A
failed inspect or plan subprocess prevents every later gate, so rollout cannot
start unless all reset-time IK and preflight checks return success. For the
shovel instruction, `--rollout` is experimental: a physical PARTIAL/FAIL or an
intentional stop is not a Task 4 PASS, and the current evidence has no
completed valid shovel recording.

Supported instructions are:

~~~text
Pick up the red block and place it left of the yellow square marker.
Pick up the green block and place it right of the yellow square marker.
Pick up the yellow block and place it in front of the yellow square marker.
Pick up the red block and stack it on the yellow block.
Pick up the shovel, scoop the red block, and place it in the target tray.
~~~

The shovel instruction is inspectable and plan-gated like the other tasks. Its
current 45 mm candidate passes Gate21 preflight, but physical rollout remains
experimental until the evaluator confirms both-finger contact, the required
tool lift, causal blade/block contact, and tray success in a complete recording.

## Expanded gate commands

This runbook uses the approved public
Isaac-Stack-RgyBlock-G129-Dex1-Joint scene and always opens a visible Isaac
Sim GUI. Run from the repository root, close any stale simulator window before
starting a new gate, and never start a physical run before its plan-only gate
reports every waypoint and preflight check as successful.

## 1. Prepare the environment

The paths below match this checkout. Replace only the three repository paths if
your sibling checkouts differ.

~~~bash
cd g1pickandplace

export UNITREE_ROOT=../unitree_sim_isaaclab
export G1_URDF=../unitree_ros/robots/g1_description/g1_29dof.urdf
export G1_PACKAGE=../unitree_ros/robots/g1_description
export ASSIMP_LIB="$CONDA_PREFIX/lib/python3.11/site-packages/cmeel.prefix/lib/libassimp.so.5"

unset CYCLONEDDS_HOME
test -f "$ASSIMP_LIB"
export LD_PRELOAD="$ASSIMP_LIB"
mkdir -p outputs/entrance_task3
~~~

The Assimp preload avoids the Pinocchio/HPP-FCL symbol collision observed on
this host. Leaving CYCLONEDDS_HOME unset avoids the Unitree DDS initializer
exiting silently. Do not add private projects or assets to the environment.

The entrance-test demo resolves the calibrated hand profile, reset offsets,
heights, gripper values, and timing defaults from the task. Do not copy the
long legacy calibration command unless you are intentionally reproducing an
older evidence artifact. Explicit legacy flags must exactly agree with the
selected profile or the runner fails closed.

## 2. Gate B: visible inspect-only

Inspect the public scene without constructing an expert or trajectory:

~~~bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n unitree_sim_env \
  python scripts/run_unitree_mvp.py \
  --unitree-root "$UNITREE_ROOT" \
  --instruction "Pick up the red block and stack it on the yellow block." \
  --device cuda:0 \
  --enable_cameras \
  --inspect-only \
  --phase-boundary-frame-root outputs/entrance_task3/inspect_frames \
  --viewport-frame outputs/entrance_task3/gui_viewport.png \
  2>&1 | tee outputs/entrance_task3/inspect.log
~~~

Confirm in the log that the public warehouse/table, G1, Dex1, red/yellow/green
blocks, marker, and front/left-wrist/right-wrist cameras are present; semantic
labels are available; reset geometry has no overlaps and consistent support;
the viewport is configured; and the expert/trajectory were not constructed.
Inspect the sensor PNGs and the actual GUI viewport PNG before continuing.

## 3. Gate C: visible plan-only

Use the same public task and instruction. This compiles the entire open-loop
program and exits before action zero:

~~~bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n unitree_sim_env \
  python scripts/run_unitree_mvp.py \
  --unitree-root "$UNITREE_ROOT" \
  --urdf "$G1_URDF" \
  --package-dir "$G1_PACKAGE" \
  --instruction "Pick up the red block and stack it on the yellow block." \
  --device cuda:0 \
  --enable_cameras \
  --plan-only \
  --trajectory-out outputs/entrance_task3/plan.npz \
  2>&1 | tee outputs/entrance_task3/plan.log
~~~

Continue only if the log contains all six reset-time solves
(staging/pregrasp/grasp/lift/preplace/place), zero IK/limit/collision/clearance
failures, and the explicit message that rollout step zero was not started.
Keep the NPZ and its printed action hash with the corresponding rollout.

## 4. Gate D/E: visible rollout and native recording

Run this only after Gate C passes. The action sequence is still the one
compiled before policy construction; recording and evaluation cannot alter it.

~~~bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n unitree_sim_env \
  python scripts/run_unitree_mvp.py \
  --unitree-root "$UNITREE_ROOT" \
  --urdf "$G1_URDF" \
  --package-dir "$G1_PACKAGE" \
  --instruction "Pick up the red block and stack it on the yellow block." \
  --device cuda:0 \
  --enable_cameras \
  --phase-boundary-frame-root outputs/entrance_task3/rollout_frames \
  --record-root outputs/entrance_task3/lerobot \
  --dataset-repo-id local/g1-entrance-task3-stack \
  --trajectory-out outputs/entrance_task3/rollout.npz \
  2>&1 | tee outputs/entrance_task3/rollout.log
~~~

Check the result, phase-boundary images, trajectory hash, native dataset
validation, timestamp cadence, and decoded front/left-wrist/right-wrist
videos. A valid recording must use the public LeRobotDataset lifecycle and
have matching recorded and trajectory action hashes.

## 5. Run Tasks 1 and 2 with the same gates

Keep the wrapper and paths above, and replace the instruction with one of these
finite commands. The instruction itself resolves the demo mode. Run inspect,
then plan-only, then a physical run only for an accepted plan.

~~~text
--instruction "Pick up the red block and place it left of the yellow square marker."

--instruction "Pick up the green block and place it right of the yellow square marker."

--instruction "Pick up the yellow block and place it in front of the yellow square marker."
~~~

All three blocks remain in the public scene. The instruction selects exactly
one object from the immutable reset snapshot. Unsupported or conflicting text
and structured fields are rejected before planning.

## 6. Shovel status

The shovel instruction may be inspected to verify the visible semantic assets
and the repository-owned compound tool/tray geometry:

~~~bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n unitree_sim_env \
  python scripts/run_unitree_mvp.py \
  --unitree-root "$UNITREE_ROOT" \
  --instruction "Pick up the shovel, scoop the red block, and place it in the target tray." \
  --device cuda:0 \
  --enable_cameras \
  --inspect-only \
  --phase-boundary-frame-root outputs/entrance_task4/inspect_frames \
  --viewport-frame outputs/entrance_task4/gui_viewport.png \
  2>&1 | tee outputs/entrance_task4/inspect.log
~~~

Run the visible plan-only gate after inspect, using the single wrapper when
possible:

~~~bash
python3 scripts/run_demo.py \
  --instruction "Pick up the shovel, scoop the red block, and place it in the target tray."
~~~

For the current 45 mm handle candidate, [Gate21](../outputs/task4_handle45_gate_21/plan.log)
passed all 22 reset-time IK waypoints and 345,138 swept checks (zero failures),
compiled 2,501 steps, and exited before rollout step zero. [Gate20](../outputs/task4_tall_handle_gate_20/plan.log)
is retained as a correctly rejected 50 mm candidate: it had 66 exact
handle/left-elbow overlaps during `tilt_blade_up`.

Only after a fresh visible inspect and a passing visible plan may an operator
request physical execution:

~~~bash
python3 scripts/run_demo.py \
  --instruction "Pick up the shovel, scoop the red block, and place it in the target tray." \
  --rollout
~~~

This shovel rollout is experimental and must be treated as non-PASS unless a
complete recording reports the required lift, causal blade-red contact, block
transport, tray placement, and stable final state. The earlier complete
[rollout-02](../outputs/task4_graspcenter_rollout_02/rollout.log) was a valid
native LeRobot recording but semantically FAIL (0.0155148506 m tool lift versus
the required 0.05 m). The current [rollout-05 retry](../outputs/task4_handle45_rollout_05_retry1/rollout.log)
was intentionally stopped after phase frames showed the shovel dropped/remained
on the left table before the hand moved behind red; it is physical PARTIAL
visual evidence only and not a completed valid recording. Never report a
push/drop-only trace as a scoop, and never bypass reset-time IK or swept
preflight gates.

## 7. Troubleshooting the observed warnings

- The Warp cuDeviceGetUuid/CUDA API error 36 warning was observed on the RTX
  5090 with driver 580.167.08; successful runs continued, but a simulator
  abort is an environment failure and must not be ignored.
- Deprecated dynamic-control/Semantics messages, duplicate plugin registration,
  disabled high-frequency spans, and missing decorative materials are extension
  diagnostics.
- DDS object-not-found initialization messages, unsupported metadata notices,
  and shared-memory cleanup messages were observed at shutdown. Judge a run by
  exit status, gate records, action hashes, frame counts, and decoded videos.
- A missing Assimp preload can produce the fatal HPP-FCL symbol error described
  above. A set CYCLONEDDS_HOME can break the public DDS startup.

For the full evidence, thresholds, versions, and artifact paths, see
[ENTRANCE_TEST_REPORT.md](ENTRANCE_TEST_REPORT.md). For the implementation
source of truth, see [entrance_test_demo_design.md](../spec/entrance_test_demo_design.md).
