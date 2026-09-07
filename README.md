# Unitree G1 pick-and-place entrance demo

This repository implements a reset-time, open-loop IK baseline for the public
Unitree G1 29-DoF Dex1 environment. The approved scene is
Isaac-Stack-RgyBlock-G129-Dex1-Joint: public warehouse, packing table, yellow
marker, red/yellow/green blocks, G1, Dex1, and public RGB cameras.

## Current acceptance status

| Task | Status |
| --- | --- |
| Task 1: red block left of yellow marker | PASS; visible rollout and native LeRobot episode |
| Task 2: instruction-conditioned selection | PASS for validated green/right instruction; red/yellow plan-gated |
| Task 3: red block stacked on yellow | PASS; calibrated8 visible gates, physical result, and Gate E episode |
| Task 4: shovel scoop | PARTIAL physical evidence only; no successful scoop or completed valid recording |

The detailed evidence, hashes, metrics, versions, and risks are in
[docs/ENTRANCE_TEST_REPORT.md](docs/ENTRANCE_TEST_REPORT.md). The copyable
visible-GUI commands are in
[docs/RUN_ENTRANCE_TEST_DEMO.md](docs/RUN_ENTRANCE_TEST_DEMO.md).

## Setup from Git

The reproducible setup starts with a normal clone and installs the validated
public dependency checkouts as siblings:

~~~bash
git clone https://github.com/uhdfhnn/g1pickandplace.git
cd g1pickandplace
bash scripts/setup_environment.sh
~~~

The bootstrap pins the Unitree, Isaac Lab, ROS-description, SDK, CycloneDDS,
Pinocchio, and LeRobot versions used by the accepted runs. Requirements,
hardware notes, Assimp handling, validation, and manual recovery are documented
in [SETUP.md](SETUP.md).

## Safety and architecture

The runner snapshots all three block poses once after reset, parses a finite
instruction, selects exactly one object, solves every IK waypoint, compiles an
immutable joint/action array, and performs limits/collision/clearance checks
before constructing OpenLoopPolicy. Its act method is observation-invariant:
observations are recorded/evaluated only and never change the action sequence.

There is no rollout-time IK, replanning, contact/error/reward transition,
recovery, attachment, weld, teleport, or kinematic object write. Every
physical run must follow visible inspect-only -> visible plan-only -> visible
rollout, and must stop if any reset-time solve or preflight gate fails.

This project uses public Unitree, Isaac Lab, Pinocchio, and LeRobot APIs/assets.
It does not access, copy, imitate, translate, or port BrickSim, BrickBench,
RoCoBrick, or related private code/assets. Dex3 and Inspire are out of scope.

## Dependency-light checks

From the repository root:

~~~bash
python3 -m pytest -q
python3 -m compileall -q src scripts tests
git diff --check
~~~

Current shared-tree validation: 168 tests passed; compilation and whitespace
checks passed.

## Visible quick start

The single entry point requires an instruction. By default it opens visible
inspect-only and plan-only gates in order and stops before rollout:

~~~bash
cd g1pickandplace
python3 scripts/run_demo.py \
  --instruction "Pick up the red block and stack it on the yellow block."
~~~

Add `--rollout` to the same command to request physical execution and native
LeRobot recording. The wrapper still runs visible inspect and plan first and
fails closed without starting rollout if either gate fails. It creates a unique
output directory and prints its path. For the shovel instruction, `--rollout`
is experimental: the current 45 mm candidate passes reset-time Gate21, but
physical evidence is only PARTIAL and must not be treated as a Task 4 PASS
until a complete evaluator-valid recording exists.

## Cosmos first-frame policy inference

The inference-only entry point starts the same visible public Unitree scene,
captures the settled first RGB frame from the front, left-wrist, and
right-wrist cameras, builds the Cosmos `concat_view`, and calls the asynchronous
Cosmos policy endpoint. Open the documented SSH tunnel first:

~~~bash
ssh -L 8080:localhost:8080 <user>@<cosmos-host>
curl http://localhost:8080/v1/models
~~~

Then run the simulation and one inference in another terminal:

~~~bash
cd g1pickandplace
python3 scripts/run_cosmos_inference.py \
  --base-url http://localhost:8080 \
  --duration-s 16 \
  --instruction "Pick up the red block and stack it on the yellow block." \
  --output-dir outputs/cosmos_stack_red_on_yellow_16s
~~~

The output directory contains `unitree_concat_view.png`,
`cosmos_policy_action.json`, `metadata.json`, `sim_inference_context.npz`, and
the simulator log. This path is deliberately dry-run only: Cosmos returns a
normalized AgiBotWorld end-effector action, not the G1 environment's ordered
joint command. The wrapper defaults to 16 seconds, producing 160 actions at the
model's 10 Hz rate (`num_frames=161`). The result is validated and saved
together with the initial G1 state, but never passed to `env.step`.

## Cosmos action replay and video

Replay a saved inference artifact in the same visible Unitree scene and record
the front/left-wrist/right-wrist concat view:

~~~bash
cd g1pickandplace
python3 scripts/replay_cosmos_action.py \
  --action outputs/cosmos_stack_red_on_yellow_16s \
  --instruction "Pick up the red block and stack it on the yellow block." \
  --output-dir outputs/cosmos_stack_red_on_yellow_16s_replay
~~~

`--action` accepts an inference directory containing
`sim_inference_context.npz`, a compatible NPZ, or the saved action JSON. The
29-D normalized AgiBotWorld action is quantile-denormalized and interpreted as
relative wrist poses plus gripper-open fractions (`0=closed`, `1=open`). Only
the hand selected by the resolved task profile is mapped to G1. Every 10 Hz
wrist pose is solved through the public G1 URDF before rollout; the resulting
joint targets are interpolated to 50 Hz and checked for joint limits, maximum
step size, IK residual, and URDF self-collision. Any failed preflight stops
before action zero.

The output directory contains:

- `cosmos_replay.mp4`: 640x720 three-camera video at 10 fps;
- `cosmos_replay_trajectory.npz`: frozen 50 Hz G1 joint trajectory;
- `replay.log`: preflight, playback, and final physical evaluation evidence.

For a 160-row action artifact, replay produces 800 simulator commands and a
160-frame, 16-second video. A completed replay does not imply task success: the
final simulator evaluator remains authoritative and may report `success=false`
when the predicted motion does not grasp or place the object.

## Repository map

~~~text
src/g1pickplace/geometry.py       rigid transforms and quaternion utilities
src/g1pickplace/offline_ik.py     reset-time Pinocchio frame IK
src/g1pickplace/planner.py        semantic pick/place compiler
src/g1pickplace/trajectory.py     immutable trajectory and open-loop player
src/g1pickplace/evaluation.py     read-only metrics
src/g1pickplace/lerobot_writer.py native LeRobot episode writer
scripts/run_unitree_mvp.py        public Unitree integration and visible gates
scripts/run_demo.py               single visible inspect/plan/rollout entry
scripts/run_cosmos_inference.py   visible first-frame Cosmos dry-run entry
scripts/replay_cosmos_action.py   validated G1 replay and MP4 wrapper
scripts/preview_plan.py           simulator-free planner smoke test
src/g1pickplace/cosmos_inference.py concat preprocessing and async Cosmos client
tests/                            dependency-light correctness tests
spec/entrance_test_demo_design.md approved design source of truth
~~~

Historical calibration logs remain under outputs/ and are labeled in the final
report; they should not be confused with the current approved baseline.
