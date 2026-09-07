# Unitree G1 entrance-test demo report

Date: 2026-09-06. This is the current evidence summary for the approved
design in [spec/entrance_test_demo_design.md](../spec/entrance_test_demo_design.md).
All simulator evidence below used the public Unitree Stack-RgyBlock Dex1 task
with a visible Isaac Sim GUI.

## Executive status

| Acceptance | Status | Evidence |
| --- | --- | --- |
| Task 1: relative placement | PASS | Red-only physical placement, relation/clearance/support checks, native LeRobot episode. |
| Task 2: type-conditioned selection | PASS for validated green instruction; red/yellow plan-gated | Green selected and moved right of marker; both distractors remained stationary. |
| Task 3: red-on-yellow stack | PASS | Calibrated8 inspect, Gate C, physical stack, recorded episode, and Gate E. |
| Task 4: shovel tool use | PARTIAL physical evidence only; no successful scoop | Gate21 reset-time preflight passed; rollout-02 failed the lift/transport evaluator and rollout-05 was intentionally stopped after visual evidence of a dropped tool. |

Task 4 is not an evaluator PASS. The current 45 mm candidate has physical
PARTIAL evidence only, and no completed valid Task 4 recording. A push-only or
drop-only result would not meet the approved scoop acceptance.

## Approved design and open-loop boundary

The final baseline is the public warehouse, packing table, yellow marker, three
dynamic blocks named red_block, yellow_block, and green_block, Unitree G1 with
Dex1, and public front/left-wrist/right-wrist cameras. The experimental
minimal white-background scene is historical only. Dex3 and Inspire are out of
scope.

~~~text
public reset -> fixed settle -> one aggregate snapshot -> parse instruction
-> resolve object/reference/relation -> solve every IK waypoint
-> compile immutable absolute targets and environment actions
-> limits/collision/clearance preflight -> save plan
-> visible plan-only exit OR construct OpenLoopPolicy
-> replay frozen actions -> read-only evaluation and recording
~~~

OpenLoopPolicy.act discards its observation and advances only the frozen action
index. No rollout-time IK, replanning, contact/error/reward transition,
recovery, attachment, weld, teleport, or kinematic object write is allowed.
All IK and preflight checks finish before rollout step zero. The finite
instruction parser rejects unsupported or conflicting text/CLI fields.

Canonical instructions are:

~~~text
Pick up the red block and place it left of the yellow square marker.
Pick up the green block and place it right of the yellow square marker.
Pick up the yellow block and place it in front of the yellow square marker.
Pick up the red block and stack it on the yellow block.
Pick up the shovel, scoop the red block, and place it in the target tray.
~~~

The frozen presentation mapping is world +X = visual-left, -X = visual-right,
-Y = in front, +Y = behind, and +Z = up. Canonical relation clearance is
0.05 m edge-to-edge. The calibrated stack reset offsets are red
(+0.025, 0, 0), yellow (+0.05, +0.05, 0), and green (-0.10, 0, 0), in world
metres with zero yaw. Viewport eye/look-at are
(-4.70,-5.20,2.05) / (-4.20,-4.08,0.95) m; sensor extrinsics are unchanged.

## Phase 0 record

The full host record is in [PROGRESS.md](../PROGRESS.md). The key values are:

~~~text
unitree_sim_isaaclab: e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc
Isaac Sim: 5.0.0-rc.45+release.23960.184afb15.gl
Isaac Lab: v2.2.0, commit 46dff135f44683f031edf346e544fcfd8456b2bb
Isaac Lab packages: isaaclab 0.44.9, isaaclab_tasks 0.10.45
Pinocchio: 2.7.0
LeRobot: not installed at Phase 0; later native validation used 0.4.4
GPU: NVIDIA GeForce RTX 5090, 32607 MiB
NVIDIA driver: 580.167.08; maximum supported CUDA: 13.0
~~~

The direct-joint action dimension is 33: 29 body joints and four Dex1 finger
joints. Runtime action order:

~~~text
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
~~~

The public fixed-base g1_29dof.urdf has the same 29 ordered body joints after
filtering simulator-only fingers and includes right_wrist_yaw_link, as does
the USD. Recorded neutral pelvis-to-wrist differences were 1.05e-8 m and
2.0815e-4 rad, consistent with USD rounding. Phase 0 also recorded the public
G1 task registry, including Stack-RgyBlock Dex1/Dex3/Inspire and the related
public pick/place IDs.

## Task 1: relative placement

Accepted artifacts:

- [rollout log](../outputs/design_task1_rollout_accept_recorded.log)
- [trajectory](../outputs/design_task1_rollout_accept_recorded.npz)
- [LeRobot validation](../outputs/design_task1_lerobot_accept/meta/g1pickandplace_validation.json)

The selected red block lifted 0.074059 m and transported 0.133814 m. Measured
edge clearance was 0.047375 m for requested 0.050000 m (error 0.002625 m);
signed relation separation was 0.123672 m. Yellow and green displacement were
0; final speed was 0.000267 m/s. Height, stability, support, relation, and
clearance passed. The native episode has 1550 state/action rows at 50 Hz,
three 480x640 RGB video streams, and matching action hashes.

The approved visible Stack-scene positive variant (x=+0.010 m, y=0,
yaw=+0.15 rad) passed Gate C and physical evaluation, ending at 0.000254 m/s.
The negative variant (x=-0.010 m, y=+0.010 m, yaw=-0.15 rad) passed Gate C but
failed physical stability: final speed was 0.052611 m/s, and extending the
settle phase still left 0.018343 m/s. It is rejected, not an accepted variant.
Older Phase 3 variants under the superseded PickPlace scene remain historical
artifacts and are not evidence for this approved Stack-scene design. The legal
reset box is therefore not claimed to be uniformly reachable or stable.

## Task 2: type-conditioned selection

The scene always contains all three blocks; the aggregate snapshot is immutable
and the instruction selects exactly one. Red and yellow were plan-gated in
[red Gate C](../outputs/design_task2_red_current_gate_c.log) and
[yellow Gate C](../outputs/design_task2_yellow_calibrated2_gate_c.log). The
green physical/recorded artifact is
[design_task2_green_recorded_rollout.log](../outputs/design_task2_green_recorded_rollout.log).

The validated green run selected only green, moved 0.169501 m, lifted 0.058911 m,
transported 0.197043 m, measured 0.054717 m edge clearance, left red/yellow at
0 displacement, and ended at 0.000307 m/s. Relation, support, lift, transport,
stability, and distractor checks pass. The retained diagnostic reports
center_error_m=0.011851 and inside_target_xy=false; that extra center box is
not an approved Task 2 gate. The corrected evaluator composes the explicit
relation/clearance and movement checks and reports overall success. Native
dataset validation is in
[design_task2_green_lerobot_accept](../outputs/design_task2_green_lerobot_accept/meta/g1pickandplace_validation.json).

## Task 3: two-object stack

Gate B inspect is recorded in the final
[inspect log](../outputs/design_task3_stack_final_inspect.log), with the actual
[GUI viewport capture](../outputs/design_task3_stack_gui_viewport.png) and
[sensor frames](../outputs/design_task3_stack_final_inspect_frames/).
Reset geometry passed with no overlaps, support-plane consistency, semantic
labels, required public scene entities, and configured viewport.

The final post-integration Gate C rerun is recorded in
[final plan log](../outputs/design_task3_stack_final_gate_c.log) and
[final plan](../outputs/design_task3_stack_final_gate_c.npz).
It reports zero swept-clearance failures, 1650 samples at 50 Hz, and minimum
airborne-object clearance 0.121780 m. Reset-time iteration counts were staging
71, pregrasp 94, grasp 71, lift 73, preplace 68, and place 64. The frozen
action SHA-256 is
a82eee52269fb756850ab70e45bc780e52d9fee197cbb64a92befcff1efefdf2; rollout
step zero was not started.

The calibrated8 physical result in
[calibrated8 rollout](../outputs/design_task3_stack_calibrated8_rollout.log)
selected red and reports physical lift 0.071563 m, vertical separation
0.049998 m, top XY error 0.003331 m, vertical error 0.000002 m, yellow
up-alignment 0.999999/upright, bottom displacement 0.000156 m, bottom speed
0.000471 m/s, top speed 0.001141 m/s, and green displacement 0. All stack
acceptance fields and overall success are true.

The final recorded artifacts are
[recorded rollout](../outputs/design_task3_stack_recorded_accept_rollout.log),
[recorded trajectory](../outputs/design_task3_stack_recorded_accept_rollout.npz),
[phase frames](../outputs/design_task3_stack_recorded_accept_frames/), and
[Gate E validation](../outputs/design_task3_stack_lerobot_accept2/meta/g1pickandplace_validation.json).
Gate E is PASS: native LeRobotDataset, one 1650-frame episode, float32
state/action shape 1650 x 33, 50 Hz, three 480x640 RGB videos, maximum
timestamp error 1.831e-6 s, and recorded/trajectory hashes equal the Gate C
hash.

A failed physical variant remains honestly recorded in
[calibrated7 rollout](../outputs/design_task3_stack_calibrated7_rollout.log):
red metrics were acceptable and yellow upright, but yellow displacement
0.019106 m and top speed 0.019599 m/s exceeded 0.01. The reset-to-preclose
trace moved yellow from approximately (-4.160002,-4.000052) to
(-4.177031,-4.010790) m. It was not called a pass; calibrated8 was then
inspected, plan-gated, and validated.

## Task 4: shovel

[Shovel inspect](../outputs/task4_handle45_gate_21/inspect.log) confirms the
public Stack Dex1 scene plus semantic `compound_shovel_tool` and `target_tray`
entities. The repository-owned shovel path now has a reset-time tool-frame
transform, explicit named finger/tool contacts, and conservative swept
handle/blade checks before `OpenLoopPolicy` construction.

[Gate19](../outputs/task4_inward_flat_grip_gate_19/plan.log) used the 35 mm handle,
-30 degree grasp orientation, and tool-local Y=0.060 m calibration. It solved
all 22 reset-time IK waypoints, checked 345,138 swept samples with zero
failures, compiled 2,501 action steps, and did not start rollout step zero.
[Gate20](../outputs/task4_tall_handle_gate_20/plan.log) rejected the 50 mm
handle candidate before rollout because 66 exact handle/left-elbow overlaps
occurred during `tilt_blade_up`. The current [Gate21 plan]
(../outputs/task4_handle45_gate_21/plan.log) is the bounded 45 mm candidate:
all 22 waypoints solved, all 345,138 swept checks passed, the trajectory has
2,501 steps, and its frozen action SHA-256 is
`87b8d7c777dcf53e45cb4ed5b790e27a2ac7bb6251d5d519e4ac60d6838fb322`.

Gate21 is preflight evidence, not physical success. The earlier complete
[rollout-02](../outputs/task4_graspcenter_rollout_02/rollout.log) at the prior
X=-0.025 m calibration generated a valid native LeRobot recording with 2,515
frames at 50 Hz, three RGB streams, and matching action hash
`3b73273a7084e87e6e13efdb1dceba66998084b48df3babf0fb0e79c90166c97`.
Both named finger contacts were true and table contact was lost, but tool lift
was only 0.0155148506 m, below the required 0.05 m; there was no causal
blade-red contact or tray success, so its semantic result was FAIL. The current
[rollout-05 retry](../outputs/task4_handle45_rollout_05_retry1/rollout.log) was
intentionally stopped after phase frames showed the shovel had dropped/remained
on the left table before the hand moved behind red. It is only physical PARTIAL
visual evidence (brief pickup/drop), not an evaluator PASS, and it did not
produce a completed valid Task 4 recording.

The strict open-loop boundary remained intact: every IK and preflight check was
completed before rollout step zero, and runtime contacts/poses were diagnostic
only. No attachment, teleport, kinematic manipulation, feedback transition, or
private BrickSim/BrickBench/RoCoBrick code or asset was used. The shovel
`--rollout` command remains experimental and must be treated as non-PASS until
a complete evaluator-valid result is captured.

## Validation, warnings, and provenance

Current dependency-light validation:

~~~text
pytest: 168 passed
compileall: passed
git diff --check: passed
~~~

The final visible inspect and Gate C reruns passed after integration. The legal
reset bounds do not guarantee IK reachability; new variants must pass visible
inspect and Gate C before rollout.

Observed warnings include Warp CUDA error 36 for missing cuDeviceGetUuid on
this RTX 5090/driver pair, deprecated Isaac extensions, duplicate plugin
registration, disabled high-frequency spans, missing decorative materials,
DDS object-not-found initialization messages, unsupported metadata notices, and
shared-memory cleanup warnings. Successful runs still produced the expected
scene, cameras, frozen actions, and datasets; an abort must be treated as an
environment failure. This host needs the Assimp preload in the runbook to
avoid the Pinocchio/HPP-FCL symbol collision, and CYCLONEDDS_HOME must remain
unset.

Only public Unitree, Isaac Lab, Pinocchio, and LeRobot APIs/assets were used.
BrickSim, BrickBench, RoCoBrick, and related private code/assets were not
accessed, copied, imitated, translated, or ported. The approved-design
implementation and validation occupied approximately 5 wall-clock hours,
based on artifact timestamps from 03:58 through 08:57 local time. Earlier
Phase 0 and historical calibration time was not separately tracked.
