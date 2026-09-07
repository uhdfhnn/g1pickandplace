# Unitree G1 Entrance-Test Demo Design Specification

## 1. Purpose

This specification defines a time-bounded demonstration suite for the VLA
entrance test. The suite contains three core tasks of increasing difficulty
and one optional tool-use acceptance task. Every task uses the public Unitree
G1 Isaac Lab integration and the same Dex1 embodiment.

The implementation must demonstrate:

- public `unitree_sim_isaaclab` scene reuse;
- model-free physical interaction through a reset-time-compiled expert;
- text-conditioned task selection;
- deterministic setup variants;
- native LeRobot data collection; and
- evidence-based evaluation with visible Isaac Sim runs.

This specification supersedes the experimental minimal white-background and
custom typed-primitive scene as the intended final demonstration design.
Existing experimental results remain historical evidence and must not be
silently deleted or presented as validation of this specification.

## 2. Non-negotiable boundaries

1. Use the public task
   `Isaac-Stack-RgyBlock-G129-Dex1-Joint` as the scene and robot baseline.
2. Retain the public warehouse, packing table, red/yellow/green blocks, Dex1
   robot, and public robot cameras.
3. Run every Isaac Sim inspection, plan, and rollout with a visible GUI. Do not
   use headless mode.
4. Complete every IK call and every preflight safety check before rollout step
   zero.
5. Keep `OpenLoopPolicy.act(observation)` observation-invariant.
6. Do not introduce contact-, error-, reward-, or image-driven expert phase
   transitions.
7. Do not attach, weld, teleport, or make a manipulated object or tool
   kinematic after reset.
8. Runtime observations may be recorded and evaluated, but they must never
   modify the frozen action sequence.
9. Use the installed public `LeRobotDataset` API. Do not create a lookalike
   custom dataset format.
10. Do not access, copy, imitate, translate, or port BrickSim, BrickBench,
    RoCoBrick, or related private code or assets.
11. Dex3 and Inspire are out of scope. A Dex1 hand profile may be introduced
    for clean configuration, but no multi-hand abstraction is required.

## 3. Public baseline

The public stack task provides:

- one fixed-base Unitree G1 29-DoF robot with Dex1 grippers;
- the public small-warehouse stage;
- the public packing table and yellow square marker;
- three dynamic cuboids named `red_block`, `yellow_block`, and `green_block`;
- front, left-wrist, and right-wrist cameras; and
- direct joint-position actions.

The runner must record the live task ID, robot/USD identity, action dimension,
ordered action joint names, joint limits, robot base pose, object poses, camera
outputs, and end-effector frame before planning. The matching public URDF must
be checked against the live USD and must expose the configured right-wrist
frame.

### 3.1 Allowed scene overrides

Only the following configuration-time overrides are allowed:

- deterministic, non-overlapping object reset poses;
- semantic text labels/metadata for the three blocks and marker;
- GUI viewport eye/look-at configuration;
- task-specific static target or tray geometry;
- documented mass, friction, and contact calibration supported by visible
  evidence;
- a collision-safe robot reset posture; and
- disabling an upstream termination that would truncate a frozen program.

These overrides must be reported in diagnostics. They must not replace the
public warehouse or table with a reconstructed scene.

## 4. Viewpoint and sensors

The initial GUI viewport must resemble the supplied public red-block image:

- camera centered approximately in front of the robot;
- camera elevated above the table and looking downward;
- robot, both hands, all relevant blocks, and the marker visible;
- enough table visible in front of and beside the marker to judge spatial
  relations and stacking; and
- warehouse visible in the background.

The viewport is a presentation camera only. It must not modify the public
front or wrist camera extrinsics used for observations and LeRobot recording.
The exact world-frame eye and look-at values must be documented next to their
definitions after one visible inspection confirms the composition. Values that
place the camera behind warehouse geometry or hide a task object are invalid.

## 5. Control architecture

Every task uses the following lifecycle:

```text
public environment reset
  -> fixed settling period
  -> one aggregate reset snapshot
  -> parse and validate the text instruction
  -> select object(s), reference, and task relation
  -> compute all task-space targets
  -> solve every IK waypoint
  -> compile one immutable action array
  -> validate limits and collision/clearance gates
  -> save the plan
  -> exit in plan-only mode OR create OpenLoopPolicy
  -> replay the frozen array without feedback
  -> evaluate and record after execution
```

The aggregate reset snapshot contains all three block poses, the marker pose,
robot base pose, ordered joint state, and ordered default joint state. Task 4
also includes the shovel and tray poses. No later state read may be used for
planning.

### 5.1 Text instruction contract

The first implementation uses a finite deterministic grammar, not a general
language model. It must support the exact documented instructions plus
unambiguous case- and punctuation-insensitive variants. Unsupported or
ambiguous instructions fail before environment construction or planning.

Structured CLI fields remain available for debugging and must resolve to the
same internal task specification as the text instruction.

Suggested interface:

```text
--demo-task {relative-place,type-place,stack,shovel}
--instruction TEXT
--object {red,green,yellow}
--relation {left-of,right-of,in-front-of,behind,on-top-of}
--reference {yellow-marker,red,green,yellow,target-tray}
--clearance-m FLOAT
```

Conflicting text and structured fields must be rejected rather than silently
overwritten.

### 5.2 Spatial relation contract

World `+Z` is up. Horizontal natural-language relations are defined from the
default presentation viewport, so "left" means visually left in the supplied
view. A visible inspection must establish the corresponding signed world axis
before implementation freezes the relation mapping.

For a relation beside a marker, the desired object center is:

```text
marker center
 relation unit vector
   * (marker half-extent + object half-extent + requested edge clearance)
```

Clearance therefore means edge-to-edge clearance, not center distance.

The canonical clearance is provisionally `0.05 m`. It represents a visible
five-centimetre gap on the public table, is large enough to prevent the block
from being mistaken for a placement on the marker, and remains small enough
for the fixed-base right arm's demonstrated workspace. A smaller value can
make the relation visually ambiguous or cause marker contact; a much larger
value can make IK unreachable. It is intentionally CLI-configurable and must
be confirmed by visible plan-only runs before physical rollout.

## 6. Core Task 1: relative single-object placement

### Instruction

```text
Pick up the red block and place it left of the yellow square marker.
```

### Behavior

The red block is grasped, lifted clear of the table, transported, and released
on the tabletop to the specified side of the yellow marker. The yellow and
green blocks remain distractors.

A fixed pre-close dwell must hold the already-solved open grasp pose before
closing. A fixed post-close dwell must allow the physical grasp to settle.
The lift waypoint must provide visibly positive object/table clearance before
horizontal transport begins. These are precompiled timed holds, not runtime
contact gates.

### Variants

Provide at least three deterministic, non-overlapping red-block XY/yaw reset
variants. The same planner and calibration must be used for all accepted
variants. An unreachable variant is rejected before rollout.

### PASS criteria

- the red block is the only selected object;
- the red block is physically lifted and transported without attachment or
  teleportation;
- its final relation to the marker is correct;
- measured edge clearance matches the requested clearance within the reported
  tolerance;
- the block is supported by the table, not the hand or marker edge;
- final block speed is below the stable-object threshold; and
- unselected blocks remain within the distractor-displacement tolerance.

## 7. Core Task 2: type-conditioned placement

### Instructions and mapping

The canonical instructions are:

```text
Pick up the red block and place it left of the yellow square marker.
Pick up the green block and place it right of the yellow square marker.
Pick up the yellow block and place it in front of the yellow square marker.
```

All three blocks remain in the scene for every instruction. The instruction
selects exactly one block and one relation; color is not inferred from camera
pixels by the open-loop expert.

### PASS criteria

- the parsed selection equals the requested color;
- the selected block moves at least `0.05 m` horizontally, demonstrating a
  meaningful manipulation rather than settling noise;
- both unselected blocks move no more than the distractor-displacement
  tolerance;
- the final spatial relation and clearance are correct; and
- changing only the instruction changes the selected object/target without a
  code or calibration change.

The `0.05 m` movement threshold is provisional and equals one public stack
block width. A smaller threshold can accept contact jitter as manipulation;
a much larger threshold can reject a valid nearby relation. It is a fixed
evaluation threshold, not a control transition, and must be checked against
the final authored geometry.

## 8. Core Task 3: two-object stacking

### Instruction

```text
Pick up the red block and stack it on the yellow block.
```

The green block remains a distractor. Additional color orderings are variants,
not separate implementations.

### Target construction

At reset, compute the desired red-block center from the frozen yellow-block
pose:

```text
target_x = yellow_center_x
target_y = yellow_center_y
target_z = yellow_center_z
         + yellow_half_height
         + red_half_height
```

The release approach must be above the stack target, and horizontal transport
must begin only after the frozen program has completed the higher lift and its
timed settle phase.

### PASS criteria

- the red block leaves the table before transport;
- the yellow block remains upright and within the reported base-displacement
  tolerance;
- final red/yellow center error is at most `0.01 m` independently in world X
  and Y;
- final vertical separation matches one block height within `0.01 m`;
- after release and a fixed settle interval, both blocks are below the
  stable-object speed threshold;
- the red block is supported by the yellow block, not the hand, marker, or
  table; and
- the green distractor remains within its displacement tolerance.

The `0.01 m` position tolerances are provisional one-centimetre evaluation
bounds. For equal five-centimetre blocks they allow limited overhang while
rejecting a visually separate or edge-only placement. Smaller bounds can
reject harmless PhysX/contact jitter; larger bounds can accept an unstable
stack. They are evaluation-only and require confirmation from visible traces.

## 9. Optional Task 4: shovel tool use

### Instruction

```text
Pick up the shovel, scoop the red block, and place it in the target tray.
```

### Scene extension

Add an original, repository-owned analytic or USD shovel and a static target
tray to the public Dex1 stack scene. Do not source either asset from a private
project. The shovel should contain:

- a Dex1-graspable handle;
- a thin blade wider than the red block with documented lateral margin;
- a shallow leading wedge capable of entering under the block; and
- a small backstop that reduces loss during lift.

Every dimension, mass, friction coefficient, contact offset, pose, and material
value must have an adjacent rationale covering units, frame/sign convention,
source or derivation, valid range, expected high/low failure, configurability,
and validation evidence.

### Frozen program

```text
open hand
  -> approach shovel handle
  -> grasp shovel
  -> timed tool-grasp settle
  -> lift shovel
  -> orient blade parallel to table
  -> approach behind red block
  -> lower blade
  -> insert blade under red block
  -> tilt blade upward
  -> lift loaded shovel
  -> transport over target tray
  -> tilt to unload
  -> retreat
  -> return home
```

All robot IK and tool-envelope checks must finish before rollout. The ordinary
robot URDF does not contain the held shovel, so the preflight gate must add a
conservative handle/blade collision envelope and check its swept clearance
from the robot, table, tray, and unselected blocks.

### Result levels

`PASS` requires all of the following:

- the hand physically grasps and lifts the shovel without an attachment;
- the red block has no direct hand or finger contact;
- the shovel blade causes the red block to leave the table;
- the blade supports the red block during a nontrivial horizontal transport;
- the red block finishes inside and supported by the tray;
- the red block is stable after the fixed final settle period; and
- yellow/green distractors remain within tolerance.

`PARTIAL` may be reported when the shovel is grasped and physically pushes the
red block into the tray but never supports it above the table. A push-only
result must never be reported as a successful scoop.

## 10. Evaluation thresholds

The initial stable-object speed threshold is `0.01 m/s`, measured after the
frozen program and never used to control it. It comes from earlier public-scene
settled traces whose speeds were below this scale. A lower bound may reject
normal PhysX jitter; a higher bound may accept a sliding or falling object.
It is fixed for comparable reports and must be revisited if the physics step,
mass, material, or solver settings change.

The initial distractor-displacement tolerance is `0.01 m` in world-space
Euclidean translation from the aggregate reset snapshot. It distinguishes a
stationary distractor from a visibly bumped object while tolerating small
reset/contact settling. Too small a value can fail normal solver jitter; too
large a value can hide unintended multi-object contact. It is evaluation-only,
fixed across task variants, and must be supported by the inspect-only settle
trace before use.

Evaluation reads may occur at phase boundaries and after execution solely for
diagnostics. They must not select, skip, repeat, or modify a frozen action.

## 11. Validation gates

### Gate A: dependency-light

Before any simulator run:

- run the complete unit-test suite;
- run compile checks and `git diff --check`;
- verify instruction parsing, relation geometry, stack target geometry, and
  conflicting-input rejection;
- prove aggregate reset snapshots are immutable;
- prove different observations yield byte-identical open-loop action arrays;
- prove injected waypoint failure prevents policy construction; and
- prove no rollout-time IK or planning call exists.

### Gate B: visible inspect-only

Without constructing an expert:

- confirm the exact public task, stage, table, robot, hand, cameras, blocks,
  marker, and any Task 4 assets;
- confirm no initial overlaps, interpenetration, or unsupported objects;
- confirm the viewport composition and horizontal relation-axis mapping; and
- save viewport plus front/left-wrist/right-wrist images.

### Gate C: visible plan-only

For every task and accepted variant:

- print the instruction and resolved structured task;
- print the complete aggregate reset snapshot;
- print every target pose, IK residual, and iteration count;
- validate ordered action dimensions, joint limits, self-collision, and
  environment/tool clearance;
- save the immutable trajectory; and
- exit with an explicit statement that rollout step zero was not started.

No physical trajectory may run unless every required waypoint succeeds.

### Gate D: visible physical rollout

Only after Gate C passes:

- replay the exact saved frozen trajectory;
- capture phase-boundary state and all requested camera views;
- calculate task metrics without altering control;
- save a visible video; and
- report `PASS`, `FAIL`, `PARTIAL` where allowed, or `NOT RUN`.

### Gate E: native LeRobot data

- record the exact executed action sequence, ordered state, public camera
  images, instruction text, and task metadata;
- reopen the episode with the installed public LeRobot API;
- validate shapes, dtypes, timestamps, frame counts, task string, episode
  boundaries, and video decode; and
- associate the trajectory hash with the episode so planning, execution, and
  recording can be shown to use the same frozen action array.

## 12. Deliverables

| Task | Required code/config | Required evidence | Data |
| --- | --- | --- | --- |
| Task 1 | Relative-place task spec, target geometry, three variants | Inspect log/images, plan-only logs, at least one successful visible rollout, metrics, video | At least one valid LeRobot episode; one per accepted variant is preferred |
| Task 2 | Finite instruction parser and three color/relation mappings | Plan-only evidence for all three instructions, correct-object metrics, at least one visible rollout | At least one valid typed-instruction episode; all three are preferred |
| Task 3 | Stack target construction and stack evaluator | Plan-only log, phase images, one genuine stable stack, metrics, video | At least one valid stack episode |
| Task 4 | Original shovel/tray assets, tool frames, held-tool envelope checker | Inspect/plan logs, contact evidence, metrics, video, honest PASS/PARTIAL status | One valid episode if a physical rollout is attempted |

The final project report must also include:

- design and task rationale;
- files changed and exact commands run;
- environment/dependency versions and public repository commit;
- results and failure counts;
- encountered problems and unresolved risks;
- approximate hours spent; and
- a clear distinction between plan-only, physical success, partial tool use,
  failure, and work not run.

## 13. Priority and stopping rule

Implementation priority is:

1. Task 1 reproducible relative placement;
2. Task 2 correct instruction-conditioned selection;
3. Task 3 one stable red-on-yellow stack;
4. Task 4 shovel tool use as a stretch task.

Do not weaken Tasks 1-3 or their evidence to claim Task 4 completion. If time
expires during Task 4, deliver the strongest evidence-backed `PARTIAL` result
and document the exact failed acceptance criteria.

An external VLA policy is optional and outside these four expert acceptance
tasks. Any later VLA integration must use a separate observation-conditioned
runner and must never be inserted into the strict open-loop expert or presented
as equivalent to it.
