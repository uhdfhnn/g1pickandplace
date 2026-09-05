# MVP design

## Goal

Create the smallest defensible vertical slice for a Unitree G1 VLA simulation project:

> Pick up the red cuboid and place it in the green target area.

The MVP intentionally avoids locomotion, bimanual handoff, articulated appliances, and learned control. The robot base is fixed; one arm and the Dex1 parallel gripper perform the task.

## Why this object interaction

A cuboid is preferable to a sphere, cone, or deep bin for the first open-loop task:

- broad opposing grasp faces;
- stable table pose;
- no rolling before contact;
- simple final XY containment metric;
- already available in the public Unitree red-block scene.

The interaction must remain physical:

1. fingers contact the cuboid;
2. friction supports it during lift and transport;
3. opening the gripper releases it;
4. gravity and contact settle it in the target.

No object teleport, fixed joint, weld, attachment API, or direct pose write is allowed during the demonstration.

## Strict open-loop boundary

### Allowed before rollout

- read one reset snapshot;
- use privileged object and robot-base poses;
- generate Cartesian waypoints;
- solve every waypoint with Pinocchio;
- reject an unreachable reset;
- interpolate all absolute joint targets;
- convert targets to the Unitree task's default-offset action convention.

### Forbidden after rollout starts

- any IK call;
- re-reading an object pose to change an action;
- measured-pose convergence transitions;
- contact-triggered phase changes;
- retry, recovery, or regrasp behavior;
- trajectory mutation;
- action dependence on a DAgger observation.

Success monitoring and logging may observe the environment, but may not alter the action sequence.

## Public backend

The default public backend is Unitree's direct-joint red-block task. Its action term is configured as `JointPositionActionCfg(..., scale=1.0, use_default_offset=True)`. Accordingly:

```text
environment_action = absolute_joint_target - default_joint_position
```

The repository preserves both absolute targets and exact environment actions in the compiled trajectory.

## Reset-time program

The planner computes five Cartesian IK waypoints:

```text
pregrasp, grasp, lift, preplace, place
```

It then composes those solutions with gripper presets into twelve timed joint-space phases:

```text
open_at_home
move_to_pregrasp
descend_to_grasp
close_gripper
grasp_settle
lift
transport
descend_to_place
open_gripper
release_settle
retreat
return_home
```

Each IK solution seeds the next solve. If any solve fails, planning raises before rollout.

## IK implementation

`PinocchioFrameIK` is an independent fixed-base frame solver using the standard public damped least-squares pattern:

1. forward kinematics and frame placements;
2. body-frame pose error through `log6`;
3. local frame Jacobian and `Jlog6` correction;
4. damped least-squares velocity over an explicit active-joint subset;
5. Pinocchio manifold integration;
6. joint-limit clipping;
7. convergence or an explicit planning error.

Only the selected right-arm joints move by default. Simulator-only gripper joints are composed into the full action separately.

## VLA data design

Core policy inputs/targets:

- front/head RGB;
- right-wrist RGB;
- robot joint positions;
- exact joint action;
- language instruction.

Diagnostic-only fields:

- privileged object pose;
- target pose;
- semantic phase index;
- final outcome and failure reason in the report.

The phase and privileged poses should not automatically be treated as policy observations; they are present to audit the expert and build controlled ablations.

## Task progression

### Task 1 — MVP

```text
Pick up the red block and place it in the green target area.
```

### Task 2 — after Task 1 is stable

Two visible primitives, one instructed pick:

```text
Place the blue cylinder in the right target area.
```

The distractor must remain outside the goal and below a displacement threshold.

### Task 3 — stretch

```text
Pick up the gray pusher and use it to push the yellow puck into the blue target area.
```

The tool task should record tool–puck contact and reject direct hand–puck displacement. A two-block stack is the lower-risk fallback.
