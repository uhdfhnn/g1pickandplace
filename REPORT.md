# VLA entrance-test report

## Task design

Instruction: **Pick up the red block and place it in the green target area.**

Robot/hand: Unitree G1 29-DoF with fixed base and Dex1 parallel gripper.

Object: public Unitree scene's red cuboid, mass reduced for the first physical grasp.

Controller: reset-time Pinocchio IK, fully compiled joint trajectory, no model and no online feedback.

## Setup variants

| Variant | Object reset | Target | Accepted by IK precheck | Result |
|---|---:|---:|---:|---:|
| baseline | TODO | TODO | TODO | TODO |

## Results

| Metric | Value |
|---|---:|
| Episodes | TODO |
| Grasp success | TODO |
| Lift success | TODO |
| Placement success | TODO |
| Stable final placement | TODO |

## Data collection

Describe camera streams, control/data rate, joint/action conventions, language instruction, privileged diagnostics, number of episodes, total frames, and validation/replay results.

## Problems encountered

Document IK failures, frame mismatches, collision/contact issues, object slip, gripper tuning, simulator crashes, camera timing, and dataset finalization problems with evidence.

## Approximate time spent

Use actual tracked time only.

## Reproduction

List exact environment versions, repository commits, commands, seeds, configuration flags, and video/dataset locations.
