# Final VLA entrance-test report

The canonical report is
[docs/ENTRANCE_TEST_REPORT.md](docs/ENTRANCE_TEST_REPORT.md); the copyable
visible-GUI procedure is
[docs/RUN_ENTRANCE_TEST_DEMO.md](docs/RUN_ENTRANCE_TEST_DEMO.md).

## Result

| Task | Result |
| --- | --- |
| Relative red placement | PASS |
| Type-conditioned selection | PASS for the validated green/right instruction; red/yellow are plan-gated |
| Red-on-yellow stack | PASS on calibrated8, including native LeRobot Gate E |
| Shovel tool use | PARTIAL physical evidence only; Gate21 preflight passed, but no successful scoop or completed valid recording |

## Keyboard teleoperation deliverable

The final deliverable also includes a visible keyboard teleoperation demo for
the same public G1 29-DoF Dex1 Stack-RgyBlock scene:

~~~bash
python3 scripts/run_demo.py --keyboard-teleop
~~~

After focusing the Isaac Sim viewport, `Tab` switches arms, `1`–`7` selects a
shoulder-to-wrist joint, the arrow keys jog the selected target by 2 degrees,
`O`/`C` opens or closes that side's gripper, and `Q`/`Esc` exits. Targets are
clamped against the live soft joint limits and all other joints hold their
captured reset targets. The run is identified as `manual_joint_jog` in its
`teleop.log` completion record.

This optional manual mode is intentionally outside the autonomous entrance-test
acceptance evidence. It exits before reset-snapshot planning, constructs no
`OpenLoopPolicy`, performs no online IK, does not record a LeRobot episode, and
does not report a manipulation PASS. Its purpose is operator inspection,
joint-direction checks, and simple manual scene interaction without changing
the validated open-loop deliverable.

The accepted Task 1 red run physically lifted 0.074059 m, transported 0.133814
m, measured 0.047375 m edge clearance for a 0.05 m request, left both
distractors at 0 displacement, and ended at 0.000267 m/s. Its native episode
has 1550 frames and matching action hashes.

The accepted Task 2 green run selected only green, moved 0.169501 m, lifted
0.058911 m, transported 0.197043 m, measured 0.054717 m edge clearance, left
red/yellow at 0 displacement, and ended at 0.000307 m/s. The retained
center-error diagnostic is 0.011851 m; the approved evaluator gates on the
explicit relation/clearance and movement criteria, not that extra center box.

The calibrated8 Task 3 stack has physical lift 0.071563 m, vertical separation
0.049998 m, top XY error 0.003331 m, bottom displacement 0.000156 m, bottom
speed 0.000471 m/s, top speed 0.001141 m/s, upright yellow, and green
displacement 0. Gate C and Gate E use frozen action SHA-256
a82eee52269fb756850ab70e45bc780e52d9fee197cbb64a92befcff1efefdf2.

The shovel implementation now uses repository-owned compound tool/tray geometry,
tool-frame reset-time IK, a conservative swept handle/blade envelope, and
read-only public contact/pose evidence. Gate19 (35 mm handle, -30 degree
orientation, Y=0.060 m) passed 22 reset-time waypoints and 345,138 swept
checks with zero failures, producing 2,501 steps without starting rollout step
zero. The 50 mm handle Gate20 candidate was correctly rejected before rollout
for 66 exact handle/left-elbow overlaps during `tilt_blade_up`; the current
45 mm candidate passed Gate21 preflight with the same 22 waypoints, 345,138
zero-failure swept checks, 2,501 steps, and frozen action hash
`87b8d7c777dcf53e45cb4ed5b790e27a2ac7bb6251d5d519e4ac60d6838fb322`.

The earlier complete rollout-02 at the prior X=-0.025 m calibration produced a
valid native LeRobot artifact (2,515 frames at 50 Hz, three RGB streams, and
matching action hash `3b73273a7084e87e6e13efdb1dceba66998084b48df3babf0fb0e79c90166c97`),
but its semantic shovel result was FAIL: both finger contacts were true and
table contact was lost, yet measured tool lift was only 0.0155148506 m, below
the 0.05 m requirement, with no blade-red contact or tray success. The current
45 mm rollout-05 retry was intentionally stopped after phase frames showed the
shovel had dropped/remained on the left table before the hand moved behind the
red block. It is physical PARTIAL evidence only; its artifacts are not a
completed valid Task 4 recording, and no push/drop is reported as a scoop.

## Reproduction and boundaries

Use only the visible command sequence in the runbook with the public
Isaac-Stack-RgyBlock-G129-Dex1-Joint task. Every IK/preflight check completes
before rollout step zero, and OpenLoopPolicy ignores observations during replay.
No private BrickSim, BrickBench, RoCoBrick, or related code/assets were used.
Dex3 and Inspire are out of scope.

Current keyboard-focused validation is 21 tests passed, with compileall and
`git diff --check` also passing. The latest full shared-tree run is blocked at
test collection by an unrelated in-progress shovel refactor that no longer
exports `SHOVEL_HANDLE_SIZE_M`; no keyboard test fails. The final visible
inspect and plan-only reruns also passed. Phase 0 versions, commit SHAs, ordered 33-joint
action names, and URDF/USD compatibility are recorded in the canonical report.
Observed Warp/DDS/deprecation/material/cleanup messages are summarized there;
an actual simulator abort remains an environment failure. The approved-design
implementation and validation occupied approximately 5 wall-clock hours, based
on artifact timestamps from 03:58 through 08:57 local time; earlier Phase 0 and
historical calibration time was not separately tracked.
