# Codex execution instructions

Work from the repository root. Preserve existing user changes and do not replace working public integration code with invented Isaac Lab APIs.

## Non-negotiable boundaries

- Do not copy, paste, port, translate, mechanically rewrite, or closely imitate BrickSim, BrickBench, RoCoBrick, or related private code.
- Do not import those projects or reuse distinctive identifiers from them.
- Generic manipulation phases are allowed, but keep this repository's reset-time compiler architecture.
- Every IK call must finish before rollout step zero.
- `OpenLoopPolicy.act(observation)` must remain observation-invariant.
- Do not add contact-, error-, or observation-based state transitions to the expert.
- Record with the installed public `LeRobotDataset` API; do not create a superficially similar custom format.

## Phase 0 — verify the host

1. Run `git status`.
2. Read `README.md`, `docs/MVP_DESIGN.md`, and `docs/REFERENCES.md`.
3. Run `python -m pytest` and `python scripts/preview_plan.py`.
4. Locate the Unitree checkout and record its commit SHA.
5. Record Isaac Sim, Isaac Lab, Pinocchio, LeRobot, driver, and GPU versions.
6. Confirm the task ID and action dimension.
7. Confirm the fixed-base URDF's body-joint names and `right_wrist_yaw_link` frame match the simulator USD.
8. Write results to `PROGRESS.md` before changing implementation.

## Phase 1 — simulator smoke test

1. Launch the public Unitree red-block task without this policy.
2. Confirm robot, cube, table, target marker, front camera, and wrist camera.
3. Print ordered action joint names, defaults, limits, base pose, object pose, and end-effector pose.
4. Run the planner only. Do not start rollout until all five waypoint solves succeed.
5. Save and inspect the NPZ trajectory.

## Phase 2 — fixed-seed physical validation

1. Begin with one fixed object pose.
2. Tune only documented configuration parameters: wrist offset, grasp orientation, gripper positions, target location, and segment durations.
3. Verify contact-based lift and release; never attach or teleport the object.
4. Run ten fixed-seed episodes and report grasp, lift, transport, and placement failure counts.
5. Save one video and one valid LeRobot episode.

## Phase 3 — variants

After Task 1 is reproducible:

1. add bounded object XY/yaw reset variants;
2. reject unreachable variants before rollout;
3. collect at least one episode per accepted variant;
4. validate dataset shapes, timestamps, episode boundaries, task strings, and video decode.

## Required update after each phase

Report:

1. files changed;
2. exact commands run;
3. observed result;
4. blockers and uncertainty;
5. next implementation action.
