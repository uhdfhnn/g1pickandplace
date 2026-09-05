# Progress log

## 2026-09-05 — repository MVP scaffold

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

Not yet validated:

- Isaac Sim launch;
- Unitree asset availability;
- URDF/USD kinematic equivalence;
- real frame name and quaternion convention;
- actual grasp orientation and wrist offset;
- Dex1 contact/closure values;
- task success rate;
- camera recording and LeRobot video finalization in the target environment.

Hours: fill in actual human time before submission. Do not infer or fabricate it from commit timestamps.
