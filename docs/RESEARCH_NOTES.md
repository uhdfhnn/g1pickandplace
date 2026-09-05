# Research findings translated into MVP decisions

## Closest available public tasks

| Source | Public capability | Decision |
|---|---|---|
| Unitree `Isaac-PickPlace-RedBlock-G129-Dex1-Joint` | fixed-base G1, red `CuboidCfg`, Dex1, direct joint-position action, front and wrist cameras | use as the MVP scene/backend |
| Unitree cylinder task | same robot/hand with cylindrical object | reserve for Task 2 shape conditioning |
| Unitree RGB/Y/G stacking | multi-object rigid interaction | use as a lower-risk Task 3 fallback |
| NVIDIA fixed-base G1 Pink IK task | bilateral wrist-pose actions plus TriHand joints | use as an API/design reference, not runtime control |
| NVIDIA G1 locomanipulation task | manipulation plus balance/locomotion | defer; it adds unrelated control failure modes |
| NVIDIA G1 Inspire-hand task | five-finger manipulation | defer; more contact and action tuning than the MVP needs |
| Isaac Lab Mimic | demonstration annotation and setup variation | borrow evaluation ideas; write native LeRobot data instead |

## Why the Unitree direct-joint task is the better strict-open-loop base

NVIDIA's public Pink action is convenient, but its IK solver runs as the action is applied. That would make the command sequence open-loop while leaving IK online. The Unitree red-block task exposes direct joint-position actions, so this repository can solve all waypoints at reset and send only a frozen joint sequence afterward.

The Unitree task sets `scale=1.0` and `use_default_offset=True`. NVIDIA's public joint-action implementation defines the processed target as:

```text
offset + scale × input
```

and uses the articulation's default joint positions as the offset. The MVP therefore stores and sends:

```text
input = desired_absolute_position - default_position
```

## Available Isaac Lab primitives for later tasks

Useful geometry spawners:

- `CuboidCfg`: grasp block, target marker, pusher, tray walls, obstacles;
- `CylinderCfg`: puck, can, cylindrical distractor;
- `CapsuleCfg`: rounded handle or rod;
- `SphereCfg`: distractor, not a good first open-loop grasp;
- `ConeCfg`: type-conditioning distractor, not a good first grasp.

Useful asset abstractions:

- `RigidObjectCfg`: one dynamic object;
- `RigidObjectCollectionCfg`: named selected object plus distractors;
- `MultiAssetSpawnerCfg`: controlled shape/asset variation;
- `ArticulationCfg`: drawer/door mechanisms, deferred from MVP;
- `CameraCfg`: RGB/depth/segmentation;
- `ContactSensorCfg`: verify tool–object contact without feeding it into control.

## Tool-use recommendation

After pick/place and type-conditioned selection work:

```text
Pick up the gray cuboid pusher and use it to push the yellow cylindrical puck into the blue target area.
```

Use one straight precompiled push. Log tool–puck contact, puck displacement, final target containment, and direct hand–puck contact. Do not add curved pushing, obstacle avoidance, or corrective pushes until the basic tool interaction is reproducible.

## Version-sensitive risks

1. Unitree currently documents Isaac Sim 4.5 and 5.x support; exact Isaac Lab API behavior differs across those builds.
2. Quaternion order changed across Isaac Lab generations/integrations. The runner exposes it explicitly.
3. A public G1 URDF and the loaded USD may differ in root frame, wrist frame, collision additions, or joint ordering.
4. The Dex1 asset has two simulator joints per gripper side while the physical gripper is driven by one motor; the configured paired targets must be verified.
5. The existing Unitree camera observation term returns a placeholder while writing images through shared memory. The MVP reads camera sensor outputs directly.
6. The original red cube is 1 kg; the MVP lowers it to 0.15 kg for a first grasp but still requires physical contact.
