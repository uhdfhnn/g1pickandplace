# Public references and comparable implementations

Reviewed September 5, 2026. These sources informed API choices and scope. They were not copied into this repository.

## NVIDIA Isaac Lab

- Environment catalog and registered manipulation tasks: https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html
- Primitive spawners (`CuboidCfg`, `CylinderCfg`, `CapsuleCfg`, `SphereCfg`, `ConeCfg`): https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.spawners.html
- Joint action implementation and default-offset convention: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py
- Public fixed-base G1 upper-body IK environment: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/fixed_base_upper_body_ik_g1_env_cfg.py
- Public G1 Pink controller layout: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/configs/pink_controller_cfg.py
- Pink task-space action implementation: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/actions/pink_task_space_actions.py
- Camera configuration examples: https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html
- Isaac Lab Mimic data-generation framework: https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/isaaclab_mimic.html

NVIDIA's fixed-base G1 task is a useful integration reference, but this MVP does not use its runtime task-space IK action. It pre-solves joint targets instead so the rollout is strictly open-loop.

## Unitree

- Official simulation repository: https://github.com/unitreerobotics/unitree_sim_isaaclab
- Official direct-joint red-block task: https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/main/tasks/g1_tasks/pick_place_redblock_g1_29dof_dex1/pickplace_redblock_g1_29dof_dex1_joint_env_cfg.py
- Official red cuboid scene: https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/main/tasks/common_scene/base_scene_pickplace_redblock.py
- Official G1/Dex1 asset configuration: https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/main/robots/unitree.py
- Unitree LeRobot bridge: https://github.com/unitreerobotics/unitree_lerobot
- Unitree XR teleoperation/data recording: https://github.com/unitreerobotics/xr_teleoperate

The official Unitree simulator already includes G1 red-block and cylinder pick/place, RGB/Y/G block stacking, whole-body cylinder movement, three hand variants, camera streams, replay, and visual data augmentation. This MVP reuses the public environment registration as a dependency but implements its own reset-time planner, IK, trajectory representation, metrics, and LeRobot writer.

## Pinocchio

- Public inverse-kinematics example: https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_i-inverse-kinematics.html

The local solver follows the generic algorithmic pattern from this public example: frame error, frame Jacobian, damped least squares, and integration. It is not derived from any private assembly code.

## LeRobot

- LeRobotDataset v3 documentation: https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3
- Dataset implementation: https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/lerobot_dataset.py
- Official recording example: https://github.com/huggingface/lerobot/blob/main/examples/isaac_teleop_to_so101/record.py

The writer uses the native lifecycle `create → add_frame(task=...) → save_episode → finalize` rather than inventing an HDF5 imitation.

## Similar community work reviewed

- https://github.com/Lurrkkking/isaaclab-g1-pickplace — G1 locomanipulation/Mimic reproduction and imitation-learning experiments.
- https://github.com/ARC-KIST/isaaclab_g1 — G1 work in Isaac Lab.
- https://github.com/iit-DLSLab/locomanipulation-teleop-isaaclab — locomanipulation teleoperation in Isaac Lab.
- https://github.com/JeffrinSam/IsaacLab-Arena-G1 — G1-oriented Isaac Lab Arena work.

These repositories demonstrate the surrounding ecosystem and common setup pitfalls. None is an implementation dependency, vendored source, or basis for a mechanical rewrite.
