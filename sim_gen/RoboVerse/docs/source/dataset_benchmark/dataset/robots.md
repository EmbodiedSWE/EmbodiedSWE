# Robots in RoboVerse

## Scenes

RoboVerse currently includes some robots.

| Robot Name | Category | Number of DoFs | Config Name |
| ------ | --------- | ---------------- | ------------ |
| Fetch  | Wheeled | 14 | fetch |
| Franka | Arm | 9 | franka |
| Franka Slide | Arm with moving base | 11 | franka_slide |
| Google Robot | Arm | 11 | google_robot_static |
| H1 | Humanoid | 19 | h1 |
| H1 Hand | Humanoid with dexterous hand | 69 | h1_hand |
| H1 Simple Hand | Humanoid with simple hand | 45 | h1_smple_hand |
| IIWA | Arm | 9 | iiwa |
| Sawyer | Arm | 10 | sawyer |
| Sawyer Mujoco | Arm | 9 | sawyer_mujoco |
| UR5e | Arm | 6 | ur5e_2f85 |
| Walker | Bipedal | 6 | walker |
| Ant | Quadreuped | 12 | ant |

## Dexterous Hands

RoboVerse includes support for various dexterous hands for manipulation tasks.

| Robot Name | Number of DoFs | Config Name | Notes |
| ------ | ---------------- | ------------ | ----- |
| Allegro Hand | 16 | allegrohand | 4-finger anthropomorphic hand |
| BrainCo Hand (Left) | 11 | brainco_hand_left | 6 actuated + 5 mimic/passive joints, prosthetic hand |
| BrainCo Hand (Right) | 11 | brainco_hand_right | 6 actuated + 5 mimic/passive joints, prosthetic hand |
| Inspire Hand (Left) | 12 | inspire_hand_left | 6 actuated + 6 mimic/passive (coupled) joints |
| Inspire Hand (Right) | 12 | inspire_hand_right | 6 actuated + 6 mimic/passive (coupled) joints |
| PSIHand (Left) | 21 | psihand_left | All 21 joints actuated. Known issues with IsaacGym, use MuJoCo/Genesis |
| PSIHand (Right) | 21 | psihand_right | All 21 joints actuated. Known issues with IsaacGym, use MuJoCo/Genesis |
| XHand | 12 | xhand | Compact dexterous hand |
