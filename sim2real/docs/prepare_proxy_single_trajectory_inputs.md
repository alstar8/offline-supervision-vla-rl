# Prepare Proxy Single Trajectory Inputs

Эта инструкция описывает, что должно уже существовать перед запуском текущего
`proxy planner`.

## 1. Главная точка настройки

Основной файл, который обычно нужно редактировать:

- `config/config_debug.yaml`

В нём лежат:

- настройки сцены
- `manip_object_id`
- `include_objects`
- `object_placements`
- robot base/init настройки
- task tuning и proxy tuning

Связанные profile-конфиги:

- `config/hand_contact_profiles.yaml`
- `config/hand_controller_profiles.yaml`
- `config/teleop_profiles.yaml`
- `config/lighting_profiles.yaml`

## 2. Что должно уже лежать в репозитории

Scene bundle:

- `assets/scenes/airi_table_new_empty3_image/simulation/scene.json`
- `assets/scenes/airi_table_new_empty3_image/simulation/scene_optimized.glb`
- `assets/scenes/airi_table_new_empty3_image/simulation/background_registered.glb`
- `assets/scenes/airi_table_new_empty3_image/simulation/background_registered_collision.glb`
- `assets/scenes/airi_table_new_empty3_image/simulation/background.png`

Object assets, на которые ссылается текущий `config/config_debug.yaml`:

- `assets/object_bank/container_blue_ext/`
- `assets/object_bank/orange_cube_ext/`
- `assets/object_bank/blue_cube_ext/`
- `assets/object_bank/green_cube_ext/`
- `assets/object_bank/yellow_cube_ext/`
- `assets/object_bank/white_cube_ext/`
- `assets/object_bank/tube_ext/`
- `assets/object_bank/plate_ext/`
- `assets/object_bank/coke_can_ext/`
- `assets/object_bank/banana_ext/`
- `assets/object_bank/apple_ext/`

Robot assets текущего runtime:

- `openreal2sim/simulation/maniskill/robot_assets/rc5_aero_hand/urdf_rc5_right_hand/`

## 3. Текущий baseline

Текущий рабочий baseline:

- `key=airi_table_new_empty3_image`
- `robot_uids=rc5_aero_hand_openr2s_rl`
- `task_type=pick_up`
- `control_mode=arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`
- `num_envs=1`

## 4. Что важно помнить

Этот pipeline не строит новые scene assets. Он ожидает, что:

- scene bundle уже лежит в `assets/scenes/`
- reusable object assets уже лежат в `assets/object_bank/`
