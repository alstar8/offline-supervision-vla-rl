# release.v4: September Left-Camera Scene

## Scope

Created from `release.v3` (`22325d4`), not from `prerelease`.
Source: OpenReal2Sim-ROLA-ORIGINAL checkpoint `0cb0895`, accepted locally on
2026-09-14. Scene geometry migration is implemented; final grasp-quality
acceptance remains pending. No planner/controller implementation was replaced.

The default scene is `airy_table_scene14sep26_left_image`. The old
`airi_table_new_empty3_image` scene and its global/local settings are unchanged.
The primary config is `config/config_debug.yaml`; baseline/test configs were not
changed. `Makefile` derives `SCENE` from `KEY`, unless explicitly overridden.

```bash
docker start sim2real-simulation
make planner TASK_OBJECT=blue_cube_ext
make planner KEY=airi_table_new_empty3_image TASK_OBJECT=blue_cube_ext
make planner-headless TASK_OBJECT=spray_bottle_ext
make test
```

Use the existing persistent GPU container with the repository mounted at `/app`.
No image rebuild or container removal is required. The existing release.v3
runner auto-starts its proxy macro; this import does not add OpenReal2Sim's newer
manual viewer controls or object-specific hand-pregrasp stage.

## Assets and Coordinates

Bundle: `assets/scenes/airy_table_scene14sep26_left_image/simulation/`.
All file references in `scene.json` now point into this bundle. Referenced coin
assets/trajectories, background image and combined scene mesh are included.
The runtime uses the flattened `background_registered.glb` and
`background_registered_collision.glb`, not the combined reconstruction preview
`scene_optimized.glb`. The combined preview was not flattened.

The two runtime meshes were copied byte-for-byte, without rotation or rescaling:

| File | SHA-256 |
| --- | --- |
| `background_registered.glb` | `9f50f3c584d5d38cf54e02498050bd35b4514964aa8def33df045d4c45600029` |
| `background_registered_collision.glb` | `94f0ece4fc4c8a747c50020edd57ae4e4e3b0a3bc7d6368ea7a639fca0ed8699` |

Scene-specific settings include the explicit tabletop height
`0.3246851381617871`, disabled automatic height estimation, base XYZ/quaternion,
object placements/orientations and transformed world-frame grasp offsets.
Object Z remains relative to support, with a 2 mm configured spawn clearance.
Camera intrinsics and extrinsics are preserved (1886x1059).

Object selection and layout come from the OLD sim2real KEY, not from the
OpenReal2Sim scene section. All 13 object definitions and their properties are
copied; the same eight objects are enabled, including `plastic_cup_ext`.
The default target remains `blue_cube_ext`. Existing object-bank assets are
reused, including paths for inactive objects.

Only world XY positions and orientations are transformed to the new robot base:
`R_delta = R_base_new * R_base_old^-1`,
`p_xy_new = p_base_new + R_delta * (p_xy_old - p_base_old)`.
The base yaw difference is about +3.793235 degrees. Thus each object's pose
relative to the robot is preserved; object Z, scale, physics and mesh settings
are unchanged. New background, camera, tabletop/base calibration and grasp
calibrations are retained. The old scene files were not overwritten. No source
repository paths or external symlinks are required.

## Explicit Schema Adaptation

- Source `planner_proxy_object_calibrations` maps to this proxy-only runner's
  `planner_object_calibrations`. Source MPlib calibrations are not used.
- Source startup vectors map to `robot_init_qpos_profiles` and
  `robot_init_qpos_profile_by_object`, with the approved new side pose.
- Bottle targets use the existing `current_tcp` orientation mode. Pregrasp and
  descend retain identical XY offsets; no extra horizontal approach is added.
- Unsupported `hand_pregrasp_*` fields are not inserted as ineffective options.
  Hand defaults, contact/controller profiles, adaptive stepping and macro logic
  remain those of release.v3. Behavior is therefore not claimed to be identical
  to the newer OpenReal2Sim runner.

## Validation on 2026-09-14

- Baseline suite: 213 passed in `sim2real-simulation` before changes.
- Migration TDD: 7 expected failures before adding scene/config/Make support.
- Complete suite after migration: 224 passed (dependency deprecation warnings).
- Layout correction TDD: two new regression tests failed before correction,
  then the complete suite passed: 226 tests in `sim2real-simulation`.
  Tests check selection, all 13 robot-relative poses, relative Z and all other
  object properties against the old KEY; the old config remains unchanged.
- Corrected layout support check: 441 downward rays per active object over its
  visual/collision bounding footprint plus 10 mm margin; no missing support
  for any of the eight objects. Maximum table-plane deviation was 0.526 mm
  under the container, effectively zero under the other seven objects.

The following viewer, grasp and replay checks preceded the layout correction
and used the initial seven-object layout imported from OpenReal2Sim. They do
NOT validate grasps or contact behavior for the current corrected layout.
Grasps have not yet been rerun after the correction.

- GPU viewer smoke: window opened/rendered/closed; sensor frame 1886x1059;
  actual base position matched config. Frame:
  `runs/release_v4_validation/new_scene_base_camera.png`.
- New scene blue cube: `task_success=True`, `is_grasping=True` after lift,
  lift about 82.8 mm. Run suffix `20260914_184452`.
- Old scene blue cube: `task_success=True`, `is_grasping=True` after lift,
  lift about 84.1 mm. Run suffix `20260914_184532`.
- New scene bottle: executor reported success using
  `close_grasp_plus_lift_with_object_near_tcp`, lift about 86.6 mm, but
  `is_grasping=False` after lift. This is NOT confirmed stable retention.
  Run suffix `20260914_184654`.
- New blue NPZ replay dry-run resolved embedded runtime config, new scene and
  432 actions/images/infos. Full physical replay was not tested.

Runs are under `runs/manual/proxy_ee_delta_headless_<KEY>_pick_up_<OBJECT>_<suffix>/`.
Runtime logs and NPZ files are local ignored artifacts, not part of the scene bundle.

## Open Acceptance Items

The initial-layout new-scene blue execution reported non-target contacts with
the bottle and white cube, including the prehand camera collider. Recheck these
contacts in the corrected layout. Success labels alone do not make trajectories
clean training data. Inspect collateral object motion and bottle retention
before collection/release publication. Do not hide warnings or weaken success
criteria to pass validation.

Other target objects, long-duration retention, full physical replay and real
robot transfer remain unvalidated. Metric camera/base/TCP calibration remains
separate from this scene import.
