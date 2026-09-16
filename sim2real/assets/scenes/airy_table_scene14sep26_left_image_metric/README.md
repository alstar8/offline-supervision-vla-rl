# airy_table_scene14sep26_left_image_metric

Metric-corrected copy of `airy_table_scene14sep26_left_image` (2026-09-15).

The source reconstruction is monocular, so it lines up with the ZED 2i image but is
metrically too small. Against live ZED 2i depth (S/N 35596233, same fixed mount) the
static background's depth ratio ZED/sim is 1.0915 (IQR 1.079-1.102, flat across depth and
image regions); the table plane measured 0.621 m from the camera vs 0.560 m in the
source scene.

The whole scene is scaled by `s = 1.0915` about the camera centre
`c = [0.055108, -0.936989, 0.885042]`: `x' = c + s (x - c)`. Pixels are unchanged, so the
camera intrinsics/extrinsics and `background.png` stay as they are.

- `background_registered*.glb`: same buffers byte-for-byte; the similarity is composed into
  the `geometry_0` node matrix (`S · M`). Not on the root `world` node: trimesh treats the
  scene root as its base frame and ignores that node's own transform, SAPIEN does not.
- `scene.json`: background paths, `groundplane_in_sim`, `groundplane_in_cam`, `aabb` and
  object centres/bounds transformed. Unused assets still point at the source bundle.
- Robot base position and object XY are scaled the same way in `config/config_debug.yaml`.
  Object sizes, grasp offsets and the robot are metric already and are not scaled.
- Base yaw is unchanged: a single static depth snapshot cannot resolve it (flat residual
  over [-1.25, +6.25] deg). Pending multi-pose ChArUco hand-eye.

Evidence: `runs/calibration/` (local, gitignored).
