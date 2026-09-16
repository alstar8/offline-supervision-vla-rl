#!/usr/bin/env python3
"""
Render the simulator side of a recording made by record_paired_replay.py and write paired
sim | real videos for the ZED 2i scene camera and the D405 wrist camera.

For every output frame, on a regular grid at --video-fps over the recording:
  * the sim arm is set to the robot's measured joints at that moment, converted back to the sim joint
    convention (the replay's joint0 offset and 360-degree shifts are removed);
  * the hand joints come from the trajectory state the replay was tracking; the target object keeps its
    recorded pose until the hand closes and afterwards its recorded offset to the TCP, so it stays in the
    hand of the arm at its measured pose;
  * base_camera, the ZED 2i view the scene was reconstructed from, renders the scene;
  * a wrist camera at the calibrated D405 pose in the `prehand` link renders the same state with an
    inverted 360-photo sphere around the workspace, since the reconstructed background ends at the table.
    The sphere is hidden while the scene camera renders.
The real frame is the recorded frame nearest in time. Nothing is simulated: states are set directly.

Run inside the simulation container (record_paired_replay.py does this after a run):
    python real_replay/render_paired_videos.py runs/paired/<recording> [--pano runs/360_photos/<photo>.jpg]

Writes pair_zed2i.mp4, pair_d405.mp4, sim_zed2i.mp4 and sim_d405.mp4 into the recording directory.
"""

import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from sim_scene import TrajectoryScene, gripper_events  # noqa: E402  (also sets up the sim environment)
import sapien  # noqa: E402
import sapien.render  # noqa: E402

# Pose of the real D405 in the `prehand` link: the ChArUco eye-in-hand calibration of
# offline-supervision-vla-rl (sim2real/real_replay/calibrate_wrist_handeye.py, 2026-09-11) moved 15 mm
# back along the optical axis, which matched real wrist frames best there (hand silhouette IoU 0.753,
# 2026-09-14). The orientation is for the colour frame rotated by np.rot90(k=-1). That repository mounts
# `prehand` exactly as this URDF does (gripper_mount xyz 0 0 0.0885, rpy 0 3.1415 0).
WRIST_CAMERA_LINK = "prehand"
WRIST_CAMERA_LOCAL_P = [0.00863, 0.06805, 0.08004]
WRIST_CAMERA_LOCAL_Q = [0.711969, -0.017879, 0.701977, 0.002846]
D405_RAW_W, D405_RAW_H = 640, 480
D405_DEFAULT_FX = 391.8          # colour stream at 640x480, same source; used when the recording has no intrinsics
D405_ROT90_K = -1
PANO_RADIUS_M = 20.0
DEFAULT_PANO_GLOB = "runs/360_photos/IMG_*_00_038.jpg"
PAIR_ZED_SIZE = (960, 540)       # per side, width x height
PAIR_D405_SIZE = (480, 640)      # per side, rotated D405 frame


def _parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help="directory written by record_paired_replay.py")
    ap.add_argument("--video-fps", type=float, default=15.0)
    ap.add_argument("--pano", default=None,
                    help=f"equirectangular 360 photo for the wrist camera (default {DEFAULT_PANO_GLOB}); "
                         "Insta360 .insp captures are converted with ffmpeg v360, see real_replay/README.md")
    ap.add_argument("--pano-yaw-deg", type=float, default=0.0, help="rotation of the photo about the vertical axis")
    ap.add_argument("--no-pano", action="store_true", help="render the wrist camera without the 360 photo")
    return ap.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO / path


def _pose_matrix(p7) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_quat([p7[4], p7[5], p7[6], p7[3]]).as_matrix()
    T[:3, 3] = p7[:3]
    return T


def _matrix_pose(T) -> np.ndarray:
    q = R.from_matrix(T[:3, :3]).as_quat()
    return np.array([*T[:3, 3], q[3], q[0], q[1], q[2]])


def _inverted_uv_sphere(radius: float, n_lat: int = 48, n_lon: int = 96):
    """UV sphere seen from inside with equirectangular UVs, v=0 at +Z (after offline-supervision-vla-rl pano_sphere.py)."""
    vv, uu = np.meshgrid(np.linspace(0.0, 1.0, n_lat + 1), np.linspace(0.0, 1.0, n_lon + 1), indexing="ij")
    phi, theta = vv * np.pi, uu * 2.0 * np.pi
    unit = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], -1).reshape(-1, 3)
    uvs = np.stack([1.0 - uu, vv], -1).reshape(-1, 2)          # u flipped: the photo is not mirrored from inside
    i, j = np.meshgrid(np.arange(n_lat), np.arange(n_lon), indexing="ij")
    a = (i * (n_lon + 1) + j).reshape(-1)
    b = a + n_lon + 1
    tris = np.concatenate([np.stack([a, b, a + 1], 1), np.stack([a + 1, b, b + 1], 1)])
    tris = np.concatenate([tris, tris[:, ::-1]])                 # both windings survive backface culling
    return ((unit * radius).astype(np.float32), tris.astype(np.uint32), (-unit).astype(np.float32),
            uvs.astype(np.float32))


def _add_pano_sphere(sub_scene, photo: Path, center, yaw_deg: float):
    texture = sapien.render.RenderTexture2D(filename=str(photo), mipmap_levels=4, srgb=True)
    material = sapien.render.RenderMaterial()
    # Lit by the scene lights from inside, emission off so the photo does not clip to white (as in pano_sphere.py).
    material.set_base_color([1.0, 1.0, 1.0, 1.0])
    material.set_base_color_texture(texture)
    material.set_emission([0.0, 0.0, 0.0, 1.0])
    material.set_metallic(0.0)
    material.set_roughness(1.0)
    material.set_specular(0.0)
    verts, tris, normals, uvs = _inverted_uv_sphere(PANO_RADIUS_M)
    body = sapien.render.RenderBodyComponent()
    body.attach(sapien.render.RenderShapeTriangleMesh(vertices=verts, triangles=tris, normals=normals, uvs=uvs,
                                                      material=material))
    entity = sapien.Entity()
    entity.name = "pano_sphere"
    entity.add_component(body)
    q = R.from_euler("z", yaw_deg, degrees=True).as_quat()
    entity.set_pose(sapien.Pose(p=[float(v) for v in center], q=[q[3], q[0], q[1], q[2]]))
    sub_scene.add_entity(entity)
    return body


def _hide_ground_visual(env_unwrapped) -> None:
    """Keep the physics floor but hide its checkerboard, which would cover the 360 photo (as in pano_sphere's env)."""
    for obj in getattr(getattr(env_unwrapped, "ground", None), "_objs", None) or []:
        body = obj.find_component_by_type(sapien.render.RenderBodyComponent)
        if body is not None:
            body.visibility = 0.0


def _add_wrist_camera(sub_scene, d405_meta: dict):
    fx = float(d405_meta.get("fx", D405_DEFAULT_FX))
    fy = float(d405_meta.get("fy", D405_DEFAULT_FX))
    cx = float(d405_meta.get("cx", (D405_RAW_W - 1) / 2))
    cy = float(d405_meta.get("cy", (D405_RAW_H - 1) / 2))
    width, height = D405_RAW_H, D405_RAW_W                      # the frame after np.rot90(k=-1)
    camera = sub_scene.add_camera("d405_wrist", width, height, float(2 * np.arctan(height / (2 * fx))), 0.01, 100.0)
    # np.rot90(k=-1) sends raw (column, row) to (H - 1 - row, column): the rotated frame's horizontal focal
    # length and principal point come from the raw vertical ones.
    camera.set_focal_lengths(fy, fx)
    camera.set_principal_point(D405_RAW_H - 1 - cy, cx)
    return camera


def _color_uint8(picture: np.ndarray) -> np.ndarray:
    """RGB uint8 from a camera 'Color' picture: SAPIEN 3.0.0b1 returns uint8 RGBA, float builds return 0..1."""
    rgb = np.asarray(picture)[..., :3]
    if rgb.dtype == np.uint8:
        return np.ascontiguousarray(rgb)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


class _FrameReader:
    """Nearest-timestamp frames of a recorded camera video, read forward only."""

    def __init__(self, video: Path, timestamps: Path):
        self.cap = cv2.VideoCapture(str(video))
        self.t = np.load(timestamps)
        count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count and count != len(self.t):
            print(f"[Render] WARNING: {video.name} has {count} frames but {len(self.t)} timestamps")
            self.t = self.t[:count]
        self.index, self.frame = -1, None

    def at(self, t: float):
        want = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        if want > 0 and abs(self.t[want - 1] - t) <= abs(self.t[want] - t):
            want -= 1
        while self.index < want:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.index, self.frame = self.index + 1, frame
        return self.frame


def _label(img: np.ndarray, text: str, org=(12, 34), scale: float = 0.9) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def _pair(sim_bgr: np.ndarray, real_bgr, size, footer: str) -> np.ndarray:
    sim = cv2.resize(sim_bgr, size, interpolation=cv2.INTER_AREA)
    real = cv2.resize(real_bgr, size, interpolation=cv2.INTER_AREA) if real_bgr is not None else np.zeros_like(sim)
    _label(sim, "SIM")
    _label(real, "REAL" if real_bgr is not None else "REAL: no frame")
    out = np.concatenate([sim, real], axis=1)
    _label(out, footer, org=(12, out.shape[0] - 14), scale=0.6)
    return out


class _Writers:
    def __init__(self, out_dir: Path, fps: float):
        self.out_dir, self.fps, self.writers = out_dir, fps, {}

    def write(self, name: str, frame: np.ndarray) -> None:
        if name not in self.writers:
            path = self.out_dir / f"{name}.mp4"
            size = (frame.shape[1], frame.shape[0])
            self.writers[name] = (path, cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size))
        self.writers[name][1].write(frame)

    def close(self) -> list:
        for _path, writer in self.writers.values():
            writer.release()
        return [path for path, _writer in self.writers.values()]


def _to_h264(path: Path) -> bool:
    """Re-encode OpenCV's mp4v output as H.264 so common players open it; keeps mp4v if that fails."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    tmp = path.with_suffix(".h264.mp4")
    result = subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(path),
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(tmp)])
    if result.returncode == 0 and tmp.is_file():
        tmp.replace(path)
        return True
    tmp.unlink(missing_ok=True)
    return False


def main() -> int:
    args = _parse()
    rec = _resolve(args.recording).resolve()
    meta = json.loads((rec / "meta.json").read_text(encoding="utf-8"))
    samples = np.load(rec / "samples.npz")
    t_s, joints_deg, tracked = samples["t"], samples["joints_deg"], samples["state"]
    if len(t_s) < 2:
        raise SystemExit(f"{rec}: fewer than two joint samples")
    traj = REPO / meta["trajectory_repo_relative"] if meta.get("trajectory_repo_relative") else Path(meta["trajectory"])

    readers = {}
    for name in ("zed2i", "d405"):
        video, stamps = rec / f"real_{name}.mp4", rec / f"real_{name}_t.npy"
        if video.is_file() and stamps.is_file():
            readers[name] = _FrameReader(video, stamps)
    if not readers:
        print("[Render] no real camera recordings; the REAL halves stay black")

    scene = TrajectoryScene(traj)
    sub_scene = scene.u.scene.sub_scenes[0]
    _hide_ground_visual(scene.u)
    wrist = _add_wrist_camera(sub_scene, meta.get("cameras", {}).get("d405", {}))
    wrist_local = sapien.Pose(p=WRIST_CAMERA_LOCAL_P, q=WRIST_CAMERA_LOCAL_Q)
    pano_body = None
    if not args.no_pano:
        photo = _resolve(args.pano) if args.pano else next(iter(sorted(REPO.glob(DEFAULT_PANO_GLOB))), None)
        if photo is None or not photo.is_file():
            print(f"[Render] WARNING: no 360 photo ({args.pano or DEFAULT_PANO_GLOB}); the wrist background stays black")
        else:
            center = np.asarray(scene.robot.pose.p.cpu()).reshape(-1)[:3]
            pano_body = _add_pano_sphere(sub_scene, photo, center, args.pano_yaw_deg)
            print(f"[Render] 360 photo {photo.name}, radius {PANO_RADIUS_M:g} m, yaw {args.pano_yaw_deg:g} deg")

    offset = np.array([meta.get("j0_offset_deg", 0.0), 0, 0, 0, 0, 0]) + np.asarray(meta.get("shifts_deg", [0.0] * 6))
    events = gripper_events(scene.data["action"])
    close_state = None if meta.get("reverse") else next((i for i, n in sorted(events.items()) if n == "close"), None)
    t0 = max([t_s[0]] + [r.t[0] for r in readers.values()])
    t1 = min([t_s[-1]] + [r.t[-1] for r in readers.values()])
    grid = np.arange(t0, t1, 1.0 / args.video_fps)
    if len(grid) == 0:
        raise SystemExit("joint samples and camera frames do not overlap in time")
    print(f"[Render] {rec.name}: {len(grid)} frames at {args.video_fps:g} fps over {t1 - t0:.1f} s; "
          f"cameras {sorted(readers) or 'none'}; mode {meta.get('mode')}; hand close at state {close_state}")

    writers = _Writers(rec, args.video_fps)
    n_states = len(scene.qpos)
    try:
        for i, t in enumerate(grid):
            joints = np.array([np.interp(t, t_s, joints_deg[:, j]) for j in range(6)]) - offset
            k = int(np.clip(round(float(np.interp(t, t_s, tracked))), 0, n_states - 1))
            qpos = scene.qpos[k].astype(np.float64).copy()
            qpos[:6] = np.radians(joints)
            scene.set_robot_qpos(qpos)
            if close_state is not None and k >= close_state:
                grip = np.linalg.inv(_pose_matrix(scene.tcp_pose[k])) @ _pose_matrix(scene.object_pose[k])
                scene.set_object_pose(_matrix_pose(_pose_matrix(scene.tcp_pose7()) @ grip))
            else:
                scene.set_object_pose(scene.object_pose[k])

            if pano_body is not None:
                pano_body.visibility = 0.0
            sim_zed = cv2.cvtColor(scene.capture_scene_camera(), cv2.COLOR_RGB2BGR)
            if pano_body is not None:
                pano_body.visibility = 1.0
            link = scene.link_pose7(WRIST_CAMERA_LINK)
            wrist.entity.set_pose(sapien.Pose(p=link[:3], q=link[3:]) * wrist_local)
            sub_scene.update_render()
            wrist.take_picture()
            sim_d405 = cv2.cvtColor(_color_uint8(wrist.get_picture("Color")), cv2.COLOR_RGB2BGR)

            real_zed = readers["zed2i"].at(t) if "zed2i" in readers else None
            real_d405 = readers["d405"].at(t) if "d405" in readers else None
            if real_d405 is not None:
                real_d405 = np.ascontiguousarray(np.rot90(real_d405, k=D405_ROT90_K))
            footer = f"t {t - t0:6.2f} s   state {k}   {meta.get('mode')}"
            writers.write("pair_zed2i", _pair(sim_zed, real_zed, PAIR_ZED_SIZE, footer))
            writers.write("pair_d405", _pair(sim_d405, real_d405, PAIR_D405_SIZE, footer))
            writers.write("sim_zed2i", cv2.resize(sim_zed, PAIR_ZED_SIZE, interpolation=cv2.INTER_AREA))
            writers.write("sim_d405", cv2.resize(sim_d405, PAIR_D405_SIZE, interpolation=cv2.INTER_AREA))
            if i % 50 == 0 or i == len(grid) - 1:
                print(f"[Render] frame {i + 1}/{len(grid)}  state {k}", flush=True)
    finally:
        paths = writers.close()
        scene.close()
    for path in paths:
        print(f"[Render] {path}{'' if _to_h264(path) else '  (mp4v, H.264 re-encode unavailable)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
