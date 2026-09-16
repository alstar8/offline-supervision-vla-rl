#!/usr/bin/env python3
"""
Find a coloured object on the table with the fixed ZED 2i and compare its position with
where a sim scene places it — to put the real object where the sim expects it.

The camera pose comes from the scene's scene.json (camera_opencv_to_world). Use the
metric-corrected scene key: the original reconstruction is ~9% too small, so real depth
mapped through its camera pose would not match its object positions.

The robot is not touched; the camera is only read.

Usage (host, needs pyzed):
    python3 real_replay/locate_object_zed.py                     # green_cube_ext, metric scene
    python3 real_replay/locate_object_zed.py --object yellow_cube_ext --hsv 20 40 100 255 80 255
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]

# HSV ranges (OpenCV: H 0..180) for the object-bank cubes.
DEFAULT_HSV = {
    "green_cube_ext": (35, 85, 90, 255, 30, 255),
    "yellow_cube_ext": (20, 35, 100, 255, 80, 255),
    "blue_cube_ext": (95, 125, 90, 255, 40, 255),
    "orange_cube_ext": (5, 18, 120, 255, 80, 255),
}


def _parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", default="airy_table_scene14sep26_left_image_metric")
    ap.add_argument("--object", default="green_cube_ext")
    ap.add_argument("--hsv", type=int, nargs=6, default=None, metavar=("HLO", "HHI", "SLO", "SHI", "VLO", "VHI"))
    ap.add_argument("--half-height", type=float, default=0.025, help="object centre height above the table (m)")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--save", default=None, help="annotated image path (default runs/calibration/locate_<object>.png)")
    return ap.parse_args()


def _find(node, key):
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


def _capture(frames: int):
    import pyzed.sl as sl
    zed = sl.Camera(); ip = sl.InitParameters()
    ip.camera_resolution = sl.RESOLUTION.HD2K
    ip.depth_mode = sl.DEPTH_MODE.QUALITY
    ip.coordinate_units = sl.UNIT.METER
    ip.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE      # x right, y down, z forward (OpenCV)
    ip.depth_minimum_distance = 0.3
    ip.sdk_verbose = 0
    # The first open after the camera has been idle sometimes fails with CAMERA FAILED TO
    # SETUP / STREAM FAILED TO START and succeeds on the next attempt.
    import time
    for attempt in range(4):
        err = zed.open(ip)
        if err == sl.ERROR_CODE.SUCCESS:
            break
        print(f"ZED open attempt {attempt + 1}: {err}", flush=True)
        time.sleep(3)
    else:
        raise SystemExit(f"ZED open failed: {err}")
    try:
        rt = sl.RuntimeParameters(); xyz = sl.Mat(); img = sl.Mat(); clouds = []
        for i in range(frames + 25):
            if zed.grab(rt) != sl.ERROR_CODE.SUCCESS or i < 25:
                continue
            zed.retrieve_measure(xyz, sl.MEASURE.XYZ)
            clouds.append(xyz.get_data()[:, :, :3].astype(np.float32).copy())
        zed.retrieve_image(img, sl.VIEW.LEFT)
        cal = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        K = np.array([[cal.fx, 0, cal.cx], [0, cal.fy, cal.cy], [0, 0, 1]])
        bgr = cv2.cvtColor(img.get_data(), cv2.COLOR_BGRA2BGR)
    finally:
        zed.close()
    with np.errstate(all="ignore"):
        cloud = np.nanmedian(np.stack(clouds), axis=0)
    return bgr, cloud, K


def _table_plane(cloud, n_prior):
    P = cloud.reshape(-1, 3)
    P = P[np.isfinite(P).all(1) & (P[:, 2] > 0.3) & (P[:, 2] < 1.5)]
    rng = np.random.default_rng(0)
    Q = P[rng.choice(len(P), min(len(P), 200000), replace=False)]
    best = (0, None, None)
    for _ in range(500):
        a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n); n = n if n @ n_prior > 0 else -n
        if n @ n_prior < np.cos(np.radians(15)):
            continue
        k = (np.abs((Q - a) @ n) < 0.004).sum()
        if k > best[0]:
            best = (k, n, a)
    _, n, a = best
    sel = np.abs((Q - a) @ n) < 0.004
    p0 = Q[sel].mean(0); n = np.linalg.svd(Q[sel] - p0)[2][2]
    up = n if n @ (-p0) > 0 else -n                          # toward the camera
    return p0, up


def main() -> int:
    args = _parse()
    scene_path = REPO / "assets/scenes" / args.key / "simulation/scene.json"
    scene = json.loads(scene_path.read_text())
    T_w_cam = np.array(scene["camera"]["camera_opencv_to_world"], float)
    cfg = yaml.safe_load((REPO / "config/config_debug.yaml").read_text())["local"][args.key]
    placements = _find(cfg, "object_placements") or {}
    if args.object not in placements:
        raise SystemExit(f"{args.object} not in object_placements of {args.key}")
    table_z = float(scene["groundplane_in_sim"]["point"][2])
    target_w = np.array([*placements[args.object]["position"][:2], table_z + args.half_height])
    base = _find(cfg, "robot_base_pose")
    R_base = R.from_quat([base[4], base[5], base[6], base[3]]).as_matrix()

    hsv_rng = args.hsv or DEFAULT_HSV.get(args.object)
    if hsv_rng is None:
        raise SystemExit(f"no default HSV range for {args.object}; pass --hsv")

    bgr, cloud, K = _capture(args.frames)
    n_prior = np.array(scene["groundplane_in_cam"]["normal"], float); n_prior /= np.linalg.norm(n_prior)
    p0, up = _table_plane(cloud, n_prior)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo, hi = np.array(hsv_rng[0::2]), np.array(hsv_rng[1::2])
    mask = cv2.inRange(hsv, lo, hi) > 0
    height = np.where(np.isfinite(cloud).all(-1), (cloud - p0) @ up, np.nan)
    mask &= (height > 0.005) & (height < 2.5 * args.half_height + 0.01)   # sits on the table, not the arm
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_cc <= 1:
        raise SystemExit("object not found: no pixels in the HSV range sitting on the table")
    cc = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[cc, cv2.CC_STAT_AREA] < 300:
        raise SystemExit(f"object not found: largest blob only {stats[cc, cv2.CC_STAT_AREA]} px")
    pts = cloud[labels == cc]; pts = pts[np.isfinite(pts).all(1)]
    # Project visible surface points onto the table, then lift to the centre height.
    foot = pts - np.outer((pts - p0) @ up, up)
    centre_cam = np.median(foot, axis=0) + up * args.half_height
    centre_w = (T_w_cam @ np.append(centre_cam, 1))[:3]

    d_w = centre_w - target_w
    d_base = R_base.T @ d_w                                   # sim robot base axes (URDF)
    def to_px(pw):
        pc = np.linalg.inv(T_w_cam) @ np.append(pw, 1); uv = K @ pc[:3]; return uv[:2] / uv[2]
    px_real, px_target = to_px(centre_w), to_px(target_w)

    print(f"scene key : {args.key}  (table z {table_z:.4f} m)")
    print(f"object    : {args.object}  blob {stats[cc, cv2.CC_STAT_AREA]} px, {len(pts)} 3D points")
    print(f"real      : world {np.round(centre_w, 4).tolist()} m   pixel {np.round(px_real, 1).tolist()}")
    print(f"sim target: world {np.round(target_w, 4).tolist()} m   pixel {np.round(px_target, 1).tolist()}")
    print(f"offset real - target:")
    print(f"  world XY : dx={1000*d_w[0]:+.1f} dy={1000*d_w[1]:+.1f} mm   |dxy|={1000*np.hypot(d_w[0], d_w[1]):.1f} mm")
    print(f"  robot base axes: forward(+x)={1000*d_base[0]:+.1f} left(+y)={1000*d_base[1]:+.1f} mm")
    print(f"  image    : du={px_real[0]-px_target[0]:+.1f} px (right +), dv={px_real[1]-px_target[1]:+.1f} px (down +)")

    vis = bgr.copy()
    cnts, _ = cv2.findContours((labels == cc).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 3)
    cv2.circle(vis, tuple(int(v) for v in px_real), 10, (0, 255, 255), -1)
    cv2.drawMarker(vis, tuple(int(v) for v in px_target), (0, 0, 255), cv2.MARKER_CROSS, 60, 4)
    cv2.putText(vis, "sim target", (int(px_target[0]) + 20, int(px_target[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    save = Path(args.save) if args.save else REPO / "runs/calibration" / f"locate_{args.object}.png"
    save.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save), vis)
    print(f"image     : {save}  (yellow = detected, red cross = sim target)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
