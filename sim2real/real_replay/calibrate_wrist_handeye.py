#!/usr/bin/env python3
"""Eye-in-hand calibration of the D405 wrist camera against a ChArUco board.

Solves for the transform from the robot flange (the frame the RC5 reports as
TCP) to the camera optical frame, which is what WRIST_CAMERA_LOCAL_P /
WRIST_CAMERA_LOCAL_Q in the sim env should encode.

    # 1. print a board (skip if you already have one)
    calibrate_wrist_handeye.py board --out /tmp/charuco.png

    # 2. clamp the board in the workspace, then jog the arm by hand and shoot
    calibrate_wrist_handeye.py capture --dir runs/handeye_20260910

    # 3. solve
    calibrate_wrist_handeye.py solve --dir runs/handeye_20260910

Images are stored raw, exactly as the D405 delivers them. The rot90 that
eval_openvla_real.py applies is a model-input concern and must not leak in
here: calibration lives in the camera's own frame.

Orientation convention for the RC5 TCP is extrinsic xyz degrees, matching
replay_npz_real.py, which validates it against live robot poses.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

RS_W, RS_H = 640, 480
CAMERA_FPS = 30
RC5_IP = "10.10.10.10"

# The lab board: 25 mm checkers, 18 mm DICT_5X5 markers.
DEFAULT_DICT = "DICT_5X5_50"
DEFAULT_SQUARES_X = 11
DEFAULT_SQUARES_Y = 7
DEFAULT_SQUARE_MM = 25.0
DEFAULT_MARKER_MM = 18.0

HAND_EYE_METHODS = {
    "tsai": "CALIB_HAND_EYE_TSAI",
    "park": "CALIB_HAND_EYE_PARK",
    "horaud": "CALIB_HAND_EYE_HORAUD",
    "andreff": "CALIB_HAND_EYE_ANDREFF",
    "daniilidis": "CALIB_HAND_EYE_DANIILIDIS",
}


def _board(args):
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    return cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_mm / 1000.0,
        args.marker_mm / 1000.0,
        dictionary,
    )


def _add_board_args(parser):
    parser.add_argument("--dict", default=DEFAULT_DICT, help="cv2.aruco predefined dictionary name")
    parser.add_argument("--squares-x", type=int, default=DEFAULT_SQUARES_X)
    parser.add_argument("--squares-y", type=int, default=DEFAULT_SQUARES_Y)
    parser.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM)
    parser.add_argument("--marker-mm", type=float, default=DEFAULT_MARKER_MM)


def cmd_board(args) -> int:
    import cv2

    board = _board(args)
    px_per_mm = args.dpi / 25.4
    w = int(round(args.squares_x * args.square_mm * px_per_mm))
    h = int(round(args.squares_y * args.square_mm * px_per_mm))
    img = board.generateImage((w, h), marginSize=int(round(10 * px_per_mm)))
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"wrote {out}  ({w}x{h} px at {args.dpi} dpi)")
    print(
        f"print at 100% scale, then verify one square measures {args.square_mm:.1f} mm "
        "with a ruler -- a scaling error here becomes a scale error in the result"
    )
    return 0


def _init_realsense():
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, CAMERA_FPS)
    profile = pipeline.start(cfg)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]], dtype=float
    )
    dist = np.array(intr.coeffs, dtype=float).reshape(1, -1)
    return pipeline, camera_matrix, dist, intr


def _grab_bgr(pipeline) -> np.ndarray:
    frames = pipeline.wait_for_frames()
    return np.asanyarray(frames.get_color_frame().get_data())


def _pose_change(a: list[float], b: list[float]) -> tuple[float, float]:
    """Translation (mm) and rotation (deg) between two RC5 TCP poses."""
    from scipy.spatial.transform import Rotation

    moved_mm = float(np.linalg.norm(np.subtract(b[:3], a[:3]))) * 1000.0
    ra = Rotation.from_euler("xyz", a[3:6], degrees=True)
    rb = Rotation.from_euler("xyz", b[3:6], degrees=True)
    return moved_mm, float(np.degrees((ra.inv() * rb).magnitude()))


def _make_announcer(speak: bool):
    """Spoken cues for auto mode.

    stdout of this process goes to whoever launched it, which is not necessarily
    the person standing at the robot, so the cadence is announced out loud.
    """
    import shutil
    import subprocess

    exe = shutil.which("spd-say") if speak else None
    if speak and exe is None:
        print("[warn] --speak requested but spd-say was not found; falling back to stdout only")

    def say(text: str) -> None:
        print(f"  >> {text}", flush=True)
        if exe is not None:
            try:
                subprocess.run([exe, "-w", "-r", "20", text], check=False, timeout=10)
            except Exception:
                pass

    return say


def _connect_rc5(robot_ip: str, read_only: bool):
    """Connect for pose readback.

    Read-only by default: this SDK cannot command freedrive (motion.mode.set
    accepts only move/move_adv/pause/hold), so the arm is hand-guided from the
    pendant while this runs. Taking the control session would fight it, and
    powering the servos would make the arm impossible to move by hand.
    """
    import eval_openvla_real as ev

    if not read_only:
        return ev._init_rc5(robot_ip)
    api_root = str(ev._resolve_python_api())
    if api_root not in sys.path:
        sys.path.insert(0, api_root)
    from API.rc_api import RobotApi

    return RobotApi(robot_ip, read_only=True, show_std_traceback=True)


def cmd_capture(args) -> int:
    import cv2

    out_dir = Path(args.dir).expanduser().resolve()
    (out_dir / "shots").mkdir(parents=True, exist_ok=True)

    print("Starting RealSense D405...")
    pipeline, camera_matrix, dist, intr = _init_realsense()
    (out_dir / "intrinsics.json").write_text(
        json.dumps(
            {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy,
                "model": str(intr.model),
                "coeffs": list(intr.coeffs),
            },
            indent=2,
        )
    )
    print(f"  fx={intr.fx:.1f} fy={intr.fy:.1f} ppx={intr.ppx:.1f} ppy={intr.ppy:.1f}")

    print(f"Connecting to RC5 ({'read-only' if not args.take_control else 'control'})...")
    robot = _connect_rc5(args.robot_ip, read_only=not args.take_control)
    mode = robot.motion.mode.get()
    print(f"  motion mode: {mode}")
    if mode != "freedrive" and not args.take_control:
        print(
            f"[info] motion mode is '{mode}': hand-guiding needs freedrive on the pendant, but "
            "jogging from the pendant works from here too. Either way, reach a new pose and "
            "hold still for a shot."
        )

    board = _board(args)
    detector = cv2.aruco.CharucoDetector(board)

    # Identical robot poses carry no hand-eye information at all: with the arm
    # held in place every A_i is the same and X drops out of AX = XB.
    accepted_tcps: list[list[float]] = []

    def take(shot: int) -> int:
        """Grab one frame; return the shot index after a successful save."""
        for _ in range(5):  # let auto-exposure settle
            _grab_bgr(pipeline)
        bgr = _grab_bgr(pipeline)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _corners, charuco_ids, _, _ = detector.detectBoard(gray)
        n = 0 if charuco_ids is None else len(charuco_ids)
        if n < args.min_corners:
            print(f"  rejected: {n} charuco corners < --min-corners {args.min_corners}")
            return shot
        tcp = [float(v) for v in robot.motion.linear.get_actual_position(orientation_units="deg")]
        joints = [float(v) for v in robot.motion.joint.get_actual_position(units="deg")]
        if accepted_tcps:
            moved_mm, turned_deg = _pose_change(accepted_tcps[-1], tcp)
            if moved_mm < args.min_move_mm and turned_deg < args.min_turn_deg:
                print(
                    f"  rejected: arm barely moved since the last shot "
                    f"({moved_mm:.1f} mm, {turned_deg:.1f} deg); need >= {args.min_turn_deg:.0f} deg "
                    f"or {args.min_move_mm:.0f} mm -- move the arm (pendant jog or freedrive)"
                )
                return shot
        accepted_tcps.append(tcp)
        cv2.imwrite(str(out_dir / "shots" / f"shot_{shot:03d}.png"), bgr)
        (out_dir / "shots" / f"shot_{shot:03d}.json").write_text(
            json.dumps({"tcp_m_deg": tcp, "joints_deg": joints, "charuco_corners": int(n)}, indent=2)
        )
        print(f"  shot {shot:03d}: {n} corners, tcp={[round(v, 4) for v in tcp[:3]]}")
        return shot + 1

    shot = 0
    if args.auto:
        say = _make_announcer(args.speak)
        print(
            f"\nAuto mode: {args.shots} shots. Cadence per shot: {args.interval:.0f} s to\n"
            f"reposition, then {args.hold:.0f} s holding still, then the frame is taken.\n"
            "VARY THE ORIENTATION between shots, not just the position -- hand-eye leaves\n"
            "the translation unobservable without rotation diversity.\n"
        )
        say(f"starting in {args.warmup:.0f} seconds")
        time.sleep(args.warmup)
        attempts = 0
        while shot < args.shots and attempts < args.shots * 3:
            attempts += 1
            say(f"move. shot {shot + 1} of {args.shots}")
            remaining = args.interval
            while remaining > 3.5:
                time.sleep(1.0)
                remaining -= 1.0
            for count in range(int(remaining), 0, -1):
                say(str(count))
                time.sleep(1.0)
            say("hold still")
            time.sleep(args.hold)
            before = shot
            shot = take(shot)
            say("good" if shot > before else "rejected, see the reason above")
    elif args.on_settle:
        # Shoot whenever the arm has reached a new pose and stopped there. Fits
        # pendant jogging as well as freedrive: no cadence to keep up with, and
        # a settled arm means no motion blur.
        print(
            f"\nOn-settle mode: collecting {args.shots} shots. Move the arm to a new pose\n"
            f"(>= {args.min_turn_deg:.0f} deg or {args.min_move_mm:.0f} mm from the last shot) and stop;\n"
            f"a frame is taken once it has been still for {args.settle_sec:.1f} s.\n"
            "Rotate the wrist 20-40 deg about DIFFERENT axes between shots.\n"
        )
        deadline = time.monotonic() + args.timeout
        prev = None
        still_since = time.monotonic()
        waiting_printed = False
        while shot < args.shots and time.monotonic() < deadline:
            tcp = [float(v) for v in robot.motion.linear.get_actual_position(orientation_units="deg")]
            if prev is not None:
                d_mm, d_deg = _pose_change(prev, tcp)
                if d_mm > 0.5 or d_deg > 0.2:
                    still_since = time.monotonic()
                    waiting_printed = False
            prev = tcp
            far_enough = True
            if accepted_tcps:
                m_mm, m_deg = _pose_change(accepted_tcps[-1], tcp)
                far_enough = m_mm >= args.min_move_mm or m_deg >= args.min_turn_deg
            settled = time.monotonic() - still_since >= args.settle_sec
            if settled and far_enough:
                print(f"  >> settled at a new pose -- shooting {shot + 1} of {args.shots}", flush=True)
                shot = take(shot)
                still_since = time.monotonic()
            elif settled and not waiting_printed:
                print(
                    f"  >> waiting: move the arm (need >= {args.min_turn_deg:.0f} deg "
                    f"or {args.min_move_mm:.0f} mm from the last shot)", flush=True
                )
                waiting_printed = True
            time.sleep(0.1)
        if shot < args.shots:
            print(f"[warn] stopped after {args.timeout:.0f} s with {shot} of {args.shots} shots")
    else:
        print(
            "\nJog the arm so the board fills a good part of the wrist view, then press ENTER to shoot.\n"
            "Vary the ORIENTATION between shots, not just the position -- hand-eye needs rotation\n"
            "diversity or the translation stays unobservable. 'q' + ENTER to finish.\n"
        )
        while True:
            reply = input(f"[{shot} captured] ENTER = shoot, q = done: ").strip().lower()
            if reply == "q":
                break
            shot = take(shot)

    pipeline.stop()
    print(f"\n{shot} shots in {out_dir / 'shots'}")
    if shot < args.min_shots:
        print(f"[warn] hand-eye wants at least {args.min_shots} well-spread poses")
    return 0


def _pose_to_Rt(tcp_m_deg: list[float]) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    # Matches replay_npz_real.py, which validates this convention against the robot.
    rot = Rotation.from_euler("xyz", tcp_m_deg[3:6], degrees=True)
    return rot.as_matrix(), np.asarray(tcp_m_deg[:3], dtype=float).reshape(3, 1)


def cmd_solve(args) -> int:
    import cv2
    from scipy.spatial.transform import Rotation

    in_dir = Path(args.dir).expanduser().resolve()
    intr = json.loads((in_dir / "intrinsics.json").read_text())
    camera_matrix = np.array(
        [[intr["fx"], 0.0, intr["ppx"]], [0.0, intr["fy"], intr["ppy"]], [0.0, 0.0, 1.0]], dtype=float
    )
    dist = np.array(intr["coeffs"], dtype=float).reshape(1, -1)

    board = _board(args)
    detector = cv2.aruco.CharucoDetector(board)

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    used, skipped = [], []
    for img_path in sorted((in_dir / "shots").glob("shot_*.png")):
        meta = json.loads(img_path.with_suffix(".json").read_text())
        gray = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)
        if ids is None or len(ids) < args.min_corners:
            skipped.append((img_path.name, 0 if ids is None else len(ids)))
            continue
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist)
        if not ok:
            skipped.append((img_path.name, -1))
            continue
        Rb, tb = _pose_to_Rt(meta["tcp_m_deg"])
        R_g2b.append(Rb)
        t_g2b.append(tb)
        R_t2c.append(cv2.Rodrigues(rvec)[0])
        t_t2c.append(np.asarray(tvec, dtype=float).reshape(3, 1))
        used.append((img_path.name, len(ids)))

    print(f"usable shots: {len(used)}")
    for name, n in skipped:
        print(f"  skipped {name}: {n} corners")
    if len(used) < 3:
        print("need at least 3 usable shots (and realistically 10+ with varied orientation)")
        return 1

    print(f"\n{'method':12s} {'t_cam2gripper (mm)':>28s}   {'rpy xyz (deg)':>26s}")
    results = {}
    for name, attr in HAND_EYE_METHODS.items():
        try:
            R_c2g, t_c2g = cv2.calibrateHandEye(
                R_g2b, t_g2b, R_t2c, t_t2c, method=getattr(cv2, attr)
            )
        except cv2.error as exc:
            print(f"{name:12s} failed: {exc}")
            continue
        rpy = Rotation.from_matrix(R_c2g).as_euler("xyz", degrees=True)
        t_mm = t_c2g.ravel() * 1000.0
        results[name] = (R_c2g, t_c2g)
        print(
            f"{name:12s} "
            f"[{t_mm[0]:8.2f} {t_mm[1]:8.2f} {t_mm[2]:8.2f}]   "
            f"[{rpy[0]:8.2f} {rpy[1]:8.2f} {rpy[2]:8.2f}]"
        )

    if not results:
        return 1
    spread_mm = np.ptp(
        np.array([t.ravel() for _, t in results.values()]) * 1000.0, axis=0
    )
    print(f"\nspread across methods: {spread_mm.round(2).tolist()} mm")
    print(
        "A spread of more than a few mm means the poses were not diverse enough; "
        "add shots with the wrist rotated, not just translated."
    )

    R_c2g, t_c2g = results[args.method]
    quat_xyzw = Rotation.from_matrix(R_c2g).as_quat()
    quat_wxyz = [float(quat_xyzw[3]), *(float(v) for v in quat_xyzw[:3])]
    out = {
        "method": args.method,
        "shots_used": len(used),
        "t_cam2gripper_m": [float(v) for v in t_c2g.ravel()],
        "R_cam2gripper": R_c2g.tolist(),
        "quat_wxyz_cam2gripper": quat_wxyz,
        "note": (
            "Frame is the RC5-reported TCP. WRIST_CAMERA_LOCAL_P/Q are expressed in the "
            "URDF 'prehand' link, so compose this with the TCP->prehand transform from the "
            "robot URDF before editing the sim constants."
        ),
    }
    (in_dir / "handeye_result.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {in_dir / 'handeye_result.json'}")
    print("\nIn the RC5 TCP frame:")
    print(f"  p = {[round(v, 5) for v in out['t_cam2gripper_m']]}")
    print(f"  q (wxyz) = {[round(v, 6) for v in quat_wxyz]}")
    print(
        "\nThis is NOT yet WRIST_CAMERA_LOCAL_P/Q: those are relative to the URDF 'prehand'\n"
        "link. Compose with TCP->prehand before touching openr2s_ms_env.py."
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("board", help="generate a printable ChArUco board")
    _add_board_args(p)
    p.add_argument("--out", default="/tmp/charuco.png")
    p.add_argument("--dpi", type=float, default=300.0)
    p.set_defaults(func=cmd_board)

    p = sub.add_parser("capture", help="shoot board images together with robot poses")
    _add_board_args(p)
    p.add_argument("--dir", required=True)
    p.add_argument("--robot-ip", default=RC5_IP)
    p.add_argument("--min-corners", type=int, default=8)
    p.add_argument("--min-shots", type=int, default=10)
    p.add_argument("--auto", action="store_true", help="shoot on a timer instead of on ENTER")
    p.add_argument("--shots", type=int, default=20, help="auto mode: shots to collect")
    p.add_argument("--interval", type=float, default=3.0, help="auto mode: seconds between shots")
    p.add_argument("--warmup", type=float, default=10.0, help="auto mode: delay before the first shot")
    p.add_argument("--hold", type=float, default=2.0, help="auto mode: seconds to hold still before each shot")
    p.add_argument("--speak", action="store_true", help="announce the cadence out loud via spd-say")
    p.add_argument("--on-settle", action="store_true",
                   help="shoot automatically each time the arm stops at a new pose")
    p.add_argument("--settle-sec", type=float, default=1.5,
                   help="on-settle mode: how long the arm must be still before a shot")
    p.add_argument("--timeout", type=float, default=900.0,
                   help="on-settle mode: give up after this many seconds")
    p.add_argument("--min-turn-deg", type=float, default=5.0,
                   help="reject a shot unless the wrist turned at least this much since the last one...")
    p.add_argument("--min-move-mm", type=float, default=15.0,
                   help="...or moved at least this far")
    p.add_argument("--take-control", action="store_true",
                   help="power the servos instead of connecting read-only (blocks hand-guiding)")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("solve", help="run hand-eye on captured shots")
    _add_board_args(p)
    p.add_argument("--dir", required=True)
    p.add_argument("--min-corners", type=int, default=8)
    p.add_argument("--method", choices=sorted(HAND_EYE_METHODS), default="park")
    p.set_defaults(func=cmd_solve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
