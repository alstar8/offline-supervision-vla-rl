#!/usr/bin/env python3
"""
Run a joint trajectory on the real RC5 + AeroHand while recording the ZED 2i scene camera and the
D405 wrist camera, then render the same moments in the simulator and write paired sim | real videos.

Recording runs on the host (aero-env python). The robot is driven by replay_joints_real.py; every
control cycle logs the measured joints and the trajectory state being tracked, and every camera frame
gets a monotonic timestamp. Rendering runs in the simulation container (render_paired_videos.py): the
sim arm is placed at the measured joints, the hand and the object at the tracked state, and both the
scene camera and a wrist camera at the calibrated D405 pose are rendered, with a 360 photo of the lab
around the scene.

Usage (host):
    /home/aermakov/github/aero-env/bin/python real_replay/record_paired_replay.py <run>/joint_trajectory.npz
    ... --until-height 0.15            # replay options are passed to replay_joints_real.py unchanged
    ... --camera-test                  # no robot, no hand: cameras record while the sim follows the plan
    ... --no-render                    # record only; render later with render_paired_videos.py

Output: runs/paired/<trajectory run>__<time>/ with real_zed2i.mp4, real_d405.mp4 (+ *_t.npy frame
timestamps), samples.npz, meta.json and, after rendering, pair_zed2i.mp4 and pair_d405.mp4.
"""

import os
import sys
import json
import time
import queue
import argparse
import threading
import subprocess
import importlib.util
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CONTAINER = "sim2real-simulation"
CONTAINER_REPO = Path("/app")

_spec = importlib.util.spec_from_file_location("replay_joints_real", HERE / "replay_joints_real.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)

D405_WIDTH, D405_HEIGHT = 640, 480


class _VideoSink:
    """Encodes frames on its own thread; a frame and its timestamp are kept or dropped together."""

    def __init__(self, path: Path, fps: float, size: tuple):
        self.path = path
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot open video writer {path}")
        self.queue = queue.Queue(maxsize=64)
        self.timestamps, self.dropped = [], 0
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"sink-{path.stem}")
        self.thread.start()

    def push(self, t: float, frame: np.ndarray) -> None:
        try:
            self.queue.put_nowait((t, frame))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                return
            t, frame = item
            self.writer.write(frame)
            self.timestamps.append(t)

    def close(self) -> int:
        self.queue.put(None)
        self.thread.join()
        self.writer.release()
        np.save(self.path.with_name(self.path.stem + "_t.npy"), np.asarray(self.timestamps, dtype=np.float64))
        return len(self.timestamps)


class _Capture:
    """Grabs a camera continuously on its own thread and writes frames only while recording."""

    name = "camera"

    def __init__(self, out_dir: Path, fps: int):
        self.out_dir, self.fps = out_dir, fps
        self.meta, self.sink, self.thread = {}, None, None
        self.recording, self.closed = False, False

    def start_thread(self, width: int, height: int) -> None:
        self.sink = _VideoSink(self.out_dir / f"real_{self.name}.mp4", self.fps, (width, height))
        self.thread = threading.Thread(target=self._run, daemon=True, name=self.name)
        self.thread.start()

    def close(self) -> int:
        self.recording, self.closed = False, True
        if self.thread is not None:
            self.thread.join(timeout=5)
        self._release()
        frames = self.sink.close() if self.sink is not None else 0
        self.meta.update(frames=frames, dropped=self.sink.dropped if self.sink is not None else 0)
        return frames

    def _run(self):
        raise NotImplementedError

    def _release(self):
        pass


class _ZedCapture(_Capture):
    name = "zed2i"

    def open(self) -> None:
        import pyzed.sl as sl
        self.sl = sl
        self.zed = sl.Camera()
        params = sl.InitParameters()
        params.camera_resolution = sl.RESOLUTION.HD2K
        params.camera_fps = int(self.fps)
        params.depth_mode = sl.DEPTH_MODE.NONE
        params.sdk_verbose = 0
        # The first open after the camera has been idle often fails (CAMERA FAILED TO SETUP /
        # STREAM FAILED TO START) and succeeds on the next attempt.
        for attempt in range(4):
            err = self.zed.open(params)
            if err == sl.ERROR_CODE.SUCCESS:
                break
            print(f"[Record] ZED open attempt {attempt + 1}: {err}", flush=True)
            time.sleep(3)
        else:
            raise RuntimeError(f"ZED 2i open failed: {err}")
        info = self.zed.get_camera_information()
        cfg = info.camera_configuration
        cam = cfg.calibration_parameters.left_cam
        width, height = int(cfg.resolution.width), int(cfg.resolution.height)
        self.meta = dict(serial=int(info.serial_number), width=width, height=height, fps=float(cfg.fps),
                         fx=float(cam.fx), fy=float(cam.fy), cx=float(cam.cx), cy=float(cam.cy), view="left rectified")
        self.start_thread(width, height)

    def _run(self) -> None:
        sl = self.sl
        runtime, image = sl.RuntimeParameters(), sl.Mat()
        while not self.closed:
            if self.zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                time.sleep(0.005)
                continue
            t = time.monotonic()
            if self.recording:
                self.zed.retrieve_image(image, sl.VIEW.LEFT)
                self.sink.push(t, cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR))

    def _release(self) -> None:
        self.zed.close()


class _D405Capture(_Capture):
    name = "d405"

    def open(self) -> None:
        import pyrealsense2 as rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, D405_WIDTH, D405_HEIGHT, rs.format.bgr8, int(self.fps))
        profile = self.pipeline.start(config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.meta = dict(serial=profile.get_device().get_info(rs.camera_info.serial_number),
                         width=D405_WIDTH, height=D405_HEIGHT, fps=float(self.fps),
                         fx=float(intr.fx), fy=float(intr.fy), cx=float(intr.ppx), cy=float(intr.ppy),
                         view="raw colour stream; the sim wrist camera matches it after np.rot90(k=-1)")
        self.start_thread(D405_WIDTH, D405_HEIGHT)

    def _run(self) -> None:
        while not self.closed:
            try:
                frames = self.pipeline.wait_for_frames(1000)
            except RuntimeError:
                continue
            t = time.monotonic()
            color = frames.get_color_frame()
            if self.recording and color:
                self.sink.push(t, np.asanyarray(color.get_data()).copy())

    def _release(self) -> None:
        self.pipeline.stop()


class PairedRecorder:
    """Hook object for replay_joints_real.main: cameras, joint samples and metadata of one run."""

    def __init__(self, out_dir: Path, trajectory: Path, mode: str, zed_fps: int, d405_fps: int,
                 use_zed: bool = True, use_d405: bool = True):
        self.out_dir, self.trajectory, self.mode = out_dir, trajectory, mode
        self.captures = ([_ZedCapture(out_dir, zed_fps)] if use_zed else []) + \
                        ([_D405Capture(out_dir, d405_fps)] if use_d405 else [])
        self.samples, self.lock = [], threading.Lock()
        self.plan_meta, self.t_start, self.t_end = {}, None, None
        self.opened = self.stopped = False

    def open(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for cap in self.captures:
            cap.open()
            print(f"[Record] {cap.name} open: {cap.meta}", flush=True)
        self.opened = True

    def start(self, plan: dict, args) -> None:
        self.plan_meta = dict(
            j0_offset_deg=float(args.j0_offset), shifts_deg=[float(v) for v in plan["shifts"]],
            path_first=int(plan["path"][0]), path_last=int(plan["path"][-1]), control_hz=float(plan["hz"]),
            speed=float(args.speed), reverse=bool(args.reverse), until_height=args.until_height,
            until_step=args.until_step, stop_reason=plan["reason"],
            hand_events=[[int(i), n] for i, n in plan["events"]])
        self.t_start = time.monotonic()
        for cap in self.captures:
            cap.recording = True
        print(f"[Record] recording to {self.out_dir}", flush=True)

    def sample(self, t: float, joints, tcp, state: float) -> None:
        row = [t, float(state), *[float(v) for v in joints]]
        row += [float(v) for v in tcp] if tcp is not None else [np.nan] * 6
        with self.lock:
            self.samples.append(row)

    def stop(self, completed: bool) -> None:
        if self.stopped or not self.opened:
            return
        self.stopped = True
        self.t_end = time.monotonic()
        cameras = {}
        for cap in self.captures:
            cap.close()
            cameras[cap.name] = cap.meta
        with self.lock:
            rows = np.asarray(self.samples, dtype=np.float64).reshape(-1, 14)
        np.savez(self.out_dir / "samples.npz", t=rows[:, 0], state=rows[:, 1], joints_deg=rows[:, 2:8],
                 tcp_ctrl=rows[:, 8:14])
        try:
            traj_rel = str(self.trajectory.resolve().relative_to(REPO))
        except ValueError:
            traj_rel = None
        meta = dict(mode=self.mode, completed=bool(completed), created=time.strftime("%Y-%m-%d %H:%M:%S"),
                    trajectory=str(self.trajectory.resolve()), trajectory_repo_relative=traj_rel,
                    t_start=self.t_start, t_end=self.t_end, samples=int(len(rows)), cameras=cameras, **self.plan_meta)
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        frames = ", ".join(f"{k} {v.get('frames')} frames ({v.get('dropped')} dropped)" for k, v in cameras.items())
        print(f"[Record] stopped: {len(rows)} joint samples, {frames}; completed={completed}", flush=True)

    @property
    def n_samples(self) -> int:
        return len(self.samples)


def _parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trajectory", help="joint_trajectory.npz")
    ap.add_argument("--out", default=None, help="output directory (default runs/paired/<trajectory run>__<time>)")
    ap.add_argument("--camera-test", action="store_true",
                    help="no robot and no hand: record the cameras while the sim follows the planned timeline")
    ap.add_argument("--no-zed", action="store_true")
    ap.add_argument("--no-d405", action="store_true")
    ap.add_argument("--zed-fps", type=int, default=15, help="ZED 2i frame rate at HD2K (15 is the maximum)")
    ap.add_argument("--d405-fps", type=int, default=30)
    ap.add_argument("--no-render", action="store_true", help="record only")
    ap.add_argument("--video-fps", type=float, default=15.0, help="frame rate of the paired videos")
    ap.add_argument("--pano", default=None, help="360 photo for the sim wrist camera (repo-relative), see render_paired_videos.py")
    return ap.parse_known_args()


def _camera_test(rec: PairedRecorder, replay_argv: list) -> int:
    """Record the cameras for the planned duration while samples follow the plan exactly."""
    rargs = replay._parse(replay_argv)
    replay._check_trajectory_file(rargs.trajectory)
    plan = replay._plan(rargs)
    path, hz = plan["path"], plan["hz"]
    if len(path) < 2:
        raise SystemExit("trajectory selection has fewer than two states")
    replay._report(rargs, plan)
    q = plan["q_real"][np.array(path)]
    rec.open()
    input("Cameras open. ENTER starts the camera test (the robot is not connected and does not move)...")
    rec.start(plan, rargs)
    length = (len(path) - 1) / hz
    t0, completed = time.monotonic(), False
    try:
        while not replay._stop:
            t = (time.monotonic() - t0) * rargs.speed
            if t >= length:
                completed = True
                break
            x = t * hz
            i = min(int(x), len(path) - 2)
            a = x - i
            rec.sample(time.monotonic(), q[i] + a * (q[i + 1] - q[i]), None, path[i] + a * (path[i + 1] - path[i]))
            time.sleep(1.0 / 30.0)
    finally:
        rec.stop(completed)
    return 0 if completed else 1


def _render_in_container(out_dir: Path, video_fps: float, pano: str | None) -> int:
    try:
        rel = out_dir.resolve().relative_to(REPO)
    except ValueError:
        print(f"[Record] {out_dir} is outside the repository; render it manually with render_paired_videos.py")
        return 1
    target = CONTAINER_REPO / rel
    cmd = (f"cd {CONTAINER_REPO} && PYTHONPATH={CONTAINER_REPO} /opt/conda/bin/python real_replay/render_paired_videos.py "
           f"{target} --video-fps {video_fps:g}" + (f" --pano {pano}" if pano else "") +
           f"; rc=$?; chown -R {os.getuid()}:{os.getgid()} {target}; exit $rc")
    print(f"[Record] rendering in {CONTAINER}: {target}", flush=True)
    return subprocess.call(["docker", "exec", CONTAINER, "bash", "-lc", cmd])


def main() -> int:
    args, replay_options = _parse()
    traj = Path(args.trajectory).resolve()
    run_name = traj.parent.name.removeprefix("proxy_ee_delta_headless_")
    out_dir = Path(args.out) if args.out else REPO / "runs" / "paired" / f"{run_name}__{time.strftime('%Y%m%d_%H%M%S')}"
    rec = PairedRecorder(out_dir, traj, "camera_test" if args.camera_test else "robot",
                         args.zed_fps, args.d405_fps, use_zed=not args.no_zed, use_d405=not args.no_d405)
    replay_argv = [str(traj)] + replay_options
    rc = _camera_test(rec, replay_argv) if args.camera_test else replay.main(replay_argv, recorder=rec)
    if not rec.opened or rec.n_samples == 0:
        print("[Record] nothing was recorded (dry run, refused start or aborted before motion).")
        return rc
    if not args.no_render:
        render_rc = _render_in_container(out_dir, args.video_fps, args.pano)
        if render_rc != 0:
            print(f"[Record] rendering failed ({render_rc}); the recording is kept in {out_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
