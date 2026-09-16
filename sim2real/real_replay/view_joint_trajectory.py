#!/usr/bin/env python3
"""
Show a recorded joint trajectory (joint_trajectory.npz from export_sim_joint_trajectory.py) in
the simulator.

The scene is rebuilt from the runtime config embedded in the run's rl4vla_raw_episode*.npz, and
every frame sets the recorded 22 robot joints and the target object's recorded pose. Nothing is
simulated, so what you see is exactly the recorded states. The states shown are chosen as in the
real-robot replay (replay_joints_real.py): idle settle skipped, --until-height, --until-step,
--reverse.

Why not `make replay`: it steps the rl4vla NPZ actions open-loop, but that NPZ holds only part of
the planner's env steps, so the replayed run diverges and misses the grasp.

Run inside the simulation container (or `make view-joints TRAJ=...`):
    python real_replay/view_joint_trajectory.py <run>/joint_trajectory.npz [--until-height 0.15] [--speed 0.5]
    python real_replay/view_joint_trajectory.py <run>/joint_trajectory.npz --video <out>.mp4    # headless

Viewer keys: n autoplay (from the first state again once finished), SPACE pause / one state
forward, r back to the first state, q quit.
"""

import sys
import json
import time
import signal
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sim_scene import TrajectoryScene, gripper_events, npz_viewer  # noqa: E402  (also sets up the sim environment)

# State selection shared with the real-robot replay.
_spec = importlib.util.spec_from_file_location("replay_joints_real", HERE / "replay_joints_real.py")
_joint_replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_joint_replay)
signal.signal(signal.SIGINT, signal.default_int_handler)   # that module installs a robot E-stop handler


def _parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trajectory", help="joint_trajectory.npz")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed multiplier on the sim timeline (1.0 = control_freq states per second)")
    ap.add_argument("--until-height", type=float, default=None,
                    help="stop before the grasp once the sim TCP is this high above the table (m), as the real replay")
    ap.add_argument("--until-step", type=int, default=None, help="stop at this trajectory state index")
    ap.add_argument("--reverse", action="store_true", help="play from the stop state back to the start")
    ap.add_argument("--from-start", action="store_true", help="include the idle settle states at the start")
    ap.add_argument("--autoplay", action="store_true", help="start playing as soon as the viewer opens")
    ap.add_argument("--video", default=None, help="render the base camera headless into this .mp4 instead of a viewer")
    ap.add_argument("--sim-backend", default="physx_cpu",
                    help="states are set directly and never stepped; the CPU backend needs no GPU buffer syncs")
    ap.add_argument("--render-backend", default="gpu")
    ap.add_argument("--window-width", type=int, default=1920)
    ap.add_argument("--window-height", type=int, default=1080)
    return ap.parse_args()


def _select_states(args, traj: Path, meta: dict):
    plan_args = SimpleNamespace(trajectory=str(traj), scene_key=meta.get("key"), j0_offset=None,
                                until_height=args.until_height, until_step=args.until_step, reverse=args.reverse)
    try:
        plan = _joint_replay._plan(plan_args)
    except SystemExit:
        # No real-robot joint calibration for this scene; the sim joints are shown unchanged anyway.
        plan_args.j0_offset = 0.0
        plan = _joint_replay._plan(plan_args)
    path = plan["path"]
    if args.from_start:
        path = list(range(plan["stop"], -1, -1)) if args.reverse else list(range(0, plan["stop"] + 1))
    return plan, path


def main() -> int:
    args = _parse()
    if args.speed <= 0:
        raise SystemExit("--speed must be > 0")
    traj = Path(args.trajectory).resolve()
    _joint_replay._check_trajectory_file(str(traj))
    data = np.load(traj, allow_pickle=True)
    meta = json.loads(str(data["meta_json"]))
    plan, path = _select_states(args, traj, meta)
    hz, tcp_h, events = plan["hz"], plan["tcp_h"], gripper_events(data["action"])
    obj_pose = data["object_pose_world"]

    print(f"[View] trajectory : {traj}")
    print(f"[View] states     : {path[0]} -> {path[-1]} ({len(path)} of {len(data['qpos'])}); stop: {plan['reason']}")
    print(f"[View] timing     : {hz:g} Hz x speed {args.speed:g} -> {(len(path) - 1) / hz / args.speed:.1f} s")
    print(f"[View] hand events: {events or 'none'}")

    scene = TrajectoryScene(traj, render_mode="rgb_array" if args.video else "human", sim_backend=args.sim_backend,
                            render_backend=args.render_backend, window=(args.window_width, args.window_height))
    print(f"[View] scene      : {scene.context.key}  object {scene.context.task_object_id}  "
          f"(config embedded in {scene.episode.name})")
    try:
        def log_state(i: int, force: bool = False) -> None:
            k = path[i]
            if force or k in events or i % 20 == 0 or i == len(path) - 1:
                note = f"  <- hand {events[k]}" if k in events else ""
                obj_z = f"  object z {obj_pose[k, 2]:.3f} m" if np.isfinite(obj_pose[k]).all() else ""
                print(f"[View] state {k:4d} ({i + 1}/{len(path)})  TCP {100 * tcp_h[k]:5.1f} cm above table{obj_z}{note}",
                      flush=True)

        if args.video:
            return _write_video(args, scene, path, hz, log_state)

        i = 0
        scene.set_state(path[i])
        viewer = scene.env.render()
        if viewer is None:
            raise RuntimeError("Viewer initialization returned None.")
        viewer.paused = False
        log_state(i, force=True)
        print("Viewer keys: n autoplay | SPACE pause / one state forward | r first state | q quit")
        autoplay, key_states, last = bool(args.autoplay), {}, time.time()
        frame_dt = 1.0 / (hz * args.speed)

        def advance() -> bool:
            nonlocal i
            if i >= len(path) - 1:
                print("[View] Trajectory finished. n plays it again, r returns to the first state.")
                return False
            i += 1
            scene.set_state(path[i])
            log_state(i)
            return True

        while viewer is not None and not getattr(viewer, "closed", True):
            npz_viewer._refresh_viewer(scene.env, viewer)
            if npz_viewer._viewer_key_pressed_once(viewer, key_states, "q"):
                break
            if npz_viewer._viewer_key_pressed_once(viewer, key_states, "r"):
                i, autoplay = 0, False
                scene.set_state(path[i])
                log_state(i, force=True)
            if npz_viewer._viewer_key_pressed_once(viewer, key_states, "n"):
                if i >= len(path) - 1:
                    i = 0
                    scene.set_state(path[i])
                    log_state(i, force=True)
                autoplay, last = True, time.time()
            if npz_viewer._viewer_key_pressed_once(viewer, key_states, " "):
                if autoplay:
                    autoplay = False
                    print(f"[View] Paused at state {path[i]}.")
                else:
                    advance()
            if autoplay and time.time() - last >= frame_dt:
                autoplay = advance()
                last = time.time()
            time.sleep(0.005)
        return 0
    finally:
        scene.close()


def _write_video(args, scene, path, hz, log_state) -> int:
    import imageio
    out = Path(args.video)
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(round(hz * args.speed)))
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=7, macro_block_size=1)
    try:
        for i, k in enumerate(path):
            scene.set_state(k)
            frame = scene.capture_scene_camera()
            h, w = frame.shape[:2]
            writer.append_data(np.ascontiguousarray(frame[: h - h % 2, : w - w % 2]))   # libx264 needs even sizes
            log_state(i)
    finally:
        writer.close()
    print(f"[View] video: {out}  ({len(path)} frames at {fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
