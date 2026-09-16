#!/usr/bin/env python3
"""
Run the proxy planner headless for one object and record the joint trajectory the simulated
arm actually follows, for joint-space replay on the real robot.

Why not step the saved NPZ's actions through the env: those actions are only part of what
the closed-loop planner fed the env, and replaying them open-loop does not reproduce the
grasp (tried 2026-09-15: replayed success=0 for a run that had succeeded). Recording during
the planner run captures every env.step — settle, approach, close, lift — of the run whose
success is reported.

The runner is executed in-process; rc5_unified_proxy_setup.make_env is wrapped so the env
it creates records state on every step(). The runner code is not modified.

Run inside the simulation container:
    docker exec sim2real-simulation bash -lc 'cd /app && PYTHONPATH=/app \\
      /opt/conda/bin/python real_replay/export_sim_joint_trajectory.py \\
      --key airy_table_scene14sep26_left_image_metric --object green_cube_ext'

Writes <planner run dir>/joint_trajectory.npz with, for N env steps:
    qpos              float64 [N+1, 22]  arm joint0..5 then 16 hand joints (rad, active-joint order)
    tcp_pose_world    float64 [N+1, 7]   right_tcp_link p xyz + q wxyz
    object_pose_world float64 [N+1, 7]   target object p + q (NaN if the actor is not found)
    action            float32 [N, 7]     action passed to env.step; gripper channel is column 6
    joint_names, control_freq, source_npz, instruction, meta_json
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app")
from openreal2sim.simulation.maniskill.scripts import rc5_unified_proxy_setup as unified_setup
from openreal2sim.simulation.maniskill.scripts import run_rc5_unified as runner


def _parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", default="airy_table_scene14sep26_left_image_metric")
    ap.add_argument("--object", default="green_cube_ext")
    ap.add_argument("--task", default="pick_up")
    ap.add_argument("--config", default="config/config_debug.yaml")
    ap.add_argument("--descend-offset-dz", type=float, default=0.0,
                    help="raise the object's descend_offset_xyz z by this many metres for this run only. "
                         "The real hand sits ~1 cm below plan; +0.015 gave the first safe real grasp.")
    ap.add_argument("--approach-yaw-deg", type=float, default=0.0,
                    help="rotate the object's grasp calibration (pregrasp/descend offsets and target_quat, all "
                         "world-frame) about the vertical through the object, for this run only. The grasp on a "
                         "cube is unchanged at multiples of 90 deg; only the side the arm approaches from changes.")
    ap.add_argument("--viewer", action="store_true", help="run with the SAPIEN viewer instead of headless")
    return ap.parse_args()


def _np(value):
    return np.asarray(value.cpu() if hasattr(value, "cpu") else value)


def _pose7(pose) -> np.ndarray:
    return np.concatenate([_np(pose.p).reshape(-1)[:3], _np(pose.q).reshape(-1)[:4]]).astype(np.float64)


class _Recorder:
    def __init__(self, object_id: str):
        self.object_id = object_id
        self.qpos, self.tcp, self.obj, self.actions = [], [], [], []
        self.joint_names, self.control_freq, self.object_actor_name = None, None, None
        self._obj_actor, self._looked_up = None, False

    def _object_actor(self, u):
        if not self._looked_up:
            self._looked_up = True
            short = self.object_id.removesuffix("_ext")
            actors = getattr(u.scene, "actors", {}) or {}
            for cand in (f"object_{short}", self.object_id, short):
                if cand in actors:
                    self._obj_actor, self.object_actor_name = actors[cand], cand
                    break
            if self._obj_actor is None:
                print(f"[Export] object actor for {self.object_id!r} not found among {sorted(actors)[:20]}", flush=True)
        return self._obj_actor

    def snapshot(self, env):
        u = env.unwrapped
        robot = u.agent.robot
        if self.joint_names is None:
            self.joint_names = [j.name for j in robot.active_joints]
            self.control_freq = float(getattr(u, "control_freq", 0) or 0)
        self.qpos.append(_np(robot.get_qpos()).reshape(-1).astype(np.float64))
        self.tcp.append(_pose7(u.agent.tcp.pose))
        actor = self._object_actor(u)
        self.obj.append(_pose7(actor.pose) if actor is not None else np.full(7, np.nan))

    def wrap(self, env):
        original_step = env.step

        def step(action, *a, **kw):
            if not self.qpos:
                self.snapshot(env)                      # state before the first action
            out = original_step(action, *a, **kw)
            self.actions.append(_np(action).astype(np.float32).reshape(-1)[:7])
            self.snapshot(env)
            return out

        env.step = step
        return env


def _robot_base_pose(config_path: str, key: str):
    """robot_base_pose the run used; the joint replay derives the joint0 convention from its yaw."""
    import yaml
    try:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        return [float(v) for v in cfg["local"][key]["simulation"]["robot_base_pose"]]
    except Exception:
        return None


def main() -> int:
    args = _parse()
    scene = f"assets/scenes/{args.key}/simulation/scene.json"
    if not args.key.endswith("_metric"):
        print(f"[Export] WARNING: {args.key} is not metric-corrected; its geometry is ~9% too small for the real robot.")
    config_path, tmp_config, descend_used, approach_used = args.config, None, None, None
    if args.descend_offset_dz or args.approach_yaw_deg:
        import yaml
        from scipy.spatial.transform import Rotation
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        cal = cfg["local"][args.key]["simulation"].get("planner_object_calibrations", {}).get(args.object, {})
        needed = ["descend_offset_xyz"] + (["pregrasp_offset_xyz", "target_quat"] if args.approach_yaw_deg else [])
        missing = [k for k in needed if k not in cal]
        if missing:
            print(f"[Export] {args.object} has no {missing} in {args.key}; cannot apply the calibration overrides.")
            return 1
        if args.approach_yaw_deg:
            rz = Rotation.from_euler("z", args.approach_yaw_deg, degrees=True)
            for field in ("pregrasp_offset_xyz", "descend_offset_xyz"):
                cal[field] = [round(float(v), 12) for v in rz.apply(cal[field])]
            w, x, y, z = cal["target_quat"]                                    # wxyz, as ManiSkill poses
            qx, qy, qz, qw = (rz * Rotation.from_quat([x, y, z, w])).as_quat()
            cal["target_quat"] = [round(float(v), 12) for v in (qw, qx, qy, qz)]
            approach_used = {"yaw_deg": args.approach_yaw_deg, "pregrasp_offset_xyz": cal["pregrasp_offset_xyz"],
                             "target_quat": cal["target_quat"]}
            print(f"[Export] approach rotated {args.approach_yaw_deg:+g} deg: {approach_used}", flush=True)
        descend_used = [float(cal["descend_offset_xyz"][0]), float(cal["descend_offset_xyz"][1]),
                        round(float(cal["descend_offset_xyz"][2]) + args.descend_offset_dz, 6)]
        cal["descend_offset_xyz"] = descend_used
        tmp_config = Path("config") / f"_tmp_joint_export_{args.key}_{args.object}.yaml"
        tmp_config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        config_path = str(tmp_config)
        print(f"[Export] descend_offset_xyz for this run: {descend_used} (+{args.descend_offset_dz} m)", flush=True)
    rec = _Recorder(args.object)

    original_make_env = unified_setup.make_env
    unified_setup.make_env = lambda *a, **kw: rec.wrap(original_make_env(*a, **kw))

    runs = Path("runs/manual")
    before = {p.name for p in runs.iterdir()} if runs.exists() else set()
    t0 = time.time()
    sys.argv = ["run_rc5_unified.py", "--motion_backend", "proxy_ee_delta", "--config_path", config_path,
                "--key", args.key, "--scene", scene, "--task_type", args.task, "--task_object_id", args.object]
    if not args.viewer:
        sys.argv.append("--headless")
    try:
        exit_code = runner.main()
    except SystemExit as e:
        exit_code = e.code
    finally:
        unified_setup.make_env = original_make_env
        if tmp_config is not None:
            tmp_config.unlink(missing_ok=True)

    if not rec.actions:
        print("[Export] no env.step calls were recorded — the runner did not use the wrapped env.")
        return 1

    new_dirs = sorted((p for p in runs.iterdir() if p.is_dir() and p.name not in before
                       and args.key in p.name and args.object in p.name and p.stat().st_mtime >= t0 - 1),
                      key=lambda p: p.stat().st_mtime)
    if not new_dirs:
        print("[Export] planner run directory not found.")
        return 1
    run_dir = new_dirs[-1]
    raw = sorted(run_dir.glob("rl4vla_raw_episode*.npz"))
    success = any("success" in p.name for p in raw)

    qpos, tcp, obj = np.stack(rec.qpos), np.stack(rec.tcp), np.stack(rec.obj)
    actions = np.stack(rec.actions)
    meta = {
        "recorded_via": "planner in-process, wrapped env.step",
        "key": args.key, "object_id": args.object, "run_dir": str(run_dir),
        "planner_exit_code": exit_code, "planner_success": success,
        "env_steps": int(len(actions)), "control_freq": rec.control_freq,
        "object_actor": rec.object_actor_name,
        "config": args.config, "descend_offset_dz_m": args.descend_offset_dz, "descend_offset_xyz_used": descend_used,
        "robot_base_pose": _robot_base_pose(args.config, args.key),
        "approach_override": approach_used,
        "object_lift_m": None if np.isnan(obj).all() else float(obj[-1, 2] - obj[0, 2]),
        "wall_time_s": round(time.time() - t0, 1),
    }
    out = run_dir / "joint_trajectory.npz"
    np.savez_compressed(
        out, qpos=qpos, tcp_pose_world=tcp, object_pose_world=obj, action=actions,
        joint_names=np.array(rec.joint_names), control_freq=rec.control_freq,
        source_npz=str(raw[0]) if raw else "", instruction=f"{args.task}:{args.object}",
        meta_json=json.dumps(meta),
    )
    arm = np.degrees(qpos[:, :6])
    print("### EXPORT ###")
    print(f"saved: {out}")
    print(f"env steps: {len(actions)}  states: {len(qpos)}  control_freq: {rec.control_freq}")
    print(f"arm start (deg): {np.round(arm[0], 2).tolist()}")
    print(f"arm end   (deg): {np.round(arm[-1], 2).tolist()}")
    print("meta:", json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
