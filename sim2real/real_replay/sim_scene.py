"""
Rebuild the simulator scene of a recorded joint trajectory and set recorded states directly.

The scene comes from the runtime config embedded in the run's rl4vla_raw_episode*.npz, so it is the
scene the trajectory was recorded in. States are written straight into the articulation and the
target object; physics is never stepped. Used by view_joint_trajectory.py and render_paired_videos.py
inside the simulation container.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# Importing the NPZ viewer sets the display/GPU environment and registers the RC5 agents and envs.
from openreal2sim.simulation.maniskill.scripts import rc5_replay_rl4vla_npz_viewer as npz_viewer  # noqa: E402
import torch  # noqa: E402
from mani_skill.utils.structs.pose import Pose  # noqa: E402

GRIPPER_HOLD_EPS = 1e-4   # RCPresetHandController hold_epsilon, as in replay_npz_real.py
CURRENT_CONFIG = HERE.parent / "config" / "config_debug.yaml"


def current_hand_visual_profile():
    """global.simulation.hand_visual_profile of the current config (None if it is unset)."""
    import yaml
    cfg = yaml.safe_load(CURRENT_CONFIG.read_text(encoding="utf-8")) or {}
    return ((cfg.get("global") or {}).get("simulation") or {}).get("hand_visual_profile")


def gripper_events(action: np.ndarray) -> dict:
    """{state index: 'open' | 'close'} where the gripper command changes."""
    events, cmd = {}, "hold"
    for i, g in enumerate(action[:, 6]):
        new = "close" if g < -GRIPPER_HOLD_EPS else ("open" if g > GRIPPER_HOLD_EPS else None)
        if new is not None and new != cmd:
            events[i] = new
            cmd = new
    return events


class TrajectoryScene:
    """OpenReal2Sim env rebuilt for one joint_trajectory.npz, with direct state setters."""

    def __init__(self, traj, *, render_mode="rgb_array", sim_backend="physx_cpu", render_backend="gpu",
                 window=(1920, 1080), hand_profile="current"):
        """hand_profile: "current" uses the recording's hand_visual_profile, or the current config's
        when the recording predates hand materials; "recorded" keeps exactly what was recorded."""
        self.traj = Path(traj).resolve()
        self.data = np.load(self.traj, allow_pickle=True)
        self.meta = json.loads(str(self.data["meta_json"]))
        episodes = sorted(self.traj.parent.glob("rl4vla_raw_episode*.npz"), key=lambda p: "success" not in p.name)
        if not episodes:
            raise SystemExit(f"No rl4vla_raw_episode*.npz next to {self.traj}: its embedded runtime config rebuilds the scene.")
        self.episode = episodes[0]
        ctx_args = argparse.Namespace(
            npz_path=str(self.episode), config_path=None, scene=None, key=self.meta.get("key"),
            task_object_id=self.meta.get("object_id"), task_type=None, instruction=None,
            object_placements_json_path=None, control_mode=None, render_backend=render_backend,
            sim_backend=sim_backend, window_width=window[0], window_height=window[1])
        self.context = npz_viewer._load_replay_context(ctx_args)
        env_kwargs, _sim_cfg, _hand_pose_cfg, _startup_cfg = npz_viewer._build_env_kwargs(self.context, ctx_args)
        env_kwargs["render_mode"] = render_mode
        if env_kwargs.get("hand_visual_profile") is None and hand_profile == "current":
            env_kwargs["hand_visual_profile"] = current_hand_visual_profile()
            if env_kwargs["hand_visual_profile"] is not None:
                print(f"[Scene] recording has no hand_visual_profile; using "
                      f"{env_kwargs['hand_visual_profile'].get('name')} from {CURRENT_CONFIG.name}")
        self.env = npz_viewer.envs.OpenReal2SimEnv(**env_kwargs)
        self.env.reset(seed=0, options=dict(reconfigure=True))
        self.u = self.env.unwrapped
        self.robot = self.u.agent.robot
        self.device = self.u.device

        names = [j.name for j in self.robot.active_joints]
        recorded = [str(n) for n in self.data["joint_names"]]
        if names != recorded:
            raise SystemExit(f"Robot joints differ from the recording:\n  env      {names}\n  recorded {recorded}")
        base = self.meta.get("robot_base_pose")
        if base is not None:
            env_q = np.asarray(self.robot.pose.q.cpu()).reshape(-1)
            env_yaw = np.degrees(2 * np.arctan2(env_q[3], env_q[0]))
            rec_yaw = np.degrees(2 * np.arctan2(base[6], base[3]))
            if abs((env_yaw - rec_yaw + 180) % 360 - 180) > 0.1:
                print(f"[Scene] WARNING: robot base yaw {env_yaw:.2f} deg in the rebuilt scene, recorded {rec_yaw:.2f} deg")
        short = str(self.meta.get("object_id", "")).removesuffix("_ext")
        self.actor = self.u.scene.actors.get(self.meta.get("object_actor") or f"object_{short}")
        if self.actor is None:
            print(f"[Scene] WARNING: object actor for {self.meta.get('object_id')} not found; only the robot is animated")

    @property
    def qpos(self) -> np.ndarray:
        return self.data["qpos"]

    @property
    def object_pose(self) -> np.ndarray:
        return self.data["object_pose_world"]

    @property
    def tcp_pose(self) -> np.ndarray:
        return self.data["tcp_pose_world"]

    def _tensor(self, values):
        return torch.as_tensor(np.asarray(values, dtype=np.float32), device=self.device)[None]

    def _sync(self) -> None:
        if getattr(self.u, "gpu_sim_enabled", False):
            self.u.scene._gpu_apply_all()
            self.u.scene.px.gpu_update_articulation_kinematics()
            self.u.scene._gpu_fetch_all()

    def set_robot_qpos(self, qpos) -> None:
        self.robot.set_qpos(self._tensor(qpos))
        self.robot.set_qvel(torch.zeros((1, len(qpos)), dtype=torch.float32, device=self.device))
        self._sync()

    def set_object_pose(self, pose7) -> None:
        if self.actor is None or not np.isfinite(pose7).all():
            return
        self.actor.set_pose(Pose.create_from_pq(p=self._tensor(pose7[:3]), q=self._tensor(pose7[3:])))
        self._sync()

    def set_state(self, k: int) -> None:
        """Recorded robot joints and object pose of trajectory state k."""
        self.set_robot_qpos(self.qpos[k])
        self.set_object_pose(self.object_pose[k])

    @staticmethod
    def _pose7(pose) -> np.ndarray:
        return np.concatenate([np.asarray(pose.p.cpu()).reshape(-1)[:3], np.asarray(pose.q.cpu()).reshape(-1)[:4]])

    def link_pose7(self, name: str) -> np.ndarray:
        """World pose (p xyz, q wxyz) of a robot link in the current state."""
        return self._pose7(self.robot.links_map[name].pose)

    def tcp_pose7(self) -> np.ndarray:
        return self._pose7(self.u.agent.tcp.pose)

    def capture_scene_camera(self) -> np.ndarray:
        """RGB uint8 frame of base_camera, the ZED 2i view the scene was reconstructed from."""
        return npz_viewer._capture_base_camera_frame(self.env)

    def close(self) -> None:
        self.env.close()
