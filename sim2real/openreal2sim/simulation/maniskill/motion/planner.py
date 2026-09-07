import os
import time
import inspect
import mplib
import numpy as np
from pathlib import Path
import sapien
import torch
import trimesh
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.pose import \
    to_sapien_pose as to_sapien_pose_mani_skill
from transforms3d import quaternions
from transforms3d.quaternions import mat2quat, qinverse, qmult

try:
    from ..planner_core.grasp_state import (
        infer_gripper_state_from_hand_target,
        restore_planner_grasp_state,
        save_planner_grasp_state,
    )
except ImportError:
    from openreal2sim.simulation.maniskill.planner_core.grasp_state import (
        infer_gripper_state_from_hand_target,
        restore_planner_grasp_state,
        save_planner_grasp_state,
    )

_Y = "\033[33m"
_R = "\033[0m"


def _log_planner_dispatch(message: str) -> None:
    print(f"{_Y}[PLANNER_DISPATCH] {message}{_R}")


def _select_first_env_pose(pose):
    if hasattr(pose, "raw_pose"):
        raw_pose = pose.raw_pose
        if isinstance(raw_pose, torch.Tensor) and raw_pose.ndim > 1:
            return Pose(raw_pose=raw_pose[0])
    return pose


class BaseMotionPlanningSolver:
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        record_frames=None,  # List to append frames to for video recording
        normalize_frame_fn=None,  # Function to normalize frame before appending
        record_frames_from="render",  # "render" or "base_camera" - use sensor image directly
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.normalize_frame_fn = normalize_frame_fn
        self.record_frames_from = record_frames_from
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self.base_pose = to_sapien_pose_mani_skill(base_pose)

        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info

        self.elapsed_steps = 0

        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None
        
        self.record_frames = record_frames  # Store frames list for video recording

    def _batch_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        num_envs = getattr(self.base_env.scene, "num_envs", 1)
        if num_envs and num_envs > 1 and action.ndim == 1:
            action = np.repeat(action[None, :], num_envs, axis=0)
        return action

    def _get_record_frame(self, obs=None):
        """Get frame for video recording. Uses base_camera sensor or render."""
        if self.record_frames_from == "base_camera" and obs is not None:
            # ManiSkill rgbd: sensor_data[uid]["rgb"] or image[uid]
            sd = obs.get("sensor_data", obs.get("image", {}))
            bc = sd.get("base_camera", {})
            frame = bc.get("rgb", bc.get("Color"))
            if frame is not None and hasattr(frame, "cpu"):
                frame = frame.cpu().numpy()
            if frame is not None and frame.ndim == 4:
                frame = frame[0]
            if frame is not None and frame.shape[-1] == 4:
                frame = frame[..., :3]
            return frame
        return self.env.render()

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        try:
            viewer = self.base_env.render_human()
            if viewer is None or not hasattr(viewer, 'window') or viewer.window is None:
                # Viewer is not available (e.g., in interactive mode or rgb_array mode)
                # Skip waiting for key press
                return
            auto_continue_time = getattr(self, "auto_continue_time", None)
            if auto_continue_time is not None:
                timeout_s = max(float(auto_continue_time), 0.0)
                if timeout_s <= 0.0:
                    self.base_env.render_human()
                    return
                deadline = time.monotonic() + timeout_s
                while time.monotonic() < deadline:
                    if viewer.window.key_down("c"):
                        break
                    self.base_env.render_human()
                    time.sleep(0.01)
                return
            print("Press [c] to continue")
            while True:
                if viewer.window.key_down("c"):
                    break
                self.base_env.render_human()
        except (AttributeError, RuntimeError):
            # Viewer window may be closed or not available in interactive mode
            return

    def setup_planner(self):
        move_group = self._get_move_group()
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self._get_planning_urdf_path(),
            srdf=self._get_planning_srdf_path(),
            new_package_keyword="",
            use_convex=False,
            user_link_names=link_names,
            user_joint_names=joint_names,
            joint_vel_limits=None,
            joint_acc_limits=None,
            objects=[],
            move_group=move_group,
            verbose=False,
        )
        self._apply_modern_mplib_adjacent_link_acm(planner)
        base_pose_arr = np.hstack([self.base_pose.p, self.base_pose.q])
        planner.set_base_pose(self._to_mplib_goal_pose(base_pose_arr, planner_obj=planner))
        print(f"[PLANNER_IK_DEBUG] setup_planner: base_pose passed to mplib: p={base_pose_arr[:3]}, q={base_pose_arr[3:7]}")
        planner.joint_vel_limits = (
            np.asarray(planner.joint_vel_limits) * self.joint_vel_limits
        )
        planner.joint_acc_limits = (
            np.asarray(planner.joint_acc_limits) * self.joint_acc_limits
        )
        self._log_mplib_runtime(planner)
        return planner

    def _log_mplib_runtime(self, planner) -> None:
        mplib_version = getattr(mplib, "__version__", "<unknown>")
        mplib_file = getattr(mplib, "__file__", "<unknown>")
        planner_cls = type(planner).__name__
        has_legacy_pose_planner = callable(getattr(planner, "plan_qpos_to_pose", None))
        has_modern_pose_planner = callable(getattr(planner, "plan_pose", None))
        has_plan_qpos = callable(getattr(planner, "plan_qpos", None))
        ik_fn = getattr(planner, "IK", None)
        if not callable(ik_fn):
            raise RuntimeError("Installed mplib Planner does not expose IK(); cannot log planner capabilities.")
        try:
            ik_signature = inspect.signature(ik_fn)
            ik_supports_return_closest = "return_closest" in ik_signature.parameters
        except (TypeError, ValueError):
            ik_supports_return_closest = False
        pose_api = "legacy" if has_legacy_pose_planner else "modern" if has_modern_pose_planner else "missing"
        print(
            "[PLANNER_MPLIB_RUNTIME] "
            f"version={mplib_version} file={mplib_file} planner_class={planner_cls} "
            f"pose_api={pose_api} has_plan_qpos={has_plan_qpos} "
            f"ik_return_closest={ik_supports_return_closest}"
        )

    def _get_move_group(self) -> str:
        return self.MOVE_GROUP if hasattr(self, "MOVE_GROUP") else "eef"

    def _planner_uses_mplib_pose_api_v2(self, planner_obj) -> bool:
        has_legacy_pose_planner = callable(getattr(planner_obj, "plan_qpos_to_pose", None))
        has_modern_pose_planner = callable(getattr(planner_obj, "plan_pose", None))
        if has_legacy_pose_planner:
            return False
        if has_modern_pose_planner:
            return True
        raise RuntimeError(
            "Installed mplib Planner exposes neither legacy plan_qpos_to_pose() nor modern "
            "plan_pose(); cannot run pose planning."
        )

    def _uses_mplib_pose_api_v2(self) -> bool:
        return self._planner_uses_mplib_pose_api_v2(self.planner)

    def _planner_ik_supports_return_closest(self):
        try:
            params = inspect.signature(self.planner.IK).parameters
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Unable to introspect installed mplib IK signature for return_closest support."
            ) from exc
        return "return_closest" in params

    def _to_mplib_goal_pose(self, goal_pose, planner_obj=None):
        goal_pose = np.asarray(goal_pose, dtype=np.float32).reshape(-1)
        planner_obj = self.planner if planner_obj is None else planner_obj
        if not self._planner_uses_mplib_pose_api_v2(planner_obj):
            return goal_pose
        if goal_pose.size != 7:
            raise ValueError(
                f"Expected goal pose vector of shape (7,) for mplib Pose conversion, got shape {goal_pose.shape}"
            )
        return mplib.Pose(goal_pose[:3].tolist(), goal_pose[3:7].tolist())

    def _get_rrtconnect_plan_kwargs(self):
        return {
            "time_step": self.base_env.control_timestep,
            "rrt_range": 0.1,
            "planning_time": 1,
            "fix_joint_limits": True,
            "simplify": True,
            "constraint_function": None,
            "constraint_jacobian": None,
            "constraint_tolerance": 1e-3,
            "verbose": False,
        }

    def _get_screw_plan_kwargs(self, *, wrt_world: bool):
        return {
            "qpos_step": 0.1,
            "time_step": self.base_env.control_timestep,
            "wrt_world": bool(wrt_world),
            "verbose": False,
        }

    def _iter_adjacent_link_name_pairs(self):
        pairs = []
        seen = set()
        get_joints = getattr(self.robot, "get_joints", None)
        if not callable(get_joints):
            return pairs
        for joint in get_joints():
            try:
                parent_link = joint.get_parent_link()
                child_link = joint.get_child_link()
            except Exception:
                continue
            if parent_link is None or child_link is None:
                continue
            try:
                parent_name = str(parent_link.get_name())
                child_name = str(child_link.get_name())
            except Exception:
                continue
            if not parent_name or not child_name or parent_name == child_name:
                continue
            pair = tuple(sorted((parent_name, child_name)))
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append((parent_name, child_name))
        return pairs

    def _apply_modern_mplib_adjacent_link_acm(self, planner) -> None:
        if not self._planner_uses_mplib_pose_api_v2(planner):
            return
        planning_world = getattr(planner, "planning_world", None)
        if planning_world is None or not hasattr(planning_world, "get_allowed_collision_matrix"):
            raise RuntimeError(
                "Installed modern mplib planner does not expose planning_world.get_allowed_collision_matrix(); "
                "cannot apply explicit adjacent-link ACM compatibility shim."
            )
        acm = planning_world.get_allowed_collision_matrix()
        if acm is None or not hasattr(acm, "set_entry"):
            raise RuntimeError(
                "Installed modern mplib planner returned an ACM object without set_entry(); "
                "cannot apply explicit adjacent-link ACM compatibility shim."
            )
        adjacent_pairs = self._iter_adjacent_link_name_pairs()
        if not adjacent_pairs:
            raise RuntimeError(
                "Failed to derive adjacent link pairs from the robot articulation; "
                "refusing to proceed without explicit modern mplib ACM setup."
            )
        for parent_name, child_name in adjacent_pairs:
            acm.set_entry(parent_name, child_name, True)
        print(
            f"[PLANNER_MPLIB_DEBUG][ACM] Applied explicit adjacent-link ACM allowlist for "
            f"{len(adjacent_pairs)} parent-child link pair(s) for modern mplib."
        )

    def _plan_pose_rrtstar(self, goal_pose, start_qpos):
        if self._uses_mplib_pose_api_v2():
            raise RuntimeError(
                "Installed mplib uses the modern plan_pose()/plan_qpos() API; "
                "RRTStar is not supported by this compatibility layer. "
                "Use RRTConnect or add an explicit plan_qpos()-based RRTStar path."
            )
        return self.planner.plan_qpos_to_pose(
            goal_pose,
            start_qpos,
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            rrt_range=0.0,
            planning_time=1,
            planner_name="RRTstar",
            wrt_world=True,
        )

    def _plan_pose_rrtconnect(self, goal_pose, start_qpos):
        if self._uses_mplib_pose_api_v2():
            start_qpos = np.asarray(start_qpos, dtype=np.float32).reshape(-1)
            start_qpos = np.clip(
                start_qpos,
                np.asarray(self.planner.joint_limits)[:, 0],
                np.asarray(self.planner.joint_limits)[:, 1],
            )
            goal_pose_obj = self._to_mplib_goal_pose(goal_pose)
            goal_pose_obj = self.planner._transform_goal_to_wrt_base(goal_pose_obj)
            if not self._planner_ik_supports_return_closest():
                raise RuntimeError(
                    "Installed modern mplib planner requires IK(return_closest=True) support "
                    "for deterministic RC5 goal-branch selection."
                )
            ik_status, goal_qposes = self.planner.IK(
                goal_pose_obj,
                start_qpos,
                mask=[],
                n_init_qpos=20,
                threshold=1e-3,
                return_closest=True,
                verbose=False,
            )
            if ik_status != "Success":
                return {"status": ik_status}
            goal_qposes = np.asarray(goal_qposes, dtype=np.float32)
            if goal_qposes.ndim == 1:
                goal_qposes = [goal_qposes]
            return self.planner.plan_qpos(
                goal_qposes,
                start_qpos,
                **self._get_rrtconnect_plan_kwargs(),
            )
        return self.planner.plan_qpos_to_pose(
            goal_pose,
            start_qpos,
            time_step=self.base_env.control_timestep,
            rrt_range=0.1,
            planning_time=1,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )

    def _plan_pose_screw(self, goal_pose, start_qpos, *, wrt_world: bool):
        if self._uses_mplib_pose_api_v2():
            return self.planner.plan_screw(
                self._to_mplib_goal_pose(goal_pose),
                start_qpos,
                **self._get_screw_plan_kwargs(wrt_world=wrt_world),
            )
        return self.planner.plan_screw(
            goal_pose,
            start_qpos,
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
        )

    def _get_planning_urdf_path(self) -> str:
        return self.env_agent.urdf_path

    def _get_planning_srdf_path(self) -> str:
        return self.env_agent.urdf_path.replace(".urdf", ".srdf")

    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        return None

    def _transform_pose_for_planning(self, target: sapien.Pose) -> sapien.Pose:
        return target

    def _log_target_transform_debug(
        self,
        original_pose: sapien.Pose,
        transformed_pose: sapien.Pose,
        context: str,
    ) -> None:
        solver_name = type(self).__name__
        if not self.debug and solver_name not in {
            "RC5ArmMotionPlanningSolver",
            "WidowXArmMotionPlanningSolver",
        }:
            return
        original_pose = to_sapien_pose_mani_skill(original_pose)
        transformed_pose = to_sapien_pose_mani_skill(transformed_pose)
        orig_p = np.asarray(original_pose.p, dtype=np.float32).reshape(-1)[:3]
        orig_q = np.asarray(original_pose.q, dtype=np.float32).reshape(-1)[:4]
        trans_p = np.asarray(transformed_pose.p, dtype=np.float32).reshape(-1)[:3]
        trans_q = np.asarray(transformed_pose.q, dtype=np.float32).reshape(-1)[:4]
        print(
            f"[PLANNER_TARGET_DEBUG][{solver_name}][{context}] "
            f"target_before_p={np.array2string(orig_p, precision=4, suppress_small=True)} "
            f"target_before_q={np.array2string(orig_q, precision=4, suppress_small=True)}"
        )
        print(
            f"[PLANNER_TARGET_DEBUG][{solver_name}][{context}] "
            f"target_after_p={np.array2string(trans_p, precision=4, suppress_small=True)} "
            f"target_after_q={np.array2string(trans_q, precision=4, suppress_small=True)}"
        )

    def _get_planner_qpos(self) -> np.ndarray:
        qpos = self.robot.get_qpos().cpu().numpy()
        if qpos.ndim > 1:
            qpos = qpos[0]
        return np.asarray(qpos)

    def _get_active_joint_names(self):
        return [joint.get_name() for joint in self.robot.get_active_joints()]

    def _get_planner_joint_names(self):
        planner_joint_names = list(getattr(self.planner, "user_joint_names", []) or [])
        if planner_joint_names:
            return planner_joint_names
        return self._get_active_joint_names()

    @staticmethod
    def _format_array_debug(arr, precision: int = 4) -> str:
        return np.array2string(
            np.asarray(arr, dtype=np.float32).reshape(-1),
            precision=precision,
            suppress_small=True,
        )

    def _log_joint_delta_debug(self, label: str, start_qpos, target_qpos) -> None:
        start_qpos = np.asarray(start_qpos, dtype=np.float32).reshape(-1)
        target_qpos = np.asarray(target_qpos, dtype=np.float32).reshape(-1)
        if start_qpos.shape != target_qpos.shape:
            print(
                f"{_Y}[PLANNER_MPLIB_DEBUG][{label}] WARNING: qpos shape mismatch "
                f"start={start_qpos.shape} target={target_qpos.shape}{_R}"
            )
            return
        raw_delta = target_qpos - start_qpos
        wrapped_delta = ((raw_delta + np.pi) % (2.0 * np.pi)) - np.pi
        wrap_extra = raw_delta - wrapped_delta
        joint_names = self._get_planner_joint_names()
        suspicious = np.where(np.abs(wrap_extra) > 1.0)[0]
        if suspicious.size == 0:
            return
        parts = []
        for idx in suspicious:
            joint_name = joint_names[idx] if idx < len(joint_names) else f"joint{idx}"
            parts.append(
                f"{joint_name}: start={start_qpos[idx]:+.4f} target={target_qpos[idx]:+.4f} "
                f"raw_delta={raw_delta[idx]:+.4f} wrapped_delta={wrapped_delta[idx]:+.4f} "
                f"extra={wrap_extra[idx]:+.4f}"
            )
        print(
            f"{_Y}[PLANNER_MPLIB_DEBUG][{label}] WARNING: suspicious >pi joint delta(s): "
            f"{'; '.join(parts)}{_R}"
        )

    def _log_mplib_request(self, label: str, goal_pose, start_qpos, **kwargs) -> None:
        goal_pose = np.asarray(goal_pose, dtype=np.float32).reshape(-1)
        start_qpos = np.asarray(start_qpos, dtype=np.float32).reshape(-1)
        goal_p = goal_pose[:3]
        goal_q = goal_pose[3:7]
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        print(
            f"[PLANNER_MPLIB_DEBUG][{label}][request] move_group={self._get_move_group()} "
            f"goal_p={self._format_array_debug(goal_p)} goal_q={self._format_array_debug(goal_q)} "
            f"start_qpos={self._format_array_debug(start_qpos)}"
            + (f" {extra}" if extra else "")
        )

    def _log_mplib_plan_result(self, label: str, result, start_qpos) -> None:
        status = result.get("status", "unknown")
        start_qpos = np.asarray(start_qpos, dtype=np.float32).reshape(-1)
        positions = result.get("position", None)
        n_steps = 0 if positions is None else int(np.asarray(positions).shape[0])
        print(
            f"[PLANNER_MPLIB_DEBUG][{label}][response] status={status} n_steps={n_steps}"
        )
        if positions is None or n_steps <= 0:
            return
        positions = np.asarray(positions, dtype=np.float32)
        first_qpos = positions[0].reshape(-1)
        last_qpos = positions[-1].reshape(-1)
        print(
            f"[PLANNER_MPLIB_DEBUG][{label}][trajectory] "
            f"first_qpos={self._format_array_debug(first_qpos)} "
            f"last_qpos={self._format_array_debug(last_qpos)}"
        )
        self._log_joint_delta_debug(f"{label}:start_to_first", start_qpos, first_qpos)
        self._log_joint_delta_debug(f"{label}:start_to_last", start_qpos, last_qpos)
        if n_steps > 1:
            step_deltas = np.diff(positions, axis=0)
            max_abs_step = np.max(np.abs(step_deltas), axis=0)
            print(
                f"[PLANNER_MPLIB_DEBUG][{label}][trajectory] "
                f"max_abs_step={self._format_array_debug(max_abs_step)}"
            )
            suspicious_steps = np.argwhere(np.abs(step_deltas) > np.pi)
            if suspicious_steps.size > 0:
                joint_names = self._get_planner_joint_names()
                parts = []
                for row_idx, joint_idx in suspicious_steps[:10]:
                    joint_name = (
                        joint_names[joint_idx] if joint_idx < len(joint_names) else f"joint{joint_idx}"
                    )
                    parts.append(
                        f"step={int(row_idx)}->{int(row_idx)+1} {joint_name} "
                        f"delta={float(step_deltas[row_idx, joint_idx]):+.4f}"
                    )
                print(
                    f"{_Y}[PLANNER_MPLIB_DEBUG][{label}] WARNING: suspicious >pi trajectory step(s): "
                    f"{'; '.join(parts)}{_R}"
                )

    def _get_current_move_group_pose(self):
        """Return the current world pose for the EE frame used by the active planner."""
        move_group = self._get_move_group()
        if move_group == "eef" and hasattr(self.base_env.agent, "tcp"):
            return _select_first_env_pose(self.base_env.agent.tcp.pose)
        if hasattr(self.robot, "links_map") and move_group in self.robot.links_map:
            return _select_first_env_pose(self.robot.links_map[move_group].pose)
        if hasattr(self.base_env.agent, "tcp"):
            return _select_first_env_pose(self.base_env.agent.tcp.pose)
        return _select_first_env_pose(self.robot.pose)

    def _log_ik_debug_info(self, target_pose: sapien.Pose) -> None:
        """Log robot base, active move-group EE, and target positions in world frame."""
        base_p = np.asarray(self.base_pose.p).flatten()[:3]
        ee_pose = self._get_current_move_group_pose()
        tcp_sp = to_sapien_pose_mani_skill(ee_pose)
        tcp_p = tcp_sp.p
        tcp_q = tcp_sp.q
        if isinstance(tcp_p, torch.Tensor):
            tcp_p = tcp_p.cpu().numpy()
        if isinstance(tcp_q, torch.Tensor):
            tcp_q = tcp_q.cpu().numpy()
        tcp_p = np.asarray(tcp_p)
        tcp_q = np.asarray(tcp_q)
        if tcp_p.ndim > 1:
            tcp_p = tcp_p[0]
        if tcp_q.ndim > 1:
            tcp_q = tcp_q[0]
        tcp_p = tcp_p.flatten()[:3]
        tcp_q = tcp_q.flatten()[:4]
        target_p = np.asarray(target_pose.p).flatten()[:3]
        target_q = np.asarray(target_pose.q).flatten()[:4]
        q_rel = qmult(qinverse(tcp_q), target_q)
        dist_base_target = float(np.linalg.norm(target_p - base_p))
        dist_tcp_target = float(np.linalg.norm(target_p - tcp_p))
        move_group = self._get_move_group()
        print("[PLANNER_IK_DEBUG] ===== World frame positions =====")
        print(f"[PLANNER_IK_DEBUG] Robot base: {base_p}")
        print(f"[PLANNER_IK_DEBUG] EE ({move_group}):  {tcp_p}")
        print(f"[PLANNER_IK_DEBUG] EE q ({move_group}): {tcp_q}")
        print(f"[PLANNER_IK_DEBUG] Target:    {target_p}")
        print(f"[PLANNER_IK_DEBUG] Target q:  {target_q}")
        print(f"[PLANNER_IK_DEBUG] q_rel:     {q_rel}")
        print(f"[PLANNER_IK_DEBUG] Distance base->target: {dist_base_target:.4f} m")
        print(f"[PLANNER_IK_DEBUG] Distance EE({move_group})->target: {dist_tcp_target:.4f} m")
        print("[PLANNER_IK_DEBUG] (Compare with mplib 'IK Failed! Distance X' to detect frame mismatch)")

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = self._batch_action(np.hstack([qpos, qvel]))
            else:
                action = self._batch_action(np.hstack([qpos]))
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                try:
                    self.base_env.render_human()
                except (AttributeError, RuntimeError):
                    # Viewer window may be closed or None in interactive mode
                    pass
            # Record frame for video if needed - record after EVERY step for smooth video
            if self.record_frames is not None:
                frame = self._get_record_frame(obs)
                if frame is not None:
                    if self.normalize_frame_fn is not None:
                        frame = self.normalize_frame_fn(frame)
                    if frame is not None:
                        self.record_frames.append(frame)
        return obs, reward, terminated, truncated, info

    def move_to_pose_with_RRTStar(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose_mani_skill(pose)
        self._update_grasp_visual(pose)
        original_pose = pose
        pose = self._transform_pose_for_planning(pose)
        self._log_target_transform_debug(original_pose, pose, "rrtstar")
        start_qpos = self._get_planner_qpos()
        goal_pose = np.concatenate([pose.p, pose.q])
        self._log_mplib_request(
            "RRTStar",
            goal_pose,
            start_qpos,
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )
        result = self._plan_pose_rrtstar(goal_pose, start_qpos)
        self._log_mplib_plan_result("RRTStar", result, start_qpos)
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        _log_planner_dispatch(
            f"{type(self).__name__}.move_to_pose_with_RRTConnect(dry_run={dry_run}, refine_steps={refine_steps})"
        )
        pose = to_sapien_pose_mani_skill(pose)
        self._update_grasp_visual(pose)
        original_pose = pose
        pose = self._transform_pose_for_planning(pose)
        self._log_target_transform_debug(original_pose, pose, "rrtconnect")
        start_qpos = self._get_planner_qpos()
        goal_pose = np.concatenate([pose.p, pose.q])
        self._log_mplib_request(
            "RRTConnect",
            goal_pose,
            start_qpos,
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )
        result = self._plan_pose_rrtconnect(goal_pose, start_qpos)
        self._log_mplib_plan_result("RRTConnect", result, start_qpos)
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        _log_planner_dispatch(
            f"{type(self).__name__}.move_to_pose_with_screw(dry_run={dry_run}, refine_steps={refine_steps})"
        )
        pose = to_sapien_pose_mani_skill(pose)
        # try screw two times before giving up
        self._update_grasp_visual(pose)
        original_pose = pose
        pose = self._transform_pose_for_planning(pose)
        self._log_target_transform_debug(original_pose, pose, "screw")
        self._log_ik_debug_info(pose)
        start_qpos = self._get_planner_qpos()
        goal_pose = np.concatenate([pose.p, pose.q])
        self._log_mplib_request(
            "Screw",
            goal_pose,
            start_qpos,
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
        )
        result = self._plan_pose_screw(goal_pose, start_qpos, wrt_world=True)
        self._log_mplib_plan_result("Screw", result, start_qpos)
        if result["status"] != "Success":
            if self.debug:
                print(f"[PLANNER][screw] First attempt failed with status='{result['status']}'")
                print(f"[PLANNER][screw] Target pose p={pose.p}, q={pose.q}")
            start_qpos_retry = self._get_planner_qpos()
            self._log_mplib_request(
                "ScrewRetry",
                goal_pose,
                start_qpos_retry,
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
            )
            result = self._plan_pose_screw(goal_pose, start_qpos_retry, wrt_world=True)
            self._log_mplib_plan_result("ScrewRetry", result, start_qpos_retry)
            if result["status"] != "Success":
                print(result["status"])
                if self.debug:
                    print(f"[PLANNER][screw] Second attempt failed with status='{result['status']}'")
                    print(f"[PLANNER][screw] Current qpos={self.robot.get_qpos().cpu().numpy()[0]}")
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        """Try screw first, fallback to RRTConnect (for WidowX when screw fails)."""
        _log_planner_dispatch(
            f"{type(self).__name__}.move_to_pose -> screw first (dry_run={dry_run}, refine_steps={refine_steps})"
        )
        res = self.move_to_pose_with_screw(pose, dry_run=dry_run, refine_steps=refine_steps)
        if res != -1:
            return res
        _log_planner_dispatch(
            f"{type(self).__name__}.move_to_pose fallback: screw failed -> RRTConnect"
        )
        if self.debug:
            print("[PLANNER] screw failed, trying RRTConnect fallback")
        return self.move_to_pose_with_RRTConnect(pose, dry_run=dry_run, refine_steps=refine_steps)

    def add_box_collision(self, extents: np.ndarray, pose: sapien.Pose):
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=pose.to_transformation_matrix())
        pts, _ = trimesh.sample.sample_surface(box, 256)
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def add_collision_pts(self, pts: np.ndarray):
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def clear_collisions(self):
        self.all_collision_pts = None
        self.use_point_cloud = False

    def close(self):
        pass


class TwoFingerGripperMotionPlanningSolver(BaseMotionPlanningSolver):
    OPEN = 1
    CLOSED = -1

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        record_frames=None,  # List to append frames to for video recording
        normalize_frame_fn=None,  # Function to normalize frame before appending
        grasp_pose_visual_initial_pose=None,  # [x,y,z] from config to avoid ManiSkill warning
        record_frames_from="render",  # "render" or "base_camera"
    ):
        super().__init__(
            env,
            debug,
            vis,
            base_pose,
            print_env_info,
            joint_vel_limits,
            joint_acc_limits,
            record_frames,
            normalize_frame_fn,
            record_frames_from,
        )
        self.gripper_state = self.OPEN
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_two_finger_gripper_grasp_pose_visual(
                    self.base_env.scene,
                    initial_pose=grasp_pose_visual_initial_pose,
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            self.grasp_pose_visual.set_pose(self.base_env.agent.tcp.pose)

    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target)

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = self._batch_action(np.hstack([qpos, qvel, self.gripper_state]))
            else:
                action = self._batch_action(np.hstack([qpos, self.gripper_state]))
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                try:
                    self.base_env.render_human()
                except (AttributeError, RuntimeError):
                    # Viewer window may be closed or None in interactive mode
                    pass
            # Record frame for video if needed - record after EVERY step for smooth video
            if self.record_frames is not None:
                frame = self._get_record_frame(obs)
                if frame is not None:
                    if self.normalize_frame_fn is not None:
                        frame = self.normalize_frame_fn(frame)
                    if frame is not None:
                        self.record_frames.append(frame)
        return obs, reward, terminated, truncated, info

    def open_gripper(self, t=6, gripper_state=None):
        if gripper_state is None:
            gripper_state = self.OPEN
        self.gripper_state = gripper_state
        qpos = (
            self.robot.get_qpos()[0, : len(self.planner.joint_vel_limits)].cpu().numpy()
        )
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = self._batch_action(np.hstack([qpos, self.gripper_state]))
            else:
                action = self._batch_action(np.hstack([qpos, qpos * 0, self.gripper_state]))
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                try:
                    self.base_env.render_human()
                except (AttributeError, RuntimeError):
                    # Viewer window may be closed or None in interactive mode
                    pass
            # Record frame for video if needed
            if self.record_frames is not None:
                frame = self._get_record_frame(obs)
                if frame is not None:
                    if self.normalize_frame_fn is not None:
                        frame = self.normalize_frame_fn(frame)
                    if frame is not None:
                        self.record_frames.append(frame)
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6, gripper_state=None):
        if gripper_state is None:
            gripper_state = self.CLOSED
        self.gripper_state = gripper_state
        qpos = (
            self.robot.get_qpos()[0, : len(self.planner.joint_vel_limits)].cpu().numpy()
        )
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = self._batch_action(np.hstack([qpos, self.gripper_state]))
            else:
                action = self._batch_action(np.hstack([qpos, qpos * 0, self.gripper_state]))
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                try:
                    self.base_env.render_human()
                except (AttributeError, RuntimeError):
                    # Viewer window may be closed or None in interactive mode
                    pass
            # Record frame for video if needed
            if self.record_frames is not None:
                frame = self._get_record_frame(obs)
                if frame is not None:
                    if self.normalize_frame_fn is not None:
                        frame = self.normalize_frame_fn(frame)
                    if frame is not None:
                        self.record_frames.append(frame)
        return obs, reward, terminated, truncated, info


def build_two_finger_gripper_grasp_pose_visual(
    scene: ManiSkillScene,
    initial_pose=None,
):
    """
    Build grasp pose visual. initial_pose [x,y,z] or sapien.Pose avoids ManiSkill warning.
    """
    builder = scene.create_actor_builder()
    if initial_pose is not None:
        if isinstance(initial_pose, (list, tuple)) and len(initial_pose) >= 3:
            builder.initial_pose = sapien.Pose(p=initial_pose[:3])
        elif hasattr(initial_pose, "p"):
            builder.initial_pose = initial_pose
    grasp_pose_visual_width = 0.01
    grasp_width = 0.05

    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7]),
    )

    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                grasp_width + grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                -grasp_width - grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )
    grasp_pose_visual = builder.build_kinematic(name="grasp_pose_visual")
    return grasp_pose_visual


def build_pose_axes_visual(
    scene: ManiSkillScene,
    initial_pose=None,
    actor_name: str = "pose_axes_visual",
):
    """Build a simple RGB axes marker that is easy to spot in the viewer."""
    builder = scene.create_actor_builder()
    if initial_pose is not None:
        if isinstance(initial_pose, (list, tuple)) and len(initial_pose) >= 3:
            builder.initial_pose = sapien.Pose(p=initial_pose[:3])
        elif hasattr(initial_pose, "p"):
            builder.initial_pose = initial_pose

    axis_length = 0.16
    axis_radius = 0.012
    tip_radius = 0.022

    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0]),
        radius=tip_radius,
        material=sapien.render.RenderMaterial(base_color=[1.0, 1.0, 0.0, 1.0]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[axis_length * 0.5, 0, 0]),
        half_size=[axis_length * 0.5, axis_radius, axis_radius],
        material=sapien.render.RenderMaterial(base_color=[1.0, 0.1, 0.1, 1.0]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, axis_length * 0.5, 0]),
        half_size=[axis_radius, axis_length * 0.5, axis_radius],
        material=sapien.render.RenderMaterial(base_color=[0.1, 1.0, 0.1, 1.0]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, axis_length * 0.5]),
        half_size=[axis_radius, axis_radius, axis_length * 0.5],
        material=sapien.render.RenderMaterial(base_color=[0.1, 0.6, 1.0, 1.0]),
    )

    return builder.build_kinematic(name=actor_name)


class HeuristicManipulationAgent:
    """
    An agent that performs heuristic grasp trials and follows a trajectory.
    """

    def __init__(
        self,
        env: BaseEnv,
        planner: "PandaArmMotionPlanningSolver",
        lift_height: float = 0.1,
    ):
        self.env = env
        self.planner = planner
        self.lift_height = lift_height

    def check_grasp_success(self, target_object_id: str) -> bool:
        """
        Checks if the object is successfully grasped by using the agent's
        built-in contact force checking method.
        """
        target_object: Actor = self.env.unwrapped.object_actors[target_object_id]
        agent: BaseAgent = self.env.unwrapped.agent

        # This check needs to be done over a few steps to be reliable
        for _ in range(5):
            self.env.step(None)  # Let physics settle

        is_grasping = agent.is_grasping(target_object)

        if is_grasping:
            print("Contact forces detected. Grasp is likely successful.")
            # Lift the gripper to confirm
            initial_gripper_pose = self.env.unwrapped.agent.tcp.pose
            lift_pose = (
                Pose.create_from_pq(p=[0, 0, self.lift_height]) * initial_gripper_pose
            )
            self.planner.move_to_pose_with_screw(lift_pose)
        else:
            # TODO: check the pre_grasp_pose to see if it is close to the object
            print("No significant contact forces detected. Grasp failed.")
            initial_gripper_pose = self.env.unwrapped.agent.tcp.pose
            # Move back to a safe position
            pre_grasp_pose = Pose.create_from_pq(
                p=initial_gripper_pose.p + np.array([0, 0, 0.1]).reshape(1, 3),
                q=initial_gripper_pose.q,
            )
            self.planner.open_gripper()
            self.planner.move_to_pose_with_RRTConnect(pre_grasp_pose)

        return is_grasping

    def attempt_grasp(
        self, grasp_pose_world: Pose, pre_grasp_offset: float = 0.1
    ) -> None:
        """
        Executes a grasp attempt from a pre-grasp position.
        """
        # Calculate pre-grasp pose (offset along the grasp's +X axis)
        # unsqueeze to 4x4 matrix (N, 4, 4) -> (4, 4)
        approach_dir = grasp_pose_world.to_transformation_matrix().reshape(4, 4)[:3, 0]

        pre_grasp_p = grasp_pose_world.p - approach_dir * pre_grasp_offset
        pre_grasp_pose = Pose.create_from_pq(p=pre_grasp_p, q=grasp_pose_world.q)

        # Execute the motion sequence
        self.planner.open_gripper()
        print("Moving to pre-grasp pose...")
        self.planner.move_to_pose_with_RRTConnect(pre_grasp_pose)
        print("Moving to grasp pose...")
        self.planner.move_to_pose_with_screw(grasp_pose_world)
        self.planner.close_gripper()

    def follow_trajectory(self, target_object_id: str, trajectory: list[Pose]) -> None:
        """
        Follows a given 6D object trajectory.
        """
        target_object: Actor = self.env.unwrapped.object_actors[target_object_id]
        initial_ee_pose_world: Pose = self.env.unwrapped.agent.tcp.pose
        initial_obj_pose_world: Pose = target_object.pose

        # Calculate the fixed transform from the object to the EE frame
        T_world_obj = initial_obj_pose_world.to_transformation_matrix()
        T_world_ee = initial_ee_pose_world.to_transformation_matrix()

        T_obj_ee = torch.linalg.inv(T_world_obj) @ T_world_ee

        print("Starting trajectory following...")
        for i, target_obj_pose_world in enumerate(trajectory):
            print(f"  Waypoint {i + 1}/{len(trajectory)}")
            T_world_obj_target = target_obj_pose_world.to_transformation_matrix()
            T_world_ee_target = T_world_obj_target @ T_obj_ee
            T_world_ee_target = T_world_ee_target.reshape(4, 4)

            # turn into numpy array
            T_world_ee_target = T_world_ee_target.cpu().numpy()

            ee_target_pose = Pose.create_from_pq(
                p=T_world_ee_target[:3, 3],
                q=mat2quat(T_world_ee_target[:3, :3]),
            )

            self.planner.move_to_pose_with_screw(ee_target_pose, refine_steps=0)

        print("Trajectory following complete.")


class PandaArmMotionPlanningSolver(TwoFingerGripperMotionPlanningSolver):
    OPEN = 1
    CLOSED = -1
    MOVE_GROUP = "panda_hand_tcp"

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        record_frames=None,  # List to append frames to for video recording
        normalize_frame_fn=None,  # Function to normalize frame before appending
        grasp_pose_visual_initial_pose=None,  # [x,y,z] from config to avoid ManiSkill warning
        record_frames_from="render",
    ):
        super().__init__(
            env,
            debug,
            vis,
            base_pose,
            visualize_target_grasp_pose,
            print_env_info,
            joint_vel_limits,
            joint_acc_limits,
            record_frames,
            normalize_frame_fn,
            grasp_pose_visual_initial_pose,
            record_frames_from,
        )


class RC5ArmMotionPlanningSolver(BaseMotionPlanningSolver):
    """Motion planner for RC5 + Aero Hand."""

    MOVE_GROUP = "right_tcp_link"
    _PLANNING_ASSET_DIR = (
        Path(__file__).resolve().parent.parent
        / "robot_assets"
        / "rc5_aero_hand_planning_prod_minus_thumb_plus_prehand_only"
    )
    # RC5 requires an EE-frame orientation adapter before passing targets to
    # mplib. Keep the target position unchanged until the rotational convention
    # is validated separately from any TCP offset.
    TCP_FROM_PREHAND_Q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _get_pose_for_link_name(self, link_name: str):
        if not link_name:
            return None
        if link_name == "right_tcp_link" and hasattr(self.base_env.agent, "tcp"):
            return to_sapien_pose_mani_skill(_select_first_env_pose(self.base_env.agent.tcp.pose))
        if link_name == "prehand" and hasattr(self.base_env.agent, "palm_link"):
            return to_sapien_pose_mani_skill(_select_first_env_pose(self.base_env.agent.palm_link.pose))
        if hasattr(self.robot, "links_map") and link_name in self.robot.links_map:
            return to_sapien_pose_mani_skill(_select_first_env_pose(self.robot.links_map[link_name].pose))
        return None

    def _format_pose_debug(self, pose: sapien.Pose) -> str:
        pose = to_sapien_pose_mani_skill(pose)
        p = np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3]
        q = np.asarray(pose.q, dtype=np.float32).reshape(-1)[:4]
        return (
            f"p={np.array2string(p, precision=4, suppress_small=True)} "
            f"q={np.array2string(q, precision=4, suppress_small=True)}"
        )

    def _log_rc5_kinematics_debug(self, context: str, target_pose: sapien.Pose = None) -> None:
        move_group = self._get_move_group()
        ee_link_name = getattr(self.env_agent, "ee_link_name", "")
        prehand_pose = self._get_pose_for_link_name("prehand")
        tcp_pose = self._get_pose_for_link_name("right_tcp_link")
        move_group_pose = self._get_pose_for_link_name(move_group)

        print(
            f"[RC5_KIN_DEBUG][{context}] move_group={move_group} "
            f"agent_ee_link_name={ee_link_name} planner_class={type(self).__name__}"
        )
        print(
            f"[RC5_KIN_DEBUG][{context}] planning_urdf={self._get_planning_urdf_path()} "
            f"planning_srdf={self._get_planning_srdf_path()}"
        )
        if ee_link_name and ee_link_name != move_group:
            print(
                f"{_Y}[RC5_KIN_DEBUG][{context}] WARNING: agent ee_link_name ('{ee_link_name}') "
                f"!= planner move_group ('{move_group}') {_R}"
            )

        if prehand_pose is not None:
            print(f"[RC5_KIN_DEBUG][{context}] prehand {self._format_pose_debug(prehand_pose)}")
        if tcp_pose is not None:
            print(f"[RC5_KIN_DEBUG][{context}] right_tcp_link {self._format_pose_debug(tcp_pose)}")
        if move_group_pose is not None:
            print(
                f"[RC5_KIN_DEBUG][{context}] active_move_group('{move_group}') "
                f"{self._format_pose_debug(move_group_pose)}"
            )

        if prehand_pose is not None and tcp_pose is not None:
            prehand_p = np.asarray(prehand_pose.p, dtype=np.float32).reshape(-1)[:3]
            tcp_p = np.asarray(tcp_pose.p, dtype=np.float32).reshape(-1)[:3]
            prehand_q = np.asarray(prehand_pose.q, dtype=np.float32).reshape(-1)[:4]
            tcp_q = np.asarray(tcp_pose.q, dtype=np.float32).reshape(-1)[:4]
            delta_p = tcp_p - prehand_p
            delta_q = qmult(qinverse(prehand_q), tcp_q)
            print(
                f"[RC5_KIN_DEBUG][{context}] prehand_to_right_tcp_link "
                f"delta_p={np.array2string(delta_p, precision=4, suppress_small=True)} "
                f"delta_q={np.array2string(delta_q, precision=4, suppress_small=True)}"
            )

        if target_pose is not None:
            target_pose = to_sapien_pose_mani_skill(target_pose)
            print(f"[RC5_KIN_DEBUG][{context}] target_world {self._format_pose_debug(target_pose)}")
            transformed_target = self._transform_pose_for_planning(target_pose)
            print(
                f"[RC5_KIN_DEBUG][{context}] target_after_transform "
                f"{self._format_pose_debug(transformed_target)}"
            )
            target_p = np.asarray(target_pose.p, dtype=np.float32).reshape(-1)[:3]
            for label, pose in (
                ("prehand", prehand_pose),
                ("right_tcp_link", tcp_pose),
                (f"move_group:{move_group}", move_group_pose),
            ):
                if pose is None:
                    continue
                pose_p = np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3]
                dist = float(np.linalg.norm(target_p - pose_p))
                print(
                    f"[RC5_KIN_DEBUG][{context}] dist_target_to_{label}={dist:.4f} m"
                )

    @staticmethod
    def _resolve_override_asset_file(asset_override: str, suffix: str) -> str:
        asset_path = Path(asset_override)
        if asset_path.is_file():
            return str(asset_path)
        matches = sorted(asset_path.glob(f"*{suffix}"))
        if len(matches) == 1:
            return str(matches[0])
        if not matches:
            raise FileNotFoundError(f"No '{suffix}' file found in override asset dir: {asset_path}")
        raise RuntimeError(f"Multiple '{suffix}' files found in override asset dir: {asset_path}")

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        record_frames=None,
        normalize_frame_fn=None,
        grasp_pose_visual_initial_pose=None,
        record_frames_from="render",
    ):
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.grasp_pose_visual = None
        super().__init__(
            env,
            debug,
            vis,
            base_pose,
            print_env_info,
            joint_vel_limits,
            joint_acc_limits,
            record_frames,
            normalize_frame_fn,
            record_frames_from,
        )
        self.arm_dof = len(getattr(self.env_agent, "arm_joint_names", [])) or 6
        self.hand_open_qpos = np.asarray(self.env_agent.hand_open_qpos, dtype=np.float32).reshape(-1)
        self.hand_close_qpos = np.asarray(self.env_agent.hand_close_qpos, dtype=np.float32).reshape(-1)
        grasp_state = restore_planner_grasp_state(
            self.base_env,
            default_target_hand_qpos=self.hand_open_qpos,
            source_stage="solver_init",
        )
        latched_target = grasp_state.target_hand_qpos
        if latched_target is not None and latched_target.shape == self.hand_open_qpos.shape:
            self.hand_target_qpos = latched_target.copy()
        else:
            self.hand_target_qpos = self.hand_open_qpos.copy()
            save_planner_grasp_state(
                self.base_env,
                target_hand_qpos=self.hand_target_qpos,
                source_stage="solver_init",
            )
        if self.control_mode not in {"pd_joint_pos", "pd_joint_pos_vel"}:
            raise RuntimeError(
                f"RC5ArmMotionPlanningSolver expects joint-space control mode, got '{self.control_mode}'"
            )
        if self.vis and self.visualize_target_grasp_pose:
            visual_name = "rc5_grasp_pose_visual"
            if visual_name not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_pose_axes_visual(
                    self.base_env.scene,
                    initial_pose=grasp_pose_visual_initial_pose,
                    actor_name=visual_name,
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors[visual_name]
            self.grasp_pose_visual.set_pose(self.base_env.agent.tcp.pose)

    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target)

    def _set_latched_hand_target_qpos(self, target_qpos: np.ndarray) -> None:
        target_qpos = np.asarray(target_qpos, dtype=np.float32).reshape(-1)
        self.hand_target_qpos = target_qpos.copy()
        save_planner_grasp_state(
            self.base_env,
            target_hand_qpos=target_qpos,
            source_stage="solver_latched_hand_target",
        )

    def _get_current_arm_qpos(self) -> np.ndarray:
        current_qpos = self.robot.get_qpos()
        if hasattr(current_qpos, "cpu"):
            current_qpos = current_qpos.cpu().numpy()
        current_qpos = np.asarray(current_qpos)
        if current_qpos.ndim > 1:
            current_qpos = current_qpos[0]
        return np.asarray(current_qpos[: self.arm_dof], dtype=np.float32)

    def _get_planner_qpos(self) -> np.ndarray:
        current_qpos = self.robot.get_qpos()
        if hasattr(current_qpos, "cpu"):
            current_qpos = current_qpos.cpu().numpy()
        current_qpos = np.asarray(current_qpos)
        if current_qpos.ndim > 1:
            current_qpos = current_qpos[0]
        current_qpos = np.asarray(current_qpos, dtype=np.float32)

        sim_joint_names = self._get_active_joint_names()
        planner_joint_names = list(getattr(self.planner, "user_joint_names", sim_joint_names))
        if planner_joint_names == sim_joint_names:
            return current_qpos

        qpos_by_name = {
            name: current_qpos[idx]
            for idx, name in enumerate(sim_joint_names)
        }
        reordered = np.asarray([qpos_by_name[name] for name in planner_joint_names], dtype=np.float32)
        if self.debug:
            print(f"[RC5][planner_qpos] sim_joint_names={sim_joint_names}")
            print(f"[RC5][planner_qpos] planner_joint_names={planner_joint_names}")
            print(f"[RC5][planner_qpos] reordered current qpos for mplib")
        return reordered

    # Temporary RC5 workaround:
    # - We intentionally route planner assets through the production-like
    #   `rc5_aero_hand_planning_prod_minus_thumb` variant.
    # - In that asset pack, `prehand.stl.convex.stl` was replaced with a
    #   box-like proxy because the current generated prehand convex sidecar
    #   causes planner degradation.
    # - Once a regenerated planner-friendly `prehand_convex.stl` arrives,
    #   restore the original sidecar file and re-run the RC5 smoke test.
    def _get_planning_urdf_path(self) -> str:
        asset_override = os.environ.get("OPENR2S_RC5_PLANNING_ASSET_DIR")
        if asset_override:
            return self._resolve_override_asset_file(asset_override, ".urdf")
        return str(
            Path(__file__).resolve().parent.parent
            / "robot_assets"
            / "rc5_aero_hand_planning_prod_minus_thumb_continuous"
            / "rc5_aero_hand_planning_prod_minus_thumb.urdf"
        )

    def _get_planning_srdf_path(self) -> str:
        asset_override = os.environ.get("OPENR2S_RC5_PLANNING_ASSET_DIR")
        if asset_override:
            return self._resolve_override_asset_file(asset_override, ".srdf")
        return str(
            Path(__file__).resolve().parent.parent
            / "robot_assets"
            / "rc5_aero_hand_planning_prod_minus_thumb_continuous"
            / "rc5_aero_hand_planning_prod_minus_thumb.srdf"
        )

    def _get_move_group(self) -> str:
        move_group_override = os.environ.get("OPENR2S_RC5_MOVE_GROUP")
        if move_group_override:
            move_group_override = move_group_override.strip()
            if move_group_override:
                return move_group_override
        return self.MOVE_GROUP

    def setup_planner(self):
        planning_urdf_path = self._get_planning_urdf_path()
        planning_srdf_path = self._get_planning_srdf_path()
        move_group = self._get_move_group()
        print(
            f"[PLANNER_RC5_ASSETS] Using planning assets: urdf={planning_urdf_path}, "
            f"srdf={planning_srdf_path}, move_group={move_group}"
        )
        use_nonconvex = os.environ.get("OPENR2S_RC5_NONCONVEX_PLANNER", "").strip().lower()
        use_nonconvex = use_nonconvex in {"1", "true", "yes", "on"}
        self._log_rc5_kinematics_debug("setup_planner")
        if not use_nonconvex:
            return super().setup_planner()

        from .mplib_non_convex import NonConvexPlanner

        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = NonConvexPlanner(
            urdf=planning_urdf_path,
            srdf=planning_srdf_path,
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=move_group,
        )
        base_pose_arr = np.hstack([self.base_pose.p, self.base_pose.q])
        planner.set_base_pose(self._to_mplib_goal_pose(base_pose_arr, planner_obj=planner))
        print(
            "[PLANNER_IK_DEBUG] setup_planner (RC5 NonConvex): "
            f"base_pose passed to mplib: p={base_pose_arr[:3]}, q={base_pose_arr[3:7]}"
        )
        planner.joint_vel_limits = (
            np.asarray(planner.joint_vel_limits) * self.joint_vel_limits
        )
        planner.joint_acc_limits = (
            np.asarray(planner.joint_acc_limits) * self.joint_acc_limits
        )
        return planner

    def _compose_action(self, arm_qpos: np.ndarray, arm_qvel: np.ndarray = None) -> np.ndarray:
        hand_qpos = self.hand_target_qpos
        if self.control_mode == "pd_joint_pos_vel":
            if arm_qvel is None:
                arm_qvel = np.zeros_like(arm_qpos)
            hand_qvel = np.zeros_like(hand_qpos)
            action = np.hstack([arm_qpos, hand_qpos, arm_qvel, hand_qvel])
        else:
            action = np.hstack([arm_qpos, hand_qpos])
        num_envs = getattr(self.base_env.scene, "num_envs", 1)
        if num_envs and num_envs > 1:
            action = np.repeat(action[None, :], num_envs, axis=0)
        return action

    def _transform_pose_for_planning(self, target: sapien.Pose) -> sapien.Pose:
        target = to_sapien_pose_mani_skill(target)
        p = np.asarray(target.p).flatten()[:3]
        q = np.asarray(target.q).flatten()[:4]
        if hasattr(q, "cpu"):
            q = q.cpu().numpy()
        # Keep the grasp-target orientation semantics and only apply an
        # RC5-specific EE-frame correction, same integration pattern as WidowX.
        return sapien.Pose(p=p, q=tuple(q)) * sapien.Pose(q=tuple(self.TCP_FROM_PREHAND_Q))

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            arm_qpos = result["position"][min(i, n_step - 1)]
            arm_qvel = None
            if self.control_mode == "pd_joint_pos_vel":
                arm_qvel = result["velocity"][min(i, n_step - 1)]
            action = self._compose_action(arm_qpos, arm_qvel)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
            if self.vis:
                try:
                    self.base_env.render_human()
                except (AttributeError, RuntimeError):
                    pass
            if self.record_frames is not None:
                frame = self._get_record_frame(obs)
                if frame is not None:
                    if self.normalize_frame_fn is not None:
                        frame = self.normalize_frame_fn(frame)
                    if frame is not None:
                        self.record_frames.append(frame)
        return obs, reward, terminated, truncated, info

    def _hold_current_arm_and_step(self, t: int = 6):
        arm_qpos = self._get_current_arm_qpos()
        obs = reward = terminated = truncated = info = None
        for _ in range(t):
            action = self._compose_action(arm_qpos)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
            if self.vis:
                try:
                    self.base_env.render_human()
                except (AttributeError, RuntimeError):
                    pass
            if self.record_frames is not None:
                frame = self._get_record_frame(obs)
                if frame is not None:
                    if self.normalize_frame_fn is not None:
                        frame = self.normalize_frame_fn(frame)
                    if frame is not None:
                        self.record_frames.append(frame)
        return obs, reward, terminated, truncated, info

    def open_gripper(self, t: int = 6):
        self._set_latched_hand_target_qpos(self.hand_open_qpos)
        return self._hold_current_arm_and_step(t=t)

    def close_gripper(self, t: int = 6):
        self._set_latched_hand_target_qpos(self.hand_close_qpos)
        return self._hold_current_arm_and_step(t=t)


class WidowXArmMotionPlanningSolver(TwoFingerGripperMotionPlanningSolver):
    """Motion planner for WidowX250S using NonConvexPlanner (non-convex geometry)."""

    OPEN = 1
    CLOSED = -1
    MOVE_GROUP = "ee_gripper_link"

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        record_frames=None,
        normalize_frame_fn=None,
        grasp_pose_visual_initial_pose=None,
        plan_time_step=None,
        record_frames_from="render",
    ):
        self.plan_time_step = plan_time_step
        super().__init__(
            env,
            debug,
            vis,
            base_pose,
            visualize_target_grasp_pose,
            print_env_info,
            joint_vel_limits,
            joint_acc_limits,
            record_frames,
            normalize_frame_fn,
            grasp_pose_visual_initial_pose,
            record_frames_from,
        )
        self.transfer_q = np.array([0.7071068, 0, -0.7071068, 0])  # -90 deg Y for WidowX
        self._sync_gripper_state_from_latched_target()

    def setup_planner(self):
        from .mplib_non_convex import NonConvexPlanner

        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = NonConvexPlanner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=self.MOVE_GROUP,
            joint_vel_limits=np.ones(6) * self.joint_vel_limits,
            joint_acc_limits=np.ones(6) * self.joint_acc_limits,
        )
        base_pose = self.base_pose
        p = np.asarray(base_pose.p).flatten()[:3]
        q = np.asarray(base_pose.q).flatten()[:4]
        if hasattr(q, "cpu"):
            q = q.cpu().numpy()
        base_pose_arr = np.hstack([p, q])
        planner.set_base_pose(self._to_mplib_goal_pose(base_pose_arr, planner_obj=planner))
        print(f"[PLANNER_IK_DEBUG] setup_planner (WidowX): base_pose passed to mplib: p={base_pose_arr[:3]}, q={base_pose_arr[3:7]}")
        return planner

    def _transform_pose_for_planning(self, target: sapien.Pose) -> sapien.Pose:
        """Apply WidowX EE frame transform for mplib."""
        target = to_sapien_pose_mani_skill(target)
        p = np.asarray(target.p).flatten()[:3]
        q = np.asarray(target.q).flatten()[:4]
        if hasattr(q, "cpu"):
            q = q.cpu().numpy()
        return sapien.Pose(p=p, q=tuple(q)) * sapien.Pose(q=tuple(self.transfer_q))

    def _set_latched_hand_target_qpos(self, target_qpos: np.ndarray) -> None:
        target_qpos = np.asarray(target_qpos, dtype=np.float32).reshape(-1)
        self.gripper_state = infer_gripper_state_from_hand_target(
            target_qpos,
            open_hand_qpos=[0.037, 0.037],
            closed_hand_qpos=[0.015, 0.015],
            open_state=self.OPEN,
            closed_state=self.CLOSED,
        )
        save_planner_grasp_state(
            self.base_env,
            target_hand_qpos=target_qpos,
            gripper_state=self.gripper_state,
            source_stage="widowx_latched_hand_target",
        )

    def _sync_gripper_state_from_latched_target(self) -> None:
        state = restore_planner_grasp_state(
            self.base_env,
            default_target_hand_qpos=[0.037, 0.037],
            default_gripper_state=self.OPEN,
            source_stage="widowx_solver_init",
        )
        latched = state.target_hand_qpos
        if latched is None:
            self.gripper_state = self.OPEN
            return
        self._set_latched_hand_target_qpos(latched)

    def _gripper_action(self):
        """WidowX: map gripper_state (1|-1) to [left, right] finger positions (m).
        RL4VLA WidowX250SBridgeDataset: open=0.037, closed=0.015 (PD controller range)."""
        if self.gripper_state == self.OPEN:
            return np.array([0.037, 0.037])  # RL4VLA: PD upper=0.037
        return np.array([0.015, 0.015])  # RL4VLA: PD lower=0.015

    def follow_path(self, result, refine_steps: int = 0):
        """WidowX: action = [qpos_arm (6), left_finger, right_finger]."""
        n_step = result["position"].shape[0]
        gripper = self._gripper_action()
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = self._batch_action(np.hstack([qpos, qvel, gripper]))
            else:
                action = self._batch_action(np.hstack([qpos, gripper]))
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                try:
                    self.base_env.render_human()
                except (AttributeError, RuntimeError):
                    pass
            if self.record_frames is not None:
                frame = self._get_record_frame(obs)
                if frame is not None:
                    if self.normalize_frame_fn is not None:
                        frame = self.normalize_frame_fn(frame)
                    if frame is not None:
                        self.record_frames.append(frame)
        return obs, reward, terminated, truncated, info

    def open_gripper(self, t=6, gripper_state=None):
        if gripper_state is None:
            gripper_state = self.OPEN
        self.gripper_state = gripper_state
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        gripper = self._gripper_action()
        self._set_latched_hand_target_qpos(gripper)
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = self._batch_action(np.hstack([qpos, gripper]))
            else:
                action = self._batch_action(np.hstack([qpos, np.zeros(6), gripper]))
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                try:
                    self.base_env.render_human()
                except (AttributeError, RuntimeError):
                    pass
            if self.record_frames is not None:
                frame = self._get_record_frame(obs)
                if frame is not None:
                    if self.normalize_frame_fn is not None:
                        frame = self.normalize_frame_fn(frame)
                    if frame is not None:
                        self.record_frames.append(frame)
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6, gripper_state=None):
        if gripper_state is None:
            gripper_state = self.CLOSED
        self.gripper_state = gripper_state
        # WidowX: больше шагов для устойчивого контакта (гриппер меньше Panda)
        t = 15 if t == 6 else t
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        gripper = self._gripper_action()
        self._set_latched_hand_target_qpos(gripper)
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = self._batch_action(np.hstack([qpos, gripper]))
            else:
                action = self._batch_action(np.hstack([qpos, np.zeros(6), gripper]))
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                try:
                    self.base_env.render_human()
                except (AttributeError, RuntimeError):
                    pass
            if self.record_frames is not None:
                frame = self._get_record_frame(obs)
                if frame is not None:
                    if self.normalize_frame_fn is not None:
                        frame = self.normalize_frame_fn(frame)
                    if frame is not None:
                        self.record_frames.append(frame)
        return obs, reward, terminated, truncated, info
