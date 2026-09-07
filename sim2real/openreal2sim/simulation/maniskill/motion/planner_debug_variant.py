import os
from pathlib import Path
import inspect

import mplib
import numpy as np
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


def _normalize_quat_np(q):
    q = np.asarray(q, dtype=np.float32).reshape(-1)[:4]
    norm = float(np.linalg.norm(q))
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def _nlerp_quat_np(q0, q1, alpha: float):
    q0 = _normalize_quat_np(q0)
    q1 = _normalize_quat_np(q1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    blended = (1.0 - float(alpha)) * q0 + float(alpha) * q1
    return _normalize_quat_np(blended)


def _quat_angle_rad(q0, q1) -> float:
    q0 = _normalize_quat_np(q0)
    q1 = _normalize_quat_np(q1)
    dot = float(np.clip(abs(np.dot(q0, q1)), -1.0, 1.0))
    return float(2.0 * np.arccos(dot))


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

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info

        self.base_pose = to_sapien_pose_mani_skill(base_pose)

        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode

        self.elapsed_steps = 0

        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None
        self._last_plan_branch_debug = None

        self.record_frames = record_frames  # Store frames list for video recording

    def _batch_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        num_envs = getattr(self.base_env.scene, "num_envs", 1)
        if num_envs and num_envs > 1 and action.ndim == 1:
            action = np.repeat(action[None, :], num_envs, axis=0)
        elif num_envs == 1 and action.ndim == 1:
            action = action[None, :]
        return action

    def _get_planning_urdf_path(self) -> str:
        return getattr(self.env_agent, "urdf_path")

    def _get_planning_srdf_path(self) -> str:
        planning_urdf_path = self._get_planning_urdf_path()
        if planning_urdf_path.endswith(".urdf"):
            return planning_urdf_path.replace(".urdf", ".srdf")
        return getattr(self.env_agent, "urdf_path").replace(".urdf", ".srdf")

    def _get_move_group(self) -> str:
        return getattr(self, "MOVE_GROUP", "eef")

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
            ik_status, goal_qposes = self._call_mplib_ik(
                np.concatenate([goal_pose_obj.p, goal_pose_obj.q]),
                start_qpos,
                mask=[],
                n_init_qpos=20,
                threshold=1e-3,
                return_closest=False,
            )
            if ik_status != "Success":
                return {"status": ik_status}
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

    def _planner_ik_supports_return_closest(self):
        try:
            params = inspect.signature(self.planner.IK).parameters
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Unable to introspect installed mplib IK signature for return_closest support."
            ) from exc
        return "return_closest" in params

    def _call_mplib_ik(
        self,
        goal_pose,
        start_qpos,
        *,
        mask=None,
        n_init_qpos: int = 20,
        threshold: float = 1e-3,
        return_closest: bool = False,
    ):
        ik_kwargs = dict(
            mask=mask,
            n_init_qpos=n_init_qpos,
            threshold=threshold,
        )
        if return_closest:
            if self._planner_ik_supports_return_closest():
                ik_kwargs["return_closest"] = True
            else:
                print(
                    f"{_Y}[PLANNER_MPLIB_DEBUG][IK] WARNING: installed mplib does not support "
                    f"return_closest=True; using legacy IK API without this flag.{_R}"
                )
        return self.planner.IK(
            self._to_mplib_goal_pose(goal_pose),
            start_qpos,
            **ik_kwargs,
        )

    def _get_record_frame(self, obs=None):
        """Get frame for video recording. Uses base_camera sensor or render."""
        if self.record_frames_from == "base_camera":
            if isinstance(obs, dict):
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
                if frame is not None:
                    return frame
            try:
                self.base_env.scene.update_render()
                self.base_env.capture_sensor_data()
                sensor = self.base_env.scene.sensors.get("base_camera")
                if sensor is not None:
                    sensor_obs = sensor.get_obs(rgb=True, depth=False, position=False, segmentation=False)
                    frame = sensor_obs.get("rgb", sensor_obs.get("Color"))
                    if frame is not None and hasattr(frame, "cpu"):
                        frame = frame.cpu().numpy()
                    if frame is not None and frame.ndim == 4:
                        frame = frame[0]
                    if frame is not None and frame.shape[-1] == 4:
                        frame = frame[..., :3]
                    if frame is not None:
                        return frame
            except Exception:
                pass
            if getattr(self.env, "render_mode", None) == "none":
                return None
        return self.env.render()

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        try:
            viewer = self.base_env.render_human()
            if viewer is None or not hasattr(viewer, 'window') or viewer.window is None:
                return
            # Keep the viewer refreshed, but do not block planner execution on a key press.
            self.base_env.render_human()
        except (AttributeError, RuntimeError):
            return

    def setup_planner(self):
        move_group = self._get_move_group()
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planning_urdf_path = self._get_planning_urdf_path()
        planning_srdf_path = self._get_planning_srdf_path()
        if self.debug:
            print(
                f"[PLANNER_DEBUG_VARIANT] Using planning assets: urdf={planning_urdf_path}, "
                f"srdf={planning_srdf_path}, move_group={move_group}"
            )
        planner = mplib.Planner(
            urdf=planning_urdf_path,
            srdf=planning_srdf_path,
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

    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        return None

    def _transform_pose_for_planning(self, target: sapien.Pose) -> sapien.Pose:
        return target

    def _transform_pose_for_planner_base(self, target: sapien.Pose) -> sapien.Pose:
        if self.base_pose is None:
            return target
        return self.base_pose.inv() * target

    def _prepare_target_pose_for_solver(self, target: sapien.Pose) -> sapien.Pose:
        return self._transform_pose_for_planner_base(
            self._transform_pose_for_planning(target)
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
        """Log robot base, active move-group EE, and target positions in world frame for IK debugging."""
        base_p = np.asarray(self.base_pose.p).flatten()[:3]
        tcp_pose = self._get_current_move_group_pose()
        tcp_sp = to_sapien_pose_mani_skill(tcp_pose)
        tcp_p = tcp_sp.p
        if isinstance(tcp_p, torch.Tensor):
            tcp_p = tcp_p.cpu().numpy()
        tcp_p = np.asarray(tcp_p)
        if tcp_p.ndim > 1:
            tcp_p = tcp_p[0]
        tcp_p = tcp_p.flatten()[:3]
        target_p = np.asarray(target_pose.p).flatten()[:3]
        dist_base_target = float(np.linalg.norm(target_p - base_p))
        dist_tcp_target = float(np.linalg.norm(target_p - tcp_p))
        move_group = self._get_move_group()
        print("[PLANNER_IK_DEBUG] ===== World frame positions =====")
        print(f"[PLANNER_IK_DEBUG] Robot base: {base_p}")
        print(f"[PLANNER_IK_DEBUG] EE ({move_group}):  {tcp_p}")
        print(f"[PLANNER_IK_DEBUG] Target:    {target_p}")
        print(f"[PLANNER_IK_DEBUG] Distance base->target: {dist_base_target:.4f} m")
        print(f"[PLANNER_IK_DEBUG] Distance EE({move_group})->target: {dist_tcp_target:.4f} m")
        print("[PLANNER_IK_DEBUG] (Compare with mplib 'IK Failed! Distance X' to detect frame mismatch)")

    def _get_planner_start_qpos(self) -> np.ndarray:
        return self.robot.get_qpos().cpu().numpy()[0]

    def _get_planner_joint_names(self):
        planner_joint_names = list(getattr(self.planner, "user_joint_names", []) or [])
        if planner_joint_names:
            return planner_joint_names
        return [joint.get_name() for joint in self.robot.get_active_joints()]

    @staticmethod
    def _format_array_debug(arr, precision: int = 4) -> str:
        return np.array2string(
            np.asarray(arr, dtype=np.float32).reshape(-1),
            precision=precision,
            suppress_small=True,
        )

    def _analyze_joint_delta_debug(self, start_qpos, target_qpos):
        start_qpos = np.asarray(start_qpos, dtype=np.float32).reshape(-1)
        target_qpos = np.asarray(target_qpos, dtype=np.float32).reshape(-1)
        joint_names = self._get_planner_joint_names()

        compare_start = start_qpos
        compare_target = target_qpos
        compare_joint_names = joint_names
        compare_scope = "full_qpos"
        shape_mismatch = start_qpos.shape != target_qpos.shape

        if shape_mismatch:
            arm_compare_len = min(int(getattr(self, "arm_dof", 0) or 0), start_qpos.shape[0], target_qpos.shape[0])
            if arm_compare_len <= 0:
                return None
            compare_start = start_qpos[:arm_compare_len]
            compare_target = target_qpos[:arm_compare_len]
            compare_joint_names = joint_names[:arm_compare_len]
            compare_scope = "arm_prefix"

        raw_delta = compare_target - compare_start
        wrapped_delta = ((raw_delta + np.pi) % (2.0 * np.pi)) - np.pi
        wrap_extra = raw_delta - wrapped_delta
        suspicious = np.where(np.abs(wrap_extra) > 1.0)[0]
        large_raw_delta = np.where(np.abs(raw_delta) > 1.0)[0]
        return {
            "compare_scope": compare_scope,
            "shape_mismatch": bool(shape_mismatch),
            "compare_start": compare_start.copy(),
            "compare_target": compare_target.copy(),
            "compare_joint_names": list(compare_joint_names),
            "raw_delta": raw_delta.copy(),
            "wrapped_delta": wrapped_delta.copy(),
            "wrap_extra": wrap_extra.copy(),
            "suspicious_indices": suspicious.copy(),
            "large_raw_delta_indices": large_raw_delta.copy(),
        }

    def _log_joint_delta_debug(self, label: str, start_qpos, target_qpos):
        analysis = self._analyze_joint_delta_debug(start_qpos, target_qpos)
        if analysis is None:
            return None
        compare_scope = analysis["compare_scope"]
        shape_mismatch = analysis["shape_mismatch"]
        compare_start = analysis["compare_start"]
        compare_target = analysis["compare_target"]
        compare_joint_names = analysis["compare_joint_names"]
        raw_delta = analysis["raw_delta"]
        wrapped_delta = analysis["wrapped_delta"]
        wrap_extra = analysis["wrap_extra"]
        suspicious = analysis["suspicious_indices"]
        large_raw_delta = analysis["large_raw_delta_indices"]
        if shape_mismatch:
            print(
                f"{_Y}[PLANNER_MPLIB_DEBUG][{label}] WARNING: qpos shape mismatch "
                f"start={np.asarray(start_qpos).shape} target={np.asarray(target_qpos).shape}{_R}"
            )
        if shape_mismatch or large_raw_delta.size > 0:
            print(
                f"[PLANNER_MPLIB_DEBUG][{label}] compare_scope={compare_scope} "
                f"start={self._format_array_debug(compare_start)} "
                f"target={self._format_array_debug(compare_target)} "
                f"raw_delta={self._format_array_debug(raw_delta)} "
                f"wrapped_delta={self._format_array_debug(wrapped_delta)}"
            )
        if large_raw_delta.size > 0:
            parts = []
            for idx in large_raw_delta[:10]:
                joint_name = (
                    compare_joint_names[idx]
                    if idx < len(compare_joint_names)
                    else f"joint{idx}"
                )
                parts.append(
                    f"{joint_name}: start={compare_start[idx]:+.4f} target={compare_target[idx]:+.4f} "
                    f"raw_delta={raw_delta[idx]:+.4f} wrapped_delta={wrapped_delta[idx]:+.4f}"
                )
            print(
                f"{_Y}[PLANNER_MPLIB_DEBUG][{label}] WARNING: large arm/joint delta(s): "
                f"{'; '.join(parts)}{_R}"
            )
        if suspicious.size == 0:
            return
        parts = []
        for idx in suspicious:
            joint_name = (
                compare_joint_names[idx]
                if idx < len(compare_joint_names)
                else f"joint{idx}"
            )
            parts.append(
                f"{joint_name}: start={compare_start[idx]:+.4f} target={compare_target[idx]:+.4f} "
                f"raw_delta={raw_delta[idx]:+.4f} wrapped_delta={wrapped_delta[idx]:+.4f} "
                f"extra={wrap_extra[idx]:+.4f}"
            )
            print(
                f"{_Y}[PLANNER_MPLIB_DEBUG][{label}] WARNING: suspicious >pi joint delta(s): "
                f"{'; '.join(parts)}{_R}"
            )
        return analysis

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

    def _log_mplib_ik_result(self, label: str, status, solutions, start_qpos) -> None:
        start_qpos = np.asarray(start_qpos, dtype=np.float32).reshape(-1)
        n_solutions = 0 if solutions is None else len(solutions)
        print(
            f"[PLANNER_MPLIB_DEBUG][{label}][response] status={status} n_solutions={n_solutions}"
        )
        if not solutions:
            return
        for idx, solution in enumerate(solutions[:3]):
            solution = np.asarray(solution, dtype=np.float32).reshape(-1)
            print(
                f"[PLANNER_MPLIB_DEBUG][{label}][solution {idx}] "
                f"qpos={self._format_array_debug(solution)}"
            )
            self._log_joint_delta_debug(f"{label}:solution_{idx}", start_qpos, solution)

    def _log_mplib_plan_result(self, label: str, result, start_qpos) -> None:
        status = result.get("status", "unknown")
        start_qpos = np.asarray(start_qpos, dtype=np.float32).reshape(-1)
        self._last_plan_branch_debug = {
            "label": label,
            "status": status,
            "n_steps": 0,
            "first_qpos": None,
            "last_qpos": None,
            "start_to_first": None,
            "start_to_last": None,
            "suspicious_step_count": 0,
        }
        positions = result.get("position", None)
        n_steps = 0 if positions is None else int(np.asarray(positions).shape[0])
        self._last_plan_branch_debug["n_steps"] = n_steps
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
        self._last_plan_branch_debug["first_qpos"] = first_qpos.copy()
        self._last_plan_branch_debug["last_qpos"] = last_qpos.copy()
        self._last_plan_branch_debug["start_to_first"] = self._log_joint_delta_debug(
            f"{label}:start_to_first",
            start_qpos,
            first_qpos,
        )
        self._last_plan_branch_debug["start_to_last"] = self._log_joint_delta_debug(
            f"{label}:start_to_last",
            start_qpos,
            last_qpos,
        )
        if n_steps > 1:
            step_deltas = np.diff(positions, axis=0)
            max_abs_step = np.max(np.abs(step_deltas), axis=0)
            print(
                f"[PLANNER_MPLIB_DEBUG][{label}][trajectory] "
                f"max_abs_step={self._format_array_debug(max_abs_step)}"
            )
            suspicious_steps = np.argwhere(np.abs(step_deltas) > np.pi)
            self._last_plan_branch_debug["suspicious_step_count"] = int(suspicious_steps.shape[0])
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

    def get_last_plan_branch_debug(self):
        return self._last_plan_branch_debug

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
        pose = self._transform_pose_for_planning(pose)
        start_qpos = self._get_planner_start_qpos()
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
        pose = self._transform_pose_for_planning(pose)
        start_qpos = self._get_planner_start_qpos()
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
        self._log_ik_debug_info(pose)
        pose = self._prepare_target_pose_for_solver(pose)
        start_qpos = self._get_planner_start_qpos()
        goal_pose = np.concatenate([pose.p, pose.q])
        self._log_mplib_request(
            "Screw",
            goal_pose,
            start_qpos,
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
        )
        result = self._plan_pose_screw(goal_pose, start_qpos, wrt_world=False)
        self._log_mplib_plan_result("Screw", result, start_qpos)
        if result["status"] != "Success":
            if self.debug:
                print(f"[PLANNER][screw] First attempt failed with status='{result['status']}'")
                print(f"[PLANNER][screw] Target pose p={pose.p}, q={pose.q}")
            start_qpos_retry = self._get_planner_start_qpos()
            self._log_mplib_request(
                "ScrewRetry",
                goal_pose,
                start_qpos_retry,
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
            )
            result = self._plan_pose_screw(goal_pose, start_qpos_retry, wrt_world=False)
            self._log_mplib_plan_result("ScrewRetry", result, start_qpos_retry)
            if result["status"] != "Success":
                print(result["status"])
                if self.debug:
                    print(f"[PLANNER][screw] Second attempt failed with status='{result['status']}'")
                    print(f"[PLANNER][screw] Current qpos={self._get_planner_start_qpos()}")
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

    def solve_ik(
        self,
        pose: sapien.Pose,
        threshold: float = 1e-3,
        mask=None,
        n_init_qpos: int = 20,
        return_closest: bool = False,
    ):
        _log_planner_dispatch(
            f"{type(self).__name__}.solve_ik(threshold={threshold}, mask={mask}, "
            f"n_init_qpos={n_init_qpos}, return_closest={return_closest})"
        )
        pose = to_sapien_pose_mani_skill(pose)
        self._update_grasp_visual(pose)
        self._log_ik_debug_info(pose)
        if self.debug:
            print(
                f"[PLANNER_DEBUG_VARIANT][solve_ik] start_qpos={np.array2string(self._get_current_qpos(), precision=4, suppress_small=True)}"
            )
        pose = self._prepare_target_pose_for_solver(pose)
        if self.debug:
            print(
                f"[PLANNER_DEBUG_VARIANT][solve_ik] planner_frame_target_p={np.array2string(np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3], precision=4, suppress_small=True)} "
                f"planner_frame_target_q={np.array2string(np.asarray(pose.q, dtype=np.float32).reshape(-1)[:4], precision=4, suppress_small=True)} "
                f"threshold={threshold:.4f} mask={mask} n_init_qpos={n_init_qpos}"
            )
        start_qpos = self._get_planner_start_qpos()
        goal_pose = np.concatenate([pose.p, pose.q])
        self._log_mplib_request(
            "IK",
            goal_pose,
            start_qpos,
            threshold=threshold,
            mask=mask,
            n_init_qpos=n_init_qpos,
            return_closest=return_closest,
        )
        ik_result = self._call_mplib_ik(
            goal_pose,
            start_qpos,
            mask=mask,
            n_init_qpos=n_init_qpos,
            threshold=threshold,
            return_closest=return_closest,
        )
        if isinstance(ik_result, tuple) and len(ik_result) == 2:
            status, solutions = ik_result
            normalized_solutions = solutions
            if return_closest:
                if solutions is None:
                    normalized_solutions = []
                else:
                    normalized_solutions = [np.asarray(solutions, dtype=np.float32).reshape(-1)]
            self._log_mplib_ik_result("IK", status, normalized_solutions, start_qpos)
            return status, normalized_solutions
        else:
            print(
                f"{_Y}[PLANNER_MPLIB_DEBUG][IK] WARNING: unexpected IK return type="
                f"{type(ik_result).__name__}{_R}"
            )
            return ik_result
        return ik_result

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
            builder.set_initial_pose(sapien.Pose(p=initial_pose[:3]))
        elif hasattr(initial_pose, "p"):
            builder.set_initial_pose(initial_pose)
    grasp_pose_visual_width = 0.02
    grasp_width = 0.08

    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[1.0, 1.0, 0.0, 1.0]),
    )

    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0.0, 1.0, 0.0, 1.0]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0.0, 1.0, 0.0, 1.0]),
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
        half_size=[0.05, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0.0, 0.8, 1.0, 1.0]),
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
        half_size=[0.05, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1.0, 0.2, 0.2, 1.0]),
    )
    grasp_pose_visual = builder.build_kinematic(name="grasp_pose_visual")
    return grasp_pose_visual


def build_pose_axes_visual(
    scene: ManiSkillScene,
    initial_pose=None,
    actor_name: str = "pose_axes_visual",
):
    """Build a large RGB axes marker that is easy to spot in the viewer."""
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

    def _prepare_target_pose_for_solver(self, target: sapien.Pose) -> sapien.Pose:
        return self._transform_pose_for_planning(target)


class RC5ArmMotionPlanningSolver(BaseMotionPlanningSolver):
    """Planner-only RC5 arm solver for the debug planner variant.

    Canonical debug-planner semantics now follow the same EE contract as the
    working RC5 teleop path: `right_tcp_link` is the default solve link.
    Legacy object-profile calibrations may still be authored in `prehand`
    semantics and must be converted explicitly at the script layer.
    """

    MOVE_GROUP = "right_tcp_link"
    _PLANNING_ASSET_DIR_DEBUG = (
        Path(__file__).resolve().parent.parent / "robot_assets" / "rc5_aero_hand_planning_debug_variant"
    )
    _PLANNING_ASSET_DIR_MAIN = (
        Path(__file__).resolve().parent.parent / "robot_assets" / "rc5_aero_hand" / "urdf_rc5_right_hand"
    )
    _USE_MAIN_ASSETS = os.environ.get("RC5_DEBUG_PLANNER_USE_MAIN_ASSETS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

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

    def _use_main_assets(self) -> bool:
        return os.environ.get("RC5_DEBUG_PLANNER_USE_MAIN_ASSETS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _get_move_group(self) -> str:
        planner_cfg = getattr(self.base_env, "_debug_planner_config", {}) or {}
        move_group_override = str(
            planner_cfg.get(
                "planner_rc5_move_group",
                os.environ.get("RC5_DEBUG_PLANNER_MOVE_GROUP", self.MOVE_GROUP),
            )
        ).strip()
        if move_group_override in {"prehand", "right_tcp_link"}:
            return move_group_override
        if move_group_override:
            print(
                f"{_Y}[RC5_KIN_DEBUG] WARNING: unsupported RC5 debug move_group='{move_group_override}', "
                f"falling back to '{self.MOVE_GROUP}'. Supported=['prehand', 'right_tcp_link']{_R}"
            )
        return self.MOVE_GROUP

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
            print(f"[RC5_KIN_DEBUG][{context}] prehand {self._format_pose_np(prehand_pose)}")
        if tcp_pose is not None:
            print(f"[RC5_KIN_DEBUG][{context}] right_tcp_link {self._format_pose_np(tcp_pose)}")
        if move_group_pose is not None:
            print(
                f"[RC5_KIN_DEBUG][{context}] active_move_group('{move_group}') "
                f"{self._format_pose_np(move_group_pose)}"
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
            print(f"[RC5_KIN_DEBUG][{context}] target_world {self._format_pose_np(target_pose)}")
            transformed_target = self._prepare_target_pose_for_solver(target_pose)
            print(
                f"[RC5_KIN_DEBUG][{context}] target_after_prepare {self._format_pose_np(transformed_target)}"
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
                print(f"[RC5_KIN_DEBUG][{context}] dist_target_to_{label}={dist:.4f} m")

    def _get_asset_dir(self) -> Path:
        asset_override = os.environ.get("OPENR2S_RC5_PLANNING_ASSET_DIR", "").strip()
        if asset_override:
            return Path(asset_override)
        return self._PLANNING_ASSET_DIR_MAIN if self._use_main_assets() else self._PLANNING_ASSET_DIR_DEBUG

    def _get_planning_urdf_path(self) -> str:
        asset_override = os.environ.get("OPENR2S_RC5_PLANNING_ASSET_DIR", "").strip()
        if asset_override:
            return self._resolve_override_asset_file(asset_override, ".urdf")
        if self._use_main_assets():
            return str(self._get_asset_dir() / "Robot _with_right_hand_colored_visual_continuous.urdf")
        return str(self._get_asset_dir() / "rc5_aero_hand_planning_debug_variant.urdf")

    def _get_planning_srdf_path(self) -> str:
        asset_override = os.environ.get("OPENR2S_RC5_PLANNING_ASSET_DIR", "").strip()
        if asset_override:
            return self._resolve_override_asset_file(asset_override, ".srdf")
        if self._use_main_assets():
            return str(self._get_asset_dir() / "Robot _with_right_hand.srdf")
        return str(self._get_asset_dir() / "rc5_aero_hand_planning_debug_variant.srdf")

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
        if os.environ.get("OPENR2S_RC5_PLANNING_ASSET_DIR", "").strip():
            print(
                "[PLANNER_DEBUG_VARIANT] OPENR2S_RC5_PLANNING_ASSET_DIR is set "
                f"-> using override planner asset dir: {self._get_asset_dir()}"
            )
        elif self._use_main_assets():
            print(
                "[PLANNER_DEBUG_VARIANT] RC5_DEBUG_PLANNER_USE_MAIN_ASSETS=1 "
                "-> using main RC5 URDF/SRDF with adjacent convex STL assets"
            )
        self.arm_dof = len(getattr(self.env_agent, "arm_joint_names", [])) or 6
        self.hand_open_qpos = np.asarray(self.env_agent.hand_open_qpos, dtype=np.float32)
        self.hand_close_qpos = np.asarray(self.env_agent.hand_close_qpos, dtype=np.float32)
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
        self._log_rc5_kinematics_debug("solver_init")

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

    def _get_current_qpos(self) -> np.ndarray:
        current_qpos = self.robot.get_qpos()
        if hasattr(current_qpos, "cpu"):
            current_qpos = current_qpos.cpu().numpy()
        if getattr(current_qpos, "ndim", 1) > 1:
            current_qpos = current_qpos[0]
        return np.asarray(current_qpos, dtype=np.float32).reshape(-1)

    def _get_planner_start_qpos(self) -> np.ndarray:
        current_qpos = self._get_current_qpos()
        sim_joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner_joint_names = list(getattr(self.planner, "user_joint_names", sim_joint_names))
        if planner_joint_names == sim_joint_names:
            return current_qpos
        qpos_by_name = {
            name: current_qpos[idx]
            for idx, name in enumerate(sim_joint_names)
        }
        reordered = np.asarray([qpos_by_name[name] for name in planner_joint_names], dtype=np.float32)
        if self.debug:
            print(f"[RC5_DEBUG][planner_qpos] sim_joint_names={sim_joint_names}")
            print(f"[RC5_DEBUG][planner_qpos] planner_joint_names={planner_joint_names}")
            print(f"[RC5_DEBUG][planner_qpos] reordered current qpos for mplib")
        return reordered

    def _get_current_arm_qpos(self) -> np.ndarray:
        current_qpos = self._get_current_qpos()
        return np.asarray(current_qpos[: self.arm_dof], dtype=np.float32)

    def _format_pose_np(self, pose: sapien.Pose) -> str:
        pose = to_sapien_pose_mani_skill(pose)
        p = np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3]
        q = np.asarray(pose.q, dtype=np.float32).reshape(-1)[:4]
        return (
            f"p={np.array2string(p, precision=4, suppress_small=True)} "
            f"q={np.array2string(q, precision=4, suppress_small=True)}"
        )

    def _log_state_snapshot(self, label: str, target_pose: sapien.Pose = None) -> None:
        current_qpos = self._get_current_qpos()
        arm_qpos = np.asarray(current_qpos[: self.arm_dof], dtype=np.float32)
        hand_qpos = np.asarray(
            current_qpos[self.arm_dof : self.arm_dof + len(self.hand_target_qpos)],
            dtype=np.float32,
        )
        tcp_pose = to_sapien_pose_mani_skill(self.base_env.agent.tcp.pose)
        tcp_p = np.asarray(tcp_pose.p, dtype=np.float32).reshape(-1)[:3]
        tcp_q = np.asarray(tcp_pose.q, dtype=np.float32).reshape(-1)[:4]
        base_p = np.asarray(self.base_pose.p, dtype=np.float32).reshape(-1)[:3]
        base_q = np.asarray(self.base_pose.q, dtype=np.float32).reshape(-1)[:4]
        print(
            f"[PLANNER_DEBUG_VARIANT][{label}] control_mode={self.control_mode} "
            f"qpos_dim={current_qpos.shape[0]} arm_dof={self.arm_dof}"
        )
        print(
            f"[PLANNER_DEBUG_VARIANT][{label}] base_pose_p={np.array2string(base_p, precision=4, suppress_small=True)} "
            f"base_pose_q={np.array2string(base_q, precision=4, suppress_small=True)}"
        )
        print(
            f"[PLANNER_DEBUG_VARIANT][{label}] current_arm_qpos={np.array2string(arm_qpos, precision=4, suppress_small=True)}"
        )
        print(
            f"[PLANNER_DEBUG_VARIANT][{label}] current_hand_qpos={np.array2string(hand_qpos, precision=4, suppress_small=True)}"
        )
        print(
            f"[PLANNER_DEBUG_VARIANT][{label}] current_tcp_world_p={np.array2string(tcp_p, precision=4, suppress_small=True)} "
            f"current_tcp_world_q={np.array2string(tcp_q, precision=4, suppress_small=True)}"
        )
        if target_pose is not None:
            target_pose = to_sapien_pose_mani_skill(target_pose)
            transformed_pose = self._prepare_target_pose_for_solver(target_pose)
            transformed_p = np.asarray(transformed_pose.p, dtype=np.float32).reshape(-1)[:3]
            transformed_q = np.asarray(transformed_pose.q, dtype=np.float32).reshape(-1)[:4]
            print(
                f"[PLANNER_DEBUG_VARIANT][{label}] target_world_{self._format_pose_np(target_pose)}"
            )
            print(
                f"[PLANNER_DEBUG_VARIANT][{label}] target_planner_base_p={np.array2string(transformed_p, precision=4, suppress_small=True)} "
                f"target_planner_base_q={np.array2string(transformed_q, precision=4, suppress_small=True)}"
            )

    def _compose_action(self, arm_qpos: np.ndarray, arm_qvel: np.ndarray = None) -> np.ndarray:
        hand_qpos = self.hand_target_qpos
        if self.control_mode == "pd_joint_pos_vel":
            if arm_qvel is None:
                arm_qvel = np.zeros_like(arm_qpos)
            hand_qvel = np.zeros_like(hand_qpos)
            return np.hstack([arm_qpos, hand_qpos, arm_qvel, hand_qvel])
        return np.hstack([arm_qpos, hand_qpos])

    def _get_arm_wraparound_mask(self) -> np.ndarray:
        mask = np.zeros(self.arm_dof, dtype=bool)
        active_joints = list(self.robot.get_active_joints())[: self.arm_dof]
        for idx, joint in enumerate(active_joints):
            try:
                limits = joint.get_limits()
            except Exception:
                continue
            if hasattr(limits, "cpu"):
                limits = limits.cpu().numpy()
            limits = np.asarray(limits, dtype=np.float32).reshape(-1)
            if limits.shape[0] < 2:
                continue
            lower, upper = float(limits[0]), float(limits[1])
            if not np.isfinite(lower) or not np.isfinite(upper):
                mask[idx] = True
                continue
            if (upper - lower) >= (2.0 * np.pi - 0.2):
                mask[idx] = True
        return mask

    def _normalize_arm_qpos_near_reference(
        self,
        raw_arm_qpos: np.ndarray,
        reference_arm_qpos: np.ndarray,
        warning_label: str,
    ) -> np.ndarray:
        raw_arm_qpos = np.asarray(raw_arm_qpos, dtype=np.float32).reshape(-1)
        reference_arm_qpos = np.asarray(reference_arm_qpos, dtype=np.float32).reshape(-1)
        adjusted = raw_arm_qpos.copy()
        wrap_mask = self._get_arm_wraparound_mask()
        if not np.any(wrap_mask):
            return adjusted
        active_joints = list(self.robot.get_active_joints())[: self.arm_dof]
        for idx in np.where(wrap_mask)[0]:
            raw_value = float(adjusted[idx])
            reference_value = float(reference_arm_qpos[idx])
            candidate_values = [raw_value + 2.0 * np.pi * k for k in range(-2, 3)]

            try:
                limits = active_joints[idx].get_limits()
                if hasattr(limits, "cpu"):
                    limits = limits.cpu().numpy()
                limits = np.asarray(limits, dtype=np.float32).reshape(-1)
            except Exception:
                limits = np.array([], dtype=np.float32)

            valid_candidates = []
            if limits.shape[0] >= 2 and np.isfinite(float(limits[0])) and np.isfinite(float(limits[1])):
                lower, upper = float(limits[0]), float(limits[1])
                for candidate in candidate_values:
                    if lower - 1e-4 <= candidate <= upper + 1e-4:
                        valid_candidates.append(candidate)
            else:
                valid_candidates = candidate_values

            if len(valid_candidates) == 0:
                valid_candidates = [raw_value]

            best_candidate = min(
                valid_candidates,
                key=lambda candidate: abs(candidate - reference_value),
            )
            adjusted[idx] = float(best_candidate)

        changed = np.where(np.abs(adjusted - raw_arm_qpos) > 1e-4)[0]
        if changed.size > 0:
            joint_names = list(getattr(self.env_agent, "arm_joint_names", []))[: self.arm_dof]
            parts = []
            for idx in changed:
                joint_name = joint_names[idx] if idx < len(joint_names) else f"joint{idx}"
                parts.append(
                    f"{joint_name}: raw={raw_arm_qpos[idx]:+.4f} -> normalized={adjusted[idx]:+.4f} "
                    f"(ref={reference_arm_qpos[idx]:+.4f})"
                )
            print(
                f"{_Y}[{warning_label}] WARNING: normalized RC5 arm qpos across 2pi seam "
                f"to keep the nearest angular branch: {'; '.join(parts)}{_R}"
            )
        return adjusted

    def _unwrap_arm_qpos_for_execution(
        self,
        raw_arm_qpos: np.ndarray,
        reference_arm_qpos: np.ndarray,
    ) -> np.ndarray:
        return self._normalize_arm_qpos_near_reference(
            raw_arm_qpos,
            reference_arm_qpos,
            warning_label="RC5_WRAP_DEBUG",
        )

    def _compute_move_group_fk_position_error(
        self,
        full_qpos: np.ndarray,
        planner_target_pose: sapien.Pose,
    ) -> float:
        full_qpos = np.asarray(full_qpos, dtype=np.float64).reshape(-1)
        planner_target_p = np.asarray(planner_target_pose.p, dtype=np.float64).reshape(-1)[:3]
        try:
            pinocchio_model = getattr(self.planner, "pinocchio_model", None)
            move_group_link_id = int(getattr(self.planner, "move_group_link_id"))
            if pinocchio_model is None:
                return float("inf")
            pinocchio_model.compute_forward_kinematics(full_qpos)
            fk_pose = pinocchio_model.get_link_pose(move_group_link_id)
            if hasattr(fk_pose, "p"):
                fk_p = np.asarray(fk_pose.p, dtype=np.float64).reshape(-1)[:3]
            else:
                fk_pose = np.asarray(fk_pose, dtype=np.float64).reshape(-1)
                fk_p = fk_pose[:3]
            return float(np.linalg.norm(fk_p - planner_target_p))
        except Exception as exc:
            if self.debug:
                print(
                    f"{_Y}[PLANNER_LOCAL_IK_FK] WARNING: failed to compute FK candidate error: "
                    f"{type(exc).__name__}: {exc}{_R}"
                )
            return float("inf")

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        prev_arm_qpos = self._get_current_arm_qpos()
        follow_path_trace = []
        for i in range(n_step + refine_steps):
            raw_arm_qpos = np.asarray(result["position"][min(i, n_step - 1)], dtype=np.float32)
            arm_qpos = self._unwrap_arm_qpos_for_execution(raw_arm_qpos, prev_arm_qpos)
            arm_qvel = None
            if self.control_mode == "pd_joint_pos_vel":
                arm_qvel = result["velocity"][min(i, n_step - 1)]
            action = self._batch_action(self._compose_action(arm_qpos, arm_qvel))
            obs, reward, terminated, truncated, info = self.env.step(action)
            # Use the actual post-step arm state as the unwrap reference for the next waypoint.
            # Reusing the commanded target here can drift seam selection when execution lags.
            actual_arm_qpos = self._get_current_arm_qpos()
            prev_arm_qpos = actual_arm_qpos
            compare_len = min(arm_qpos.shape[0], actual_arm_qpos.shape[0])
            arm_tracking_err = np.asarray([], dtype=np.float32)
            if compare_len > 0:
                arm_tracking_err = np.asarray(
                    actual_arm_qpos[:compare_len] - arm_qpos[:compare_len],
                    dtype=np.float32,
                )
            follow_path_trace.append(
                {
                    "step_index": int(i),
                    "commanded_arm_qpos": np.asarray(arm_qpos, dtype=np.float32).copy(),
                    "actual_arm_qpos": np.asarray(actual_arm_qpos, dtype=np.float32).copy(),
                    "arm_tracking_err": arm_tracking_err.copy(),
                }
            )
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
        self._last_follow_path_trace = follow_path_trace
        return obs, reward, terminated, truncated, info

    def _hold_current_arm_and_step(self, t: int = 6):
        arm_qpos = self._get_current_arm_qpos()
        obs = reward = terminated = truncated = info = None
        for _ in range(t):
            action = self._batch_action(self._compose_action(arm_qpos))
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

    def move_to_pose_with_local_ik(
        self,
        pose: sapien.Pose,
        dry_run: bool = False,
        min_steps: int = 12,
        max_steps: int = 40,
        joint_step_size: float = 0.03,
        pose_tolerance: float = 0.01,
        max_refine_iters: int = 3,
        hard_failure_tolerance: float = 0.1,
    ):
        pose = to_sapien_pose_mani_skill(pose)
        current_ee_pose = to_sapien_pose_mani_skill(self._get_current_move_group_pose())
        current_p = np.asarray(current_ee_pose.p, dtype=np.float32).reshape(-1)[:3]
        current_q = np.asarray(current_ee_pose.q, dtype=np.float32).reshape(-1)[:4]
        target_p = np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3]
        target_q = np.asarray(pose.q, dtype=np.float32).reshape(-1)[:4]
        target_dist = float(np.linalg.norm(target_p - current_p))
        target_angle = _quat_angle_rad(current_q, target_q)
        if (
            not dry_run
            and type(self).__name__ == "RC5ArmMotionPlanningSolver"
            and self._get_move_group() == "right_tcp_link"
        ):
            max_cart_step = 0.08
            max_rot_step = 0.45
            n_interp_steps = int(
                max(
                    1,
                    np.ceil(target_dist / max_cart_step),
                    np.ceil(target_angle / max_rot_step),
                )
            )
            if n_interp_steps > 1:
                if self.debug:
                    print(
                        f"[PLANNER][local_ik] RC5 staged Cartesian interpolation enabled: "
                        f"n_interp_steps={n_interp_steps} target_dist={target_dist:.4f} m "
                        f"target_angle={target_angle:.4f} rad move_group='{self._get_move_group()}'"
                    )
                for interp_idx in range(n_interp_steps):
                    alpha = float(interp_idx + 1) / float(n_interp_steps)
                    is_final_interp_step = interp_idx == (n_interp_steps - 1)
                    staged_pose = sapien.Pose(
                        p=((1.0 - alpha) * current_p + alpha * target_p).astype(np.float32),
                        q=_nlerp_quat_np(current_q, target_q, alpha),
                    )
                    stage_result = self._move_to_pose_with_local_ik_single_target(
                        staged_pose,
                        dry_run=False,
                        min_steps=min_steps,
                        max_steps=max_steps,
                        joint_step_size=joint_step_size,
                        pose_tolerance=(
                            float(pose_tolerance)
                            if is_final_interp_step
                            else max(float(pose_tolerance), 0.02)
                        ),
                        max_refine_iters=max_refine_iters,
                        hard_failure_tolerance=max(float(hard_failure_tolerance), 0.12),
                        stage_label=f"staged_step={interp_idx + 1}/{n_interp_steps}",
                    )
                    if stage_result == -1:
                        return -1
                return 0
        return self._move_to_pose_with_local_ik_single_target(
            pose,
            dry_run=dry_run,
            min_steps=min_steps,
            max_steps=max_steps,
            joint_step_size=joint_step_size,
            pose_tolerance=pose_tolerance,
            max_refine_iters=max_refine_iters,
            hard_failure_tolerance=hard_failure_tolerance,
        )

    def _move_to_pose_with_local_ik_single_target(
        self,
        pose: sapien.Pose,
        dry_run: bool = False,
        min_steps: int = 12,
        max_steps: int = 40,
        joint_step_size: float = 0.03,
        pose_tolerance: float = 0.01,
        max_refine_iters: int = 3,
        hard_failure_tolerance: float = 0.1,
        stage_label: str = None,
    ):
        pose = to_sapien_pose_mani_skill(pose)
        _log_planner_dispatch(
            f"{type(self).__name__}.move_to_pose_with_local_ik(dry_run={dry_run}, max_refine_iters={max_refine_iters})"
        )
        self._update_grasp_visual(pose)
        self._log_ik_debug_info(pose)
        if self.debug:
            if stage_label:
                print(f"[PLANNER][local_ik] {stage_label}")
            self._log_state_snapshot("local_ik:start", target_pose=pose)
        target_p = np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3]
        planner_target_pose = self._prepare_target_pose_for_solver(pose)
        last_result = -1

        for refine_idx in range(max(max_refine_iters, 1)):
            if self.debug:
                print(
                    f"[PLANNER][local_ik] refine_iter={refine_idx + 1}/{max(max_refine_iters, 1)} "
                    f"target_pose={self._format_pose_np(pose)}"
                )
            status, solutions = self.solve_ik(
                pose,
                threshold=1e-3,
                return_closest=(
                    type(self).__name__ == "RC5ArmMotionPlanningSolver"
                    and self._get_move_group() == "right_tcp_link"
                ),
            )
            if len(solutions) == 0:
                if self.debug:
                    print(f"[PLANNER][local_ik] IK failed with status='{status}' at iter {refine_idx + 1}")
                return -1

            current_arm_qpos = self._get_current_arm_qpos()
            best_arm_qpos = None
            best_score = None
            best_candidate_idx = None
            for solution_idx, solution in enumerate(solutions):
                solution = np.asarray(solution, dtype=np.float32).reshape(-1)
                if solution.shape[0] < self.arm_dof:
                    continue
                arm_qpos = self._normalize_arm_qpos_near_reference(
                    solution[: self.arm_dof],
                    current_arm_qpos,
                    warning_label=f"PLANNER_LOCAL_IK_CANDIDATE:{solution_idx}",
                )
                candidate_solution = solution.copy()
                candidate_solution[: self.arm_dof] = arm_qpos
                joint_delta_score = float(np.linalg.norm(arm_qpos - current_arm_qpos))
                fk_pos_err = self._compute_move_group_fk_position_error(
                    candidate_solution,
                    planner_target_pose,
                )
                score = (fk_pos_err, joint_delta_score)
                if self.debug:
                    print(
                        f"[PLANNER][local_ik] candidate={solution_idx} "
                        f"fk_pos_err={fk_pos_err:.4f} m "
                        f"joint_delta_score={joint_delta_score:.4f} "
                        f"arm_qpos={np.array2string(arm_qpos, precision=4, suppress_small=True)}"
                    )
                if best_score is None or score < best_score:
                    best_score = score
                    best_arm_qpos = arm_qpos
                    best_candidate_idx = int(solution_idx)

            if best_arm_qpos is None:
                if self.debug:
                    print("[PLANNER][local_ik] No IK solution had enough arm DOFs")
                return -1

            if self.debug:
                print(
                    f"[PLANNER][local_ik] selected_candidate={best_candidate_idx} "
                    f"selected_score={best_score} "
                    f"selected_best_arm_qpos={np.array2string(best_arm_qpos, precision=4, suppress_small=True)}"
                )

            n_steps = int(
                np.ceil(np.max(np.abs(best_arm_qpos - current_arm_qpos)) / max(joint_step_size, 1e-3))
            )
            n_steps = max(min_steps, min(max_steps, n_steps))
            path = np.linspace(current_arm_qpos, best_arm_qpos, num=n_steps, dtype=np.float32)
            result = {"status": status, "position": path}
            if self.debug:
                local_ik_label = "LocalIKPath"
                if stage_label:
                    local_ik_label = f"{local_ik_label}:{stage_label}"
                self._log_mplib_plan_result(local_ik_label, result, current_arm_qpos)
            if dry_run:
                return result

            if self.debug:
                print(
                    f"[PLANNER][local_ik] executing path with n_steps={n_steps} "
                    f"best_arm_qpos={np.array2string(best_arm_qpos, precision=4, suppress_small=True)} "
                    f"current_arm_qpos={np.array2string(current_arm_qpos, precision=4, suppress_small=True)}"
                )
            last_result = self.follow_path(result, refine_steps=0)
            current_ee_pose = to_sapien_pose_mani_skill(self._get_current_move_group_pose())
            current_ee_p = np.asarray(current_ee_pose.p, dtype=np.float32).reshape(-1)[:3]
            pos_err = float(np.linalg.norm(current_ee_p - target_p))
            if self.debug:
                print(
                    f"[PLANNER][local_ik] iter {refine_idx + 1}/{max(max_refine_iters, 1)} "
                    f"move_group='{self._get_move_group()}' ee_pos_err={pos_err:.4f} m"
                )
            if pos_err <= float(pose_tolerance):
                break
            if pos_err > float(hard_failure_tolerance):
                if self.debug:
                    print(
                        f"[PLANNER][local_ik] final move_group='{self._get_move_group()}' ee_pos_err={pos_err:.4f} m exceeds "
                        f"hard_failure_tolerance={float(hard_failure_tolerance):.4f} m"
                    )
                return -1

        if last_result != -1:
            current_ee_pose = to_sapien_pose_mani_skill(self._get_current_move_group_pose())
            current_ee_p = np.asarray(current_ee_pose.p, dtype=np.float32).reshape(-1)[:3]
            final_pos_err = float(np.linalg.norm(current_ee_p - target_p))
            if final_pos_err > float(hard_failure_tolerance):
                if self.debug:
                    print(
                        f"[PLANNER][local_ik] terminal move_group='{self._get_move_group()}' ee_pos_err={final_pos_err:.4f} m exceeds "
                        f"hard_failure_tolerance={float(hard_failure_tolerance):.4f} m"
                    )
                return -1

        return last_result

    def open_gripper(self, t=6):
        self._set_latched_hand_target_qpos(self.hand_open_qpos)
        return self._hold_current_arm_and_step(t=t)

    def close_gripper(self, t=6):
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

    def _prepare_target_pose_for_solver(self, target: sapien.Pose) -> sapien.Pose:
        return self._transform_pose_for_planning(target)

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
