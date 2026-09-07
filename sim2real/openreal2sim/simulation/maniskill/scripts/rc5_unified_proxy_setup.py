from __future__ import annotations

import copy
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import sapien
import torch

from openreal2sim.simulation.maniskill.scripts.rc5_unified_bootstrap import (
    apply_lighting_profile_overrides as shared_apply_lighting_profile_overrides,
    apply_hand_contact_profile_overrides as shared_apply_hand_contact_profile_overrides,
    apply_hand_controller_profile_overrides as shared_apply_hand_controller_profile_overrides,
    apply_hand_pose_config_to_agent as shared_apply_hand_pose_config_to_agent,
    apply_teleop_profile_overrides as shared_apply_teleop_profile_overrides,
    detect_config_key as shared_detect_config_key,
    load_simulation_config_sections,
    pick_simulation_value,
    resolve_rc5_move_group,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_execution import (
    is_ee_delta_control_mode as shared_is_ee_delta_control_mode,
    resolve_effective_control_mode as shared_resolve_effective_control_mode,
    select_startup_stabilize_control_mode as shared_select_startup_stabilize_control_mode,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_logging import (
    is_debug_enabled,
    logger,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_lowlevel import (
    get_required_pregrasp_joint_guard_config,
    get_required_planner_waypoints_config,
)
from openreal2sim.simulation.maniskill.utils.scene_loader import DEFAULT_CAMERAS_CONFIG, load_cameras_config

DEFAULT_RENDERER_MAX_NUM_MATERIALS = 20000
DEFAULT_RENDERER_MAX_NUM_TEXTURES = 20000
RC5_CANONICAL_TARGET_FRAME = "right_tcp_link"
RC5_EXPECTED_QPOS_SIZE = 22
_Y = "\033[33m"
_R = "\033[0m"


def _load_openreal2sim_env_class():
    from openreal2sim.simulation.maniskill.envs import OpenReal2SimEnv

    return OpenReal2SimEnv


def _to_numpy_1d(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr[0]
    return arr.astype(np.float32).copy()


def _get_robot_qpos(env):
    return _to_numpy_1d(env.unwrapped.agent.robot.get_qpos())


def _to_numpy_qpos(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _get_active_hand_joint_names(agent):
    hand_joint_names = list(getattr(agent, "hand_joint_names", []))
    if hand_joint_names:
        return hand_joint_names
    return []


def _apply_object_specific_robot_init_qpos_profile(config_overrides, *, cli_override):
    profiles = config_overrides.get("robot_init_qpos_profiles")
    profile_by_object = config_overrides.get("robot_init_qpos_profile_by_object")
    if profiles is None and profile_by_object is None:
        return
    if not isinstance(profiles, dict):
        raise TypeError("robot_init_qpos_profiles must be a mapping.")
    if not isinstance(profile_by_object, dict):
        raise TypeError("robot_init_qpos_profile_by_object must be a mapping.")
    if cli_override:
        config_overrides["resolved_robot_init_qpos_profile"] = "cli_override"
        print("[Info] Explicit CLI robot_init_qpos overrides object-specific startup profile.")
        return

    object_id = str(config_overrides.get("manip_object_id") or "").strip()
    if not object_id:
        raise RuntimeError(
            "robot_init_qpos_profile_by_object is configured, but manip_object_id is unset."
        )
    if object_id not in profile_by_object:
        available = ", ".join(sorted(str(key) for key in profile_by_object)) or "<none>"
        raise RuntimeError(
            f"Object '{object_id}' has no startup profile mapping in "
            "robot_init_qpos_profile_by_object. "
            f"Configured objects: {available}"
        )

    profile_name = str(profile_by_object[object_id] or "").strip()
    if not profile_name or profile_name not in profiles:
        available = ", ".join(sorted(str(key) for key in profiles)) or "<none>"
        raise RuntimeError(
            f"Object '{object_id}' references unknown robot_init_qpos profile "
            f"'{profile_name}'. Configured profiles: {available}"
        )
    try:
        qpos = np.asarray(profiles[profile_name], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"robot_init_qpos_profiles['{profile_name}'] must contain numeric values."
        ) from exc
    if qpos.shape != (RC5_EXPECTED_QPOS_SIZE,):
        raise ValueError(
            f"robot_init_qpos_profiles['{profile_name}'] must contain exactly "
            f"{RC5_EXPECTED_QPOS_SIZE} values, got shape {qpos.shape}."
        )
    if not np.all(np.isfinite(qpos)):
        raise ValueError(
            f"robot_init_qpos_profiles['{profile_name}'] contains non-finite values."
        )

    config_overrides["robot_init_qpos"] = qpos.tolist()
    config_overrides["resolved_robot_init_qpos_profile"] = profile_name
    print(
        "[Info] Object-specific robot_init_qpos profile: "
        f"manip_object_id={object_id} profile={profile_name!r} len={len(qpos)}"
    )


def ensure_hand_defaults(agent):
    hand_joint_names = _get_active_hand_joint_names(agent)
    if not hand_joint_names:
        raise RuntimeError(
            f"Agent uid='{getattr(agent, 'uid', 'unknown')}' does not expose hand_joint_names. "
            "Unified RC5 runtime requires explicit hand_joint_names."
        )
    if not hasattr(agent, "hand_joint_names"):
        agent.hand_joint_names = list(hand_joint_names)
    if not hasattr(agent, "hand_open_qpos"):
        raise RuntimeError(
            f"Agent uid='{getattr(agent, 'uid', 'unknown')}' is missing hand_open_qpos. "
            "Unified RC5 runtime requires explicit open-hand defaults."
        )
    if not hasattr(agent, "hand_close_qpos"):
        raise RuntimeError(
            f"Agent uid='{getattr(agent, 'uid', 'unknown')}' is missing hand_close_qpos. "
            "Unified RC5 runtime requires explicit close-hand defaults."
        )


def log_hand_joint_state(agent, *, label: str):
    if not is_debug_enabled():
        return
    hand_joint_names = list(getattr(agent, "hand_joint_names", []) or [])
    if not hand_joint_names:
        logger.debug("[HandPresetDebug] label={} hand_joint_names unavailable", label)
        return

    robot_qpos = _to_numpy_qpos(agent.robot.get_qpos())
    arm_dof = len(getattr(agent, "arm_joint_names", []) or [])
    hand_dof = len(hand_joint_names)
    if robot_qpos.ndim == 1:
        robot_hand_qpos = robot_qpos[arm_dof:arm_dof + hand_dof]
    else:
        robot_hand_qpos = robot_qpos[:, arm_dof:arm_dof + hand_dof]

    open_qpos = None
    close_qpos = None
    if hasattr(agent, "hand_open_qpos"):
        open_qpos = np.asarray(agent.hand_open_qpos, dtype=np.float32).reshape(-1)
    if hasattr(agent, "hand_close_qpos"):
        close_qpos = np.asarray(agent.hand_close_qpos, dtype=np.float32).reshape(-1)

    gripper_controller = getattr(getattr(agent, "controller", None), "controllers", {}).get("gripper")
    controller_open_qpos = None
    controller_close_qpos = None
    if gripper_controller is not None:
        config = getattr(gripper_controller, "config", None)
        if config is not None:
            if hasattr(config, "open_qpos"):
                controller_open_qpos = np.asarray(config.open_qpos, dtype=np.float32).reshape(-1)
            if hasattr(config, "close_qpos"):
                controller_close_qpos = np.asarray(config.close_qpos, dtype=np.float32).reshape(-1)

    logger.debug("[HandPresetDebug] label={} hand_joint_names={}", label, hand_joint_names)
    logger.debug(
        "[HandPresetDebug] label={} robot_hand_qpos={}",
        label,
        np.array2string(robot_hand_qpos, precision=4, suppress_small=True, max_line_width=200),
    )
    if open_qpos is not None:
        logger.debug(
            "[HandPresetDebug] label={} hand_open_qpos={}",
            label,
            np.array2string(open_qpos, precision=4, suppress_small=True, max_line_width=200),
        )
    if close_qpos is not None:
        logger.debug(
            "[HandPresetDebug] label={} hand_close_qpos={}",
            label,
            np.array2string(close_qpos, precision=4, suppress_small=True, max_line_width=200),
        )
    if controller_open_qpos is not None:
        logger.debug(
            "[HandPresetDebug] label={} controller_open_qpos={}",
            label,
            np.array2string(controller_open_qpos, precision=4, suppress_small=True, max_line_width=200),
        )
    if controller_close_qpos is not None:
        logger.debug(
            "[HandPresetDebug] label={} controller_close_qpos={}",
            label,
            np.array2string(controller_close_qpos, precision=4, suppress_small=True, max_line_width=200),
        )


def _repeat_for_envs(action, num_envs):
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    if num_envs <= 1:
        return action.astype(np.float32)
    if action.shape[0] == 1:
        action = np.repeat(action, num_envs, axis=0)
    return action.astype(np.float32)


def _get_active_action_dim(env) -> int:
    agent = getattr(env.unwrapped, "agent", None)
    controller = getattr(agent, "controller", None) if agent is not None else None
    action_space = getattr(controller, "action_space", None)
    shape = getattr(action_space, "shape", None)
    if shape is not None and len(shape) > 0 and int(shape[-1]) > 0:
        return int(shape[-1])
    return int(env.action_space.shape[-1])


def _step_env(env, action):
    batched_action = _repeat_for_envs(action, env.unwrapped.num_envs)
    return env.step(batched_action)


def _refresh_render_state(env, viewer=None):
    if viewer is not None:
        try:
            env.render_human()
            return
        except Exception:
            pass


def _apply_custom_camera_pose_if_needed(env):
    if getattr(env.unwrapped, "custom_render_camera_pose", None) is None:
        return
    eye, target = env.unwrapped.custom_render_camera_pose
    try:
        env.unwrapped.set_render_camera_pose(eye, target)
    except Exception:
        pass


def _map_teleop_gripper_signal_to_controller(signal_value, target_state):
    magnitude = abs(float(signal_value))
    if target_state == "open":
        return magnitude
    if target_state == "close":
        return -magnitude
    raise ValueError(f"Unsupported gripper target_state: {target_state}")


def is_ee_delta_control_mode(control_mode):
    return shared_is_ee_delta_control_mode(control_mode)


def _build_hold_action(env, control_mode, gripper_hold_signal=None):
    action_dim = _get_active_action_dim(env)
    if is_ee_delta_control_mode(control_mode):
        action = np.zeros(action_dim, dtype=np.float32)
        if action_dim >= 7 and gripper_hold_signal is not None:
            action[6] = _map_teleop_gripper_signal_to_controller(gripper_hold_signal, "open")
        return action
    return _get_robot_qpos(env)[:action_dim].astype(np.float32)


def _stabilize_env(env, settle_steps, control_mode, gripper_hold_signal=None):
    if settle_steps <= 0:
        return
    print(f"[Init] Stabilizing scene with {settle_steps} hold steps...")
    for _ in range(settle_steps):
        hold_action = _build_hold_action(env, control_mode, gripper_hold_signal=gripper_hold_signal)
        _step_env(env, hold_action)


@contextmanager
def _temporary_agent_control_mode(env, target_control_mode, reason: str):
    env_unwrapped = env.unwrapped
    agent = getattr(env_unwrapped, "agent", None)
    if agent is None:
        raise RuntimeError("Cannot switch control mode: env.unwrapped.agent is missing")
    previous_control_mode = getattr(agent, "control_mode", None)
    if previous_control_mode == target_control_mode:
        print(
            f"[PlannerDebug] control_mode already '{target_control_mode}' for {reason}; "
            "reusing active controller."
        )
        yield previous_control_mode
        return
    print(
        f"[PlannerDebug] Switching control_mode for {reason}: "
        f"'{previous_control_mode}' -> '{target_control_mode}'"
    )
    agent.set_control_mode(target_control_mode)
    try:
        yield previous_control_mode
    finally:
        if previous_control_mode is not None and getattr(agent, "control_mode", None) != previous_control_mode:
            print(
                f"[PlannerDebug] Restoring control_mode after {reason}: "
                f"'{getattr(agent, 'control_mode', None)}' -> '{previous_control_mode}'"
            )
            agent.set_control_mode(previous_control_mode)


def detect_config_key(scene_path, explicit_key=None):
    return shared_detect_config_key(scene_path, explicit_key)


def resolve_scene_path(scene_path, config_path, explicit_key=None):
    if scene_path:
        return scene_path

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path

    key_to_use = explicit_key
    if key_to_use is None and cfg_path.exists():
        import yaml

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        keys = cfg.get("keys") or []
        if keys:
            key_to_use = keys[0]

    if not key_to_use:
        raise ValueError(
            "--scene was not provided and no config key could be resolved. "
            "Pass --scene explicitly or provide --key / keys: in the config."
        )

    candidates = [
        Path.cwd() / "assets" / "scenes" / key_to_use / "simulation" / "scene.json",
        Path("/app/assets/scenes") / key_to_use / "simulation" / "scene.json",
        Path.cwd() / "outputs" / key_to_use / "simulation" / "scene.json",
        Path("/app/outputs") / key_to_use / "simulation" / "scene.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            print(f"[Scene] Auto-resolved scene from key '{key_to_use}': {candidate}")
            return str(candidate)

    raise FileNotFoundError(
        f"--scene was not provided and no scene.json was found for key '{key_to_use}'. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def load_runner_config(args):
    config_overrides = dict(
        scene_z_offset=0.0,
        sim_ground_offset=None,
        object_spawn_clearance=None,
        auto_placement=not args.no_auto_placement,
        bg_collision_mode="nonconvex",
        bg_use_decimated_collision_mesh=False,
        bg_collision_mesh="background_registered_collision.glb",
        obj_collision_mode="coacd",
        physx_contact_offset=0.005,
        physx_rest_offset=-0.001,
        placement_mode="scene",
        object_placements=None,
        random_placement=None,
        include_objects=None,
        exclude_objects=None,
        manip_object_id=None,
        object_material=None,
        hand_contact_material=None,
        hand_contact=None,
        arm_controller=None,
        hand_controller=None,
        cameras_config=copy.deepcopy(DEFAULT_CAMERAS_CONFIG),
        lighting_config=None,
        lighting_profile_config=None,
        lighting_profile=None,
        robot_base_pose=None,
        robot_init_qpos=None,
        robot_init_qpos_profiles=None,
        robot_init_qpos_profile_by_object=None,
        resolved_robot_init_qpos_profile=None,
        robot_uids="rc5_aero_hand_openr2s",
        robot_base_pose_z_auto=True,
        hand_pose_config=None,
        hand_contact_config=None,
        hand_contact_profile=None,
        hand_controller_config=None,
        hand_controller_profile=None,
        teleop_profile_config=None,
        teleop_profile=None,
        planner_backend=None,
        planner_approach_waypoints=0,
        planner_approach_waypoint_mode="fixed",
        planner_waypoints={"enabled": False, "points": []},
        planner_proxy_safe_clearance_z=0.10,
        planner_proxy_frame="base_camera_plane",
        planner_proxy_adaptive_steps_enabled=True,
        planner_proxy_threshold_xy_m=0.010,
        planner_proxy_threshold_z_m=0.010,
        planner_proxy_delta_remap_rpy_deg=None,
        planner_proxy_xy_step_m=0.01,
        planner_proxy_z_step_m=0.008,
        planner_proxy_lift_z_step_m=None,
        planner_proxy_rot_step_deg=6.0,
        planner_proxy_pos_tol_m=0.01,
        planner_proxy_lift_pos_tol_m=None,
        planner_proxy_rot_tol_deg=8.0,
        planner_proxy_hold_steps=1,
        planner_proxy_predescent_settle_steps=12,
        planner_proxy_relatch_ee_target_pose_between_stages=False,
        planner_proxy_pregrasp_joint_guard={
            "enabled": False,
            "run_align_stage": False,
            "joint_targets": {},
            "tolerance_rad": 0.10,
            "mismatch_policy": "warn",
            "align_solver_overrides": {
                "posture_gain": 0.45,
                "posture_gain_near_target": 0.45,
                "posture_joint_weights": [0.25, 0.25, 0.25, 0.25, 6.0, 6.0],
            },
        },
        planner_proxy_preclose_settle_steps=6,
        planner_proxy_max_stage_steps=100,
        planner_proxy_stall_steps=12,
        planner_rc5_move_group=RC5_CANONICAL_TARGET_FRAME,
        planner_rc5_frame_conversion="auto",
        planner_rc5_obb_target_semantics="right_tcp_link",
        planner_rc5_object_profile_target_semantics=RC5_CANONICAL_TARGET_FRAME,
        renderer_kwargs=None,
        video_config=None,
        sim_delta_remap_rpy_deg=None,
        gripper_open_signal=None,
        gripper_close_signal=None,
        control_mode=None,
    )

    key_to_use = detect_config_key(args.scene, args.key)
    cfg_path = Path(args.config_path)
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path
    cfg = None
    sections = None
    if key_to_use and cfg_path.exists():
        sections = load_simulation_config_sections(cfg_path, key_to_use)
        cfg = sections.raw_config

    if cfg is not None and sections is not None:
        global_sim = sections.global_sim
        local_sim = sections.local_sim

        def pick(name, default):
            if name == "cameras":
                return default
            return pick_simulation_value(sections, name, default)

        config_overrides["scene_z_offset"] = pick("scene_z_offset", 0.0)
        config_overrides["sim_ground_offset"] = pick("sim_ground_offset", None)
        config_overrides["object_spawn_clearance"] = pick("object_spawn_clearance", None)
        config_overrides["auto_placement"] = pick("auto_placement", config_overrides["auto_placement"])
        config_overrides["bg_collision_mode"] = pick("bg_collision_mode", "nonconvex")
        config_overrides["bg_use_decimated_collision_mesh"] = pick("bg_use_decimated_collision_mesh", False)
        config_overrides["bg_collision_mesh"] = pick("bg_collision_mesh", "background_registered_collision.glb")
        config_overrides["obj_collision_mode"] = pick("obj_collision_mode", "coacd")
        config_overrides["physx_contact_offset"] = pick("physx_contact_offset", 0.005)
        config_overrides["physx_rest_offset"] = pick("physx_rest_offset", -0.001)
        config_overrides["placement_mode"] = pick("placement_mode", "scene")
        config_overrides["object_placements"] = pick("object_placements", None)
        config_overrides["random_placement"] = pick("random_placement", None)
        config_overrides["include_objects"] = pick("include_objects", None)
        config_overrides["exclude_objects"] = pick("exclude_objects", None)
        config_overrides["manip_object_id"] = pick("manip_object_id", None)
        config_overrides["object_material"] = pick("object_material", None)
        config_overrides["hand_contact_material"] = pick("hand_contact_material", None)
        config_overrides["hand_contact"] = pick("hand_contact", None)
        config_overrides["arm_controller"] = pick("arm_controller", None)
        config_overrides["hand_controller"] = pick("hand_controller", None)
        config_overrides["lighting_config"] = pick("lighting", None)
        config_overrides["lighting_profile_config"] = pick("lighting_profile_config", None)
        config_overrides["lighting_profile"] = pick("lighting_profile", None)
        config_overrides["robot_base_pose"] = pick("robot_base_pose", None)
        config_overrides["robot_init_qpos"] = pick("robot_init_qpos", None)
        config_overrides["robot_init_qpos_profiles"] = copy.deepcopy(
            pick("robot_init_qpos_profiles", None)
        )
        config_overrides["robot_init_qpos_profile_by_object"] = copy.deepcopy(
            pick("robot_init_qpos_profile_by_object", None)
        )
        config_overrides["robot_uids"] = pick("robot_uids", "rc5_aero_hand_openr2s")
        config_overrides["robot_base_pose_z_auto"] = pick("robot_base_pose_z_auto", True)
        config_overrides["hand_pose_config"] = pick("hand_pose_config", None)
        config_overrides["hand_contact_config"] = pick("hand_contact_config", None)
        config_overrides["hand_contact_profile"] = pick("hand_contact_profile", None)
        config_overrides["hand_controller_config"] = pick("hand_controller_config", None)
        config_overrides["hand_controller_profile"] = pick("hand_controller_profile", None)
        config_overrides["teleop_profile_config"] = pick("teleop_profile_config", None)
        config_overrides["teleop_profile"] = pick("teleop_profile", None)
        config_overrides["planner_backend"] = pick("planner_backend", None)
        config_overrides["planner_proxy_frame"] = pick("planner_proxy_frame", config_overrides["planner_proxy_frame"])
        config_overrides["planner_proxy_delta_remap_rpy_deg"] = pick("planner_proxy_delta_remap_rpy_deg", None)
        config_overrides["renderer_kwargs"] = pick("renderer_kwargs", None)
        config_overrides["video_config"] = pick("video", None)
        config_overrides["control_mode"] = pick("control_mode", None)
        config_overrides["planner_pregrasp_approach_offset"] = pick("planner_pregrasp_approach_offset", 0.10)
        config_overrides["planner_pregrasp_radial_backoff"] = pick("planner_pregrasp_radial_backoff", 0.05)
        config_overrides["planner_pregrasp_world_tweak_xyz"] = pick("planner_pregrasp_world_tweak_xyz", [0.0, 0.0, 0.0])
        config_overrides["planner_descend_world_tweak_xyz"] = pick("planner_descend_world_tweak_xyz", [0.0, 0.0, 0.045])
        config_overrides["planner_descend_world_tweak_xyz_generic"] = pick(
            "planner_descend_world_tweak_xyz_generic",
            [0.0, 0.0, 0.0],
        )
        config_overrides["planner_finger_length"] = pick("planner_finger_length", 0.04)
        config_overrides["planner_object_calibrations"] = pick("planner_object_calibrations", None)
        config_overrides["planner_pregrasp_offset_xyz"] = pick("planner_pregrasp_offset_xyz", [0.0, 0.0, 0.0])
        config_overrides["planner_descend_offset_xyz"] = pick("planner_descend_offset_xyz", [0.0, 0.0, 0.0])
        config_overrides["planner_approach_waypoints"] = int(pick("planner_approach_waypoints", 0) or 0)
        config_overrides["planner_approach_waypoint_mode"] = str(
            pick("planner_approach_waypoint_mode", "fixed") or "fixed"
        )
        config_overrides["planner_waypoints"] = copy.deepcopy(
            pick("planner_waypoints", config_overrides["planner_waypoints"])
        )
        config_overrides["planner_proxy_frame"] = str(
            pick("planner_proxy_frame", config_overrides["planner_proxy_frame"]) or config_overrides["planner_proxy_frame"]
        )
        config_overrides["planner_proxy_adaptive_steps_enabled"] = bool(
            pick("planner_proxy_adaptive_steps_enabled", config_overrides["planner_proxy_adaptive_steps_enabled"])
        )
        config_overrides["planner_proxy_threshold_xy_m"] = float(
            pick("planner_proxy_threshold_xy_m", config_overrides["planner_proxy_threshold_xy_m"])
        )
        config_overrides["planner_proxy_threshold_z_m"] = float(
            pick("planner_proxy_threshold_z_m", config_overrides["planner_proxy_threshold_z_m"])
        )
        proxy_remap = pick("planner_proxy_delta_remap_rpy_deg", config_overrides["planner_proxy_delta_remap_rpy_deg"])
        if proxy_remap is not None:
            config_overrides["planner_proxy_delta_remap_rpy_deg"] = [float(x) for x in proxy_remap]
        config_overrides["planner_proxy_safe_clearance_z"] = pick("planner_proxy_safe_clearance_z", 0.10)
        config_overrides["planner_proxy_xy_step_m"] = pick("planner_proxy_xy_step_m", 0.01)
        config_overrides["planner_proxy_z_step_m"] = pick("planner_proxy_z_step_m", 0.008)
        lift_z_step = pick(
            "planner_proxy_lift_z_step_m",
            config_overrides["planner_proxy_lift_z_step_m"],
        )
        config_overrides["planner_proxy_lift_z_step_m"] = (
            None if lift_z_step is None else float(lift_z_step)
        )
        config_overrides["planner_proxy_rot_step_deg"] = pick("planner_proxy_rot_step_deg", 6.0)
        config_overrides["planner_proxy_pos_tol_m"] = pick("planner_proxy_pos_tol_m", 0.01)
        lift_pos_tol = pick(
            "planner_proxy_lift_pos_tol_m",
            config_overrides["planner_proxy_lift_pos_tol_m"],
        )
        config_overrides["planner_proxy_lift_pos_tol_m"] = (
            None if lift_pos_tol is None else float(lift_pos_tol)
        )
        config_overrides["planner_proxy_rot_tol_deg"] = pick("planner_proxy_rot_tol_deg", 8.0)
        config_overrides["planner_proxy_hold_steps"] = int(pick("planner_proxy_hold_steps", 1) or 1)
        config_overrides["planner_proxy_predescent_settle_steps"] = int(
            pick("planner_proxy_predescent_settle_steps", 12) or 0
        )
        config_overrides["planner_proxy_relatch_ee_target_pose_between_stages"] = bool(
            pick(
                "planner_proxy_relatch_ee_target_pose_between_stages",
                config_overrides["planner_proxy_relatch_ee_target_pose_between_stages"],
            )
        )
        config_overrides["planner_proxy_pregrasp_joint_guard"] = copy.deepcopy(
            pick(
                "planner_proxy_pregrasp_joint_guard",
                config_overrides["planner_proxy_pregrasp_joint_guard"],
            )
        )
        config_overrides["planner_proxy_preclose_settle_steps"] = int(
            pick("planner_proxy_preclose_settle_steps", 6) or 0
        )
        config_overrides["planner_proxy_max_stage_steps"] = int(pick("planner_proxy_max_stage_steps", 100) or 100)
        config_overrides["planner_proxy_stall_steps"] = int(pick("planner_proxy_stall_steps", 12) or 12)
        config_overrides["planner_lift_delta_z"] = pick("planner_lift_delta_z", 0.05)
        config_overrides["planner_rc5_move_group"] = pick("planner_rc5_move_group", RC5_CANONICAL_TARGET_FRAME)
        config_overrides["planner_rc5_frame_conversion"] = pick("planner_rc5_frame_conversion", "auto")
        config_overrides["planner_rc5_obb_target_semantics"] = pick(
            "planner_rc5_obb_target_semantics",
            "right_tcp_link",
        )
        config_overrides["planner_rc5_object_profile_target_semantics"] = pick(
            "planner_rc5_object_profile_target_semantics",
            RC5_CANONICAL_TARGET_FRAME,
        )

        if config_overrides["include_objects"] == "manip_only":
            manip_oid = local_sim.get("manip_object_id", global_sim.get("manip_object_id"))
            if manip_oid is not None:
                config_overrides["include_objects"] = [str(manip_oid)]
                print(f"[Filter] include_objects='manip_only' -> [{manip_oid}]")
            else:
                config_overrides["include_objects"] = None

        cameras_config = load_cameras_config(cfg, key_to_use)
        if cameras_config is None:
            cameras_config = load_cameras_config(cfg, None)
        if cameras_config is not None:
            config_overrides["cameras_config"] = cameras_config
            if key_to_use:
                print(f"[Camera] cameras_config loaded from config for key '{key_to_use}'")
            else:
                print("[Camera] cameras_config loaded from global config")

        if key_to_use and config_overrides["sim_ground_offset"] is not None:
            print(f"[Info] Using sim_ground_offset={config_overrides['sim_ground_offset']} from config for key '{key_to_use}'")
        if key_to_use:
            print(f"[Info] Using key '{key_to_use}' for debug runner config")

    base_camera_cfg = config_overrides["cameras_config"].setdefault("base_camera", {})
    video_cfg = dict(config_overrides.get("video_config") or {})
    if args.video_width is not None:
        base_camera_cfg["width"] = int(args.video_width)
    elif video_cfg.get("width") is not None:
        base_camera_cfg["width"] = int(video_cfg["width"])
    if args.video_height is not None:
        base_camera_cfg["height"] = int(args.video_height)
    elif video_cfg.get("height") is not None:
        base_camera_cfg["height"] = int(video_cfg["height"])

    if args.robot_base_pose is not None:
        config_overrides["robot_base_pose"] = list(args.robot_base_pose)
        print(f"[Info] CLI override: robot_base_pose={config_overrides['robot_base_pose']}")
    if args.robot_init_qpos is not None:
        config_overrides["robot_init_qpos"] = list(args.robot_init_qpos)
        print(f"[Info] CLI override: robot_init_qpos len={len(config_overrides['robot_init_qpos'])}")
    cli_manip_object_id = getattr(args, "manip_object_id", None)
    if cli_manip_object_id is None:
        cli_manip_object_id = getattr(args, "task_object_id", None)
    if cli_manip_object_id is not None:
        config_overrides["manip_object_id"] = str(cli_manip_object_id)
        print(f"[Info] CLI override: manip_object_id={config_overrides['manip_object_id']}")
    _apply_object_specific_robot_init_qpos_profile(
        config_overrides,
        cli_override=args.robot_init_qpos is not None,
    )
    if args.robot_uids is not None:
        config_overrides["robot_uids"] = args.robot_uids
        print(f"[Info] CLI override: robot_uids={config_overrides['robot_uids']}")
    if args.approach_waypoints is not None:
        config_overrides["planner_approach_waypoints"] = max(int(args.approach_waypoints), 0)
        print(
            "[Info] CLI override: planner_approach_waypoints="
            f"{config_overrides['planner_approach_waypoints']}"
        )
    if args.approach_waypoint_mode is not None:
        config_overrides["planner_approach_waypoint_mode"] = str(args.approach_waypoint_mode)
        print(
            "[Info] CLI override: planner_approach_waypoint_mode="
            f"{config_overrides['planner_approach_waypoint_mode']}"
        )
    if args.planner_backend is not None:
        config_overrides["planner_backend"] = str(args.planner_backend)
        print(f"[Info] CLI override: planner_backend={config_overrides['planner_backend']}")
    if args.planner_proxy_frame is not None:
        config_overrides["planner_proxy_frame"] = str(args.planner_proxy_frame)
        print(f"[Info] CLI override: planner_proxy_frame={config_overrides['planner_proxy_frame']}")
    if getattr(args, "disable_planner_proxy_adaptive_steps", False):
        config_overrides["planner_proxy_adaptive_steps_enabled"] = False
        print("[Info] CLI override: planner_proxy_adaptive_steps_enabled=False")
    if args.planner_proxy_safe_clearance_z is not None:
        config_overrides["planner_proxy_safe_clearance_z"] = float(args.planner_proxy_safe_clearance_z)
        print(
            "[Info] CLI override: planner_proxy_safe_clearance_z="
            f"{config_overrides['planner_proxy_safe_clearance_z']:.4f}"
        )
    if args.planner_proxy_predescent_settle_steps is not None:
        config_overrides["planner_proxy_predescent_settle_steps"] = int(args.planner_proxy_predescent_settle_steps)
        print(
            "[Info] CLI override: planner_proxy_predescent_settle_steps="
            f"{config_overrides['planner_proxy_predescent_settle_steps']}"
        )
    if args.planner_proxy_preclose_settle_steps is not None:
        config_overrides["planner_proxy_preclose_settle_steps"] = int(args.planner_proxy_preclose_settle_steps)
        print(
            "[Info] CLI override: planner_proxy_preclose_settle_steps="
            f"{config_overrides['planner_proxy_preclose_settle_steps']}"
        )
    if args.rc5_move_group is not None:
        config_overrides["planner_rc5_move_group"] = resolve_rc5_move_group(
            str(args.rc5_move_group),
            env={},
            scope="RC5UnifiedProxySetup",
            warn_on_env=False,
            warn_on_default=False,
        )
        print(f"[Info] CLI override: planner_rc5_move_group={config_overrides['planner_rc5_move_group']}")
    else:
        env_move_group = str(os.environ.get("OPENR2S_RC5_MOVE_GROUP", "")).strip()
        if env_move_group:
            config_overrides["planner_rc5_move_group"] = resolve_rc5_move_group(
                None,
                env=os.environ,
                scope="RC5UnifiedProxySetup",
                warn_on_env=True,
                warn_on_default=False,
            )
            print(
                "[Info] Environment override: planner_rc5_move_group="
                f"{config_overrides['planner_rc5_move_group']}"
            )
    if args.rc5_frame_conversion is not None:
        config_overrides["planner_rc5_frame_conversion"] = str(args.rc5_frame_conversion)
        print(
            "[Info] CLI override: planner_rc5_frame_conversion="
            f"{config_overrides['planner_rc5_frame_conversion']}"
        )
    if args.rc5_obb_target_semantics is not None:
        config_overrides["planner_rc5_obb_target_semantics"] = str(args.rc5_obb_target_semantics)
        print(
            "[Info] CLI override: planner_rc5_obb_target_semantics="
            f"{config_overrides['planner_rc5_obb_target_semantics']}"
        )
    if args.rc5_object_profile_target_semantics is not None:
        config_overrides["planner_rc5_object_profile_target_semantics"] = str(
            args.rc5_object_profile_target_semantics
        )
        print(
            "[Info] CLI override: planner_rc5_object_profile_target_semantics="
            f"{config_overrides['planner_rc5_object_profile_target_semantics']}"
        )
    if args.no_auto_placement:
        config_overrides["auto_placement"] = False
        print("[Auto] CLI override: auto_placement=False")
    if args.hand_pose_config is not None:
        config_overrides["hand_pose_config"] = args.hand_pose_config
        print(f"[HandPose] CLI override: hand_pose_config={args.hand_pose_config}")
    if getattr(args, "lighting_profile_config", None) is not None:
        config_overrides["lighting_profile_config"] = args.lighting_profile_config
        print(f"[Lighting] CLI override: lighting_profile_config={args.lighting_profile_config}")
    if getattr(args, "lighting_profile", None) is not None:
        config_overrides["lighting_profile"] = args.lighting_profile
        print(f"[Lighting] CLI override: lighting_profile={args.lighting_profile}")
    if args.hand_contact_config is not None:
        config_overrides["hand_contact_config"] = args.hand_contact_config
        print(f"[HandContact] CLI override: hand_contact_config={args.hand_contact_config}")
    if args.hand_contact_profile is not None:
        config_overrides["hand_contact_profile"] = args.hand_contact_profile
        print(f"[HandContact] CLI override: hand_contact_profile={args.hand_contact_profile}")
    if args.hand_controller_config is not None:
        config_overrides["hand_controller_config"] = args.hand_controller_config
        print(f"[HandController] CLI override: hand_controller_config={args.hand_controller_config}")
    if args.hand_controller_profile is not None:
        config_overrides["hand_controller_profile"] = args.hand_controller_profile
        print(f"[HandController] CLI override: hand_controller_profile={args.hand_controller_profile}")
    if args.teleop_profile_config is not None:
        config_overrides["teleop_profile_config"] = args.teleop_profile_config
        print(f"[Teleop] CLI override: teleop_profile_config={args.teleop_profile_config}")
    if args.teleop_profile is not None:
        config_overrides["teleop_profile"] = args.teleop_profile
        print(f"[Teleop] CLI override: teleop_profile={args.teleop_profile}")
    if args.control_mode is not None:
        config_overrides["control_mode"] = args.control_mode
        print(f"[Mode] CLI override: control_mode={args.control_mode}")
    if args.max_num_materials is not None or args.max_num_textures is not None:
        renderer_kwargs = dict(config_overrides.get("renderer_kwargs") or {})
        if args.max_num_materials is not None:
            renderer_kwargs["max_num_materials"] = int(args.max_num_materials)
        if args.max_num_textures is not None:
            renderer_kwargs["max_num_textures"] = int(args.max_num_textures)
        config_overrides["renderer_kwargs"] = renderer_kwargs
        print(f"[Renderer] CLI override: renderer_kwargs={config_overrides['renderer_kwargs']}")

    config_overrides["control_mode"] = shared_resolve_effective_control_mode(
        config_overrides["robot_uids"],
        config_overrides["control_mode"],
    )
    planner_waypoints_enabled, planner_waypoints_points = get_required_planner_waypoints_config(config_overrides)
    config_overrides["planner_waypoints"] = {
        "enabled": planner_waypoints_enabled,
        "points": planner_waypoints_points,
    }
    config_overrides["planner_proxy_pregrasp_joint_guard"] = (
        get_required_pregrasp_joint_guard_config(config_overrides)
    )
    print(f"[Mode] Effective control_mode={config_overrides['control_mode']}")
    return config_overrides


def apply_teleop_profile_overrides(args, config_overrides):
    del args
    shared_apply_teleop_profile_overrides(config_overrides, scope="Teleop")


def apply_lighting_profile_overrides(args, config_overrides):
    del args
    shared_apply_lighting_profile_overrides(config_overrides, scope="Lighting")


def apply_hand_contact_profile_overrides(args, config_overrides):
    del args
    shared_apply_hand_contact_profile_overrides(config_overrides, scope="HandContact")


def apply_hand_controller_profile_overrides(args, config_overrides):
    del args
    shared_apply_hand_controller_profile_overrides(config_overrides, scope="HandController")


def validate_teleop_runtime_requirements(config_overrides):
    control_mode = config_overrides.get("control_mode")
    if not is_ee_delta_control_mode(control_mode):
        return
    open_signal = config_overrides.get("gripper_open_signal")
    close_signal = config_overrides.get("gripper_close_signal")
    if open_signal is None or close_signal is None:
        msg = (
            "EE-delta / teleop control_mode requires gripper_open_signal and gripper_close_signal, "
            "but they are missing. Provide --teleop_profile_config + --teleop_profile, or define "
            "teleop_profile_config/teleop_profile for the active config key."
        )
        print(f"{_Y}[WARNING] {msg}{_R}")
        raise RuntimeError(msg)


def initialize_sapien_renderer(renderer_kwargs):
    if not renderer_kwargs:
        return
    max_materials = renderer_kwargs.get("max_num_materials")
    max_textures = renderer_kwargs.get("max_num_textures")
    if max_materials is None and max_textures is None:
        return
    print(
        "[Renderer] Initializing SapienRenderer with "
        f"max_num_materials={max_materials}, max_num_textures={max_textures}"
    )
    sapien.SapienRenderer(
        max_num_materials=max_materials or DEFAULT_RENDERER_MAX_NUM_MATERIALS,
        max_num_textures=max_textures or DEFAULT_RENDERER_MAX_NUM_TEXTURES,
    )


def apply_hand_pose_overrides(agent, args, config_overrides):
    hand_pose_config = config_overrides.get("hand_pose_config")
    if hand_pose_config is None:
        if args.hand_open_preset is not None or args.hand_close_preset is not None:
            raise ValueError("--hand_open_preset/--hand_close_preset require --hand_pose_config")
        print("[HandPose] No hand_pose_config override supplied; keeping agent defaults.")
        return

    required_bindings = [
        "open",
        "close",
        "full_open",
        "pinch",
        "tripod",
        "thumb_abduction_full",
        "thumb_abduction_partial",
    ]
    applied = shared_apply_hand_pose_config_to_agent(
        agent,
        hand_pose_config,
        open_preset_name=args.hand_open_preset,
        close_preset_name=args.hand_close_preset,
        required_bindings=required_bindings,
    )
    assert applied is not None
    cfg_path = applied.config_path
    bindings = applied.bindings
    all_qpos_presets = applied.presets
    open_preset_name = applied.open_preset_name
    close_preset_name = applied.close_preset_name

    agent._debug_hand_pose_config_path = str(cfg_path)
    agent._debug_hand_pose_bindings = dict(bindings)
    agent._debug_hand_pose_presets = all_qpos_presets
    print(
        f"[HandPose] Loaded hand pose presets from {cfg_path}: "
        f"open='{open_preset_name}', close='{close_preset_name}'"
    )
    print(f"[HandPose] bindings={bindings}")
    print(
        f"[HandPose] hand_open_qpos={np.array2string(agent.hand_open_qpos, precision=4, suppress_small=True, max_line_width=200)}"
    )
    print(
        f"[HandPose] hand_close_qpos={np.array2string(agent.hand_close_qpos, precision=4, suppress_small=True, max_line_width=200)}"
    )


def make_env(args, config_overrides, render_mode="human"):
    viewer_camera_configs = {
        "viewer": {
            "width": args.window_width,
            "height": args.window_height,
        }
    }
    env_kwargs = {
        "scene_json_path": args.scene,
        "robot_uids": args.robot_uids,
        "num_envs": args.num_envs,
        "obs_mode": "state",
        "control_mode": config_overrides["control_mode"],
        "render_mode": render_mode,
        "render_backend": args.render_backend,
        "viewer_camera_configs": viewer_camera_configs,
        "render_width": args.cam_width,
        "render_height": args.cam_height,
        "robot_init_qpos_noise": args.robot_init_qpos_noise,
        "settle_steps": 0,
        "auto_placement": config_overrides["auto_placement"],
        "cameras_config": config_overrides["cameras_config"],
        "lighting_config": config_overrides["lighting_config"],
        "scene_z_offset": config_overrides["scene_z_offset"],
        "sim_ground_offset": config_overrides["sim_ground_offset"],
        "object_spawn_clearance": config_overrides["object_spawn_clearance"],
        "bg_collision_mode": config_overrides["bg_collision_mode"],
        "bg_use_decimated_collision_mesh": config_overrides["bg_use_decimated_collision_mesh"],
        "bg_collision_mesh": config_overrides["bg_collision_mesh"],
        "obj_collision_mode": config_overrides["obj_collision_mode"],
        "physx_contact_offset": config_overrides["physx_contact_offset"],
        "physx_rest_offset": config_overrides["physx_rest_offset"],
        "placement_mode": config_overrides["placement_mode"],
        "robot_uids": config_overrides["robot_uids"],
    }
    if args.sim_backend is not None:
        env_kwargs["sim_backend"] = args.sim_backend
    if config_overrides["object_placements"] is not None:
        env_kwargs["object_placements"] = config_overrides["object_placements"]
    if config_overrides["random_placement"] is not None:
        env_kwargs["random_placement"] = config_overrides["random_placement"]
    if config_overrides["include_objects"] is not None:
        env_kwargs["include_objects"] = config_overrides["include_objects"]
    if config_overrides["exclude_objects"] is not None:
        env_kwargs["exclude_objects"] = config_overrides["exclude_objects"]
    if config_overrides["manip_object_id"] is not None:
        env_kwargs["manip_object_id"] = config_overrides["manip_object_id"]
    if config_overrides["object_material"] is not None:
        env_kwargs["object_material"] = config_overrides["object_material"]
    if config_overrides["hand_contact_material"] is not None:
        env_kwargs["hand_contact_material"] = config_overrides["hand_contact_material"]
    if config_overrides["hand_contact"] is not None:
        env_kwargs["hand_contact"] = config_overrides["hand_contact"]
    if config_overrides["arm_controller"] is not None:
        env_kwargs["arm_controller"] = config_overrides["arm_controller"]
    if config_overrides["hand_controller"] is not None:
        env_kwargs["hand_controller"] = config_overrides["hand_controller"]
    if config_overrides["robot_base_pose"] is not None:
        env_kwargs["robot_base_pose"] = list(config_overrides["robot_base_pose"])
    if config_overrides["robot_init_qpos"] is not None:
        env_kwargs["robot_init_qpos"] = list(config_overrides["robot_init_qpos"])
    if args.cam_eye is not None and args.cam_target is not None:
        env_kwargs["render_camera_eye"] = list(args.cam_eye)
        env_kwargs["render_camera_target"] = list(args.cam_target)
    OpenReal2SimEnv = _load_openreal2sim_env_class()
    return OpenReal2SimEnv(**env_kwargs)


def reset_and_prepare(env, args, control_mode, gripper_hold_signal=None):
    env.reset()
    _refresh_render_state(env)
    _apply_custom_camera_pose_if_needed(env)
    stabilize_control_mode = shared_select_startup_stabilize_control_mode(
        supported_control_modes=list(getattr(getattr(env.unwrapped, "agent", None), "supported_control_modes", []) or []),
        requested_control_mode=control_mode,
    )
    if stabilize_control_mode is None:
        if args.settle_steps > 0:
            raise RuntimeError(
                f"Startup stabilization requires a supported joint-space control mode, but control_mode='{control_mode}' "
                "and the agent exposes no explicit stabilization controller. "
                "Unified RC5 runtime requires an explicit stabilization path."
            )
    elif stabilize_control_mode != control_mode:
        with _temporary_agent_control_mode(env, stabilize_control_mode, reason="startup scene stabilization"):
            _stabilize_env(
                env,
                args.settle_steps,
                stabilize_control_mode,
                gripper_hold_signal=gripper_hold_signal,
            )
    else:
        _stabilize_env(
            env,
            args.settle_steps,
            control_mode,
            gripper_hold_signal=gripper_hold_signal,
        )
    _refresh_render_state(env)
    _apply_custom_camera_pose_if_needed(env)
    return _get_robot_qpos(env)
