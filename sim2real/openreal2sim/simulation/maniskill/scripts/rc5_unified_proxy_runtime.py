from __future__ import annotations

import copy
import importlib
import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import time

import numpy as np
from mani_skill.utils.structs.pose import Pose

from openreal2sim.simulation.maniskill.planner_core import (
    restore_planner_grasp_state,
    save_planner_grasp_state,
)
from openreal2sim.simulation.maniskill.planner_core.grasp_state import (
    get_planner_grasp_state,
)
from openreal2sim.simulation.maniskill.rc5_pick_retention_heuristic import (
    evaluate_rc5_pick_lift_success,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_bootstrap import (
    detect_config_key,
    load_simulation_config_sections,
    pick_simulation_value,
    reset_planner_hand_target_to_open as shared_reset_planner_hand_target_to_open,
)
from openreal2sim.simulation.maniskill.scripts.maniskill_num_envs_policy import DEFAULT_RC5_SIM_BACKEND
from openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_argv import (
    inject_unified_planner_backend_arg,
    resolve_unified_backend_request_argv,
)

_UNIFIED_PROXY_SETUP_MODULE = (
    "openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_setup"
)
_UNIFIED_PROXY_ARTIFACTS_MODULE = (
    "openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_artifacts"
)
_UNIFIED_PROXY_MACRO_MODULE = (
    "openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_macro"
)
_UNIFIED_PROXY_CONTROL_MODULE = (
    "openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_control"
)
_UNIFIED_PROXY_STAGE_POSE_MODULE = (
    "openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_stage_pose"
)
_UNIFIED_PROXY_STAGE_CLOSE_MODULE = (
    "openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_stage_close"
)
_UNIFIED_PROXY_STAGE_LIFT_MODULE = (
    "openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_stage_lift"
)
_UNIFIED_PROXY_DEBUG_MODULE = (
    "openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_debug"
)
_UNIFIED_PROXY_MOTION_MODULE = (
    "openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_motion"
)
_UNIFIED_PROXY_LOWLEVEL_MODULE = (
    "openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_lowlevel"
)
_UNIFIED_PROXY_TARGETS_MODULE = (
    "openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_targets"
)
AUTO_VIDEO_CODEC = "auto"
DEFAULT_VIDEO_FPS = 30
DEFAULT_VIDEO_FORMAT = "mp4"
_Y = "\033[33m"
_R = "\033[0m"

_UNIFIED_MACRO_FEEDBACK = None


def clear_unified_macro_feedback() -> None:
    global _UNIFIED_MACRO_FEEDBACK
    _UNIFIED_MACRO_FEEDBACK = None


def set_unified_macro_feedback(
    *,
    semantic_task_success,
    failed_stage,
    per_env_feedback=None,
    batch_size=None,
    successful_env_count=None,
    failed_env_indices=None,
    artifacts_recorded_per_env=None,
) -> None:
    global _UNIFIED_MACRO_FEEDBACK
    _UNIFIED_MACRO_FEEDBACK = {
        "semantic_task_success": bool(semantic_task_success),
        "failed_stage": failed_stage,
    }
    if per_env_feedback is not None:
        _UNIFIED_MACRO_FEEDBACK["per_env_feedback"] = copy.deepcopy(list(per_env_feedback))
    if batch_size is not None:
        _UNIFIED_MACRO_FEEDBACK["batch_size"] = int(batch_size)
    if successful_env_count is not None:
        _UNIFIED_MACRO_FEEDBACK["successful_env_count"] = int(successful_env_count)
    if failed_env_indices is not None:
        _UNIFIED_MACRO_FEEDBACK["failed_env_indices"] = [int(item) for item in failed_env_indices]
    if artifacts_recorded_per_env is not None:
        _UNIFIED_MACRO_FEEDBACK["artifacts_recorded_per_env"] = bool(artifacts_recorded_per_env)


def consume_unified_macro_feedback():
    global _UNIFIED_MACRO_FEEDBACK
    feedback = _UNIFIED_MACRO_FEEDBACK
    _UNIFIED_MACRO_FEEDBACK = None
    if feedback is None:
        return None
    return copy.deepcopy(dict(feedback))


def peek_unified_macro_feedback():
    if _UNIFIED_MACRO_FEEDBACK is None:
        return None
    return copy.deepcopy(dict(_UNIFIED_MACRO_FEEDBACK))


def _to_numpy_array(value):
    if value is None:
        return None
    if hasattr(value, "detach") and callable(getattr(value, "detach", None)):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _parse_optional_json_string_list(value, *, label: str) -> list[str] | None:
    if value is None:
        return None
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must decode to a JSON list: {exc.msg} at pos {exc.pos}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{label} must decode to a JSON list")
    return [str(item) for item in payload]


def _normalize_per_env_scalar_list(value, *, num_envs: int, dtype=float):
    if value is None:
        return [None] * int(num_envs)
    arr = _to_numpy_array(value)
    if arr is None:
        return [None] * int(num_envs)
    if arr.ndim == 0:
        arr = np.repeat(arr.reshape(1), int(num_envs), axis=0)
    else:
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[0] == 1 and int(num_envs) > 1:
            arr = np.repeat(arr, int(num_envs), axis=0)
        if arr.shape[0] != int(num_envs):
            raise ValueError(
                f"Expected per-env scalar payload with first dimension {int(num_envs)}, got shape={arr.shape}."
            )
        arr = arr[:, 0]
    if dtype is bool:
        return [bool(item) for item in arr.tolist()]
    if dtype is int:
        return [int(item) for item in arr.tolist()]
    return [float(item) for item in arr.tolist()]


def _normalize_per_env_pose_rows(value, *, num_envs: int):
    if value is None:
        return [None] * int(num_envs)
    arr = _to_numpy_array(value)
    if arr is None:
        return [None] * int(num_envs)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 1 and int(num_envs) > 1:
        arr = np.repeat(arr, int(num_envs), axis=0)
    if arr.shape[0] != int(num_envs):
        raise ValueError(
            f"Expected per-env pose payload with first dimension {int(num_envs)}, got shape={arr.shape}."
        )
    return [np.asarray(row, dtype=np.float32).reshape(-1).tolist() for row in arr]


def _build_batched_env_feedback(env, *, failed_stage):
    env_unwrapped = env.unwrapped
    num_envs = int(getattr(env_unwrapped, "num_envs", 1) or 1)
    if num_envs <= 1:
        return []

    evaluation = env_unwrapped.evaluate()
    if not isinstance(evaluation, dict):
        raise ValueError(
            "Unified proxy batched runtime requires env.evaluate() to return a mapping "
            "with per-env success metadata."
        )

    target_object = getattr(env_unwrapped, "object_actors", {}).get(getattr(env_unwrapped, "manip_object_id", None))
    if target_object is None:
        target_object = _load_unified_proxy_debug().get_debug_target_object(env_unwrapped)

    success_flags = _normalize_per_env_scalar_list(
        evaluation.get("success"),
        num_envs=num_envs,
        dtype=bool,
    )
    grasp_flags = _normalize_per_env_scalar_list(
        evaluation.get("is_src_obj_grasped"),
        num_envs=num_envs,
        dtype=bool,
    )
    instant_grasp_flags = _normalize_per_env_scalar_list(
        None if target_object is None else env_unwrapped.agent.is_grasping(target_object),
        num_envs=num_envs,
        dtype=bool,
    )
    object_heights = _normalize_per_env_scalar_list(
        evaluation.get("obj_height_above_table"),
        num_envs=num_envs,
        dtype=float,
    )
    gripper_object_distances = _normalize_per_env_scalar_list(
        evaluation.get("gripper_obj_dist"),
        num_envs=num_envs,
        dtype=float,
    )
    gripper_goal_distances = _normalize_per_env_scalar_list(
        evaluation.get("gripper_goal_dist"),
        num_envs=num_envs,
        dtype=float,
    )
    tcp_pose_rows = _normalize_per_env_pose_rows(
        getattr(getattr(getattr(env_unwrapped.agent, "tcp", None), "pose", None), "raw_pose", None),
        num_envs=num_envs,
    )
    object_pose_rows = _normalize_per_env_pose_rows(
        getattr(getattr(target_object, "pose", None), "raw_pose", None),
        num_envs=num_envs,
    )

    return [
        {
            "env_index": env_index,
            "semantic_task_success": bool(success_flags[env_index]),
            "failed_stage": (None if bool(success_flags[env_index]) else failed_stage),
            "is_src_obj_grasped": grasp_flags[env_index],
            "instant_is_src_obj_grasped": instant_grasp_flags[env_index],
            "obj_height_above_table": object_heights[env_index],
            "gripper_obj_dist": gripper_object_distances[env_index],
            "gripper_goal_dist": gripper_goal_distances[env_index],
            "tcp_pose": tcp_pose_rows[env_index],
            "object_pose": object_pose_rows[env_index],
        }
        for env_index in range(num_envs)
    ]


def _finalize_macro_feedback_for_runtime(env, macro_feedback):
    if macro_feedback is None:
        return None
    feedback = copy.deepcopy(dict(macro_feedback))
    num_envs = int(getattr(env.unwrapped, "num_envs", 1) or 1)
    if num_envs <= 1:
        return feedback

    per_env_feedback = _build_batched_env_feedback(
        env,
        failed_stage=feedback.get("failed_stage"),
    )
    existing_per_env_feedback = list(feedback.get("per_env_feedback", []) or [])
    if existing_per_env_feedback:
        existing_by_index = {
            int(item.get("env_index")): dict(item)
            for item in existing_per_env_feedback
            if isinstance(item, dict) and item.get("env_index") is not None
        }
        merged_per_env_feedback = []
        for item in per_env_feedback:
            env_index = int(item["env_index"])
            merged = dict(item)
            existing = existing_by_index.get(env_index)
            if existing is not None:
                merged.update(existing)
                if not bool(existing.get("semantic_task_success")):
                    merged["semantic_task_success"] = False
                    merged["failed_stage"] = existing.get("failed_stage")
            merged_per_env_feedback.append(merged)
        per_env_feedback = merged_per_env_feedback
    successful_env_count = sum(1 for item in per_env_feedback if item["semantic_task_success"])
    failed_env_indices = [
        int(item["env_index"]) for item in per_env_feedback if not item["semantic_task_success"]
    ]
    batch_semantic_task_success = (
        successful_env_count == len(per_env_feedback)
        if per_env_feedback
        else bool(feedback.get("semantic_task_success"))
    )

    # Batched runtime must aggregate from per-env evaluation rather than inheriting env0 semantics.
    feedback["semantic_task_success"] = batch_semantic_task_success
    if batch_semantic_task_success:
        feedback["failed_stage"] = None
    feedback["per_env_feedback"] = per_env_feedback
    feedback["batch_size"] = num_envs
    feedback["successful_env_count"] = successful_env_count
    feedback["failed_env_indices"] = failed_env_indices
    feedback["artifacts_recorded_per_env"] = False
    return feedback


def _normalize_pose_rows_payload(value, *, num_envs: int):
    arr = _to_numpy_array(value)
    if arr is None:
        raise ValueError("Expected a pose payload, got None.")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 1 and int(num_envs) > 1:
        arr = np.repeat(arr, int(num_envs), axis=0)
    if arr.shape[0] != int(num_envs):
        raise ValueError(
            f"Expected pose payload with first dimension {int(num_envs)}, got shape={arr.shape}."
        )
    return np.asarray(arr, dtype=np.float32).copy()


def _select_pose_row(pose_value, *, env_index: int, num_envs: int):
    pose_rows = _normalize_pose_rows_payload(pose_value, num_envs=num_envs)
    return np.asarray(pose_rows[int(env_index)], dtype=np.float32).reshape(-1).copy()


def _set_batched_last_task_pose_rows(
    env_unwrapped,
    *,
    position_rows,
    quaternion_rows,
    bbox_np=None,
    stage_name: str,
):
    position_rows = _normalize_pose_rows_payload(
        position_rows,
        num_envs=int(getattr(env_unwrapped, "num_envs", 1) or 1),
    )[:, :3]
    quaternion_rows = _normalize_pose_rows_payload(
        quaternion_rows,
        num_envs=int(getattr(env_unwrapped, "num_envs", 1) or 1),
    )[:, :4]
    env_unwrapped._planner_last_task_pose = Pose.create_from_pq(
        p=np.asarray(position_rows, dtype=np.float32),
        q=np.asarray(quaternion_rows, dtype=np.float32),
    )
    env_unwrapped._planner_last_task_pose_stage = str(stage_name)
    env_unwrapped._planner_last_task_bbox_np = (
        None
        if bbox_np is None
        else np.asarray(bbox_np, dtype=np.float32).reshape(-1)[:3].copy()
    )


def _load_unified_proxy_setup():
    return importlib.import_module(_UNIFIED_PROXY_SETUP_MODULE)


def _load_unified_proxy_artifacts():
    return importlib.import_module(_UNIFIED_PROXY_ARTIFACTS_MODULE)


def _load_unified_proxy_macro():
    return importlib.import_module(_UNIFIED_PROXY_MACRO_MODULE)


def _load_unified_proxy_control():
    return importlib.import_module(_UNIFIED_PROXY_CONTROL_MODULE)


def _load_unified_proxy_stage_pose():
    return importlib.import_module(_UNIFIED_PROXY_STAGE_POSE_MODULE)


def _load_unified_proxy_stage_close():
    return importlib.import_module(_UNIFIED_PROXY_STAGE_CLOSE_MODULE)


def _load_unified_proxy_stage_lift():
    return importlib.import_module(_UNIFIED_PROXY_STAGE_LIFT_MODULE)


def _load_unified_proxy_debug():
    return importlib.import_module(_UNIFIED_PROXY_DEBUG_MODULE)


def _load_unified_proxy_motion():
    return importlib.import_module(_UNIFIED_PROXY_MOTION_MODULE)


def _load_unified_proxy_lowlevel():
    return importlib.import_module(_UNIFIED_PROXY_LOWLEVEL_MODULE)


def _load_unified_proxy_targets():
    return importlib.import_module(_UNIFIED_PROXY_TARGETS_MODULE)


def _emit_yellow_warning(message: str) -> None:
    print(f"{_Y}[WARNING] [RC5UnifiedProxyRuntime] {message}{_R}")


def _raise_unified_proxy_fail_fast(reason: str) -> None:
    raise ValueError(
        "Unified proxy runtime fail-fast: "
        f"{reason}. "
        "Only the supported unified envelope is allowed."
    )


def _validate_direct_runtime_task_object_id(args, env_unwrapped) -> None:
    requested_task_object_id = getattr(args, "task_object_id", None)
    if not requested_task_object_id:
        return
    runtime_task_object_id = getattr(env_unwrapped, "manip_object_id", None)
    if runtime_task_object_id is None:
        _raise_unified_proxy_fail_fast(
            f"direct runtime lost CLI task_object_id='{requested_task_object_id}' because env.manip_object_id is unset"
        )
    if str(runtime_task_object_id) != str(requested_task_object_id):
        _raise_unified_proxy_fail_fast(
            "direct runtime task-object mismatch: "
            f"CLI requested '{requested_task_object_id}', env.manip_object_id resolved to '{runtime_task_object_id}'"
        )


def _extract_last_flag_value(argv, flag: str):
    values = []
    argv = list(argv or [])
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token != flag:
            idx += 1
            continue
        if idx + 1 >= len(argv):
            raise ValueError(f"{flag} requires a value")
        values.append(argv[idx + 1])
        idx += 2
    if not values:
        return None
    return values[-1]


def _batched_per_env_artifacts_enabled(args) -> bool:
    num_envs = int(getattr(args, "num_envs", 1) or 1)
    if num_envs <= 1:
        return False
    return bool(
        getattr(args, "dense_episode_output_dir", None)
        or getattr(args, "rl4vla_raw_episode_output_dir", None)
    )


def _resolve_startup_settle_steps_from_argv(argv) -> int:
    cli_value = _extract_last_flag_value(argv, "--settle_steps")
    if cli_value is not None:
        return int(cli_value)

    config_path = _extract_last_flag_value(argv, "--config_path") or "config/config_debug.yaml"
    key = detect_config_key(
        _extract_last_flag_value(argv, "--scene"),
        _extract_last_flag_value(argv, "--key"),
    )
    if key:
        try:
            sections = load_simulation_config_sections(config_path, key)
            return int(pick_simulation_value(sections, "settle_steps", 3) or 0)
        except Exception as exc:
            print(
                "[WARNING] [RC5UnifiedProxyRuntime] Failed to read settle_steps from "
                f"config_path={config_path!r} key={key!r}: {exc}. Falling back to 3."
            )
    return 3


def _build_direct_args_namespace(argv):
    argv = list(argv or [])
    unsupported_multi_value_flags = {
        "--robot_base_pose",
        "--robot_init_qpos",
        "--cam_eye",
        "--cam_target",
    }
    if any(flag in argv for flag in unsupported_multi_value_flags):
        return None
    if "--auto_ee_axis_probe" in argv:
        return None
    auto_pick_macro = _extract_last_flag_value(argv, "--auto_pick_macro")
    if auto_pick_macro not in {"1", "2"}:
        return None
    task_object_id = _extract_last_flag_value(argv, "--task_object_id")
    return SimpleNamespace(
        scene=_extract_last_flag_value(argv, "--scene"),
        config_path=_extract_last_flag_value(argv, "--config_path") or "config/config_debug.yaml",
        key=_extract_last_flag_value(argv, "--key"),
        task_object_id=task_object_id,
        manip_object_id=task_object_id,
        robot_uids=_extract_last_flag_value(argv, "--robot_uids"),
        control_mode=_extract_last_flag_value(argv, "--control_mode"),
        hand_pose_config=_extract_last_flag_value(argv, "--hand_pose_config"),
        hand_open_preset=_extract_last_flag_value(argv, "--hand_open_preset"),
        hand_close_preset=_extract_last_flag_value(argv, "--hand_close_preset"),
        lighting_profile_config=_extract_last_flag_value(argv, "--lighting_profile_config"),
        lighting_profile=_extract_last_flag_value(argv, "--lighting_profile"),
        hand_contact_config=_extract_last_flag_value(argv, "--hand_contact_config"),
        hand_contact_profile=_extract_last_flag_value(argv, "--hand_contact_profile"),
        hand_controller_config=_extract_last_flag_value(argv, "--hand_controller_config"),
        hand_controller_profile=_extract_last_flag_value(argv, "--hand_controller_profile"),
        teleop_profile_config=_extract_last_flag_value(argv, "--teleop_profile_config"),
        teleop_profile=_extract_last_flag_value(argv, "--teleop_profile"),
        planner_backend=_extract_last_flag_value(argv, "--planner_backend"),
        planner_proxy_frame=_extract_last_flag_value(argv, "--planner_proxy_frame"),
        planner_proxy_safe_clearance_z=_extract_last_flag_value(argv, "--planner_proxy_safe_clearance_z"),
        disable_planner_proxy_adaptive_steps="--disable_planner_proxy_adaptive_steps" in argv,
        planner_proxy_predescent_settle_steps=_extract_last_flag_value(
            argv, "--planner_proxy_predescent_settle_steps"
        ),
        planner_proxy_preclose_settle_steps=_extract_last_flag_value(argv, "--planner_proxy_preclose_settle_steps"),
        approach_waypoints=_extract_last_flag_value(argv, "--approach_waypoints"),
        approach_waypoint_mode=_extract_last_flag_value(argv, "--approach_waypoint_mode"),
        rc5_move_group=_extract_last_flag_value(argv, "--rc5_move_group"),
        rc5_frame_conversion=_extract_last_flag_value(argv, "--rc5_frame_conversion"),
        rc5_obb_target_semantics=_extract_last_flag_value(argv, "--rc5_obb_target_semantics"),
        rc5_object_profile_target_semantics=_extract_last_flag_value(
            argv, "--rc5_object_profile_target_semantics"
        ),
        max_num_materials=_extract_last_flag_value(argv, "--max_num_materials"),
        max_num_textures=_extract_last_flag_value(argv, "--max_num_textures"),
        video_width=_extract_last_flag_value(argv, "--video_width"),
        video_height=_extract_last_flag_value(argv, "--video_height"),
        robot_base_pose=None,
        robot_init_qpos=None,
        no_auto_placement="--no_auto_placement" in argv,
        cam_eye=None,
        cam_target=None,
        num_envs=int(_extract_last_flag_value(argv, "--num_envs") or 1),
        render_backend=_extract_last_flag_value(argv, "--render_backend") or "gpu",
        sim_backend=_extract_last_flag_value(argv, "--sim_backend") or DEFAULT_RC5_SIM_BACKEND,
        window_width=int(_extract_last_flag_value(argv, "--window_width") or 3840),
        window_height=int(_extract_last_flag_value(argv, "--window_height") or 2160),
        cam_width=int(_extract_last_flag_value(argv, "--cam_width") or 512),
        cam_height=int(_extract_last_flag_value(argv, "--cam_height") or 512),
        robot_init_qpos_noise=float(_extract_last_flag_value(argv, "--robot_init_qpos_noise") or 0.0),
        settle_steps=_resolve_startup_settle_steps_from_argv(argv),
        headless="--headless" in argv,
        step_by_step="--step_by_step" in argv,
        dense_episode_output=_extract_last_flag_value(argv, "--dense_episode_output"),
        dense_episode_output_dir=_extract_last_flag_value(argv, "--dense_episode_output_dir"),
        dense_episode_instruction=_extract_last_flag_value(argv, "--dense_episode_instruction"),
        episode_instruction_per_env_json=_extract_last_flag_value(argv, "--episode_instruction_per_env_json"),
        dense_episode_target_width=int(_extract_last_flag_value(argv, "--dense_episode_target_width") or 640),
        dense_episode_target_height=int(_extract_last_flag_value(argv, "--dense_episode_target_height") or 480),
        embed_runtime_bundle_in_rl4vla_raw_npz=(
            "--no-embed_runtime_bundle_in_rl4vla_raw_npz" not in argv
        ),
        runtime_request_path=_extract_last_flag_value(argv, "--runtime_request_path"),
        runtime_config_path_per_env_json=_extract_last_flag_value(argv, "--runtime_config_path_per_env_json"),
        runtime_request_path_per_env_json=_extract_last_flag_value(argv, "--runtime_request_path_per_env_json"),
        rl4vla_raw_episode_output=_extract_last_flag_value(argv, "--rl4vla_raw_episode_output"),
        rl4vla_raw_episode_output_dir=_extract_last_flag_value(argv, "--rl4vla_raw_episode_output_dir"),
        save_video_on_exit="--save_video_on_exit" in argv,
        save_video_path=_extract_last_flag_value(argv, "--save_video_path"),
        save_video_gif_on_exit="--save_video_gif_on_exit" in argv,
        save_video_gif_path=_extract_last_flag_value(argv, "--save_video_gif_path"),
        batched_save_video_output_dir=_extract_last_flag_value(argv, "--batched_save_video_output_dir"),
        batched_save_video_gif_output_dir=_extract_last_flag_value(argv, "--batched_save_video_gif_output_dir"),
        batched_object_pose_trace_output_dir=_extract_last_flag_value(argv, "--batched_object_pose_trace_output_dir"),
        video_output=_extract_last_flag_value(argv, "--video_output"),
        video_fps=int(_extract_last_flag_value(argv, "--video_fps") or DEFAULT_VIDEO_FPS),
        video_format=_extract_last_flag_value(argv, "--video_format") or DEFAULT_VIDEO_FORMAT,
        video_codec=_extract_last_flag_value(argv, "--video_codec") or AUTO_VIDEO_CODEC,
        auto_pick_macro=auto_pick_macro,
        auto_ee_axis_probe=False,
        command_mode=_extract_last_flag_value(argv, "--command_mode") or "solver_intent",
        test_mode=_extract_last_flag_value(argv, "--test_mode") or "hold",
    )


def _maybe_save_debug_side_artifacts(args, video_frames, save_video_buffer_to_path) -> None:
    unified_artifacts = _load_unified_proxy_artifacts()
    temp_video_path_for_gif = None
    try:
        if args.save_video_on_exit and args.save_video_path:
            saved_video_path = save_video_buffer_to_path(args.save_video_path)
            if not saved_video_path:
                raise RuntimeError("Failed to save debug video side artifact on exit.")
        if args.save_video_gif_on_exit and args.save_video_gif_path:
            source_video_path = saved_video_path if args.save_video_on_exit and 'saved_video_path' in locals() else None
            if source_video_path is None:
                temp_video_path_for_gif = (
                    Path(args.save_video_gif_path).expanduser().resolve().with_suffix(".tmp_debug_video.mkv")
                )
                source_video_path = save_video_buffer_to_path(temp_video_path_for_gif)
                if not source_video_path:
                    raise RuntimeError("Failed to save temporary debug video required for GIF generation.")
            unified_artifacts.write_debug_video_gif_from_video(source_video_path, args.save_video_gif_path)
    finally:
        if temp_video_path_for_gif is not None and temp_video_path_for_gif.exists():
            try:
                temp_video_path_for_gif.unlink()
            except Exception as exc:
                _emit_yellow_warning(
                    f"Failed to remove temporary GIF source video: {temp_video_path_for_gif} ({exc})"
                )


def _maybe_save_batched_debug_side_artifacts(
    args,
    *,
    video_frames_per_env,
    save_video_buffer_for_env_to_path,
    finalized_video_env_indices=None,
) -> None:
    unified_artifacts = _load_unified_proxy_artifacts()
    batched_save_video_output_dir = getattr(args, "batched_save_video_output_dir", None)
    batched_save_video_gif_output_dir = getattr(args, "batched_save_video_gif_output_dir", None)
    video_output_dir = (
        None
        if batched_save_video_output_dir is None
        else Path(batched_save_video_output_dir).expanduser().resolve()
    )
    gif_output_dir = (
        None
        if batched_save_video_gif_output_dir is None
        else Path(batched_save_video_gif_output_dir).expanduser().resolve()
    )
    if video_output_dir is None and gif_output_dir is None:
        return

    temp_video_paths = []
    try:
        for env_index, _frames in enumerate(video_frames_per_env):
            if finalized_video_env_indices and int(env_index) in finalized_video_env_indices:
                continue
            if len(_frames) == 0:
                continue
            saved_video_path = None
            if video_output_dir is not None:
                target_video_path = unified_artifacts.resolve_batched_debug_video_output_path(
                    video_output_dir,
                    env_index=env_index,
                )
                saved_video_path = save_video_buffer_for_env_to_path(env_index, target_video_path)
                if not saved_video_path:
                    raise RuntimeError(
                        f"Failed to save batched debug video side artifact for env_index={env_index}."
                    )
            if gif_output_dir is not None:
                source_video_path = saved_video_path
                temp_video_path_for_gif = None
                if source_video_path is None:
                    target_gif_path = unified_artifacts.resolve_batched_debug_video_gif_output_path(
                        gif_output_dir,
                        env_index=env_index,
                    )
                    temp_video_path_for_gif = target_gif_path.with_suffix(".tmp_debug_video.mkv")
                    source_video_path = save_video_buffer_for_env_to_path(env_index, temp_video_path_for_gif)
                    if not source_video_path:
                        raise RuntimeError(
                            f"Failed to save temporary batched debug video required for GIF generation "
                            f"for env_index={env_index}."
                        )
                    temp_video_paths.append(temp_video_path_for_gif)
                target_gif_path = unified_artifacts.resolve_batched_debug_video_gif_output_path(
                    gif_output_dir,
                    env_index=env_index,
                )
                unified_artifacts.write_debug_video_gif_from_video(source_video_path, target_gif_path)
    finally:
        for temp_video_path in temp_video_paths:
            if temp_video_path.exists():
                try:
                    temp_video_path.unlink()
                except Exception as exc:
                    _emit_yellow_warning(
                        f"Failed to remove temporary GIF source video: {temp_video_path} ({exc})"
                    )


def _build_direct_stage_callbacks(
    *,
    unified_control,
    unified_debug,
    unified_motion,
    unified_lowlevel,
    unified_targets,
    unified_stage_pose,
    unified_stage_close,
    unified_stage_lift,
):
    def execute_real_planner_pose_with_backend(solver, target_pose, **kwargs):
        planner_pose_executor = getattr(
            unified_motion,
            "execute_real_planner_pose_with_backend",
            unified_motion.execute_planner_pose_with_backend,
        )
        return planner_pose_executor(
            solver,
            target_pose,
            is_proxy_ee_delta_backend=unified_control.is_proxy_ee_delta_backend,
            is_rc5_debug_planner_agent=unified_control.is_rc5_debug_planner_agent,
            **kwargs,
        )

    def execute_planner_pose_with_backend(solver, target_pose, **kwargs):
        return execute_real_planner_pose_with_backend(
            solver,
            target_pose,
            **kwargs,
        )

    def run_linear_approach_waypoints(env, solver, target_pose, **kwargs):
        return unified_motion.run_linear_approach_waypoints(
            env,
            solver,
            target_pose,
            get_debug_planner_ee_pose=unified_debug.get_debug_planner_ee_pose,
            pose_to_numpy=unified_debug.pose_to_numpy,
            execute_planner_pose_with_backend=execute_real_planner_pose_with_backend,
            **kwargs,
        )

    def run_proxy_ee_delta_pose_stage(env, target_pose, **kwargs):
        return unified_motion.run_proxy_ee_delta_pose_stage(
            env,
            target_pose,
            get_debug_planner_config=unified_lowlevel.get_debug_planner_config,
            get_debug_planner_ee_pose_sapien=unified_debug.get_debug_planner_ee_pose_sapien,
            get_debug_planner_ee_pose=unified_debug.get_debug_planner_ee_pose_rows,
            pose_to_numpy_rows=unified_debug.pose_to_numpy_rows,
            compute_proxy_rotvec_step=unified_lowlevel.compute_proxy_rotvec_step,
            build_proxy_delta_pos=unified_lowlevel.build_proxy_delta_pos,
            apply_proxy_ee_delta_action=lambda *args, **inner_kwargs: unified_lowlevel.apply_proxy_ee_delta_action(
                *args,
                pose_to_numpy=unified_debug.pose_to_numpy,
                **inner_kwargs,
            ),
            **kwargs,
        )

    def run_proxy_full_approach_to_descend(
        env,
        target_pose,
        *,
        initial_actor_p,
        bbox_np,
        stage_label,
        safe_clearance_z,
    ):
        return unified_motion.run_proxy_full_approach_to_descend(
            env,
            target_pose,
            initial_actor_p=initial_actor_p,
            bbox_np=bbox_np,
            stage_label=stage_label,
            safe_clearance_z=safe_clearance_z,
            get_debug_planner_config=unified_lowlevel.get_debug_planner_config,
            get_debug_planner_ee_pose_sapien=unified_debug.get_debug_planner_ee_pose_sapien,
            get_debug_planner_ee_pose=unified_debug.get_debug_planner_ee_pose_rows,
            pose_to_numpy_rows=unified_debug.pose_to_numpy_rows,
            pose_to_numpy=unified_debug.pose_to_numpy,
            run_proxy_ee_delta_pose_stage=run_proxy_ee_delta_pose_stage,
            run_proxy_stationary_settle=run_proxy_stationary_settle,
            run_proxy_guarded_descend_to_object=run_proxy_guarded_descend_to_object,
        )

    def run_proxy_full_approach_to_pregrasp(
        env,
        target_pose,
        *,
        initial_actor_p,
        bbox_np,
        stage_label,
        safe_clearance_z,
    ):
        return unified_motion.run_proxy_full_approach_to_pregrasp(
            env,
            target_pose,
            initial_actor_p=initial_actor_p,
            bbox_np=bbox_np,
            stage_label=stage_label,
            safe_clearance_z=safe_clearance_z,
            get_debug_planner_config=unified_lowlevel.get_debug_planner_config,
            get_debug_planner_ee_pose_sapien=unified_debug.get_debug_planner_ee_pose_sapien,
            get_debug_planner_ee_pose=unified_debug.get_debug_planner_ee_pose_rows,
            pose_to_numpy_rows=unified_debug.pose_to_numpy_rows,
            pose_to_numpy=unified_debug.pose_to_numpy,
            run_proxy_ee_delta_pose_stage=run_proxy_ee_delta_pose_stage,
            run_proxy_stationary_settle=run_proxy_stationary_settle,
            refresh_render_state=unified_debug.refresh_render_state,
            set_debug_planner_last_task_pose=unified_debug.set_debug_planner_last_task_pose,
        )

    def run_proxy_stationary_settle(env, *, settle_steps, stage_label, gripper_target_state="hold"):
        return unified_lowlevel.run_proxy_stationary_settle(
            env,
            settle_steps=settle_steps,
            stage_label=stage_label,
            gripper_target_state=gripper_target_state,
            get_debug_planner_ee_pose_sapien=unified_debug.get_debug_planner_ee_pose_sapien,
            get_debug_planner_ee_pose=unified_debug.get_debug_planner_ee_pose_rows,
            pose_to_numpy=unified_debug.pose_to_numpy,
            pose_to_numpy_rows=unified_debug.pose_to_numpy_rows,
        )

    def run_proxy_guarded_descend_to_object(
        env,
        initial_target_pose,
        *,
        initial_actor_p,
        bbox_np,
        stage_label,
        align_orientation,
    ):
        return unified_lowlevel.run_proxy_guarded_descend_to_object(
            env,
            initial_target_pose,
            initial_actor_p=initial_actor_p,
            bbox_np=bbox_np,
            stage_label=stage_label,
            align_orientation=align_orientation,
            get_debug_planner_ee_pose_sapien=unified_debug.get_debug_planner_ee_pose_sapien,
            get_debug_planner_ee_pose=unified_debug.get_debug_planner_ee_pose_rows,
            get_debug_target_object=unified_debug.get_debug_target_object,
            get_debug_actor_position_xyz=unified_debug.get_debug_actor_position_xyz,
            get_debug_actor_position_rows=unified_debug.get_debug_actor_position_rows,
            pose_to_numpy_rows=unified_debug.pose_to_numpy_rows,
            log_debug_non_target_object_contacts=unified_debug.log_debug_non_target_object_contacts,
        )

    def run_planner_object_pregrasp_probe(
        env,
        method="auto",
        extra_clearance=0.10,
        execute=False,
        backend="proxy_ee_delta",
    ):
        pregrasp_runner = getattr(
            unified_stage_pose,
            "run_proxy_pregrasp_probe",
            unified_stage_pose.run_planner_object_pregrasp_probe,
        )
        return pregrasp_runner(
            env,
            method=method,
            extra_clearance=extra_clearance,
            execute=execute,
            backend=backend,
            extract_planner_base_pose=unified_debug.extract_planner_base_pose,
            resolve_planner_debug_solver_class=unified_debug.resolve_planner_debug_solver_class,
            is_proxy_ee_delta_backend=unified_control.is_proxy_ee_delta_backend,
            is_proxy_then_planner_backend=unified_control.is_proxy_then_planner_backend,
            is_ee_delta_control_mode=unified_lowlevel.is_ee_delta_control_mode,
            maybe_seed_proxy_start_pose=maybe_seed_proxy_start_pose,
            build_object_pregrasp_target=build_object_pregrasp_target,
            run_proxy_full_approach_to_pregrasp=run_proxy_full_approach_to_pregrasp,
            planner_visuals_supported=unified_debug.planner_visuals_supported,
            configure_debug_planner_solver_runtime=unified_debug.configure_debug_planner_solver_runtime,
            get_planner_recording_kwargs=unified_debug.get_planner_recording_kwargs,
            get_object_specific_planner_profile=unified_debug.get_object_specific_planner_profile,
            refresh_render_state=unified_debug.refresh_render_state,
            pose_to_numpy=unified_debug.pose_to_numpy,
            get_debug_planner_ee_pose=unified_debug.get_debug_planner_ee_pose,
            set_debug_planner_last_task_pose=unified_debug.set_debug_planner_last_task_pose,
        )

    def run_planner_full_approach_to_descend(
        env,
        method="auto",
        extra_clearance=0.03,
        execute=True,
        backend="proxy_ee_delta",
    ):
        execute_real_planner_pose_with_backend = getattr(
            unified_motion,
            "execute_real_planner_pose_with_backend",
            unified_motion.execute_planner_pose_with_backend,
        )
        full_approach_runner = getattr(
            unified_stage_pose,
            "run_proxy_full_approach_to_descend",
            unified_stage_pose.run_planner_full_approach_to_descend,
        )
        return full_approach_runner(
            env,
            method=method,
            extra_clearance=extra_clearance,
            execute=execute,
            backend=backend,
            extract_planner_base_pose=unified_debug.extract_planner_base_pose,
            resolve_planner_debug_solver_class=unified_debug.resolve_planner_debug_solver_class,
            is_proxy_ee_delta_backend=unified_control.is_proxy_ee_delta_backend,
            is_ee_delta_control_mode=unified_lowlevel.is_ee_delta_control_mode,
            build_object_descend_target=build_object_descend_target,
            run_proxy_full_approach_to_descend=run_proxy_full_approach_to_descend,
            refresh_render_state=unified_debug.refresh_render_state,
            pose_to_numpy=unified_debug.pose_to_numpy,
            get_debug_planner_ee_pose=unified_debug.get_debug_planner_ee_pose,
            set_debug_planner_last_task_pose=unified_debug.set_debug_planner_last_task_pose,
            planner_visuals_supported=unified_debug.planner_visuals_supported,
            configure_debug_planner_solver_runtime=unified_debug.configure_debug_planner_solver_runtime,
            get_planner_recording_kwargs=unified_debug.get_planner_recording_kwargs,
            run_linear_approach_waypoints=run_linear_approach_waypoints,
            execute_real_planner_pose_with_backend=execute_real_planner_pose_with_backend,
        )

    def run_planner_object_descend(
        env,
        method="rrtconnect",
        extra_clearance=0.03,
        execute=True,
        backend="proxy_ee_delta",
    ):
        execute_real_planner_pose_with_backend = getattr(
            unified_motion,
            "execute_real_planner_pose_with_backend",
            unified_motion.execute_planner_pose_with_backend,
        )
        descend_runner = getattr(
            unified_stage_pose,
            "run_proxy_descend",
            unified_stage_pose.run_planner_object_descend,
        )
        return descend_runner(
            env,
            method=method,
            extra_clearance=extra_clearance,
            execute=execute,
            backend=backend,
            extract_planner_base_pose=unified_debug.extract_planner_base_pose,
            resolve_planner_debug_solver_class=unified_debug.resolve_planner_debug_solver_class,
            is_proxy_ee_delta_backend=unified_control.is_proxy_ee_delta_backend,
            is_ee_delta_control_mode=unified_lowlevel.is_ee_delta_control_mode,
            build_object_descend_target=build_object_descend_target,
            run_proxy_ee_delta_pose_stage=run_proxy_ee_delta_pose_stage,
            planner_visuals_supported=unified_debug.planner_visuals_supported,
            configure_debug_planner_solver_runtime=unified_debug.configure_debug_planner_solver_runtime,
            get_planner_recording_kwargs=unified_debug.get_planner_recording_kwargs,
            refresh_render_state=unified_debug.refresh_render_state,
            pose_to_numpy=unified_debug.pose_to_numpy,
            get_debug_planner_ee_pose=unified_debug.get_debug_planner_ee_pose,
            run_linear_approach_waypoints=run_linear_approach_waypoints,
            execute_real_planner_pose_with_backend=execute_real_planner_pose_with_backend,
            set_debug_planner_last_task_pose=unified_debug.set_debug_planner_last_task_pose,
        )

    def run_planner_close_gripper(env, close_steps=20, backend="proxy_ee_delta"):
        return unified_stage_close.run_planner_close_gripper(
            env,
            close_steps=close_steps,
            backend=backend,
            extract_planner_base_pose=unified_debug.extract_planner_base_pose,
            resolve_planner_debug_solver_class=unified_debug.resolve_planner_debug_solver_class,
            is_proxy_ee_delta_backend=unified_control.is_proxy_ee_delta_backend,
            is_ee_delta_control_mode=unified_lowlevel.is_ee_delta_control_mode,
            run_proxy_close_gripper=run_proxy_close_gripper,
            configure_debug_planner_solver_runtime=unified_debug.configure_debug_planner_solver_runtime,
            get_planner_recording_kwargs=unified_debug.get_planner_recording_kwargs,
            get_debug_target_object=unified_debug.get_debug_target_object,
            get_debug_actor_position_xyz=unified_debug.get_debug_actor_position_xyz,
            log_debug_pre_close_snapshot=unified_debug.log_debug_pre_close_snapshot,
            pose_to_numpy=unified_debug.pose_to_numpy,
            get_debug_planner_ee_pose=unified_debug.get_debug_planner_ee_pose,
            log_debug_post_close_retention=unified_debug.log_debug_post_close_retention,
            get_robot_hand_qpos_debug=unified_debug.get_robot_hand_qpos_debug,
            get_robot_hand_range_debug=unified_debug.get_robot_hand_range_debug,
            save_planner_grasp_state=save_planner_grasp_state,
            refresh_render_state=unified_debug.refresh_render_state,
        )

    def run_planner_lift(
        env,
        lift_delta_z=0.05,
        method="rrtconnect",
        execute=True,
        repeat=1,
        backend="proxy_ee_delta",
    ):
        execute_real_planner_pose_with_backend = getattr(
            unified_motion,
            "execute_real_planner_pose_with_backend",
            unified_motion.execute_planner_pose_with_backend,
        )
        lift_runner = getattr(
            unified_stage_lift,
            "run_proxy_lift",
            unified_stage_lift.run_planner_lift,
        )
        return lift_runner(
            env,
            lift_delta_z=lift_delta_z,
            method=method,
            execute=execute,
            repeat=repeat,
            backend=backend,
            extract_planner_base_pose=unified_debug.extract_planner_base_pose,
            resolve_planner_debug_solver_class=unified_debug.resolve_planner_debug_solver_class,
            is_proxy_ee_delta_backend=unified_control.is_proxy_ee_delta_backend,
            is_ee_delta_control_mode=unified_lowlevel.is_ee_delta_control_mode,
            run_proxy_ee_delta_pose_stage=run_proxy_ee_delta_pose_stage,
            pose_to_numpy=unified_debug.pose_to_numpy,
            get_debug_planner_ee_pose=unified_debug.get_debug_planner_ee_pose,
            get_debug_planner_ee_pose_rows=unified_debug.get_debug_planner_ee_pose_rows,
            get_debug_planner_retention_pose=unified_debug.get_debug_planner_retention_pose,
            get_debug_planner_retention_pose_rows=unified_debug.get_debug_planner_retention_pose_rows,
            get_debug_target_object=unified_debug.get_debug_target_object,
            get_debug_actor_position_xyz=unified_debug.get_debug_actor_position_xyz,
            to_scalar_bool=unified_debug.to_scalar_bool,
            get_robot_hand_qpos_debug=unified_debug.get_robot_hand_qpos_debug,
            get_robot_hand_qpos_rows_debug=unified_debug.get_robot_hand_qpos_rows_debug,
            get_robot_qpos=unified_debug.get_robot_qpos,
            configure_debug_planner_solver_runtime=unified_debug.configure_debug_planner_solver_runtime,
            get_planner_recording_kwargs=unified_debug.get_planner_recording_kwargs,
            build_debug_lift_pose_from_policy=unified_debug.build_debug_lift_pose_from_policy,
            is_rc5_debug_planner_agent=unified_control.is_rc5_debug_planner_agent,
            evaluate_rc5_pick_lift_success=evaluate_rc5_pick_lift_success,
            get_planner_grasp_state=get_planner_grasp_state,
            save_planner_grasp_state=save_planner_grasp_state,
            refresh_render_state=unified_debug.refresh_render_state,
            execute_real_planner_pose_with_backend=execute_real_planner_pose_with_backend,
        )

    def build_object_pregrasp_target(env, extra_clearance=0.10, radial_backoff_override=None):
        return unified_targets.build_object_pregrasp_target(
            env,
            extra_clearance=extra_clearance,
            radial_backoff_override=radial_backoff_override,
            get_debug_target_object=unified_debug.get_debug_target_object,
            get_object_specific_planner_profile=unified_debug.get_object_specific_planner_profile,
            resolve_rc5_target_semantics=unified_debug.resolve_rc5_target_semantics,
            align_rc5_target_pose_to_active_move_group=unified_debug.align_rc5_target_pose_to_active_move_group,
            pose_to_numpy=unified_debug.pose_to_numpy,
            pose_to_numpy_rows=unified_debug.pose_to_numpy_rows,
            get_debug_planner_ee_pose=unified_debug.get_debug_planner_ee_pose,
            is_rc5_debug_planner_agent=unified_control.is_rc5_debug_planner_agent,
        )

    def build_object_descend_target(env, extra_clearance=0.03):
        return unified_targets.build_object_descend_target(
            env,
            extra_clearance=extra_clearance,
            get_debug_target_object=unified_debug.get_debug_target_object,
            get_object_specific_planner_profile=unified_debug.get_object_specific_planner_profile,
            resolve_rc5_target_semantics=unified_debug.resolve_rc5_target_semantics,
            align_rc5_target_pose_to_active_move_group=unified_debug.align_rc5_target_pose_to_active_move_group,
            pose_to_numpy=unified_debug.pose_to_numpy,
            pose_to_numpy_rows=unified_debug.pose_to_numpy_rows,
            get_debug_planner_ee_pose=unified_debug.get_debug_planner_ee_pose,
            is_rc5_debug_planner_agent=unified_control.is_rc5_debug_planner_agent,
        )

    def run_proxy_close_gripper(env, *, close_steps):
        return unified_lowlevel.run_proxy_close_gripper(
            env,
            close_steps=close_steps,
            get_debug_planner_ee_pose=unified_debug.get_debug_planner_ee_pose,
            get_debug_planner_ee_pose_rows=unified_debug.get_debug_planner_ee_pose_rows,
            pose_to_numpy=unified_debug.pose_to_numpy,
            pose_to_numpy_rows=unified_debug.pose_to_numpy_rows,
            get_debug_target_object=unified_debug.get_debug_target_object,
            get_debug_actor_position_xyz=unified_debug.get_debug_actor_position_xyz,
            log_debug_pre_close_snapshot=unified_debug.log_debug_pre_close_snapshot,
            log_debug_post_close_retention=unified_debug.log_debug_post_close_retention,
            get_robot_hand_qpos_debug=unified_debug.get_robot_hand_qpos_debug,
            get_robot_hand_range_debug=unified_debug.get_robot_hand_range_debug,
            refresh_render_state=unified_debug.refresh_render_state,
            run_proxy_guarded_descend_to_object=run_proxy_guarded_descend_to_object,
        )

    def maybe_seed_proxy_start_pose(env, *, reason):
        return unified_lowlevel.maybe_seed_proxy_start_pose(
            env,
            reason=reason,
            get_hybrid_jointspace_control_mode=unified_control.get_hybrid_jointspace_control_mode,
            temporary_agent_control_mode=unified_control.temporary_agent_control_mode,
            get_robot_qpos=unified_debug.get_robot_qpos,
        )

    return {
        "run_proxy_pregrasp_probe": run_planner_object_pregrasp_probe,
        "run_proxy_full_approach_to_descend": run_planner_full_approach_to_descend,
        "run_proxy_descend": run_planner_object_descend,
        "run_proxy_close_gripper": run_planner_close_gripper,
        "run_proxy_lift": run_planner_lift,
        "execute_real_planner_pose_with_backend": execute_real_planner_pose_with_backend,
        "run_planner_object_pregrasp_probe": run_planner_object_pregrasp_probe,
        "run_planner_full_approach_to_descend": run_planner_full_approach_to_descend,
        "run_planner_object_descend": run_planner_object_descend,
        "run_planner_close_gripper": run_planner_close_gripper,
        "run_planner_lift": run_planner_lift,
        "execute_planner_pose_with_backend": execute_planner_pose_with_backend,
        "build_object_descend_target": build_object_descend_target,
    }


def _explain_direct_executor_skip(argv) -> str:
    argv = list(argv or [])
    if "--auto_ee_axis_probe" in argv:
        return "direct executor does not support --auto_ee_axis_probe"
    if any(flag in argv for flag in ("--robot_base_pose", "--robot_init_qpos", "--cam_eye", "--cam_target")):
        return "direct executor does not support custom multi-value pose/camera overrides yet"
    num_envs = int(_extract_last_flag_value(argv, "--num_envs") or 1)
    if num_envs > 1 and "--headless" not in argv:
        return "direct executor batched runtime currently supports only --headless"
    if num_envs > 1 and _extract_last_flag_value(argv, "--dense_episode_output") is not None:
        return (
            "direct executor batched runtime does not support a single shared --dense_episode_output; "
            "per-env artifact persistence belongs to the collector path"
        )
    if num_envs > 1 and _extract_last_flag_value(argv, "--rl4vla_raw_episode_output") is not None:
        return (
            "direct executor batched runtime does not support a single shared --rl4vla_raw_episode_output; "
            "per-env artifact persistence belongs to the collector path"
        )
    if num_envs > 1 and (
        "--save_video_on_exit" in argv
        or "--save_video_gif_on_exit" in argv
        or _extract_last_flag_value(argv, "--video_output") is not None
        or _extract_last_flag_value(argv, "--save_video_path") is not None
        or _extract_last_flag_value(argv, "--save_video_gif_path") is not None
    ):
        return (
            "direct executor batched runtime does not support shared debug video artifacts; "
            "per-env artifact accounting belongs to the collector path"
        )
    auto_pick_macro = _extract_last_flag_value(argv, "--auto_pick_macro")
    if auto_pick_macro not in {"1", "2"}:
        return "direct executor currently supports only --auto_pick_macro=1 or 2"
    return "direct executor eligibility check did not match the validated envelope"


def _requires_direct_executor(argv) -> bool:
    argv = list(argv or [])
    return _extract_last_flag_value(argv, "--auto_pick_macro") in {"1", "2"}


def _check_viewer(viewer) -> None:
    if viewer is None:
        raise RuntimeError("Failed to initialize viewer")
    if hasattr(viewer, "window") and viewer.window is None:
        raise RuntimeError("Viewer window is None. Check GUI/X11 access.")


def _sanitize_artifact_token(value: str | None, *, default: str) -> str:
    token = "".join(ch if str(ch).isalnum() or str(ch) in {"-", "_"} else "_" for ch in str(value or "").strip())
    token = token.strip("_")
    return token or default


def _resolve_episode_outcome_token(macro_feedback) -> str:
    if not macro_feedback:
        return "fail"
    return "success" if bool(macro_feedback.get("semantic_task_success")) else "fail"


def _allocate_viewer_video_snapshot_path(
    env_unwrapped,
    args,
    config_overrides,
    *,
    video_format: str,
    video_codec: str | None,
) -> Path:
    snapshot_dir = getattr(env_unwrapped, "_debug_planner_viewer_video_dir", None)
    if snapshot_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        key_token = _sanitize_artifact_token(getattr(args, "key", None), default="no_key")
        task_token = _sanitize_artifact_token(config_overrides.get("task_type"), default="task")
        object_token = _sanitize_artifact_token(config_overrides.get("manip_object_id"), default="object")
        snapshot_dir = (Path("runs/manual") / f"viewer_video_{key_token}_{task_token}_{object_token}_{timestamp}").resolve()
        env_unwrapped._debug_planner_viewer_video_dir = snapshot_dir
        env_unwrapped._debug_planner_viewer_video_index = 0
    next_index = int(getattr(env_unwrapped, "_debug_planner_viewer_video_index", 0)) + 1
    env_unwrapped._debug_planner_viewer_video_index = next_index
    outcome_token = _sanitize_artifact_token(
        getattr(env_unwrapped, "_debug_planner_episode_outcome_token", None),
        default="fail",
    )
    requested_path = snapshot_dir / f"{next_index:03d}_{outcome_token}.{str(video_format).lstrip('.')}"
    unified_artifacts = _load_unified_proxy_artifacts()
    return unified_artifacts.resolve_output_video_path(
        requested_path,
        video_format=video_format,
        video_codec=video_codec,
    )


def _keep_viewer_open_for_inspection(env, viewer, *, unified_lowlevel) -> None:
    env_unwrapped = env.unwrapped
    print("[PlannerDebug] Proxy planner macro finished; keeping viewer open for inspection.")
    print("[PlannerDebug] Press 'v' to save the current buffered video and 'q' to exit.")
    while viewer is not None and not getattr(viewer, "closed", False):
        if getattr(viewer, "window", None) is not None:
            try:
                viewer.render()
            except Exception:
                break
            unified_lowlevel.maybe_handle_proxy_viewer_video_hotkey(env, viewer)
            try:
                if unified_lowlevel.viewer_key_pressed_once(env_unwrapped, viewer, "q"):
                    break
            except Exception:
                pass
        else:
            break
        time.sleep(0.01)


def _install_headless_render_guard(env) -> None:
    env.unwrapped._debug_planner_headless = True
    original_render_human = getattr(env, "render_human", None)
    if not callable(original_render_human):
        return

    def _forbidden_render_human(*_args, **_kwargs):
        raise RuntimeError(
            "Unified proxy headless guard: render_human() was invoked during a --headless run. "
            "This would create a viewer window and indicates a headless/runtime bug."
        )

    env.render_human = _forbidden_render_human


def _execute_independent_batched_proxy_pick_macro(
    env,
    *,
    args,
    config_overrides,
    unified_artifacts,
    unified_control,
    unified_debug,
    unified_lowlevel,
    build_object_descend_target,
    save_video_buffer_for_env_to_path,
    video_frames_per_env,
    finalized_video_env_indices,
):
    env_unwrapped = env.unwrapped
    num_envs = int(getattr(env_unwrapped, "num_envs", 1) or 1)
    if num_envs <= 1:
        raise ValueError("Independent batched proxy executor requires num_envs > 1.")

    resolve_proxy_only_macro_backend = getattr(
        unified_control,
        "resolve_proxy_only_macro_backend",
        None,
    )
    if callable(resolve_proxy_only_macro_backend):
        try:
            macro_backend = resolve_proxy_only_macro_backend(
                config_overrides,
                default_backend="proxy_ee_delta",
            )
        except ValueError as exc:
            _raise_unified_proxy_fail_fast(str(exc))
    else:
        macro_backend = unified_control.resolve_macro_backend(
            config_overrides,
            default_backend="proxy_ee_delta",
        )
        if not unified_control.is_proxy_ee_delta_backend(macro_backend):
            _raise_unified_proxy_fail_fast(
                "independent batched proxy executor currently supports only "
                "macro_backend='proxy_ee_delta' (legacy planner_backend naming)"
            )

    planner_cfg = unified_lowlevel.get_debug_planner_config(env_unwrapped)
    hold_steps = int(planner_cfg.get("planner_proxy_hold_steps", 1) or 1)
    if hold_steps != 1:
        _raise_unified_proxy_fail_fast(
            "independent batched proxy executor currently requires planner_proxy_hold_steps=1"
        )
    if any(
        int(planner_cfg.get(flag_name, 0) or 0) > 0
        for flag_name in (
            "planner_proxy_postclose_settle_steps",
            "planner_proxy_batch_close_retry_rounds",
            "planner_proxy_batch_close_retry_steps",
            "planner_proxy_batch_close_retry_settle_steps",
            "planner_proxy_batch_postclose_reseat_rounds",
            "planner_proxy_batch_postclose_reseat_close_steps",
            "planner_proxy_batch_postclose_reseat_settle_steps",
        )
    ):
        _raise_unified_proxy_fail_fast(
            "independent batched proxy executor does not support batch-only close reseat/retry heuristics"
        )

    safe_clearance_z = float(planner_cfg.get("planner_proxy_safe_clearance_z", 0.10))
    predescent_settle_steps = int(planner_cfg.get("planner_proxy_predescent_settle_steps", 12) or 0)
    preclose_settle_steps = int(planner_cfg.get("planner_proxy_preclose_settle_steps", 6) or 0)
    max_stage_steps = int(planner_cfg.get("planner_proxy_max_stage_steps", 100) or 100)
    stall_limit = int(planner_cfg.get("planner_proxy_stall_steps", 12) or 12)
    waypoint_chain_enabled, waypoint_points = unified_lowlevel.get_required_planner_waypoints_config(planner_cfg)
    _adaptive_enabled, _threshold_xy, _threshold_z, pos_tol = unified_lowlevel.get_required_proxy_adaptive_step_config(
        planner_cfg
    )
    max_xy_step = float(planner_cfg.get("planner_proxy_xy_step_m", 0.01))
    max_z_step = float(planner_cfg.get("planner_proxy_z_step_m", 0.008))
    descend_xy_step = float(min(max_xy_step, 0.004))
    close_steps = int(unified_control.get_default_debug_close_steps(getattr(env_unwrapped.agent, "uid", "unknown")))
    lift_delta_z = 2.0 * float(config_overrides.get("planner_lift_delta_z", 0.05))
    batch_env_step_timeout_scale = float(
        planner_cfg.get("planner_proxy_batch_env_step_timeout_scale", 2.0) or 2.0
    )
    batch_env_step_timeout_min_steps = int(
        planner_cfg.get("planner_proxy_batch_env_step_timeout_min_steps", 0) or 0
    )
    runaway_abs_limit_m = float(planner_cfg.get("planner_proxy_batch_runaway_abs_limit_m", 5.0) or 5.0)
    macro_route = f"pick_macro_{int(args.auto_pick_macro)}"

    get_signal = getattr(unified_lowlevel, "_get_proxy_ee_gripper_controller_signal")
    open_signal = float(get_signal(env_unwrapped, "open"))
    close_signal = float(get_signal(env_unwrapped, "close"))

    target_object = unified_debug.get_debug_target_object(env_unwrapped)
    if target_object is None:
        raise RuntimeError("Independent batched proxy executor requires a resolved target object.")

    manip_id, initial_actor_p_rows, bbox_np, descend_target_pose = build_object_descend_target(
        env,
        extra_clearance=0.03,
    )
    descend_target_p_rows, descend_target_q_rows = unified_debug.pose_to_numpy_rows(descend_target_pose)
    initial_actor_p_rows = np.asarray(initial_actor_p_rows, dtype=np.float32).reshape(num_envs, 3)
    current_tcp_p_rows, _current_tcp_q_rows = unified_debug.pose_to_numpy_rows(
        unified_debug.get_debug_planner_ee_pose_rows(env_unwrapped)
    )
    object_top_z_rows = initial_actor_p_rows[:, 2] + 0.5 * float(np.asarray(bbox_np, dtype=np.float32).reshape(-1)[2])
    safe_z_rows = np.maximum(current_tcp_p_rows[:, 2], object_top_z_rows + float(safe_clearance_z))
    rise_target_p_rows = current_tcp_p_rows.copy()
    rise_target_p_rows[:, 2] = safe_z_rows
    move_xy_target_p_rows = descend_target_p_rows.copy()
    move_xy_target_p_rows[:, 2] = safe_z_rows
    waypoint_stage_names = [f"waypoint_{idx + 1:03d}" for idx in range(len(waypoint_points))] if waypoint_chain_enabled else []
    waypoint_target_p_rows_by_stage = {}
    if waypoint_chain_enabled:
        print(
            f"[PlannerDebug] Independent batched proxy waypoint chain enabled: count={len(waypoint_points)}"
        )
        for stage_name, waypoint in zip(waypoint_stage_names, waypoint_points):
            waypoint_position = np.asarray(waypoint["position"], dtype=np.float32).reshape(3)
            waypoint_rows = np.repeat(waypoint_position.reshape(1, 3), num_envs, axis=0).astype(np.float32)
            waypoint_target_p_rows_by_stage[str(stage_name)] = waypoint_rows
            print(
                f"[PlannerDebug] Independent batched proxy {stage_name} id='{waypoint['id']}' "
                f"target_p={np.array2string(waypoint_position, precision=4, suppress_small=True)}"
            )
    _set_batched_last_task_pose_rows(
        env_unwrapped,
        position_rows=descend_target_p_rows,
        quaternion_rows=descend_target_q_rows,
        bbox_np=bbox_np,
        stage_name="descend",
    )

    initial_stage_name = waypoint_stage_names[0] if waypoint_stage_names else "rise"
    stage_rows = [initial_stage_name] * num_envs
    stage_step_rows = [0] * num_envs
    stall_count_rows = [0] * num_envs
    last_error_rows = [None] * num_envs
    settle_remaining_rows = [0] * num_envs
    close_remaining_rows = [0] * num_envs
    lift_target_snapshot_rows = np.zeros((num_envs, 3), dtype=np.float32)
    lift_target_ready_rows = [False] * num_envs
    terminal_rows = [False] * num_envs
    terminal_artifacts_flushed_rows = [False] * num_envs
    gripper_closed_rows = [False] * num_envs
    total_step_rows = [0] * num_envs
    adaptive_move_xy_logged_rows = [False] * num_envs
    adaptive_descend_xy_logged_rows = [False] * num_envs
    adaptive_descend_z_logged_rows = [False] * num_envs
    first_success_step_limit = None
    per_env_feedback = [
        {
            "env_index": env_index,
            "semantic_task_success": False,
            "failed_stage": "full_approach",
        }
        for env_index in range(num_envs)
    ]
    trace_object_pose_enabled = str(os.environ.get("RC5_DEBUG_TRACE_OBJECT_POSE", "")).strip() == "1"
    trace_object_id = str(os.environ.get("RC5_DEBUG_TRACE_OBJECT_ID", "green_cube_ext") or "green_cube_ext").strip()
    trace_env_indices_raw = str(os.environ.get("RC5_DEBUG_TRACE_ENV_INDICES", "")).strip()
    trace_env_indices = None
    batched_object_pose_trace_output_dir = getattr(args, "batched_object_pose_trace_output_dir", None)
    object_pose_trace_handles = [None] * num_envs
    if batched_object_pose_trace_output_dir:
        trace_output_root = Path(batched_object_pose_trace_output_dir).expanduser().resolve()
        trace_output_root.mkdir(parents=True, exist_ok=True)
        for env_index in range(num_envs):
            trace_path = unified_artifacts.resolve_batched_object_pose_trace_output_path(
                trace_output_root,
                env_index=env_index,
            )
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            object_pose_trace_handles[env_index] = trace_path.open("w", encoding="utf-8", buffering=1)
    if trace_env_indices_raw:
        parsed_trace_env_indices = set()
        for item in trace_env_indices_raw.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                parsed_trace_env_indices.add(int(item))
            except ValueError:
                print(
                    f"{_Y}[WARNING] [PlannerDebug] Ignoring invalid RC5_DEBUG_TRACE_ENV_INDICES token: "
                    f"{item!r}{_R}"
                )
        trace_env_indices = parsed_trace_env_indices if parsed_trace_env_indices else None

    def _actor_vector_rows(actor, attr_name: str):
        value = getattr(actor, attr_name, None)
        if value is None:
            return None
        if callable(value):
            try:
                value = value()
            except TypeError:
                return None
        try:
            rows = unified_debug.to_numpy_rows(value)[:, :3]
        except Exception:
            return None
        if rows.shape[0] == 1 and num_envs > 1:
            rows = np.repeat(rows, num_envs, axis=0)
        if rows.shape[0] != num_envs:
            return None
        return rows

    def _filter_contact_records_for_env(contact_records, *, env_index: int):
        filtered_records = []
        for record in contact_records:
            env_ids = record.get("env_ids", [])
            if env_ids and int(env_index) not in {int(item) for item in env_ids}:
                continue
            filtered_records.append(record)
        return filtered_records

    def _maybe_log_traced_object_pose(
        *,
        global_step_idx: int,
        stage_name: str,
        tcp_p_rows=None,
        tcp_q_rows=None,
        raw_delta_pos_rows=None,
        raw_delta_rpy_rows=None,
        gripper_signal_rows=None,
        robot_object_contact_records=None,
        object_object_contact_records=None,
    ) -> None:
        object_actors = getattr(env_unwrapped, "object_actors", {}) or {}
        if trace_object_pose_enabled:
            traced_actor = object_actors.get(trace_object_id)
            if traced_actor is None:
                print(
                    f"{_Y}[WARNING] [PlannerDebug] RC5 traced object_id={trace_object_id!r} is not present in "
                    f"object_actors; available={sorted(str(key) for key in object_actors.keys())}{_R}"
                )
            else:
                try:
                    traced_p_rows, traced_q_rows = unified_debug.pose_to_numpy_rows(traced_actor.pose)
                except Exception as exc:
                    print(
                        f"{_Y}[WARNING] [PlannerDebug] Failed to read traced object pose for object_id={trace_object_id}: "
                        f"{type(exc).__name__}: {exc}{_R}"
                    )
                else:
                    row_count = int(min(len(traced_p_rows), len(traced_q_rows), num_envs))
                    for env_index in range(row_count):
                        if trace_env_indices is not None and int(env_index) not in trace_env_indices:
                            continue
                        print(
                            f"[PlannerDebug] TraceObjectPose global_step={global_step_idx} "
                            f"env_id={env_index} env_index={env_index} stage={stage_name} object_id={trace_object_id} "
                            f"actor_p={np.array2string(np.asarray(traced_p_rows[env_index], dtype=np.float32), precision=4, suppress_small=True)} "
                            f"actor_q={np.array2string(np.asarray(traced_q_rows[env_index], dtype=np.float32), precision=4, suppress_small=True)}"
                        )
        if not any(handle is not None for handle in object_pose_trace_handles):
            return
        per_object_pose_rows = {}
        per_object_linear_velocity_rows = {}
        per_object_angular_velocity_rows = {}
        for object_id, traced_actor in sorted(object_actors.items(), key=lambda item: str(item[0])):
            try:
                traced_p_rows, traced_q_rows = unified_debug.pose_to_numpy_rows(traced_actor.pose)
            except Exception as exc:
                print(
                    f"{_Y}[WARNING] [PlannerDebug] Object pose trace failed to read pose for object_id={object_id}: "
                    f"{type(exc).__name__}: {exc}{_R}"
                )
                continue
            per_object_pose_rows[str(object_id)] = (traced_p_rows, traced_q_rows)
            linear_velocity_rows = _actor_vector_rows(traced_actor, "linear_velocity")
            angular_velocity_rows = _actor_vector_rows(traced_actor, "angular_velocity")
            if linear_velocity_rows is not None:
                per_object_linear_velocity_rows[str(object_id)] = linear_velocity_rows
            if angular_velocity_rows is not None:
                per_object_angular_velocity_rows[str(object_id)] = angular_velocity_rows
        robot_object_contact_records = robot_object_contact_records or []
        object_object_contact_records = object_object_contact_records or []
        for env_index, handle in enumerate(object_pose_trace_handles):
            if handle is None:
                continue
            step_payload = {
                "global_step": int(global_step_idx),
                "env_id": int(env_index),
                "env_index": int(env_index),
                "stage": str(stage_rows[env_index]),
                "terminal": bool(terminal_rows[env_index]),
                "tcp": {},
                "planned_action": {},
                "contacts": {
                    "robot_object": _filter_contact_records_for_env(
                        robot_object_contact_records,
                        env_index=env_index,
                    ),
                    "object_object": _filter_contact_records_for_env(
                        object_object_contact_records,
                        env_index=env_index,
                    ),
                },
                "objects": {},
            }
            if tcp_p_rows is not None and tcp_q_rows is not None and env_index < len(tcp_p_rows) and env_index < len(tcp_q_rows):
                step_payload["tcp"] = {
                    "position": [
                        float(x)
                        for x in np.asarray(tcp_p_rows[env_index], dtype=np.float32).reshape(-1)[:3].tolist()
                    ],
                    "orientation": [
                        float(x)
                        for x in np.asarray(tcp_q_rows[env_index], dtype=np.float32).reshape(-1)[:4].tolist()
                    ],
                }
            if raw_delta_pos_rows is not None and env_index < len(raw_delta_pos_rows):
                step_payload["planned_action"]["raw_delta_pos"] = [
                    float(x)
                    for x in np.asarray(raw_delta_pos_rows[env_index], dtype=np.float32).reshape(-1)[:3].tolist()
                ]
            if raw_delta_rpy_rows is not None and env_index < len(raw_delta_rpy_rows):
                step_payload["planned_action"]["raw_delta_rpy"] = [
                    float(x)
                    for x in np.asarray(raw_delta_rpy_rows[env_index], dtype=np.float32).reshape(-1)[:3].tolist()
                ]
            if gripper_signal_rows is not None and env_index < len(gripper_signal_rows):
                value = float(np.asarray(gripper_signal_rows[env_index], dtype=np.float32).reshape(-1)[0])
                step_payload["planned_action"]["gripper_signal"] = None if np.isnan(value) else value
            for object_id, (traced_p_rows, traced_q_rows) in per_object_pose_rows.items():
                row_count = int(min(len(traced_p_rows), len(traced_q_rows), num_envs))
                if env_index >= row_count:
                    continue
                step_payload["objects"][object_id] = {
                    "position": [
                        float(x)
                        for x in np.asarray(traced_p_rows[env_index], dtype=np.float32).reshape(-1)[:3].tolist()
                    ],
                    "orientation": [
                        float(x)
                        for x in np.asarray(traced_q_rows[env_index], dtype=np.float32).reshape(-1)[:4].tolist()
                    ],
                }
                linear_velocity_rows = per_object_linear_velocity_rows.get(object_id)
                if linear_velocity_rows is not None and env_index < len(linear_velocity_rows):
                    step_payload["objects"][object_id]["linear_velocity"] = [
                        float(x)
                        for x in np.asarray(linear_velocity_rows[env_index], dtype=np.float32).reshape(-1)[:3].tolist()
                    ]
                angular_velocity_rows = per_object_angular_velocity_rows.get(object_id)
                if angular_velocity_rows is not None and env_index < len(angular_velocity_rows):
                    step_payload["objects"][object_id]["angular_velocity"] = [
                        float(x)
                        for x in np.asarray(angular_velocity_rows[env_index], dtype=np.float32).reshape(-1)[:3].tolist()
                    ]
            handle.write(json.dumps(step_payload, sort_keys=True) + "\n")

    def _transition(env_index: int, next_stage: str) -> None:
        current_stage = str(stage_rows[env_index])
        if current_stage.startswith("waypoint_") and adaptive_move_xy_logged_rows[env_index]:
            print(
                f"[PlannerDebug] Independent batched proxy env {env_index} stage='FullApproach:{current_stage}' "
                f"released adaptive XY step on transition to '{next_stage}'; "
                f"restoring nominal_xy_step={max_xy_step:.4f} m for subsequent stages"
            )
            adaptive_move_xy_logged_rows[env_index] = False
        if current_stage == "move_xy" and adaptive_move_xy_logged_rows[env_index]:
            print(
                f"[PlannerDebug] Independent batched proxy env {env_index} stage='FullApproach:move_xy_above_target' "
                f"released adaptive XY step on transition to '{next_stage}'; "
                f"restoring nominal_xy_step={max_xy_step:.4f} m for subsequent stages"
            )
            adaptive_move_xy_logged_rows[env_index] = False
        if current_stage == "descent" and adaptive_descend_xy_logged_rows[env_index]:
            print(
                f"[PlannerDebug] Independent batched proxy env {env_index} stage='FullApproach:descent' "
                f"released adaptive XY step on transition to '{next_stage}'; "
                f"restoring nominal_xy_step={descend_xy_step:.4f} m for subsequent stages"
            )
            adaptive_descend_xy_logged_rows[env_index] = False
        if current_stage == "descent" and adaptive_descend_z_logged_rows[env_index]:
            print(
                f"[PlannerDebug] Independent batched proxy env {env_index} stage='FullApproach:descent' "
                f"released adaptive Z step on transition to '{next_stage}'; "
                f"restoring nominal_z_step={max_z_step:.4f} m for subsequent stages"
            )
            adaptive_descend_z_logged_rows[env_index] = False
        stage_rows[env_index] = str(next_stage)
        stage_step_rows[env_index] = 0
        stall_count_rows[env_index] = 0
        last_error_rows[env_index] = None
        if next_stage == "predescent_settle":
            settle_remaining_rows[env_index] = int(predescent_settle_steps)
        elif next_stage == "preclose_settle":
            settle_remaining_rows[env_index] = int(preclose_settle_steps)
        elif next_stage == "close":
            close_remaining_rows[env_index] = max(int(close_steps), 1)
            gripper_closed_rows[env_index] = True
        elif next_stage == "lift":
            lift_target_ready_rows[env_index] = False

    def _finalize_terminal_env_artifacts(env_index: int) -> None:
        if terminal_artifacts_flushed_rows[env_index]:
            return
        env_feedback = dict(per_env_feedback[env_index])
        failed_stage = env_feedback.get("failed_stage")
        semantic_task_success = bool(env_feedback.get("semantic_task_success"))
        batched_save_video_output_dir = getattr(args, "batched_save_video_output_dir", None)
        batched_save_video_gif_output_dir = getattr(args, "batched_save_video_gif_output_dir", None)
        saved_video_path = None
        temp_video_path_for_gif = None
        try:
            if (
                video_frames_per_env is not None
                and 0 <= env_index < len(video_frames_per_env)
                and len(video_frames_per_env[env_index]) > 0
            ):
                if batched_save_video_output_dir is not None:
                    target_video_path = unified_artifacts.resolve_batched_debug_video_output_path(
                        batched_save_video_output_dir,
                        env_index=env_index,
                    )
                    saved_video_path = save_video_buffer_for_env_to_path(env_index, target_video_path)
                if batched_save_video_gif_output_dir is not None:
                    source_video_path = saved_video_path
                    if source_video_path is None:
                        target_gif_path = unified_artifacts.resolve_batched_debug_video_gif_output_path(
                            batched_save_video_gif_output_dir,
                            env_index=env_index,
                        )
                        temp_video_path_for_gif = target_gif_path.with_suffix(".tmp_debug_video.mkv")
                        source_video_path = save_video_buffer_for_env_to_path(env_index, temp_video_path_for_gif)
                    if source_video_path is not None:
                        target_gif_path = unified_artifacts.resolve_batched_debug_video_gif_output_path(
                            batched_save_video_gif_output_dir,
                            env_index=env_index,
                        )
                        unified_artifacts.write_debug_video_gif_from_video(source_video_path, target_gif_path)
                video_frames_per_env[env_index].clear()
                if finalized_video_env_indices is not None:
                    finalized_video_env_indices.add(int(env_index))
            if _batched_per_env_artifacts_enabled(args):
                unified_artifacts.finalize_batched_dense_episode_env_if_available(
                    env_index=env_index,
                    planner_backend_value=macro_backend,
                    macro_route=macro_route,
                    exit_code=0 if semantic_task_success else 1,
                    semantic_task_success=semantic_task_success,
                    failed_stage=failed_stage,
                )
        except Exception as exc:
            _emit_yellow_warning(
                f"Failed to finalize per-env artifacts for env_index={env_index}: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            trace_handle = object_pose_trace_handles[env_index]
            if trace_handle is not None:
                try:
                    trace_handle.close()
                except Exception as exc:
                    _emit_yellow_warning(
                        f"Failed to close per-env object pose trace for env_index={env_index}: {exc}"
                    )
                object_pose_trace_handles[env_index] = None
            if temp_video_path_for_gif is not None and Path(temp_video_path_for_gif).exists():
                try:
                    Path(temp_video_path_for_gif).unlink()
                except Exception as exc:
                    _emit_yellow_warning(
                        f"Failed to remove temporary per-env GIF source video: {temp_video_path_for_gif} ({exc})"
                    )
            terminal_artifacts_flushed_rows[env_index] = True

    def _mark_failed(env_index: int, failed_stage: str, message: str) -> None:
        terminal_rows[env_index] = True
        stage_rows[env_index] = "failed"
        per_env_feedback[env_index] = {
            "env_index": int(env_index),
            "semantic_task_success": False,
            "failed_stage": str(failed_stage),
        }
        print(
            f"{_Y}[WARNING] [PlannerDebug] Independent batched proxy env {env_index} FAILED "
            f"at stage='{failed_stage}': {message}{_R}"
        )

    def _mark_success(env_index: int) -> None:
        nonlocal first_success_step_limit
        terminal_rows[env_index] = True
        stage_rows[env_index] = "success"
        per_env_feedback[env_index] = {
            "env_index": int(env_index),
            "semantic_task_success": True,
            "failed_stage": None,
        }
        if first_success_step_limit is None:
            baseline_steps = max(int(total_step_rows[env_index]), 1)
            first_success_step_limit = max(
                int(np.ceil(float(batch_env_step_timeout_scale) * float(baseline_steps))),
                int(batch_env_step_timeout_min_steps),
            )
            print(
                f"[PlannerDebug] Independent batched proxy timeout budget activated from env {env_index}: "
                f"baseline_steps={baseline_steps} timeout_limit={first_success_step_limit}"
            )
        print(f"[PlannerDebug] Independent batched proxy env {env_index} SUCCESS")

    def _advance_position_stage(
        *,
        env_index: int,
        target_p_rows,
        position_mask,
        failure_stage: str,
        next_stage: str | None,
        stage_label: str,
        max_xy_step_value: float,
        max_z_step_value: float,
        tcp_p_rows,
    ):
        current_p = np.asarray(tcp_p_rows[env_index], dtype=np.float32).reshape(-1)[:3]
        target_p = np.asarray(target_p_rows[env_index], dtype=np.float32).reshape(-1)[:3]
        pos_err_vec = target_p - current_p
        position_mask_arr = np.asarray(position_mask, dtype=bool).reshape(3)
        pos_err = (
            float(np.linalg.norm(pos_err_vec[position_mask_arr]))
            if np.any(position_mask_arr)
            else 0.0
        )
        xy_err = float(np.linalg.norm(pos_err_vec[:2]))
        if pos_err <= pos_tol:
            print(
                f"[PlannerDebug] Independent batched proxy env {env_index} stage='{stage_label}' "
                f"converged: pos_err={pos_err:.4f} m"
            )
            if next_stage is None:
                evaluation = env_unwrapped.evaluate()
                success_rows = _normalize_per_env_scalar_list(
                    evaluation.get("success"),
                    num_envs=num_envs,
                    dtype=bool,
                )
                if bool(success_rows[env_index]):
                    _mark_success(env_index)
                else:
                    _mark_failed(
                        env_index,
                        "lift",
                        "final evaluation did not report semantic success after lift convergence",
                    )
            else:
                _transition(env_index, next_stage)
            return None

        next_step_idx = int(stage_step_rows[env_index]) + 1
        if next_step_idx > int(max_stage_steps):
            _mark_failed(
                env_index,
                failure_stage,
                f"stage '{stage_label}' reached max_stage_steps={max_stage_steps} with pos_err={pos_err:.4f} m",
            )
            return None

        total_error = float(pos_err)
        last_error = last_error_rows[env_index]
        if last_error is not None and total_error >= (float(last_error) - 1e-4):
            stall_count_rows[env_index] += 1
        else:
            stall_count_rows[env_index] = 0
        last_error_rows[env_index] = total_error
        if stall_count_rows[env_index] >= int(stall_limit):
            _mark_failed(
                env_index,
                failure_stage,
                f"stage '{stage_label}' stalled for {stall_count_rows[env_index]} iterations",
            )
            return None

        adaptive_xy_step = unified_lowlevel.get_adaptive_proxy_xy_step(
            planner_cfg,
            xy_err=xy_err,
            nominal_xy_step=max_xy_step_value,
        )
        if (
            np.any(position_mask_arr[:2])
            and adaptive_xy_step < max_xy_step_value
            and not adaptive_move_xy_logged_rows[env_index]
        ):
            adaptive_move_xy_logged_rows[env_index] = True
            print(
                f"[PlannerDebug] Independent batched proxy env {env_index} stage='{stage_label}' switched to adaptive XY step: "
                f"xy_err={xy_err:.4f} m threshold_xy={float(planner_cfg['planner_proxy_threshold_xy_m']):.4f} m "
                f"nominal_xy_step={max_xy_step_value:.4f} m adaptive_xy_step={adaptive_xy_step:.4f} m"
            )
        delta_pos = unified_lowlevel.build_proxy_delta_pos(
            env_unwrapped,
            pos_err_vec,
            position_mask=position_mask_arr,
            max_xy_step=adaptive_xy_step,
            max_z_step=max_z_step_value,
            proxy_frame_mode=str(planner_cfg.get("planner_proxy_frame", "base_camera_plane") or "base_camera_plane"),
        )
        if float(np.linalg.norm(np.asarray(delta_pos, dtype=np.float32).reshape(-1)[:3])) <= 1e-6:
            _mark_failed(
                env_index,
                failure_stage,
                f"stage '{stage_label}' produced a near-zero proxy step before convergence",
            )
            return None

        stage_step_rows[env_index] = next_step_idx
        return np.asarray(delta_pos, dtype=np.float32).reshape(-1)[:3]

    print(
        f"[PlannerDebug] Starting independent batched proxy pick macro: "
        f"num_envs={num_envs} manip_object_id={manip_id} close_steps={close_steps} "
        f"lift_delta_z={lift_delta_z:.4f}"
    )

    global_step_idx = 0
    while not all(terminal_rows):
        global_step_idx += 1
        tcp_p_rows, tcp_q_rows = unified_debug.pose_to_numpy_rows(
            unified_debug.get_debug_planner_ee_pose_rows(env_unwrapped)
        )
        actor_p_rows = np.asarray(
            unified_debug.get_debug_actor_position_rows(target_object),
            dtype=np.float32,
        ).reshape(num_envs, 3)
        if any(
            stage_name == "lift" and not lift_target_ready_rows[env_index]
            for env_index, stage_name in enumerate(stage_rows)
        ):
            lift_target_pose, _reference_pose, _task_pose, _runtime_pose = unified_debug.build_debug_lift_pose_from_policy(
                env_unwrapped,
                lift_delta_z=float(lift_delta_z),
            )
            lift_target_candidate_rows, _lift_target_q_rows = unified_debug.pose_to_numpy_rows(lift_target_pose)
            for env_index, stage_name in enumerate(stage_rows):
                if stage_name == "lift" and not lift_target_ready_rows[env_index]:
                    lift_target_snapshot_rows[env_index] = np.asarray(
                        lift_target_candidate_rows[env_index],
                        dtype=np.float32,
                    ).reshape(-1)[:3]
                    lift_target_ready_rows[env_index] = True
                    print(
                        f"[PlannerDebug] Independent batched proxy env {env_index} lift target snapshot "
                        f"p={np.array2string(lift_target_snapshot_rows[env_index], precision=4, suppress_small=True)}"
                    )
        delta_pos_rows = np.zeros((num_envs, 3), dtype=np.float32)
        # Match singleton semantics: keep the current hand target latched through
        # approach stages and only send explicit close commands once we are ready
        # to settle/close/lift. NaN leaves apply_proxy_ee_delta_action() on its
        # default per-step "hold" behavior for that env row.
        desired_gripper_signal_rows = np.full((num_envs,), np.nan, dtype=np.float32)

        for env_index in range(num_envs):
            if terminal_rows[env_index]:
                desired_gripper_signal_rows[env_index] = (
                    close_signal if gripper_closed_rows[env_index] else np.nan
                )
                continue

            stage_name = stage_rows[env_index]
            total_step_rows[env_index] += 1
            current_tcp_p = np.asarray(tcp_p_rows[env_index], dtype=np.float32).reshape(-1)[:3]
            current_actor_p = np.asarray(actor_p_rows[env_index], dtype=np.float32).reshape(-1)[:3]
            if (
                not np.all(np.isfinite(current_tcp_p))
                or not np.all(np.isfinite(current_actor_p))
                or float(np.max(np.abs(current_tcp_p))) > float(runaway_abs_limit_m)
                or float(np.max(np.abs(current_actor_p))) > float(runaway_abs_limit_m)
            ):
                _mark_failed(
                    env_index,
                    stage_name if stage_name not in {"success", "failed"} else "full_approach",
                    "detected runaway or non-finite tcp/object pose during batched execution",
                )
                continue
            if first_success_step_limit is not None and int(total_step_rows[env_index]) > int(first_success_step_limit):
                _mark_failed(
                    env_index,
                    stage_name if stage_name not in {"success", "failed"} else "full_approach",
                    f"exceeded per-env timeout budget after first success: "
                    f"steps={int(total_step_rows[env_index])} limit={int(first_success_step_limit)}",
                )
                continue
            if stage_name in {"close", "lift"} or gripper_closed_rows[env_index]:
                desired_gripper_signal_rows[env_index] = close_signal
            else:
                desired_gripper_signal_rows[env_index] = np.nan

            if stage_name.startswith("waypoint_"):
                waypoint_index = waypoint_stage_names.index(stage_name)
                next_stage = (
                    waypoint_stage_names[waypoint_index + 1]
                    if (waypoint_index + 1) < len(waypoint_stage_names)
                    else "rise"
                )
                delta = _advance_position_stage(
                    env_index=env_index,
                    target_p_rows=waypoint_target_p_rows_by_stage[stage_name],
                    position_mask=(True, True, True),
                    failure_stage="full_approach",
                    next_stage=next_stage,
                    stage_label=f"FullApproach:{stage_name}",
                    max_xy_step_value=max_xy_step,
                    max_z_step_value=max_z_step,
                    tcp_p_rows=tcp_p_rows,
                )
                if delta is not None:
                    delta_pos_rows[env_index] = delta
                continue

            if stage_name == "rise":
                delta = _advance_position_stage(
                    env_index=env_index,
                    target_p_rows=rise_target_p_rows,
                    position_mask=(False, False, True),
                    failure_stage="full_approach",
                    next_stage="move_xy",
                    stage_label="FullApproach:rise_to_safe_z",
                    max_xy_step_value=max_xy_step,
                    max_z_step_value=max_z_step,
                    tcp_p_rows=tcp_p_rows,
                )
                if delta is not None:
                    delta_pos_rows[env_index] = delta
                continue

            if stage_name == "move_xy":
                next_stage = "predescent_settle" if predescent_settle_steps > 0 else "descent"
                delta = _advance_position_stage(
                    env_index=env_index,
                    target_p_rows=move_xy_target_p_rows,
                    position_mask=(True, True, False),
                    failure_stage="full_approach",
                    next_stage=next_stage,
                    stage_label="FullApproach:move_xy_above_target",
                    max_xy_step_value=max_xy_step,
                    max_z_step_value=max_z_step,
                    tcp_p_rows=tcp_p_rows,
                )
                if delta is not None:
                    delta_pos_rows[env_index] = delta
                continue

            if stage_name == "predescent_settle":
                settle_remaining_rows[env_index] -= 1
                if settle_remaining_rows[env_index] <= 0:
                    _transition(env_index, "descent")
                continue

            if stage_name == "descent":
                dynamic_target_p = (
                    np.asarray(descend_target_p_rows[env_index], dtype=np.float32).reshape(-1)[:3]
                    + (
                        np.asarray(actor_p_rows[env_index], dtype=np.float32).reshape(-1)[:3]
                        - np.asarray(initial_actor_p_rows[env_index], dtype=np.float32).reshape(-1)[:3]
                    )
                )
                current_p = np.asarray(tcp_p_rows[env_index], dtype=np.float32).reshape(-1)[:3]
                pos_err_vec = dynamic_target_p - current_p
                xy_err = float(np.linalg.norm(pos_err_vec[:2]))
                z_err = float(abs(pos_err_vec[2]))
                if xy_err <= pos_tol and z_err <= pos_tol:
                    final_err = float(
                        np.linalg.norm(
                            current_p - np.asarray(descend_target_p_rows[env_index], dtype=np.float32).reshape(-1)[:3]
                        )
                    )
                    if final_err > 0.1:
                        _mark_failed(
                            env_index,
                            "full_approach",
                            f"final pos_err={final_err:.4f} m exceeds acceptance threshold 0.1000 m",
                        )
                        continue
                    print(
                        f"[PlannerDebug] Independent batched proxy env {env_index} stage='FullApproach:descent' "
                        f"converged: xy_err={xy_err:.4f} m z_err={z_err:.4f} m"
                    )
                    next_stage = "preclose_settle" if preclose_settle_steps > 0 else "close"
                    _transition(env_index, next_stage)
                    continue

                next_step_idx = int(stage_step_rows[env_index]) + 1
                if next_step_idx > int(max_stage_steps):
                    _mark_failed(
                        env_index,
                        "full_approach",
                        f"descent reached max_stage_steps={max_stage_steps} with xy_err={xy_err:.4f} m z_err={z_err:.4f} m",
                    )
                    continue

                total_error = float(xy_err + z_err)
                last_error = last_error_rows[env_index]
                if last_error is not None and total_error >= (float(last_error) - 1e-4):
                    stall_count_rows[env_index] += 1
                else:
                    stall_count_rows[env_index] = 0
                last_error_rows[env_index] = total_error
                if stall_count_rows[env_index] >= int(stall_limit):
                    _mark_failed(
                        env_index,
                        "full_approach",
                        f"descent stalled for {stall_count_rows[env_index]} iterations",
                    )
                    continue

                adaptive_xy_step = unified_lowlevel.get_adaptive_proxy_xy_step(
                    planner_cfg,
                    xy_err=xy_err,
                    nominal_xy_step=descend_xy_step,
                )
                adaptive_z_step = unified_lowlevel.get_adaptive_proxy_z_step_near_target(
                    planner_cfg,
                    tcp_z=float(current_p[2]),
                    descend_target_z=float(dynamic_target_p[2]),
                    nominal_z_step=max_z_step,
                )
                if adaptive_xy_step < descend_xy_step and not adaptive_descend_xy_logged_rows[env_index]:
                    adaptive_descend_xy_logged_rows[env_index] = True
                    print(
                        f"[PlannerDebug] Independent batched proxy env {env_index} stage='FullApproach:descent' switched to adaptive XY step: "
                        f"xy_err={xy_err:.4f} m threshold_xy={float(planner_cfg['planner_proxy_threshold_xy_m']):.4f} m "
                        f"nominal_xy_step={descend_xy_step:.4f} m adaptive_xy_step={adaptive_xy_step:.4f} m"
                    )
                if adaptive_z_step < max_z_step and not adaptive_descend_z_logged_rows[env_index]:
                    adaptive_descend_z_logged_rows[env_index] = True
                    print(
                        f"[PlannerDebug] Independent batched proxy env {env_index} stage='FullApproach:descent' switched to adaptive Z step: "
                        f"tcp_z={float(current_p[2]):.4f} descend_target_z={float(dynamic_target_p[2]):.4f} "
                        f"threshold_z={float(planner_cfg['planner_proxy_threshold_z_m']):.4f} m "
                        f"nominal_z_step={max_z_step:.4f} m adaptive_z_step={adaptive_z_step:.4f} m"
                    )
                delta = unified_lowlevel.build_proxy_delta_pos(
                    env_unwrapped,
                    pos_err_vec,
                    position_mask=(True, True, True),
                    max_xy_step=adaptive_xy_step,
                    max_z_step=adaptive_z_step,
                    proxy_frame_mode=str(planner_cfg.get("planner_proxy_frame", "base_camera_plane") or "base_camera_plane"),
                )
                if float(np.linalg.norm(np.asarray(delta, dtype=np.float32).reshape(-1)[:3])) <= 1e-6:
                    _mark_failed(
                        env_index,
                        "full_approach",
                        "descent produced a near-zero proxy step before convergence",
                    )
                    continue
                stage_step_rows[env_index] = next_step_idx
                delta_pos_rows[env_index] = np.asarray(delta, dtype=np.float32).reshape(-1)[:3]
                continue

            if stage_name == "preclose_settle":
                desired_gripper_signal_rows[env_index] = close_signal
                settle_remaining_rows[env_index] -= 1
                if settle_remaining_rows[env_index] <= 0:
                    _transition(env_index, "close")
                continue

            if stage_name == "close":
                desired_gripper_signal_rows[env_index] = close_signal
                close_remaining_rows[env_index] -= 1
                if close_remaining_rows[env_index] <= 0:
                    _transition(env_index, "lift")
                continue

            if stage_name == "lift":
                desired_gripper_signal_rows[env_index] = close_signal
                if not lift_target_ready_rows[env_index]:
                    _mark_failed(
                        env_index,
                        "lift",
                        "lift target snapshot is unavailable in independent batched proxy executor",
                    )
                    continue
                delta = _advance_position_stage(
                    env_index=env_index,
                    target_p_rows=lift_target_snapshot_rows,
                    position_mask=(False, False, True),
                    failure_stage="lift",
                    next_stage=None,
                    stage_label="Lift",
                    max_xy_step_value=max_xy_step,
                    max_z_step_value=max_z_step,
                    tcp_p_rows=tcp_p_rows,
                )
                if delta is not None:
                    delta_pos_rows[env_index] = delta
                continue

            _mark_failed(
                env_index,
                "full_approach",
                f"independent batched proxy executor reached unsupported stage='{stage_name}'",
            )

        raw_delta_rpy_rows = np.zeros((num_envs, 3), dtype=np.float32)
        contact_stage_label = (
            f"IndependentBatch:step{global_step_idx}:"
            f"stages={'|'.join(str(stage_name) for stage_name in stage_rows)}"
        )
        planner_contact_min_force_raw = planner_cfg.get("planner_proxy_warn_non_target_contacts_min_force", 1e-8)
        planner_contact_min_force = (
            1e-8
            if planner_contact_min_force_raw is None
            else float(planner_contact_min_force_raw)
        )
        robot_object_contact_records = unified_debug.collect_debug_robot_object_contacts(
            env_unwrapped,
            min_force=planner_contact_min_force,
            include_target=True,
        )
        object_object_contact_records = unified_debug.collect_debug_object_object_contacts(
            env_unwrapped,
            min_force=planner_contact_min_force,
        )
        unified_debug.log_debug_non_target_object_contacts(
            env_unwrapped,
            stage_label=contact_stage_label,
            min_force=planner_contact_min_force,
        )
        _maybe_log_traced_object_pose(
            global_step_idx=global_step_idx,
            stage_name="|".join(str(stage_name) for stage_name in stage_rows),
            tcp_p_rows=tcp_p_rows,
            tcp_q_rows=tcp_q_rows,
            raw_delta_pos_rows=delta_pos_rows,
            raw_delta_rpy_rows=raw_delta_rpy_rows,
            gripper_signal_rows=desired_gripper_signal_rows,
            robot_object_contact_records=robot_object_contact_records,
            object_object_contact_records=object_object_contact_records,
        )

        for env_index in range(num_envs):
            if terminal_rows[env_index] and not terminal_artifacts_flushed_rows[env_index]:
                _finalize_terminal_env_artifacts(env_index)

        unified_lowlevel.apply_proxy_ee_delta_action(
            env,
            raw_delta_pos=delta_pos_rows,
            raw_delta_rpy=raw_delta_rpy_rows,
            hold_steps=1,
            stage_label=f"IndependentBatch:step{global_step_idx}",
            gripper_target_state="hold",
            gripper_signal_override=desired_gripper_signal_rows,
            pose_to_numpy=unified_debug.pose_to_numpy,
        )

    successful_env_count = sum(1 for item in per_env_feedback if item["semantic_task_success"])
    failed_env_indices = [
        int(item["env_index"])
        for item in per_env_feedback
        if not item["semantic_task_success"]
    ]
    batch_semantic_task_success = successful_env_count == len(per_env_feedback)
    set_unified_macro_feedback(
        semantic_task_success=batch_semantic_task_success,
        failed_stage=None,
        per_env_feedback=per_env_feedback,
        batch_size=num_envs,
        successful_env_count=successful_env_count,
        failed_env_indices=failed_env_indices,
        artifacts_recorded_per_env=False,
    )
    if batch_semantic_task_success:
        print("[PlannerDebug] Independent batched proxy pick macro EXECUTE OK")
    else:
        print(
            f"{_Y}[WARNING] [PlannerDebug] Independent batched proxy pick macro completed with "
            f"{successful_env_count}/{len(per_env_feedback)} successful envs.{_R}"
        )
    for env_index, trace_handle in enumerate(object_pose_trace_handles):
        if trace_handle is None:
            continue
        try:
            trace_handle.close()
        except Exception as exc:
            _emit_yellow_warning(
                f"Failed to close residual per-env object pose trace for env_index={env_index}: {exc}"
            )
        object_pose_trace_handles[env_index] = None
    return batch_semantic_task_success


def _execute_pick_macro_direct(resolved_argv) -> int | None:
    unified_setup = _load_unified_proxy_setup()
    unified_artifacts = _load_unified_proxy_artifacts()
    unified_macro = _load_unified_proxy_macro()
    unified_control = _load_unified_proxy_control()
    unified_stage_pose = _load_unified_proxy_stage_pose()
    unified_stage_close = _load_unified_proxy_stage_close()
    unified_stage_lift = _load_unified_proxy_stage_lift()
    unified_debug = _load_unified_proxy_debug()
    unified_motion = _load_unified_proxy_motion()
    unified_lowlevel = _load_unified_proxy_lowlevel()
    unified_targets = _load_unified_proxy_targets()
    direct_stage_callbacks = _build_direct_stage_callbacks(
        unified_control=unified_control,
        unified_debug=unified_debug,
        unified_motion=unified_motion,
        unified_lowlevel=unified_lowlevel,
        unified_targets=unified_targets,
        unified_stage_pose=unified_stage_pose,
        unified_stage_close=unified_stage_close,
        unified_stage_lift=unified_stage_lift,
    )
    args = _build_direct_args_namespace(resolved_argv)
    if args is None:
        return None

    runtime_mode = "headless" if args.headless else "viewer"
    print(
        "[RC5UnifiedProxyRuntime] Using direct unified proxy executor for validated envelope "
        f"({runtime_mode}, sim_backend={args.sim_backend})."
    )
    unified_artifacts.clear_unified_dense_episode_capture()
    args.scene = unified_setup.resolve_scene_path(args.scene, args.config_path, args.key)
    if args.headless and args.step_by_step:
        _raise_unified_proxy_fail_fast("--step_by_step is supported only in viewer mode")
    if args.num_envs > 1 and not args.headless:
        _raise_unified_proxy_fail_fast("batched unified proxy execution currently supports only --headless")
    if args.num_envs > 1 and args.dense_episode_output:
        _raise_unified_proxy_fail_fast(
            "batched unified proxy execution does not support a single shared dense episode artifact"
        )
    if args.num_envs > 1 and args.rl4vla_raw_episode_output:
        _raise_unified_proxy_fail_fast(
            "batched unified proxy execution does not support a single shared RL4VLA raw episode artifact"
        )
    if args.num_envs > 1 and (
        args.save_video_on_exit
        or args.save_video_gif_on_exit
        or args.video_output is not None
        or args.save_video_path is not None
        or args.save_video_gif_path is not None
    ):
        _raise_unified_proxy_fail_fast(
            "batched unified proxy execution does not support shared debug-video artifacts"
        )
    config_overrides = unified_setup.load_runner_config(args)
    if args.task_object_id:
        config_overrides["manip_object_id"] = str(args.task_object_id)
        print(f"[Info] CLI override: manip_object_id={config_overrides['manip_object_id']}")
    unified_setup.apply_lighting_profile_overrides(args, config_overrides)
    unified_setup.apply_teleop_profile_overrides(args, config_overrides)
    unified_setup.validate_teleop_runtime_requirements(config_overrides)
    unified_setup.apply_hand_contact_profile_overrides(args, config_overrides)
    unified_setup.apply_hand_controller_profile_overrides(args, config_overrides)
    unified_setup.initialize_sapien_renderer(config_overrides.get("renderer_kwargs"))

    render_mode = "none" if args.headless else "human"
    env = unified_setup.make_env(args, config_overrides, render_mode=render_mode)
    try:
        if args.headless:
            _install_headless_render_guard(env)

        agent = env.unwrapped.agent
        _validate_direct_runtime_task_object_id(args, env.unwrapped)
        env.unwrapped._debug_planner_config = dict(config_overrides)
        unified_setup.log_hand_joint_state(agent, label="before_hand_pose_overrides")
        unified_setup.apply_hand_pose_overrides(agent, args, config_overrides)
        unified_setup.ensure_hand_defaults(agent)
        unified_setup.log_hand_joint_state(agent, label="after_hand_pose_overrides")
        shared_reset_planner_hand_target_to_open(
            env.unwrapped,
            restore_planner_grasp_state_fn=restore_planner_grasp_state,
            save_planner_grasp_state_fn=save_planner_grasp_state,
            source_stage="init",
        )
        unified_setup.log_hand_joint_state(agent, label="after_reset_planner_hand_target_to_open")
        env.unwrapped._proxy_ee_latched_gripper_target = None
        env.unwrapped._planner_last_task_pose = None
        env.unwrapped._planner_last_task_pose_stage = None
        env.unwrapped._planner_last_task_bbox_np = None
        print("[PlannerDebug] Initialized latched proxy hand target to OPEN.")

        if (
            config_overrides["auto_placement"]
            and config_overrides["robot_base_pose_z_auto"]
            and config_overrides["robot_base_pose"] is not None
            and len(config_overrides["robot_base_pose"]) >= 3
            and env.unwrapped.auto_table_z is not None
        ):
            old_z = config_overrides["robot_base_pose"][2]
            config_overrides["robot_base_pose"][2] = env.unwrapped.auto_table_z
            print(
                f"[Auto] robot_base_pose.z: {old_z:.4f} -> {config_overrides['robot_base_pose'][2]:.4f} "
                "(table surface from raycast)"
            )
            env.unwrapped.robot_base_pose = unified_setup.sapien.Pose(
                p=list(config_overrides["robot_base_pose"][:3]),
                q=(
                    list(config_overrides["robot_base_pose"][3:7])
                    if len(config_overrides["robot_base_pose"]) >= 7
                    else [1.0, 0.0, 0.0, 0.0]
                ),
            )

        viewer = None
        if not args.headless:
            viewer = env.render()
            _check_viewer(viewer)
            viewer.paused = False
            env.unwrapped._debug_planner_viewer = viewer
            env.unwrapped._debug_planner_step_by_step = bool(args.step_by_step)
            env.unwrapped._debug_planner_last_key_states = {" ": False, "n": False, "q": False, "v": False}
            env.unwrapped._debug_planner_viewer_step_gate_announced = False

        unified_setup.reset_and_prepare(
            env,
            args,
            config_overrides["control_mode"],
            gripper_hold_signal=config_overrides.get("gripper_open_signal"),
        )
        unified_setup.log_hand_joint_state(agent, label="after_reset_and_prepare")
        log_scene_object_layout = getattr(unified_debug, "log_debug_scene_object_layout", None)
        if (
            str(os.environ.get("RC5_DEBUG_LOG_SCENE_LAYOUT", "")).strip() == "1"
            and callable(log_scene_object_layout)
        ):
            log_scene_object_layout(
                env.unwrapped,
                stage_label="post_reset_and_prepare",
            )
        if args.headless:
            print("[Info] Running debug planner in headless mode (direct unified proxy executor).")
        else:
            print("[Info] Running debug planner in viewer mode (direct unified proxy executor).")

        video_frames = []
        video_frames_per_env = None
        finalized_video_env_indices = set()
        buffer_video_for_viewer = not args.headless
        batched_save_video_output_dir = getattr(args, "batched_save_video_output_dir", None)
        batched_save_video_gif_output_dir = getattr(args, "batched_save_video_gif_output_dir", None)
        buffer_video_for_exit_artifacts = (
            args.save_video_on_exit
            or args.save_video_gif_on_exit
            or batched_save_video_output_dir is not None
            or batched_save_video_gif_output_dir is not None
        )
        if buffer_video_for_viewer or buffer_video_for_exit_artifacts:
            effective_video_settings = unified_artifacts.resolve_effective_video_settings(args, config_overrides)
            effective_video_codec, _ffmpeg_exe, effective_video_codec_reason = (
                unified_artifacts.resolve_effective_video_codec(
                    effective_video_settings["format"], effective_video_settings["requested_codec"]
                )
            )
            print(f"[VIDEO] Codec selection: {effective_video_codec_reason}")
            env.unwrapped._planner_video_frames = video_frames
            env.unwrapped._planner_video_normalize_fn = unified_artifacts.normalize_video_frame
            env.unwrapped._planner_video_record_from = "base_camera"
            if args.num_envs > 1:
                video_frames_per_env = [[] for _ in range(int(args.num_envs))]
                env.unwrapped._planner_video_frames = None

                def append_video_buffer_frame():
                    frames = unified_artifacts.capture_base_camera_frames(env)
                    if len(frames) != int(args.num_envs):
                        raise RuntimeError(
                            "Batched base_camera capture returned unexpected frame count: "
                            f"expected={int(args.num_envs)} actual={len(frames)}"
                        )
                    for env_index, frame in enumerate(frames):
                        if int(env_index) in finalized_video_env_indices:
                            continue
                        video_frames_per_env[env_index].append(frame)

                def save_video_buffer_for_env_to_path(env_index, video_path):
                    resolved_path = unified_artifacts.resolve_output_video_path(
                        video_path,
                        video_format=effective_video_settings["format"],
                        video_codec=effective_video_codec,
                    )
                    resolved_path.parent.mkdir(parents=True, exist_ok=True)
                    ok = unified_artifacts.flush_video_buffer_to_file(
                        video_frames_per_env[int(env_index)],
                        resolved_path,
                        effective_video_settings["fps"],
                        effective_video_codec,
                        effective_video_settings["output_params"],
                    )
                    return resolved_path if ok else None

                def save_video_buffer_to_path(video_path):
                    return save_video_buffer_for_env_to_path(0, video_path)
            else:
                def append_video_buffer_frame():
                    frame = unified_artifacts.capture_base_camera_frame(env)
                    video_frames.append(frame)

                def save_video_buffer_to_path(video_path):
                    resolved_path = unified_artifacts.resolve_output_video_path(
                        video_path,
                        video_format=effective_video_settings["format"],
                        video_codec=effective_video_codec,
                    )
                    resolved_path.parent.mkdir(parents=True, exist_ok=True)
                    ok = unified_artifacts.flush_video_buffer_to_file(
                        video_frames,
                        resolved_path,
                        effective_video_settings["fps"],
                        effective_video_codec,
                        effective_video_settings["output_params"],
                    )
                    return resolved_path if ok else None

                def save_video_buffer_for_env_to_path(_env_index, video_path):
                    return save_video_buffer_to_path(video_path)

            env.unwrapped._debug_planner_append_video_buffer_frame = append_video_buffer_frame
            if not args.headless:
                def save_viewer_video_buffer():
                    output_path = _allocate_viewer_video_snapshot_path(
                        env.unwrapped,
                        args,
                        config_overrides,
                        video_format=effective_video_settings["format"],
                        video_codec=effective_video_codec,
                    )
                    try:
                        saved_path = save_video_buffer_to_path(output_path)
                    except Exception as exc:
                        print(
                            f"{_Y}[WARNING] [PlannerDebug] Viewer video save failed and was ignored: "
                            f"{type(exc).__name__}: {exc}{_R}"
                        )
                        return False
                    if not saved_path:
                        print(f"{_Y}[WARNING] [PlannerDebug] Viewer video buffer is empty; nothing was saved.{_R}")
                        return False
                    if Path(saved_path).suffix.lower() != ".gif":
                        gif_path = Path(saved_path).with_suffix(".gif")
                        try:
                            unified_artifacts.write_debug_video_gif_from_video(saved_path, gif_path)
                        except Exception as exc:
                            print(
                                f"{_Y}[WARNING] [PlannerDebug] Viewer GIF sidecar save failed and was ignored: "
                                f"{type(exc).__name__}: {exc}{_R}"
                            )
                    return True

                env.unwrapped._debug_planner_save_video_buffer = save_viewer_video_buffer
        else:
            env.unwrapped._debug_planner_append_video_buffer_frame = lambda: None
            env.unwrapped._debug_planner_save_video_buffer = lambda: False

            def save_video_buffer_to_path(_video_path):
                return None

            def save_video_buffer_for_env_to_path(_env_index, _video_path):
                return None

        if (
            args.dense_episode_output
            or args.dense_episode_output_dir
            or args.rl4vla_raw_episode_output
            or args.rl4vla_raw_episode_output_dir
        ):
            dense_instruction = (
                args.dense_episode_instruction
                or f"pick_up:{config_overrides.get('manip_object_id', 'object')}"
            )
            dense_instructions_per_env = None
            if args.episode_instruction_per_env_json:
                try:
                    parsed_instructions = json.loads(args.episode_instruction_per_env_json)
                except json.JSONDecodeError as exc:
                    _raise_unified_proxy_fail_fast(
                        "invalid --episode_instruction_per_env_json payload: "
                        f"{exc.msg} at pos {exc.pos}"
                    )
                if not isinstance(parsed_instructions, list):
                    _raise_unified_proxy_fail_fast("--episode_instruction_per_env_json must decode to a JSON list")
                dense_instructions_per_env = [str(item) for item in parsed_instructions]
            runtime_config_paths_per_env = None
            runtime_request_paths_per_env = None
            if args.embed_runtime_bundle_in_rl4vla_raw_npz and args.num_envs > 1:
                try:
                    runtime_config_paths_per_env = _parse_optional_json_string_list(
                        args.runtime_config_path_per_env_json,
                        label="--runtime_config_path_per_env_json",
                    )
                    runtime_request_paths_per_env = _parse_optional_json_string_list(
                        args.runtime_request_path_per_env_json,
                        label="--runtime_request_path_per_env_json",
                    )
                except ValueError as exc:
                    _raise_unified_proxy_fail_fast(str(exc))
            unified_artifacts.start_unified_dense_episode_capture(
                output_path=args.dense_episode_output,
                output_dir=args.dense_episode_output_dir,
                instruction=dense_instruction,
                instructions_per_env=dense_instructions_per_env,
                runtime_config_path=(
                    args.config_path if args.embed_runtime_bundle_in_rl4vla_raw_npz else None
                ),
                runtime_request_path=(
                    args.runtime_request_path if args.embed_runtime_bundle_in_rl4vla_raw_npz else None
                ),
                runtime_config_paths_per_env=runtime_config_paths_per_env,
                runtime_request_paths_per_env=runtime_request_paths_per_env,
                rl4vla_raw_output_path=args.rl4vla_raw_episode_output,
                rl4vla_raw_output_dir=args.rl4vla_raw_episode_output_dir,
                camera_name="base_camera",
                image_target_width=args.dense_episode_target_width,
                image_target_height=args.dense_episode_target_height,
                num_envs=args.num_envs,
                shared_video_frame_targets_per_env=(
                    None
                    if not (buffer_video_for_viewer or buffer_video_for_exit_artifacts)
                    else (
                        video_frames_per_env
                        if args.num_envs > 1
                        else [video_frames]
                    )
                ),
            )
            unified_artifacts.append_dense_episode_initial_frame_if_enabled(env)
            if buffer_video_for_viewer or buffer_video_for_exit_artifacts:
                active_capture = unified_artifacts.get_active_unified_dense_episode_capture()
                if active_capture is not None and active_capture.shared_video_frame_targets_per_env is not None:
                    env.unwrapped._debug_planner_append_video_buffer_frame = lambda: None
            dense_target = None
            if args.dense_episode_output:
                dense_target = Path(args.dense_episode_output).expanduser().resolve()
            elif args.dense_episode_output_dir:
                dense_target = Path(args.dense_episode_output_dir).expanduser().resolve()
            rl4vla_raw_target = None
            if args.rl4vla_raw_episode_output:
                rl4vla_raw_target = Path(args.rl4vla_raw_episode_output).expanduser().resolve()
            elif args.rl4vla_raw_episode_output_dir:
                rl4vla_raw_target = Path(args.rl4vla_raw_episode_output_dir).expanduser().resolve()
            if dense_target is not None:
                print(f"[DenseEpisode] Recording enabled -> {dense_target}")
            if rl4vla_raw_target is not None:
                print(f"[RL4VLAEpisode] Recording enabled -> {rl4vla_raw_target}")
            if getattr(args, "batched_object_pose_trace_output_dir", None):
                trace_target = Path(args.batched_object_pose_trace_output_dir).expanduser().resolve()
                print(f"[ObjectPoseTrace] Recording enabled -> {trace_target}")

        append_video_buffer_frame = getattr(env.unwrapped, "_debug_planner_append_video_buffer_frame", None)
        if callable(append_video_buffer_frame):
            unified_setup.log_hand_joint_state(agent, label="before_initial_video_frame_capture")
            append_video_buffer_frame()

        if not args.headless:
            if args.step_by_step:
                print("[PlannerDebug] Viewer step-by-step mode enabled. Press SPACE or 'n' to advance one proxy step.")
            print("[PlannerDebug] Viewer hotkeys: 'v' save buffered video, 'q' exit after the episode.")

        if args.num_envs > 1:
            print("[PlannerDebug] Auto-running independent batched proxy pick macro from unified direct executor.")
            unified_setup.log_hand_joint_state(agent, label="before_auto_pick_macro_dispatch")
            _execute_independent_batched_proxy_pick_macro(
                env,
                args=args,
                config_overrides=config_overrides,
                unified_artifacts=unified_artifacts,
                unified_control=unified_control,
                unified_debug=unified_debug,
                unified_lowlevel=unified_lowlevel,
                build_object_descend_target=direct_stage_callbacks["build_object_descend_target"],
                save_video_buffer_for_env_to_path=save_video_buffer_for_env_to_path,
                video_frames_per_env=video_frames_per_env,
                finalized_video_env_indices=finalized_video_env_indices,
            )
        elif args.auto_pick_macro == "1":
            print("[PlannerDebug] Auto-running proxy pick macro from unified direct executor.")
            unified_setup.log_hand_joint_state(agent, label="before_auto_pick_macro_dispatch")
            unified_macro.run_proxy_pick_macro(
                env,
                config_overrides,
                run_proxy_pregrasp_probe=direct_stage_callbacks.get(
                    "run_proxy_pregrasp_probe",
                    direct_stage_callbacks["run_planner_object_pregrasp_probe"],
                ),
                run_proxy_full_approach_to_descend=direct_stage_callbacks.get(
                    "run_proxy_full_approach_to_descend",
                    direct_stage_callbacks["run_planner_full_approach_to_descend"],
                ),
                run_proxy_descend=direct_stage_callbacks.get(
                    "run_proxy_descend",
                    direct_stage_callbacks["run_planner_object_descend"],
                ),
                run_proxy_close_gripper=direct_stage_callbacks.get(
                    "run_proxy_close_gripper",
                    direct_stage_callbacks["run_planner_close_gripper"],
                ),
                run_proxy_lift=direct_stage_callbacks.get(
                    "run_proxy_lift",
                    direct_stage_callbacks["run_planner_lift"],
                ),
                set_unified_macro_feedback=set_unified_macro_feedback,
            )
        elif args.auto_pick_macro == "2":
            print("[PlannerDebug] Auto-running full real planner / mplib planner macro from unified direct executor.")
            unified_setup.log_hand_joint_state(agent, label="before_auto_pick_macro_dispatch")
            unified_macro.run_planner_pick_macro_full(
                env,
                config_overrides,
                run_planner_object_pregrasp_probe=direct_stage_callbacks["run_planner_object_pregrasp_probe"],
                run_planner_full_approach_to_descend=direct_stage_callbacks["run_planner_full_approach_to_descend"],
                run_planner_object_descend=direct_stage_callbacks["run_planner_object_descend"],
                run_planner_close_gripper=direct_stage_callbacks["run_planner_close_gripper"],
                run_planner_lift=direct_stage_callbacks["run_planner_lift"],
                set_unified_macro_feedback=set_unified_macro_feedback,
            )
        else:
            raise ValueError(f"Unsupported --auto_pick_macro value for direct executor: {args.auto_pick_macro}")

        env.unwrapped._debug_planner_episode_outcome_token = _resolve_episode_outcome_token(
            peek_unified_macro_feedback()
        )
        finalized_feedback = _finalize_macro_feedback_for_runtime(
            env,
            peek_unified_macro_feedback(),
        )
        if finalized_feedback is None:
            _raise_unified_proxy_fail_fast("structured macro feedback is missing after unified proxy execution")
        if _batched_per_env_artifacts_enabled(args):
            finalized_feedback["artifacts_recorded_per_env"] = True
        set_unified_macro_feedback(
            semantic_task_success=finalized_feedback["semantic_task_success"],
            failed_stage=finalized_feedback.get("failed_stage"),
            per_env_feedback=finalized_feedback.get("per_env_feedback"),
            batch_size=finalized_feedback.get("batch_size"),
            successful_env_count=finalized_feedback.get("successful_env_count"),
            failed_env_indices=finalized_feedback.get("failed_env_indices"),
            artifacts_recorded_per_env=finalized_feedback.get("artifacts_recorded_per_env"),
        )
        env.unwrapped._debug_planner_episode_outcome_token = _resolve_episode_outcome_token(
            peek_unified_macro_feedback()
        )

        _maybe_save_debug_side_artifacts(args, video_frames, save_video_buffer_to_path)
        if args.num_envs > 1:
            _maybe_save_batched_debug_side_artifacts(
                args,
                video_frames_per_env=video_frames_per_env or [],
                save_video_buffer_for_env_to_path=save_video_buffer_for_env_to_path,
                finalized_video_env_indices=finalized_video_env_indices,
            )
        if args.headless:
            print("[PlannerDebug] Proxy planner macro finished in direct headless mode.")
        else:
            _keep_viewer_open_for_inspection(env, viewer, unified_lowlevel=unified_lowlevel)
        return 0
    finally:
        viewer = getattr(env.unwrapped, "_debug_planner_viewer", None)
        if viewer is not None and not getattr(viewer, "closed", True):
            try:
                viewer.close()
            except Exception:
                pass
        env.close()


def _resolve_unified_backend_request_argv(unified_request, planner_backend_value: str):
    if unified_request is None:
        return resolve_unified_backend_request_argv(None, planner_backend_value)

    task_plan = getattr(unified_request, "task_plan", None)
    intent = getattr(task_plan, "intent", None)
    stages = list(getattr(task_plan, "stages", []) or [])
    task_type = getattr(intent, "task_type", None)
    object_id = getattr(intent, "object_id", None)
    prompt = getattr(intent, "prompt", None)
    bootstrap = getattr(unified_request, "bootstrap", None)
    key = getattr(bootstrap, "key", None)
    robot_uids = getattr(bootstrap, "robot_uids", None)
    trace_seed = getattr(unified_request, "trace_seed", None)
    stage_names = ",".join(str(getattr(stage, "name", "<unnamed>")) for stage in stages) or "<none>"
    stage_kinds = ",".join(str(getattr(stage, "kind", "<unknown>")) for stage in stages) or "<none>"
    trace_stage_kinds = ",".join(str(kind) for kind in getattr(trace_seed, "stage_kinds", ()) or ()) or "<none>"
    trace_object_id = getattr(trace_seed, "object_id", None)
    trace_key = getattr(trace_seed, "config_key", None)

    if task_type is not None and task_type != "pick_up":
        raise ValueError(
            f"RC5 unified proxy runtime currently supports only task_type='pick_up', got '{task_type}'"
        )

    print(
        f"[RC5UnifiedProxyRuntime] backend={planner_backend_value} "
        f"task_type={task_type or '<none>'} "
        f"object_id={object_id or '<auto>'} "
        f"stage_names={stage_names} "
        f"stage_kinds={stage_kinds} "
        f"trace_object_id={trace_object_id or '<auto>'} "
        f"trace_key={trace_key or '<none>'} "
        f"trace_stage_kinds={trace_stage_kinds} "
        f"key={key or '<none>'} "
        f"robot_uids={robot_uids or '<none>'} "
        f"prompt={'yes' if prompt else 'no'}"
    )

    return resolve_unified_backend_request_argv(unified_request, planner_backend_value)


def run_unified_proxy_backend(argv=None):
    resolved_argv = inject_unified_planner_backend_arg(argv, "proxy_ee_delta")
    reason = _explain_direct_executor_skip(resolved_argv)
    _raise_unified_proxy_fail_fast(
        "argv-based unified proxy backend entrypoint is no longer allowed "
        f"because {reason}"
    )


def run_unified_hybrid_backend(argv=None):
    resolved_argv = inject_unified_planner_backend_arg(argv, "proxy_then_planner")
    reason = _explain_direct_executor_skip(resolved_argv)
    _raise_unified_proxy_fail_fast(
        "argv-based unified hybrid backend entrypoint is no longer allowed "
        f"because {reason}"
    )


def _write_dense_episode_artifact_if_available(
    *,
    planner_backend_value,
    macro_route,
    exit_code,
    macro_feedback,
):
    unified_artifacts = _load_unified_proxy_artifacts()
    return unified_artifacts.write_dense_episode_artifact_if_available(
        planner_backend_value=planner_backend_value,
        macro_route=macro_route,
        exit_code=exit_code,
        macro_feedback=macro_feedback,
    )


def _build_runtime_events(unified_request, *, planner_backend_value: str, macro_route: str):
    task_plan = getattr(unified_request, "task_plan", None)
    stages = list(getattr(task_plan, "stages", []) or [])
    task_type = getattr(getattr(task_plan, "intent", None), "task_type", None)
    stage_names = [str(getattr(stage, "name", "<unnamed>")) for stage in stages]
    stage_kinds = [str(getattr(stage, "kind", "<unknown>")) for stage in stages]
    return [
        {
            "event_type": "backend_request_received",
            "payload": {
                "macro_backend": planner_backend_value,
                "planner_backend": planner_backend_value,
                "macro_route": macro_route,
                "task_type": task_type,
                "stage_count": len(stages),
            },
        },
        {
            "event_type": "macro_route_selected",
            "payload": {
                "macro_backend": planner_backend_value,
                "planner_backend": planner_backend_value,
                "macro_route": macro_route,
            },
        },
        {
            "event_type": "stage_sequence_received",
            "payload": {
                "stage_names": stage_names,
                "stage_kinds": stage_kinds,
                "stage_count": len(stages),
            },
        },
        *[
            {
                "event_type": "stage_submitted_to_macro",
                "payload": {
                    "order_index": idx,
                    "stage_name": stage_names[idx],
                    "stage_kind": stage_kinds[idx],
                    "macro_backend": planner_backend_value,
                    "planner_backend": planner_backend_value,
                    "macro_route": macro_route,
                },
            }
            for idx in range(len(stages))
        ],
        {
            "event_type": "macro_started",
            "payload": {
                "macro_backend": planner_backend_value,
                "planner_backend": planner_backend_value,
                "macro_route": macro_route,
            },
        },
    ]


def _run_unified_backend_request_with_feedback(
    unified_request,
    planner_backend_value: str,
    macro_route: str,
):
    resolved_argv = _resolve_unified_backend_request_argv(unified_request, planner_backend_value)
    runtime_events = _build_runtime_events(
        unified_request,
        planner_backend_value=planner_backend_value,
        macro_route=macro_route,
    )
    clear_unified_macro_feedback()
    exit_code = _execute_pick_macro_direct(resolved_argv)
    executor_path = "direct_unified"
    skip_reason = None
    if exit_code is None:
        skip_reason = _explain_direct_executor_skip(resolved_argv)
        _raise_unified_proxy_fail_fast(
            "direct unified proxy executor rejected the request "
            f"because {skip_reason}"
        )
    runtime_events.append(
        {
            "event_type": "executor_path_selected",
            "payload": {
                "executor_path": executor_path,
                "compatibility_reason": None,
                "skip_reason": skip_reason,
            },
        }
    )
    macro_feedback = consume_unified_macro_feedback()
    normalized_exit_code = 0 if exit_code is None else int(exit_code)
    macro_finished_payload = {
        "macro_backend": planner_backend_value,
        "planner_backend": planner_backend_value,
        "macro_route": macro_route,
        "exit_code": normalized_exit_code,
        "execution_outcome": "success" if normalized_exit_code == 0 else "failed",
    }
    if macro_feedback is None:
        _raise_unified_proxy_fail_fast(
            "structured macro feedback is missing after unified proxy execution"
        )
    else:
        per_env_feedback = list(macro_feedback.get("per_env_feedback", []) or [])
        if per_env_feedback:
            runtime_events.append(
                {
                    "event_type": "batch_runtime_feedback",
                    "payload": {
                        "batch_size": int(macro_feedback.get("batch_size", len(per_env_feedback))),
                        "successful_env_count": int(
                            macro_feedback.get(
                                "successful_env_count",
                                sum(1 for item in per_env_feedback if item.get("semantic_task_success")),
                            )
                        ),
                        "failed_env_indices": [
                            int(item)
                            for item in (macro_feedback.get("failed_env_indices") or [])
                        ],
                        "artifacts_recorded_per_env": bool(
                            macro_feedback.get("artifacts_recorded_per_env", False)
                        ),
                        "per_env_feedback": per_env_feedback,
                    },
                }
            )
        semantic_task_success = bool(macro_feedback["semantic_task_success"])
        macro_finished_payload["semantic_task_success"] = semantic_task_success
        macro_finished_payload["failed_stage"] = macro_feedback.get("failed_stage")
        if "batch_size" in macro_feedback:
            macro_finished_payload["batch_size"] = int(macro_feedback["batch_size"])
        if "successful_env_count" in macro_feedback:
            macro_finished_payload["successful_env_count"] = int(macro_feedback["successful_env_count"])
        if "failed_env_indices" in macro_feedback:
            macro_finished_payload["failed_env_indices"] = [
                int(item) for item in (macro_feedback.get("failed_env_indices") or [])
            ]
        if not semantic_task_success:
            macro_finished_payload["execution_outcome"] = "failed"
    runtime_events.append(
        {
            "event_type": "macro_finished",
            "payload": macro_finished_payload,
        }
    )
    dense_episode_artifact_path = _write_dense_episode_artifact_if_available(
        planner_backend_value=planner_backend_value,
        macro_route=macro_route,
        exit_code=normalized_exit_code,
        macro_feedback=macro_feedback,
    )
    if dense_episode_artifact_path is not None:
        runtime_events.append(
            {
                "event_type": "dense_episode_artifact_written",
                "payload": {
                    "artifact_path": str(dense_episode_artifact_path),
                },
            }
        )
    return {
        "exit_code": normalized_exit_code,
        "runtime_events": runtime_events,
    }


def run_unified_proxy_backend_request(unified_request: Any):
    return _run_unified_backend_request_with_feedback(
        unified_request,
        "proxy_ee_delta",
        macro_route="pick_macro_1",
    )


def run_unified_hybrid_backend_request(unified_request: Any):
    return _run_unified_backend_request_with_feedback(
        unified_request,
        "proxy_then_planner",
        macro_route="pick_macro_1",
    )
