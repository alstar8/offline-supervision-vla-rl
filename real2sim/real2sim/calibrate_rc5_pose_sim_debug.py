from pathlib import Path

import numpy as np
import torch
import tyro
from dataclasses import dataclass

import gymnasium as gym

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.robots.rc5.rc5 import RC5AeroHandRight
from real2sim.calibrate_rc5_pose import (
    Args as BaseArgs,
    DEFAULT_LOAD_STATE_FILE,
    _apply_sim_arm_target_qpos_direct,
    _load_state_file,
    _print_help,
    _reset_current_state,
    _remap_sim_delta,
    _save_snapshot_from_obs,
    _teleop_mode,
    _base_jog_mode,
    _joint_jog_mode,
)
from real2sim.openreal2sim_validation import (
    observation_camera_lookat_state,
    observation_camera_override,
    pose_from_eye_target_roll,
)


TARGET_SIM_CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
JOINT_SIM_CONTROL_MODE = "arm_pd_joint_pos_gripper_pd_joint_pos"


@dataclass
class Args(BaseArgs):
    image_prefix: str = "openreal2sim_rc5_pose_sim_debug"
    load_state_file: str = DEFAULT_LOAD_STATE_FILE
    control_mode_name: str = TARGET_SIM_CONTROL_MODE
    arm_stiffness_scale: float = 1.0
    arm_damping_scale: float = 1.0
    arm_force_limit_scale: float = 1.0
    arm_friction_override: float | None = None
    arm_stiffness_override: str = ""
    arm_damping_override: str = ""
    arm_force_limit_override: str = ""
    auto_sequence_steps: int = 3
    auto_sequence_settle_steps: int = 0
    auto_sequence_log_name: str = "openreal2sim_rc5_pose_sim_debug_sequence.log"
    joint_test_index: int = 0
    joint_test_delta_rad: float = 0.2


def _tensor_to_numpy(data) -> np.ndarray:
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _format_array(values: np.ndarray, *, precision: int = 3) -> str:
    return np.array2string(
        np.asarray(values),
        precision=precision,
        separator=", ",
        suppress_small=False,
    )


def _pose_raw_to_xyzquat(raw_pose) -> np.ndarray:
    return np.asarray(raw_pose, dtype=np.float64).reshape(-1)


def _sim_arm_controller(env: BaseEnv):
    controller = env.unwrapped.agent.controller
    if hasattr(controller, "controllers"):
        return controller.controllers.get("arm")
    return controller


def _parse_joint_override(raw: str) -> list[float] | None:
    if not raw.strip():
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 6:
        raise ValueError(
            f"Expected 6 comma-separated values for a joint override, got {raw!r}"
        )
    return [float(part) for part in parts]


def _sim_qpos(env: BaseEnv, obs: dict | None = None) -> np.ndarray:
    if obs is not None:
        return _tensor_to_numpy(obs["agent"]["qpos"])[0].astype(np.float64)
    return _tensor_to_numpy(env.unwrapped.agent.robot.get_qpos())[0].astype(np.float64)


def _sim_tcp_pose_raw(env: BaseEnv) -> np.ndarray:
    return _tensor_to_numpy(env.unwrapped.agent.ee_pose_at_robot_base.raw_pose)[0].astype(np.float64)


def _step_action_with_debug(
    env: BaseEnv,
    delta_pos: list[float],
    delta_rot: list[float],
    gripper_cmd: float,
    *,
    apply_ik_qpos_direct: bool,
) -> tuple[dict, dict]:
    mapped_delta_pos, mapped_delta_rot = _remap_sim_delta(delta_pos, delta_rot)
    action = torch.tensor(
        [[
            float(mapped_delta_pos[0]),
            float(mapped_delta_pos[1]),
            float(mapped_delta_pos[2]),
            float(mapped_delta_rot[0]),
            float(mapped_delta_rot[1]),
            float(mapped_delta_rot[2]),
            float(gripper_cmd),
        ]],
        dtype=torch.float32,
        device=env.unwrapped.device,
    )
    obs, _, terminated, truncated, info = env.step(action)
    if apply_ik_qpos_direct:
        arm_controller = _sim_arm_controller(env)
        target_qpos = None if arm_controller is None else getattr(arm_controller, "_target_qpos", None)
        obs, info = _apply_sim_arm_target_qpos_direct(env, target_qpos)
    if bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item()):
        raise RuntimeError("Simulation episode ended during scripted debug sequence.")
    return obs, info


def _step_zero_action(
    env: BaseEnv,
    gripper_cmd: float,
    *,
    apply_ik_qpos_direct: bool,
) -> tuple[dict, dict]:
    action = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(gripper_cmd)]],
        dtype=torch.float32,
        device=env.unwrapped.device,
    )
    obs, _, terminated, truncated, info = env.step(action)
    if apply_ik_qpos_direct:
        arm_controller = _sim_arm_controller(env)
        target_qpos = None if arm_controller is None else getattr(arm_controller, "_target_qpos", None)
        obs, info = _apply_sim_arm_target_qpos_direct(env, target_qpos)
    if bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item()):
        raise RuntimeError("Simulation episode ended during settle steps.")
    return obs, info


def _step_joint_target_action(
    env: BaseEnv,
    target_qpos: np.ndarray,
    gripper_cmd: float,
) -> tuple[dict, dict]:
    action = torch.tensor(
        [[
            float(target_qpos[0]),
            float(target_qpos[1]),
            float(target_qpos[2]),
            float(target_qpos[3]),
            float(target_qpos[4]),
            float(target_qpos[5]),
            float(gripper_cmd),
        ]],
        dtype=torch.float32,
        device=env.unwrapped.device,
    )
    obs, _, terminated, truncated, info = env.step(action)
    if bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item()):
        raise RuntimeError("Simulation episode ended during joint settle steps.")
    return obs, info


def _run_auto_sequence(
    env: BaseEnv,
    args: Args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    save_idx: int,
) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / args.auto_sequence_log_name
    step_size = 0.02
    gripper_cmd = -1.0
    sequence = [
        ("down", np.array([0.0, 0.0, -step_size], dtype=np.float32), np.zeros(3, dtype=np.float32)),
        ("left", np.array([-step_size, 0.0, 0.0], dtype=np.float32), np.zeros(3, dtype=np.float32)),
        ("right", np.array([step_size, 0.0, 0.0], dtype=np.float32), np.zeros(3, dtype=np.float32)),
    ]

    lines = [
        "# calibrate_rc5_pose_sim_debug auto-sequence log",
        f"control_mode={args.control_mode_name}",
        f"sim_backend={args.sim_backend}",
        f"sim_apply_ik_qpos_direct={args.sim_apply_ik_qpos_direct}",
        f"arm_stiffness={RC5AeroHandRight.arm_stiffness}",
        f"arm_damping={RC5AeroHandRight.arm_damping}",
        f"arm_force_limit={RC5AeroHandRight.arm_force_limit}",
        f"arm_friction={RC5AeroHandRight.arm_friction}",
        f"robot_base_pose_p={robot_base_pose_p.tolist()}",
        f"robot_base_pose_q={robot_base_pose_q.tolist()}",
        "",
    ]

    info = env.unwrapped.get_info()
    obs = env.unwrapped.get_obs(info)

    command_index = 0
    for label, delta_pos, delta_rot in sequence:
        for repeat_idx in range(args.auto_sequence_steps):
            arm_controller = _sim_arm_controller(env)
            prev_pose_raw = None
            expected_target_pose_raw = None
            qpos_before = _sim_qpos(env, obs)
            tcp_before = _sim_tcp_pose_raw(env)
            mapped_delta_pos, mapped_delta_rot = _remap_sim_delta(delta_pos.tolist(), delta_rot.tolist())
            action_arm = torch.tensor(
                [[
                    float(mapped_delta_pos[0]),
                    float(mapped_delta_pos[1]),
                    float(mapped_delta_pos[2]),
                    float(mapped_delta_rot[0]),
                    float(mapped_delta_rot[1]),
                    float(mapped_delta_rot[2]),
                ]],
                dtype=torch.float32,
                device=env.unwrapped.device,
            )
            if arm_controller is not None:
                prev_pose = arm_controller._target_pose if getattr(arm_controller.config, "use_target", False) else arm_controller.ee_pose_at_base
                prev_pose_raw = _pose_raw_to_xyzquat(prev_pose.raw_pose[0].detach().cpu().numpy())
                expected_target_pose = arm_controller.compute_target_pose(prev_pose, action_arm)
                expected_target_pose_raw = _pose_raw_to_xyzquat(
                    expected_target_pose.raw_pose[0].detach().cpu().numpy()
                )

            obs, info = _step_action_with_debug(
                env,
                delta_pos.tolist(),
                delta_rot.tolist(),
                gripper_cmd,
                apply_ik_qpos_direct=args.sim_apply_ik_qpos_direct,
            )

            qpos_after = _sim_qpos(env, obs)
            tcp_after = _sim_tcp_pose_raw(env)
            target_qpos = None
            recorded_target_pose_raw = None
            if arm_controller is not None:
                if getattr(arm_controller, "_target_qpos", None) is not None:
                    target_qpos = _tensor_to_numpy(arm_controller._target_qpos)[0].astype(np.float64)
                if getattr(arm_controller, "_target_pose", None) is not None:
                    recorded_target_pose_raw = _pose_raw_to_xyzquat(
                        arm_controller._target_pose.raw_pose[0].detach().cpu().numpy()
                    )

            settled_obs = obs
            settled_info = info
            for _ in range(args.auto_sequence_settle_steps):
                settled_obs, settled_info = _step_zero_action(
                    env,
                    gripper_cmd,
                    apply_ik_qpos_direct=args.sim_apply_ik_qpos_direct,
                )
            settled_qpos = _sim_qpos(env, settled_obs)
            settled_tcp = _sim_tcp_pose_raw(env)

            tcp_target_error = None
            if expected_target_pose_raw is not None:
                tcp_target_error = expected_target_pose_raw[:3] - tcp_after[:3]
            qpos_tracking_error = None
            if target_qpos is not None:
                qpos_tracking_error = target_qpos - qpos_after[:6]
            settled_tcp_target_error = None
            if expected_target_pose_raw is not None:
                settled_tcp_target_error = expected_target_pose_raw[:3] - settled_tcp[:3]
            settled_qpos_tracking_error = None
            if target_qpos is not None:
                settled_qpos_tracking_error = target_qpos - settled_qpos[:6]

            lines.extend(
                [
                    f"=== command {command_index:04d} {label}_{repeat_idx + 1} ===",
                    f"mapped_delta_pos={_format_array(mapped_delta_pos)}",
                    f"tcp_before_xyz={_format_array(tcp_before[:3])}",
                    f"tcp_target_xyz={_format_array(expected_target_pose_raw[:3]) if expected_target_pose_raw is not None else 'None'}",
                    f"tcp_after_xyz={_format_array(tcp_after[:3])}",
                    f"tcp_target_error_xyz={_format_array(tcp_target_error) if tcp_target_error is not None else 'None'}",
                    f"settled_tcp_xyz={_format_array(settled_tcp[:3])}",
                    f"settled_tcp_target_error_xyz={_format_array(settled_tcp_target_error) if settled_tcp_target_error is not None else 'None'}",
                    f"target_qpos={_format_array(target_qpos) if target_qpos is not None else 'None'}",
                    f"actual_qpos={_format_array(qpos_after[:6])}",
                    f"qpos_tracking_error={_format_array(qpos_tracking_error) if qpos_tracking_error is not None else 'None'}",
                    f"settled_qpos={_format_array(settled_qpos[:6])}",
                    f"settled_qpos_tracking_error={_format_array(settled_qpos_tracking_error) if settled_qpos_tracking_error is not None else 'None'}",
                    "",
                ]
            )
            save_idx += 1
            _save_snapshot_from_obs(settled_obs, settled_info, args, robot_base_pose_p, robot_base_pose_q, save_idx)
            command_index += 1

    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved auto-sequence log to {log_path}")
    return save_idx


def _run_joint_auto_sequence(
    env: BaseEnv,
    args: Args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    save_idx: int,
) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / args.auto_sequence_log_name
    joint_index = int(args.joint_test_index)
    if joint_index < 0 or joint_index > 5:
        raise ValueError("--joint-test-index must be in [0, 5]")
    joint_delta = float(args.joint_test_delta_rad)
    gripper_cmd = -1.0
    sequence = [("pos", joint_delta), ("neg", -joint_delta)]

    info = env.unwrapped.get_info()
    obs = env.unwrapped.get_obs(info)
    current_target_qpos = _sim_qpos(env, obs)[:6].copy()

    lines = [
        "# calibrate_rc5_pose_sim_debug joint auto-sequence log",
        f"control_mode={args.control_mode_name}",
        f"sim_backend={args.sim_backend}",
        f"joint_test_index={joint_index}",
        f"joint_test_delta_rad={joint_delta}",
        f"joint_test_start_qpos={_format_array(current_target_qpos)}",
        "",
    ]

    command_index = 0
    for label, signed_delta in sequence:
        for repeat_idx in range(args.auto_sequence_steps):
            qpos_before = _sim_qpos(env, obs)
            tcp_before = _sim_tcp_pose_raw(env)
            current_target_qpos[joint_index] += signed_delta
            action = torch.tensor(
                [[
                    float(current_target_qpos[0]),
                    float(current_target_qpos[1]),
                    float(current_target_qpos[2]),
                    float(current_target_qpos[3]),
                    float(current_target_qpos[4]),
                    float(current_target_qpos[5]),
                    float(gripper_cmd),
                ]],
                dtype=torch.float32,
                device=env.unwrapped.device,
            )
            obs, _, terminated, truncated, info = env.step(action)
            if bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item()):
                raise RuntimeError("Simulation episode ended during joint auto-sequence.")

            qpos_after = _sim_qpos(env, obs)
            tcp_after = _sim_tcp_pose_raw(env)
            settled_obs = obs
            settled_info = info
            for _ in range(args.auto_sequence_settle_steps):
                settled_obs, settled_info = _step_joint_target_action(
                    env,
                    current_target_qpos,
                    gripper_cmd,
                )
            settled_qpos = _sim_qpos(env, settled_obs)
            settled_tcp = _sim_tcp_pose_raw(env)

            lines.extend(
                [
                    f"=== command {command_index:04d} joint{joint_index}_{label}_{repeat_idx + 1} ===",
                    f"target_qpos={_format_array(current_target_qpos)}",
                    f"qpos_before={_format_array(qpos_before[:6])}",
                    f"qpos_after={_format_array(qpos_after[:6])}",
                    f"qpos_tracking_error={_format_array(current_target_qpos - qpos_after[:6])}",
                    f"settled_qpos={_format_array(settled_qpos[:6])}",
                    f"settled_qpos_tracking_error={_format_array(current_target_qpos - settled_qpos[:6])}",
                    f"tcp_before_xyz={_format_array(tcp_before[:3])}",
                    f"tcp_after_xyz={_format_array(tcp_after[:3])}",
                    f"settled_tcp_xyz={_format_array(settled_tcp[:3])}",
                    "",
                ]
            )
            save_idx += 1
            _save_snapshot_from_obs(settled_obs, settled_info, args, robot_base_pose_p, robot_base_pose_q, save_idx)
            obs, info = settled_obs, settled_info
            command_index += 1

    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved joint auto-sequence log to {log_path}")
    return save_idx


def _configure_rc5_debug_drives(args: Args) -> None:
    base_stiffness = [900.0, 900.0, 800.0, 450.0, 250.0, 180.0]
    base_damping = [220.0, 220.0, 180.0, 100.0, 60.0, 40.0]
    base_force_limit = [120.0, 120.0, 100.0, 70.0, 50.0, 30.0]
    stiffness = [
        v * args.arm_stiffness_scale for v in base_stiffness
    ]
    damping = [
        v * args.arm_damping_scale for v in base_damping
    ]
    force_limit = [
        v * args.arm_force_limit_scale for v in base_force_limit
    ]
    stiffness_override = _parse_joint_override(args.arm_stiffness_override)
    damping_override = _parse_joint_override(args.arm_damping_override)
    force_limit_override = _parse_joint_override(args.arm_force_limit_override)
    if stiffness_override is not None:
        stiffness = stiffness_override
    if damping_override is not None:
        damping = damping_override
    if force_limit_override is not None:
        force_limit = force_limit_override
    RC5AeroHandRight.arm_stiffness = stiffness
    RC5AeroHandRight.arm_damping = damping
    RC5AeroHandRight.arm_force_limit = force_limit
    RC5AeroHandRight.arm_friction = (
        RC5AeroHandRight.arm_friction
        if args.arm_friction_override is None
        else args.arm_friction_override
    )


def _build_env(
    args: Args,
    camera_mode: str,
    camera_eye: np.ndarray | None = None,
    camera_target: np.ndarray | None = None,
    camera_roll_deg: float | None = None,
    camera_fov: float | None = None,
) -> BaseEnv:
    if camera_eye is not None and camera_target is not None:
        pose = pose_from_eye_target_roll(
            eye=camera_eye,
            target=camera_target,
            roll_deg=0.0 if camera_roll_deg is None else camera_roll_deg,
        )
        camera_cfg = {
            "pose": pose,
            "intrinsic": None,
            "fov": 1.0 if camera_fov is None else camera_fov,
            "near": 0.01,
            "far": 10.0,
        }
    else:
        camera_cfg = observation_camera_override(camera_mode)

    return gym.make(
        "OpenReal2SimValidation-v1",
        obs_mode="rgb+segmentation",
        control_mode=args.control_mode_name,
        reward_mode="none",
        render_mode="rgb_array",
        sensor_configs={
            "shader_pack": args.shader,
            "3rd_view_camera": camera_cfg,
        },
        human_render_camera_configs={"shader_pack": args.shader},
        num_envs=1,
        sim_backend=args.sim_backend,
        robot_uids="rc5_aero_hand_right_openreal2sim_validation",
    )


def _rebuild_env_with_camera(
    env: BaseEnv,
    args: Args,
    camera_mode: str,
    camera_eye: np.ndarray,
    camera_target: np.ndarray,
    camera_roll_deg: float,
    camera_fov: float,
) -> BaseEnv:
    env.close()
    return _build_env(
        args,
        camera_mode=camera_mode,
        camera_eye=camera_eye,
        camera_target=camera_target,
        camera_roll_deg=camera_roll_deg,
        camera_fov=camera_fov,
    )


def main(args: Args):
    _configure_rc5_debug_drives(args)
    robot_base_pose_p, robot_base_pose_q, robot_init_qpos = _load_state_file(args.load_state_file)
    robot_base_pose_p += np.array([args.base_dx, args.base_dy, args.base_dz], dtype=np.float32)

    camera_mode = args.observation_camera_mode
    cam_state = observation_camera_lookat_state(camera_mode)
    camera_eye = np.array(cam_state["eye"], dtype=np.float32)
    camera_target = np.array(cam_state["target"], dtype=np.float32)
    camera_roll_deg = float(cam_state.get("roll_deg", 0.0))
    camera_fov = float(cam_state.get("fov", 1.0))

    env = _build_env(
        args,
        camera_mode=camera_mode,
        camera_eye=camera_eye,
        camera_target=camera_target,
        camera_roll_deg=camera_roll_deg,
        camera_fov=camera_fov,
    )
    try:
        save_idx = 0
        obs, info = _reset_current_state(
            env,
            args,
            robot_base_pose_p,
            robot_base_pose_q,
            robot_init_qpos,
            reconfigure=True,
        )
        _save_snapshot_from_obs(obs, info, args, robot_base_pose_p, robot_base_pose_q, save_idx)
        print(f"Control mode: {args.control_mode_name}")
        print(f"arm_stiffness={RC5AeroHandRight.arm_stiffness}")
        print(f"arm_damping={RC5AeroHandRight.arm_damping}")
        print(f"arm_force_limit={RC5AeroHandRight.arm_force_limit}")
        print(f"arm_friction={RC5AeroHandRight.arm_friction}")
        if args.control_mode_name == JOINT_SIM_CONTROL_MODE:
            save_idx = _run_joint_auto_sequence(
                env,
                args,
                robot_base_pose_p,
                robot_base_pose_q,
                save_idx,
            )
        else:
            save_idx = _run_auto_sequence(
                env,
                args,
                robot_base_pose_p,
                robot_base_pose_q,
                save_idx,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
