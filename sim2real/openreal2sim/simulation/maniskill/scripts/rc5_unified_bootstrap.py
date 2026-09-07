from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import yaml

_Y = "\033[33m"
_R = "\033[0m"
_VALID_RC5_MOVE_GROUPS = ("right_tcp_link",)


def emit_warning(scope: str, message: str) -> None:
    print(f"{_Y}[WARNING] [{scope}] {message}{_R}")


@dataclass(frozen=True)
class RC5BootstrapConfig:
    config_path: Path
    key: str
    robot_uids: Optional[str]
    control_mode: Optional[str]
    lighting_profile_config: Optional[str]
    lighting_profile: Optional[str]
    hand_pose_config: Optional[str]
    hand_contact_config: Optional[str]
    hand_contact_profile: Optional[str]
    hand_controller_config: Optional[str]
    hand_controller_profile: Optional[str]
    teleop_profile_config: Optional[str]
    teleop_profile: Optional[str]


@dataclass(frozen=True)
class AppliedHandPoseConfig:
    config_path: Path
    bindings: Dict[str, str]
    presets: Dict[str, np.ndarray]
    open_preset_name: str
    close_preset_name: str


@dataclass(frozen=True)
class SimulationConfigSections:
    config_path: Path
    key: str
    raw_config: Dict[str, Any]
    global_sim: Dict[str, Any]
    local_sim: Dict[str, Any]


@dataclass(frozen=True)
class UnifiedBootstrapRequest:
    config_path: Path
    scene_path: Optional[str]
    key: str
    cli_bootstrap_overrides: Dict[str, str]
    bootstrap: RC5BootstrapConfig


@dataclass(frozen=True)
class ResolvedTeleopProfile:
    config_path: Path
    profile_name: str
    sim_delta_remap_rpy_deg: list[float]
    gripper_open_signal: float
    gripper_close_signal: float


@dataclass(frozen=True)
class ResolvedLightingProfile:
    config_path: Path
    profile_name: str
    profile: Dict[str, Any]


@dataclass(frozen=True)
class ResolvedHandContactProfile:
    config_path: Path
    profile_name: str
    profile: Dict[str, Any]
    material_names: list[str]
    link_names: list[str]


@dataclass(frozen=True)
class ResolvedHandControllerProfile:
    config_path: Path
    profile_name: str
    profile: Dict[str, Any]
    summary_fields: Dict[str, Any]


def resolve_rc5_move_group(
    explicit_move_group: str | None,
    *,
    env: Mapping[str, str] | None = None,
    default: str = "right_tcp_link",
    scope: str = "RC5Bootstrap",
    warn_on_env: bool = True,
    warn_on_default: bool = True,
) -> str:
    env = os.environ if env is None else env
    move_group = (explicit_move_group or "").strip()
    if move_group:
        source = "explicit"
    else:
        move_group = str(env.get("OPENR2S_RC5_MOVE_GROUP", "")).strip()
        if move_group:
            source = "environment"
            if warn_on_env:
                emit_warning(scope, f"Using OPENR2S_RC5_MOVE_GROUP from environment: {move_group}")
        else:
            source = "default"
            move_group = str(default).strip()
            if warn_on_default:
                print(f"[{scope}] Using canonical RC5 move group default: {move_group}")

    if move_group not in _VALID_RC5_MOVE_GROUPS:
        valid = ", ".join(_VALID_RC5_MOVE_GROUPS)
        raise ValueError(
            f"Invalid RC5 move group '{move_group}' from {source}. "
            f"Expected one of: {valid}"
        )
    return move_group


def load_runner_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Runner config must be a YAML mapping: {path}")
    return data


def extract_flag_values(argv: Sequence[str], flag: str) -> List[str]:
    values: List[str] = []
    idx = 0
    argv = list(argv)
    while idx < len(argv):
        token = argv[idx]
        if token == flag:
            next_idx = idx + 1
            if next_idx >= len(argv):
                values.append("")
                idx += 1
                continue
            values.append(argv[next_idx])
            idx += 2
            continue
        idx += 1
    return values


def has_flag(argv: Sequence[str], flag: str) -> bool:
    return bool(extract_flag_values(argv, flag))


def collect_cli_bootstrap_overrides(passthrough_argv: Sequence[str]) -> Dict[str, str]:
    override_flags = (
        "--robot_uids",
        "--control_mode",
        "--lighting_profile_config",
        "--lighting_profile",
        "--hand_pose_config",
        "--hand_contact_config",
        "--hand_contact_profile",
        "--hand_controller_config",
        "--hand_controller_profile",
        "--teleop_profile_config",
        "--teleop_profile",
    )
    result: Dict[str, str] = {}
    for flag in override_flags:
        values = extract_flag_values(passthrough_argv, flag)
        if values:
            result[flag[2:]] = values[-1]
    return result


def detect_config_key(scene_path: str | Path | None, explicit_key: str | None = None) -> str | None:
    if explicit_key:
        return explicit_key
    if scene_path is None:
        return None
    scene_path = Path(scene_path)
    if "outputs" in scene_path.parts:
        outputs_idx = scene_path.parts.index("outputs")
        if outputs_idx + 1 < len(scene_path.parts):
            return scene_path.parts[outputs_idx + 1]
    if "assets" in scene_path.parts and "scenes" in scene_path.parts:
        scenes_idx = scene_path.parts.index("scenes")
        if scenes_idx > 0 and scene_path.parts[scenes_idx - 1] == "assets" and scenes_idx + 1 < len(scene_path.parts):
            return scene_path.parts[scenes_idx + 1]
    return None


def resolve_unified_bootstrap_request(
    passthrough_argv: Sequence[str],
    *,
    default_config_path: str | Path = "config/config_debug.yaml",
) -> UnifiedBootstrapRequest:
    config_values = extract_flag_values(passthrough_argv, "--config_path")
    key_values = extract_flag_values(passthrough_argv, "--key")
    scene_values = extract_flag_values(passthrough_argv, "--scene")
    config_path = Path(config_values[-1] if config_values else default_config_path).expanduser().resolve()
    explicit_key = key_values[-1] if key_values else None
    scene_path = scene_values[-1] if scene_values else None
    key = detect_config_key(scene_path, explicit_key)
    cli_bootstrap_overrides = collect_cli_bootstrap_overrides(passthrough_argv)
    bootstrap = extract_rc5_bootstrap_config(config_path, key, cli_overrides=cli_bootstrap_overrides)
    return UnifiedBootstrapRequest(
        config_path=config_path,
        scene_path=scene_path,
        key=bootstrap.key,
        cli_bootstrap_overrides=cli_bootstrap_overrides,
        bootstrap=bootstrap,
    )


def load_simulation_config_sections(config_path: str | Path, key: str) -> SimulationConfigSections:
    data = load_runner_config(config_path)
    keys = data.get("keys", [])
    local = data.get("local", {})
    global_sim = ((data.get("global") or {}).get("simulation") or {})
    if not key:
        raise ValueError("RC5 unified bootstrap requires an explicit config key.")
    if key not in local:
        available = ", ".join(sorted(local.keys()))
        raise KeyError(f"Config key '{key}' not found. Available keys: {available}")
    if keys and key not in keys:
        raise ValueError(f"Config key '{key}' exists in local but is not declared under top-level keys.")

    local_sim = ((local.get(key) or {}).get("simulation") or {})
    if not isinstance(local_sim, dict):
        raise ValueError(f"local.{key}.simulation must be a mapping")

    return SimulationConfigSections(
        config_path=Path(config_path).expanduser().resolve(),
        key=key,
        raw_config=data,
        global_sim=global_sim,
        local_sim=local_sim,
    )


def pick_simulation_value(
    sections: SimulationConfigSections,
    name: str,
    default: Any = None,
    *,
    cli_value: Any = None,
) -> Any:
    if cli_value is not None:
        return cli_value
    if name in sections.local_sim:
        return sections.local_sim[name]
    if name in sections.global_sim:
        return sections.global_sim[name]
    return default


def resolve_local_config_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def load_hand_pose_config(hand_pose_config: str | Path) -> tuple[Path, Dict[str, Any]]:
    cfg_path = resolve_local_config_path(hand_pose_config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Hand pose config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    poses = cfg.get("poses")
    if not isinstance(poses, dict) or not poses:
        raise ValueError(f"Hand pose config must define non-empty 'poses': {cfg_path}")
    return cfg_path, cfg


def load_lighting_profile_config(lighting_profile_config: str | Path) -> tuple[Path, Dict[str, Any]]:
    cfg_path = resolve_local_config_path(lighting_profile_config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Lighting profile config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    profiles = cfg.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"Lighting profile config must define non-empty 'profiles': {cfg_path}")
    return cfg_path, cfg


def load_teleop_profile_config(teleop_profile_config: str | Path) -> tuple[Path, Dict[str, Any]]:
    cfg_path = resolve_local_config_path(teleop_profile_config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Teleop profile config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    profiles = cfg.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"Teleop profile config must define non-empty 'profiles': {cfg_path}")
    return cfg_path, cfg


def load_hand_contact_config(hand_contact_config: str | Path) -> tuple[Path, Dict[str, Any]]:
    cfg_path = resolve_local_config_path(hand_contact_config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Hand contact config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    profiles = cfg.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"Hand contact config must define non-empty 'profiles': {cfg_path}")
    return cfg_path, cfg


def load_hand_controller_config(hand_controller_config: str | Path) -> tuple[Path, Dict[str, Any]]:
    cfg_path = resolve_local_config_path(hand_controller_config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Hand controller config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    profiles = cfg.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"Hand controller config must define non-empty 'profiles': {cfg_path}")
    return cfg_path, cfg


def resolve_lighting_profile(
    lighting_profile_config: str | Path,
    lighting_profile: str,
) -> ResolvedLightingProfile:
    cfg_path, cfg = load_lighting_profile_config(lighting_profile_config)
    profiles = cfg["profiles"]
    if lighting_profile not in profiles:
        raise KeyError(
            f"Lighting profile '{lighting_profile}' not found in {cfg_path}. "
            f"Available profiles: {', '.join(sorted(profiles.keys()))}"
        )
    profile = profiles[lighting_profile]
    if not isinstance(profile, dict) or not profile:
        raise ValueError(
            f"Lighting profile '{lighting_profile}' in {cfg_path} must define a non-empty mapping"
        )
    return ResolvedLightingProfile(
        config_path=cfg_path,
        profile_name=lighting_profile,
        profile=dict(profile),
    )


def apply_lighting_profile_overrides(
    config_overrides: Dict[str, Any],
    *,
    scope: str = "Lighting",
) -> ResolvedLightingProfile | None:
    profile_name = config_overrides.get("lighting_profile")
    profile_cfg_path = config_overrides.get("lighting_profile_config")
    if profile_name is None and profile_cfg_path is None:
        return None
    if profile_name is None:
        raise ValueError("lighting_profile_config was provided but lighting_profile is missing")
    if profile_cfg_path is None:
        raise ValueError("lighting_profile was provided but lighting_profile_config is missing")

    resolved = resolve_lighting_profile(profile_cfg_path, profile_name)
    config_overrides["lighting_profile_config"] = str(resolved.config_path)
    config_overrides["lighting_profile"] = resolved.profile_name
    config_overrides["lighting_config"] = dict(resolved.profile)
    print(
        f"[{scope}] Loaded profile '{resolved.profile_name}' from {resolved.config_path}: "
        f"ambient_light={resolved.profile.get('ambient_light')}, "
        f"directional_lights={len(resolved.profile.get('directional_lights') or [])}, "
        f"point_lights={len(resolved.profile.get('point_lights') or [])}"
    )
    return resolved


def resolve_teleop_profile(
    teleop_profile_config: str | Path,
    teleop_profile: str,
) -> ResolvedTeleopProfile:
    cfg_path, cfg = load_teleop_profile_config(teleop_profile_config)
    profiles = cfg["profiles"]
    if teleop_profile not in profiles:
        raise KeyError(
            f"Teleop profile '{teleop_profile}' not found in {cfg_path}. "
            f"Available profiles: {', '.join(sorted(profiles.keys()))}"
        )
    profile = profiles[teleop_profile]
    remap = profile.get("sim_delta_remap_rpy_deg")
    open_signal = profile.get("gripper_open_signal")
    close_signal = profile.get("gripper_close_signal")
    if remap is None:
        raise ValueError(
            f"Teleop profile '{teleop_profile}' in {cfg_path} must define sim_delta_remap_rpy_deg"
        )
    if len(remap) != 3:
        raise ValueError(
            f"Teleop profile '{teleop_profile}' in {cfg_path} must define exactly 3 values for "
            f"sim_delta_remap_rpy_deg, got {len(remap)}"
        )
    if open_signal is None or close_signal is None:
        raise ValueError(
            f"Teleop profile '{teleop_profile}' in {cfg_path} must define both "
            f"gripper_open_signal and gripper_close_signal"
        )
    return ResolvedTeleopProfile(
        config_path=cfg_path,
        profile_name=teleop_profile,
        sim_delta_remap_rpy_deg=[float(x) for x in remap],
        gripper_open_signal=float(open_signal),
        gripper_close_signal=float(close_signal),
    )


def apply_teleop_profile_overrides(
    config_overrides: Dict[str, Any],
    *,
    scope: str = "Teleop",
) -> ResolvedTeleopProfile | None:
    profile_name = config_overrides.get("teleop_profile")
    profile_cfg_path = config_overrides.get("teleop_profile_config")
    if profile_name is None and profile_cfg_path is None:
        return None
    if profile_name is None:
        raise ValueError("teleop_profile_config was provided but teleop_profile is missing")
    if profile_cfg_path is None:
        raise ValueError("teleop_profile was provided but teleop_profile_config is missing")

    resolved = resolve_teleop_profile(profile_cfg_path, profile_name)
    config_overrides["teleop_profile_config"] = str(resolved.config_path)
    config_overrides["teleop_profile"] = resolved.profile_name
    config_overrides["sim_delta_remap_rpy_deg"] = list(resolved.sim_delta_remap_rpy_deg)
    config_overrides["gripper_open_signal"] = float(resolved.gripper_open_signal)
    config_overrides["gripper_close_signal"] = float(resolved.gripper_close_signal)
    print(
        f"[{scope}] Loaded profile '{resolved.profile_name}' from {resolved.config_path}: "
        f"sim_delta_remap_rpy_deg={config_overrides['sim_delta_remap_rpy_deg']}, "
        f"gripper_open_signal={config_overrides['gripper_open_signal']:+.3f}, "
        f"gripper_close_signal={config_overrides['gripper_close_signal']:+.3f}"
    )
    return resolved


def resolve_hand_contact_profile(
    hand_contact_config: str | Path,
    hand_contact_profile: str,
) -> ResolvedHandContactProfile:
    cfg_path, cfg = load_hand_contact_config(hand_contact_config)
    profiles = cfg["profiles"]
    if hand_contact_profile not in profiles:
        raise KeyError(
            f"Hand contact profile '{hand_contact_profile}' not found in {cfg_path}. "
            f"Available profiles: {', '.join(sorted(profiles.keys()))}"
        )
    profile = profiles[hand_contact_profile]
    materials_cfg = profile.get("materials", profile.get("_materials"))
    links_cfg = profile.get("links", profile.get("link"))
    if not isinstance(materials_cfg, dict) or not materials_cfg:
        raise ValueError(
            f"Hand contact profile '{hand_contact_profile}' in {cfg_path} must define non-empty "
            f"'materials' (or '_materials')"
        )
    if not isinstance(links_cfg, dict) or not links_cfg:
        raise ValueError(
            f"Hand contact profile '{hand_contact_profile}' in {cfg_path} must define non-empty "
            f"'links' (or 'link')"
        )
    return ResolvedHandContactProfile(
        config_path=cfg_path,
        profile_name=hand_contact_profile,
        profile=dict(profile),
        material_names=sorted(materials_cfg.keys()),
        link_names=sorted(links_cfg.keys()),
    )


def apply_hand_contact_profile_overrides(
    config_overrides: Dict[str, Any],
    *,
    scope: str = "HandContact",
) -> ResolvedHandContactProfile | None:
    profile_name = config_overrides.get("hand_contact_profile")
    profile_cfg_path = config_overrides.get("hand_contact_config")
    if profile_name is None and profile_cfg_path is None:
        return None
    if profile_name is None:
        raise ValueError("hand_contact_config was provided but hand_contact_profile is missing")
    if profile_cfg_path is None:
        raise ValueError("hand_contact_profile was provided but hand_contact_config is missing")

    resolved = resolve_hand_contact_profile(profile_cfg_path, profile_name)
    config_overrides["hand_contact_config"] = str(resolved.config_path)
    config_overrides["hand_contact_profile"] = resolved.profile_name
    config_overrides["hand_contact"] = dict(resolved.profile)
    print(
        f"[{scope}] Loaded profile '{resolved.profile_name}' from {resolved.config_path}: "
        f"materials={resolved.material_names}, links={resolved.link_names}"
    )
    return resolved


def resolve_hand_controller_profile(
    hand_controller_config: str | Path,
    hand_controller_profile: str,
) -> ResolvedHandControllerProfile:
    cfg_path, cfg = load_hand_controller_config(hand_controller_config)
    profiles = cfg["profiles"]
    if hand_controller_profile not in profiles:
        raise KeyError(
            f"Hand controller profile '{hand_controller_profile}' not found in {cfg_path}. "
            f"Available profiles: {', '.join(sorted(profiles.keys()))}"
        )
    profile = profiles[hand_controller_profile]
    if not isinstance(profile, dict):
        raise ValueError(f"Hand controller profile '{hand_controller_profile}' in {cfg_path} must be a mapping")
    summary_fields = {
        k: profile[k] for k in ("stiffness", "damping", "force_limit", "friction") if k in profile
    }
    if not summary_fields:
        raise ValueError(
            f"Hand controller profile '{hand_controller_profile}' in {cfg_path} must define at least one of "
            f"stiffness/damping/force_limit/friction"
        )
    return ResolvedHandControllerProfile(
        config_path=cfg_path,
        profile_name=hand_controller_profile,
        profile=dict(profile),
        summary_fields=summary_fields,
    )


def apply_hand_controller_profile_overrides(
    config_overrides: Dict[str, Any],
    *,
    scope: str = "HandController",
) -> ResolvedHandControllerProfile | None:
    profile_name = config_overrides.get("hand_controller_profile")
    profile_cfg_path = config_overrides.get("hand_controller_config")
    if profile_name is None and profile_cfg_path is None:
        return None
    if profile_name is None:
        raise ValueError("hand_controller_config was provided but hand_controller_profile is missing")
    if profile_cfg_path is None:
        raise ValueError("hand_controller_profile was provided but hand_controller_config is missing")

    resolved = resolve_hand_controller_profile(profile_cfg_path, profile_name)
    config_overrides["hand_controller_config"] = str(resolved.config_path)
    config_overrides["hand_controller_profile"] = resolved.profile_name
    config_overrides["hand_controller"] = dict(resolved.profile)
    summary = ", ".join(f"{k}={v}" for k, v in resolved.summary_fields.items())
    print(f"[{scope}] Loaded profile '{resolved.profile_name}' from {resolved.config_path}: {summary}")
    return resolved


def _build_hand_qpos_from_pose(agent, pose_cfg_path: Path, pose_name: str, pose_data: Dict[str, Any]) -> np.ndarray:
    if not isinstance(pose_data, dict):
        raise ValueError(f"Hand pose '{pose_name}' in {pose_cfg_path} must be a mapping")
    joints = pose_data.get("joints")
    if not isinstance(joints, dict):
        raise ValueError(f"Hand pose '{pose_name}' in {pose_cfg_path} must define a 'joints' mapping")
    hand_names = list(getattr(agent, "hand_joint_names", []))
    missing = [name for name in hand_names if name not in joints]
    extra = [name for name in joints.keys() if name not in hand_names]
    if missing or extra:
        raise RuntimeError(
            f"Hand pose '{pose_name}' joint-name mismatch in {pose_cfg_path}\n"
            f"expected_hand_joint_names={hand_names}\n"
            f"missing={missing}\n"
            f"extra={extra}"
        )
    return np.asarray([float(joints[name]) for name in hand_names], dtype=np.float32)


def apply_hand_pose_config_to_agent(
    agent,
    hand_pose_config: str | Path | None,
    *,
    open_preset_name: str | None = None,
    close_preset_name: str | None = None,
    required_bindings: Sequence[str] | None = None,
) -> AppliedHandPoseConfig | None:
    if hand_pose_config is None:
        return None

    hand_names = list(getattr(agent, "hand_joint_names", []))
    if not hand_names:
        raise RuntimeError(
            f"Agent uid='{getattr(agent, 'uid', 'unknown')}' does not expose hand_joint_names; "
            f"cannot apply hand pose config '{hand_pose_config}'. "
            "Unified RC5 runtime requires explicit hand_joint_names."
        )

    cfg_path, cfg = load_hand_pose_config(hand_pose_config)
    poses = cfg["poses"]
    bindings = cfg.get("bindings", {})
    if not isinstance(bindings, dict):
        raise ValueError(f"Hand pose config bindings must be a mapping: {cfg_path}")

    required_bindings = list(required_bindings or [])
    missing_bindings = [name for name in required_bindings if name not in bindings]
    if missing_bindings:
        raise KeyError(f"Hand pose config {cfg_path} is missing required bindings: {missing_bindings}")

    resolved_open_preset_name = open_preset_name or bindings.get("open")
    resolved_close_preset_name = close_preset_name or bindings.get("close")
    if resolved_open_preset_name is None or resolved_close_preset_name is None:
        raise ValueError(f"Hand pose config {cfg_path} must define bindings.open and bindings.close")
    if resolved_open_preset_name not in poses or resolved_close_preset_name not in poses:
        raise KeyError(
            f"Hand pose config {cfg_path} is missing open/close presets referenced by bindings: "
            f"open={resolved_open_preset_name}, close={resolved_close_preset_name}"
        )

    all_qpos_presets = {
        pose_name: _build_hand_qpos_from_pose(agent, cfg_path, pose_name, pose_data)
        for pose_name, pose_data in poses.items()
    }

    agent.hand_open_qpos = all_qpos_presets[resolved_open_preset_name].copy()
    agent.hand_close_qpos = all_qpos_presets[resolved_close_preset_name].copy()
    controller = getattr(agent, "controller", None)
    gripper_controller = getattr(controller, "controllers", {}).get("gripper") if controller is not None else None
    if gripper_controller is not None and hasattr(gripper_controller, "config"):
        if hasattr(gripper_controller.config, "open_qpos"):
            gripper_controller.config.open_qpos = agent.hand_open_qpos.tolist()
        if hasattr(gripper_controller.config, "close_qpos"):
            gripper_controller.config.close_qpos = agent.hand_close_qpos.tolist()
        print(f"[HandPose] Injected runtime gripper controller open_qpos/close_qpos from '{cfg_path}'.")
    else:
        print(
            "[HandPose] No runtime gripper controller config is available; "
            "keeping loaded hand_open_qpos/hand_close_qpos only."
        )
    print(
        f"[HandPose] Loaded hand pose presets from {cfg_path}: "
        f"open='{resolved_open_preset_name}', close='{resolved_close_preset_name}'"
    )
    print(
        f"[HandPose] hand_open_qpos={np.array2string(agent.hand_open_qpos, precision=4, suppress_small=True, max_line_width=200)}"
    )
    print(
        f"[HandPose] hand_close_qpos={np.array2string(agent.hand_close_qpos, precision=4, suppress_small=True, max_line_width=200)}"
    )
    return AppliedHandPoseConfig(
        config_path=cfg_path,
        bindings=dict(bindings),
        presets={name: qpos.copy() for name, qpos in all_qpos_presets.items()},
        open_preset_name=resolved_open_preset_name,
        close_preset_name=resolved_close_preset_name,
    )


def extract_rc5_bootstrap_config(
    config_path: str | Path,
    key: str,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> RC5BootstrapConfig:
    sections = load_simulation_config_sections(config_path, key)
    cli_overrides = dict(cli_overrides or {})

    def pick(name: str, default: Any = None) -> Any:
        return pick_simulation_value(sections, name, default, cli_value=cli_overrides.get(name, None))

    return RC5BootstrapConfig(
        config_path=sections.config_path,
        key=sections.key,
        robot_uids=pick("robot_uids", None),
        control_mode=pick("control_mode", None),
        lighting_profile_config=pick("lighting_profile_config", None),
        lighting_profile=pick("lighting_profile", None),
        hand_pose_config=pick("hand_pose_config", None),
        hand_contact_config=pick("hand_contact_config", None),
        hand_contact_profile=pick("hand_contact_profile", None),
        hand_controller_config=pick("hand_controller_config", None),
        hand_controller_profile=pick("hand_controller_profile", None),
        teleop_profile_config=pick("teleop_profile_config", None),
        teleop_profile=pick("teleop_profile", None),
    )


def validate_rc5_bootstrap_for_backend(bootstrap: RC5BootstrapConfig, motion_backend: str) -> None:
    if bootstrap.robot_uids is None:
        raise ValueError(
            f"RC5 unified runner requires robot_uids to be defined for key '{bootstrap.key}' "
            f"in {bootstrap.config_path} or via CLI override."
        )

    if (bootstrap.lighting_profile_config is None) != (bootstrap.lighting_profile is None):
        raise ValueError(
            "lighting_profile_config and lighting_profile must be provided together "
            f"for key '{bootstrap.key}'."
        )
    if (bootstrap.hand_contact_config is None) != (bootstrap.hand_contact_profile is None):
        raise ValueError(
            "hand_contact_config and hand_contact_profile must be provided together "
            f"for key '{bootstrap.key}'."
        )
    if (bootstrap.hand_controller_config is None) != (bootstrap.hand_controller_profile is None):
        raise ValueError(
            "hand_controller_config and hand_controller_profile must be provided together "
            f"for key '{bootstrap.key}'."
        )

    if motion_backend in {"proxy_ee_delta", "hybrid"}:
        if bootstrap.teleop_profile_config is None or bootstrap.teleop_profile is None:
            raise ValueError(
                f"Backend '{motion_backend}' requires teleop_profile_config and teleop_profile "
                f"for key '{bootstrap.key}'."
            )


def reset_planner_hand_target_to_open(
    env_unwrapped,
    restore_planner_grasp_state_fn,
    save_planner_grasp_state_fn,
    source_stage: str,
) -> bool:
    agent = getattr(env_unwrapped, "agent", None)
    if agent is None or not hasattr(agent, "hand_open_qpos"):
        return False

    open_qpos = np.asarray(agent.hand_open_qpos, dtype=np.float32).reshape(-1)
    restore_planner_grasp_state_fn(
        env_unwrapped,
        default_target_hand_qpos=open_qpos,
        source_stage=source_stage,
    )
    save_planner_grasp_state_fn(
        env_unwrapped,
        target_hand_qpos=open_qpos,
        realized_hand_qpos=open_qpos,
        grasp_flag=False,
        object_id=None,
        source_stage=source_stage,
    )
    try:
        robot_qpos = agent.robot.get_qpos()
        if hasattr(robot_qpos, "cpu"):
            robot_qpos = robot_qpos.cpu().numpy()
        robot_qpos = np.asarray(robot_qpos)
        if robot_qpos.ndim > 1:
            robot_qpos = robot_qpos[0]
        arm_dof = len(getattr(agent, "arm_joint_names", []))
        if open_qpos.size > 0 and robot_qpos.shape[0] >= arm_dof + open_qpos.size:
            robot_qpos = np.asarray(robot_qpos, dtype=np.float32).copy()
            robot_qpos[arm_dof:arm_dof + open_qpos.size] = open_qpos
            agent.robot.set_qpos(robot_qpos)
    except Exception as exc:
        print(f"[WARN] Failed to explicitly restore open hand state on reset: {type(exc).__name__}: {exc}")
    return True
