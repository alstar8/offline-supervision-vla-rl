from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_REAL2SIM_ROOT = PACKAGE_ROOT.parent
DEBUG_ROOT = REPO_REAL2SIM_ROOT / "debug"

RC5_POSE_DIR = DEBUG_ROOT / "rc5_pose"
RC5_POSE_LIVE_DIR = DEBUG_ROOT / "rc5_pose_live"
RC5_TCP_POSE_LIVE_DIR = DEBUG_ROOT / "rc5_tcp_pose_live"
RC5_SYNC_SHELL_DIR = DEBUG_ROOT / "rc5_sync_shell"
RC5_JOINT_ZERO_SHELL_DIR = DEBUG_ROOT / "rc5_joint_zero_shell"
RC5_POSE_SWEEP_DIR = DEBUG_ROOT / "rc5_pose_sweep"
RC5_POSE_SWEEP_TIGHT_DIR = DEBUG_ROOT / "rc5_pose_sweep_tight"
RC5_POSE_SWEEP_TRAINING_VIEW_DIR = DEBUG_ROOT / "rc5_pose_sweep_training_view"
VIDEOS_DIR = DEBUG_ROOT / "videos"
GRID_ALIGNMENT_DIR = DEBUG_ROOT / "grid_alignment"
VR_DEMOS_DEBUG_DIR = DEBUG_ROOT / "vr_demos"
WRIST_CAMERA_CALIBRATION_DIR = DEBUG_ROOT / "wrist_camera_calibration"

HUMAN_DEMOS_DIR = REPO_REAL2SIM_ROOT / "human_demos"

RC5_POSE_DEBUG_STATE_FILE = RC5_POSE_DIR / "openreal2sim_rc5_pose_debug.txt"
RC5_POSE_LIVE_LATEST_STATE_FILE = RC5_POSE_LIVE_DIR / "openreal2sim_rc5_live_latest.txt"
RC5_SYNC_COMPARE_LOG_FILE = RC5_SYNC_SHELL_DIR / "sync_rc5_compare.log"
RC5_SYNC_STARTUP_HOME_FILE = RC5_SYNC_SHELL_DIR / "sync_rc5_0000_startup_home.txt"

PRESETS_GRID_ALIGNMENT_DIR = PACKAGE_ROOT / "presets" / "grid_alignment"


def resolve_spawn_grid_state_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None

    requested = Path(raw)
    candidates = [requested]
    if not requested.is_absolute():
        candidates.extend(
            [
                REPO_REAL2SIM_ROOT / requested,
                PACKAGE_ROOT / requested,
                DEBUG_ROOT / requested.name,
                PRESETS_GRID_ALIGNMENT_DIR / requested.name,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return requested
