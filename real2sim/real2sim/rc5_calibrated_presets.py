import ast
from pathlib import Path

import numpy as np


RC5_POSE_DIR = Path(__file__).resolve().parent / "presets" / "rc5_pose"
HOME_POSE_FILE = RC5_POSE_DIR / "home_pos_convenient.txt"
HAND_OPEN_POSE_FILE = RC5_POSE_DIR / "hand_pose_open.txt"
HAND_GRAB_POSE_FILE = RC5_POSE_DIR / "hand_pose_grab.txt"

GRIPPER_JOINT_ORDER = [
    "right_thumb_cmc_abd",
    "right_thumb_cmc_flex",
    "right_thumb_mcp",
    "right_thumb_ip",
    "right_index_mcp_flex",
    "right_index_pip",
    "right_index_dip",
    "right_middle_mcp_flex",
    "right_middle_pip",
    "right_middle_dip",
    "right_ring_mcp_flex",
    "right_ring_pip",
    "right_ring_dip",
    "right_pinky_mcp_flex",
    "right_pinky_pip",
    "right_pinky_dip",
]


def _parse_state_file(path: Path) -> dict:
    state = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            try:
                state[key] = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                continue
    return state


def _load_hand_qpos(path: Path) -> np.ndarray:
    state = _parse_state_file(path)
    hand_joint_qpos = state.get("hand_joint_qpos")
    if not isinstance(hand_joint_qpos, dict):
        raise ValueError(f"{path} is missing hand_joint_qpos data")
    return np.array([hand_joint_qpos[name] for name in GRIPPER_JOINT_ORDER], dtype=np.float32)


def _load_home_state(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = _parse_state_file(path)
    return (
        np.array(state["robot_base_pose_p"], dtype=np.float32),
        np.array(state["robot_base_pose_q"], dtype=np.float32),
        np.array(state["robot_init_qpos"], dtype=np.float32),
    )


RC5_HOME_BASE_POSE_P, RC5_HOME_BASE_POSE_Q, RC5_HOME_INIT_QPOS = _load_home_state(HOME_POSE_FILE)
RC5_HAND_OPEN_QPOS = _load_hand_qpos(HAND_OPEN_POSE_FILE)
RC5_HAND_GRAB_QPOS = _load_hand_qpos(HAND_GRAB_POSE_FILE)
