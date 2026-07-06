from __future__ import annotations

import numpy as np

# Inferred from the RC5 URDF fixed-frame construction and from the observed
# near-constant joint delta between the real robot's visually matched "manual
# home" pose and the equivalent sim pose.
#
# Interpretation:
#   q_sim ~= wrap(q_real + REAL_TO_SIM_ZERO_OFFSET_RAD)
#
# The first five entries are strong, structurally motivated candidates:
#   J1 -90 deg, J2 +120 deg, J3 -120 deg, J4 +90 deg, J5 +90 deg.
# J6 is left at 0 because the observed few-degree difference is likely not a
# zero-definition issue.
RC5_REAL_TO_SIM_ZERO_OFFSET_DEG = np.array(
    [-90.0, 120.0, -120.0, 90.0, 90.0, 0.0],
    dtype=np.float64,
)
RC5_REAL_TO_SIM_ZERO_OFFSET_RAD = np.deg2rad(RC5_REAL_TO_SIM_ZERO_OFFSET_DEG)


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return ((values + np.pi) % (2.0 * np.pi)) - np.pi


def real_to_sim_arm_joints(
    real_q_rad: np.ndarray,
    *,
    zero_offset_rad: np.ndarray | None = None,
) -> np.ndarray:
    real_q_rad = np.asarray(real_q_rad, dtype=np.float64)
    if real_q_rad.shape != (6,):
        raise ValueError(f"expected shape (6,), got {real_q_rad.shape}")
    if zero_offset_rad is None:
        zero_offset_rad = RC5_REAL_TO_SIM_ZERO_OFFSET_RAD
    zero_offset_rad = np.asarray(zero_offset_rad, dtype=np.float64)
    return wrap_to_pi(real_q_rad + zero_offset_rad)


def sim_to_real_arm_joints(
    sim_q_rad: np.ndarray,
    *,
    zero_offset_rad: np.ndarray | None = None,
) -> np.ndarray:
    sim_q_rad = np.asarray(sim_q_rad, dtype=np.float64)
    if sim_q_rad.shape != (6,):
        raise ValueError(f"expected shape (6,), got {sim_q_rad.shape}")
    if zero_offset_rad is None:
        zero_offset_rad = RC5_REAL_TO_SIM_ZERO_OFFSET_RAD
    zero_offset_rad = np.asarray(zero_offset_rad, dtype=np.float64)
    return wrap_to_pi(sim_q_rad - zero_offset_rad)
