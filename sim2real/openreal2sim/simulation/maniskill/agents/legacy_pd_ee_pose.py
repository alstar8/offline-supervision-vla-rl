"""Legacy EE-delta pose controller variants for teleop compatibility.

These reproduce the older `ee_align` / `ee_align2` semantics used in the
historical real2sim / teleop stack, while still integrating with the current
ManiSkill controller API.
"""

import importlib.util
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController, PDEEPoseControllerConfig
from mani_skill.utils.geometry.rotation_conversions import euler_angles_to_matrix, matrix_to_quaternion
from mani_skill.utils.structs import Pose

_VENDOR_KINEMATICS_PATH = (
    Path(__file__).resolve().parents[4]
    / "mani_skill"
    / "agents"
    / "controllers"
    / "utils"
    / "kinematics.py"
)


def _load_vendored_kinematics_cls():
    spec = importlib.util.spec_from_file_location(
        "openr2s_vendored_kinematics",
        _VENDOR_KINEMATICS_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load vendored kinematics from {_VENDOR_KINEMATICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Kinematics


_VENDORED_KINEMATICS_CLS = _load_vendored_kinematics_cls()


def _debug_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _debug_float(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


class LegacyAlignPDEEPoseController(PDEEPoseController):
    """PDEEPoseController with legacy `ee_align` / `ee_align2` frame semantics."""

    _preferred_arm_qpos = None
    _ik_config_validated = False

    def _check_gpu_sim_works(self):
        # The legacy teleop stack used custom EE-aligned frames even when running
        # on GPU PhysX. We intentionally bypass the stricter upstream assertion
        # because this controller computes a full target pose and then relies on
        # the normal IK solve, which works in our single-env debug setup.
        return

    def compute_target_pose(self, prev_ee_pose_at_base: Pose, action):
        if not self.config.use_delta:
            return super().compute_target_pose(prev_ee_pose_at_base, action)

        frame = str(self.config.frame)
        if frame not in {"ee_align", "ee_align2"}:
            return super().compute_target_pose(prev_ee_pose_at_base, action)

        delta_pos, delta_rot = action[:, 0:3], action[:, 3:6]
        delta_quat = matrix_to_quaternion(euler_angles_to_matrix(delta_rot, "XYZ"))
        delta_pose = Pose.create_from_pq(delta_pos, delta_quat)

        if frame == "ee_align":
            target_pose = delta_pose * prev_ee_pose_at_base
            target_pose.set_p(prev_ee_pose_at_base.p + delta_pos)
            return target_pose

        cur_ee_pose_at_base = self.ee_pose_at_base
        pivot = Pose.create_from_pq(p=cur_ee_pose_at_base.p, device=cur_ee_pose_at_base.device)
        return (pivot * delta_pose * pivot.inv()) * prev_ee_pose_at_base

    def _validate_ik_runtime_config(self):
        if self._ik_config_validated:
            return
        gains = {
            "ik_seed_preferred_gain": float(self.config.ik_seed_preferred_gain),
            "ik_seed_joint5_gain": float(self.config.ik_seed_joint5_gain),
        }
        for name, value in gains.items():
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise RuntimeError(f"{name} must be within [0, 1], got {value}")
        if float(self.config.diagnostic_joint5_delta_threshold_rad) < 0.0:
            raise RuntimeError(
                "diagnostic_joint5_delta_threshold_rad must be non-negative, "
                f"got {self.config.diagnostic_joint5_delta_threshold_rad}"
            )
        if float(self.config.diagnostic_arm_delta_norm_threshold_rad) < 0.0:
            raise RuntimeError(
                "diagnostic_arm_delta_norm_threshold_rad must be non-negative, "
                f"got {self.config.diagnostic_arm_delta_norm_threshold_rad}"
            )
        preferred_arm_qpos = self.config.preferred_arm_qpos
        if preferred_arm_qpos is not None:
            preferred_arm_qpos = np.asarray(preferred_arm_qpos, dtype=np.float32).reshape(-1)
            expected = len(self.config.joint_names)
            if preferred_arm_qpos.shape[0] != expected:
                raise RuntimeError(
                    f"preferred_arm_qpos length mismatch: expected {expected}, got {preferred_arm_qpos.shape[0]}"
                )
        solver_config = dict(self.config.delta_solver_config)
        solver_type = str(solver_config.get("type", ""))
        solver_mode = str(solver_config.get("mode", "single_stage"))
        if solver_type not in {"levenberg_marquardt", "pseudo_inverse"}:
            raise RuntimeError(
                "delta_solver_config.type must be 'levenberg_marquardt' or 'pseudo_inverse', "
                f"got {solver_type!r}"
            )
        if solver_mode not in {"single_stage", "hierarchical_position_first"}:
            raise RuntimeError(
                "delta_solver_config.mode must be 'single_stage' or 'hierarchical_position_first', "
                f"got {solver_mode!r}"
            )
        alpha = float(solver_config.get("alpha", 1.0))
        damping = float(solver_config.get("damping", 1e-4))
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise RuntimeError(f"delta_solver_config.alpha must be > 0, got {alpha}")
        if not np.isfinite(damping) or damping < 0.0:
            raise RuntimeError(f"delta_solver_config.damping must be >= 0, got {damping}")
        if solver_mode == "hierarchical_position_first":
            for name in [
                "orientation_alpha",
                "orientation_alpha_near_target",
                "orientation_position_threshold",
                "position_damping",
                "orientation_damping",
                "posture_gain",
                "posture_gain_near_target",
            ]:
                if name not in solver_config:
                    raise RuntimeError(
                        f"delta_solver_config.{name} is required when mode='hierarchical_position_first'"
                    )
                value = float(solver_config[name])
                if not np.isfinite(value) or value < 0.0:
                    raise RuntimeError(f"delta_solver_config.{name} must be >= 0, got {value}")
            posture_joint_weights = np.asarray(solver_config.get("posture_joint_weights"), dtype=np.float32).reshape(-1)
            expected = len(self.config.joint_names)
            if posture_joint_weights.shape[0] != expected:
                raise RuntimeError(
                    "delta_solver_config.posture_joint_weights length mismatch: "
                    f"expected {expected}, got {posture_joint_weights.shape[0]}"
                )
        self._ik_config_validated = True

    def _compute_ik_target_qpos(self, target_pose, q0, **kwargs):
        # Planner uses sim2real's Kinematics (current_pose / solver_config / preferred_qpos).
        # PPO uses the parent ManiSkill Kinematics; attach the local target-delta solver so
        # RC5 actions stay on the same IK as the real robot / planner.
        supported = getattr(self, "_compute_ik_params", None)
        if supported is None:
            supported = inspect.signature(self.kinematics.compute_ik).parameters
            self._compute_ik_params = supported
        if "current_pose" in supported:
            filtered = {key: value for key, value in kwargs.items() if key in supported}
            return self.kinematics.compute_ik(target_pose, q0, **filtered)
        kinematics = self.kinematics
        if not getattr(kinematics, "_openr2s_local_delta_ik", False):
            kinematics._solve_damped_least_squares = (
                _VENDORED_KINEMATICS_CLS._solve_damped_least_squares
            )
            kinematics._openr2s_local_delta_ik = True
        return _VENDORED_KINEMATICS_CLS.compute_ik(kinematics, target_pose, q0, **kwargs)

    def reset(self):
        super().reset()
        self._validate_ik_runtime_config()
        current_arm_qpos = self.qpos.detach().clone()
        preferred_from_config = self.config.preferred_arm_qpos
        if preferred_from_config is not None:
            preferred_tensor = torch.as_tensor(
                np.asarray(preferred_from_config, dtype=np.float32).reshape(1, -1),
                device=current_arm_qpos.device,
                dtype=current_arm_qpos.dtype,
            )
            if current_arm_qpos.shape[0] > 1:
                preferred_tensor = preferred_tensor.repeat(current_arm_qpos.shape[0], 1)
            self._preferred_arm_qpos = preferred_tensor
            return
        if (
            self._preferred_arm_qpos is None
            or self._preferred_arm_qpos.shape != current_arm_qpos.shape
        ):
            self._preferred_arm_qpos = current_arm_qpos.clone()
            return
        reset_mask = getattr(self.scene, "_reset_mask", None)
        if reset_mask is None:
            self._preferred_arm_qpos = current_arm_qpos.clone()
            return
        self._preferred_arm_qpos[reset_mask] = current_arm_qpos[reset_mask]

    def _build_preferred_q_seed(self, q0_full):
        preferred_arm_qpos = self._preferred_arm_qpos
        if preferred_arm_qpos is None:
            raise RuntimeError("Preferred arm qpos is not initialized; reset() must run before set_action().")
        q_seed = q0_full.clone()
        current_arm_qpos = q0_full[:, self.active_joint_indices]
        base_gain = float(self.config.ik_seed_preferred_gain)
        if base_gain > 0.0:
            blended = (1.0 - base_gain) * current_arm_qpos + base_gain * preferred_arm_qpos
        else:
            blended = current_arm_qpos.clone()
        joint_names = list(self.config.joint_names)
        if "joint5" in joint_names:
            joint5_idx = joint_names.index("joint5")
            joint5_gain = float(self.config.ik_seed_joint5_gain)
            if joint5_gain > 0.0:
                blended[:, joint5_idx] = (
                    (1.0 - joint5_gain) * blended[:, joint5_idx]
                    + joint5_gain * preferred_arm_qpos[:, joint5_idx]
                )
        q_seed[:, self.active_joint_indices] = blended
        return q_seed

    def _log_ik_joint_diagnostics(self, target_qpos):
        current_arm_qpos = self.qpos.detach()
        delta_arm_qpos = target_qpos - current_arm_qpos
        arm_delta_norm = torch.linalg.norm(delta_arm_qpos, dim=1)
        arm_norm_threshold = float(self.config.diagnostic_arm_delta_norm_threshold_rad)
        joint_names = list(self.config.joint_names)
        if "joint5" in joint_names:
            joint5_idx = joint_names.index("joint5")
            joint5_delta = delta_arm_qpos[:, joint5_idx].abs()
            joint5_threshold = float(self.config.diagnostic_joint5_delta_threshold_rad)
            should_log = torch.logical_or(
                joint5_delta > joint5_threshold,
                arm_delta_norm > arm_norm_threshold,
            )
            for env_idx in torch.nonzero(should_log, as_tuple=False).reshape(-1).tolist():
                print(
                    "[EEIKDebug] Large arm delta detected: "
                    f"env={env_idx} "
                    f"joint5_before={_debug_float(current_arm_qpos[env_idx, joint5_idx]):.4f} "
                    f"joint5_after={_debug_float(target_qpos[env_idx, joint5_idx]):.4f} "
                    f"joint5_delta={_debug_float(delta_arm_qpos[env_idx, joint5_idx]):+.4f} rad "
                    f"arm_delta_norm={_debug_float(arm_delta_norm[env_idx]):.4f} rad "
                    f"preferred_joint5={_debug_float(self._preferred_arm_qpos[env_idx, joint5_idx]):.4f}"
                )
            return
        for env_idx in torch.nonzero(arm_delta_norm > arm_norm_threshold, as_tuple=False).reshape(-1).tolist():
            print(
                "[EEIKDebug] Large arm delta detected: "
                f"env={env_idx} arm_delta_norm={_debug_float(arm_delta_norm[env_idx]):.4f} rad"
            )

    def set_action(self, action):
        self._validate_ik_runtime_config()
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos

        if self.config.use_target:
            prev_ee_pose_at_base = self._target_pose
        else:
            prev_ee_pose_at_base = self.ee_pose_at_base

        self._target_pose = self.compute_target_pose(prev_ee_pose_at_base, action)
        pos_only = type(self.config) == PDEEPoseControllerConfig

        q0_current = self.articulation.get_qpos()
        use_local_target_ik = bool(self.config.use_target and self.config.use_delta)
        q0_seed = self._build_preferred_q_seed(q0_current)
        ik_q0 = q0_current if use_local_target_ik else q0_seed
        current_pose = self.ee_pose_at_base if use_local_target_ik else None
        solver_config = dict(self.config.delta_solver_config) if use_local_target_ik else None
        self._target_qpos = self._compute_ik_target_qpos(
            self._target_pose,
            ik_q0,
            pos_only=pos_only,
            action=action,
            use_delta_ik_solver=self.config.use_delta and not self.config.use_target,
            current_pose=current_pose,
            solver_config=solver_config,
            preferred_qpos=self._preferred_arm_qpos if use_local_target_ik else None,
        )
        if self._target_qpos is None:
            tcp_pose = self.ee_pose_at_base
            message = (
                "[EEIKFailure] compute_ik returned None for LegacyAlignPDEEPoseController. "
                f"frame={self.config.frame} use_target={self.config.use_target} "
                f"target_p={np.array2string(_debug_numpy(self._target_pose.p), precision=4, suppress_small=True)} "
                f"target_q={np.array2string(_debug_numpy(self._target_pose.q), precision=4, suppress_small=True)} "
                f"tcp_p={np.array2string(_debug_numpy(tcp_pose.p), precision=4, suppress_small=True)} "
                f"tcp_q={np.array2string(_debug_numpy(tcp_pose.q), precision=4, suppress_small=True)}"
            )
            print(message)
            if bool(self.config.raise_on_ik_failure):
                raise RuntimeError(message)
            print("[EEIKFailure] Falling back to current controller qpos because raise_on_ik_failure=False")
            self._target_qpos = self._start_qpos
        else:
            self._log_ik_joint_diagnostics(self._target_qpos)
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)


@dataclass
class LegacyAlignPDEEPoseControllerConfig(PDEEPoseControllerConfig):
    frame: str = "ee_align2"
    delta_solver_config: dict = field(
        default_factory=lambda: dict(type="levenberg_marquardt", alpha=1.0)
    )
    preferred_arm_qpos: Optional[Sequence[float]] = None
    ik_seed_preferred_gain: float = 0.35
    ik_seed_joint5_gain: float = 0.85
    diagnostic_joint5_delta_threshold_rad: float = 0.35
    diagnostic_arm_delta_norm_threshold_rad: float = 0.75
    raise_on_ik_failure: bool = True
    controller_cls = LegacyAlignPDEEPoseController
