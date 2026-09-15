"""OpenVLA-compatible RL agent for RC5 + Aero Hand."""
import numpy as np

from mani_skill.agents.registration import register_agent

from .legacy_pd_ee_pose import LegacyAlignPDEEPoseControllerConfig
from .rc5_aero_hand_openr2s import RC5AeroHandOpenR2S
from .rc5_hand_adapter import RCLevelHandControllerConfig


@register_agent()
class RC5AeroHandOpenR2S_RL(RC5AeroHandOpenR2S):
    """RC5 agent with a 7D OpenVLA-compatible interface.

    Action layout matches WidowX RL mode:
    [dx, dy, dz, droll, dpitch, dyaw, gripper]
    where the last scalar is an absolute hand-openness level in [0, 1]
    (1 = fully open, 0 = fully closed), quantized to 0.2 steps by the pipeline.
    """

    uid = 'rc5_aero_hand_openr2s_rl'

    @property
    def _controller_configs(self):
        controller_configs = dict(super()._controller_configs)
        arm_controller = LegacyAlignPDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-1.0,
            pos_upper=1.0,
            rot_lower=-np.pi / 2,
            rot_upper=np.pi / 2,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            normalize_action=False,
            use_delta=True,
            use_target=True,
            frame="ee_align2",
            delta_solver_config=dict(
                type="levenberg_marquardt",
                mode="hierarchical_position_first",
                alpha=0.35,
                damping=0.01,
                position_damping=0.01,
                orientation_alpha=0.15,
                orientation_alpha_near_target=0.75,
                orientation_position_threshold=0.02,
                orientation_damping=0.02,
                posture_gain=0.10,
                posture_gain_near_target=0.02,
                posture_joint_weights=[0.25, 0.25, 0.25, 0.25, 1.50, 1.50],
            ),
            preferred_arm_qpos=None,
            ik_seed_preferred_gain=0.35,
            ik_seed_joint5_gain=0.85,
            diagnostic_joint5_delta_threshold_rad=0.35,
            diagnostic_arm_delta_norm_threshold_rad=0.75,
            raise_on_ik_failure=False,
        )
        hand_controller = RCLevelHandControllerConfig(
            joint_names=self.hand_joint_names,
            open_qpos=self.hand_open_qpos,
            close_qpos=self.hand_close_qpos,
            stiffness=[self.hand_stiffness] * len(self.hand_joint_names),
            damping=[self.hand_damping] * len(self.hand_joint_names),
            force_limit=[self.hand_force_limit] * len(self.hand_joint_names),
            normalize_action=False,
            drive_mode='force',
        )
        controller_configs.update(
            dict(
                arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos=dict(
                    arm=arm_controller,
                    gripper=hand_controller,
                )
            )
        )
        return controller_configs
