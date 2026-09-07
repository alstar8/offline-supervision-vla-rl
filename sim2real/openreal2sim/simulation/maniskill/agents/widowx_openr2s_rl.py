"""
WidowX250S RL agent for OpenReal2Sim.
Uses PDEEPoseController (7D EE delta actions) compatible with OpenVLA output.
is_grasping() inherited from WidowX250SOpenR2S.
Controller config mirrors WidowX250SSimpler from RL4VLA.
"""
import numpy as np

from mani_skill.agents.controllers import PDEEPoseControllerConfig, PDJointPosMimicControllerConfig
from mani_skill.agents.registration import register_agent

from .widowx_openr2s import WidowX250SOpenR2S


@register_agent(asset_download_ids=["widowx250s"])
class WidowX250SOpenR2S_RL(WidowX250SOpenR2S):
    """WidowX250S for RL training with OpenVLA.

    Replaces pd_joint_pos (8D) with arm_pd_ee_target_delta_pose_align2 (7D EE delta)
    compatible with OpenVLA action output: [dx, dy, dz, droll, dpitch, dyaw, gripper].
    is_grasping() and build_grasp_pose() are inherited from WidowX250SOpenR2S.
    """

    uid = "widowx250s_openr2s_rl"

    @property
    def _controller_configs(self):
        controller_configs = dict(super()._controller_configs)
        arm_controller = PDEEPoseControllerConfig(
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
        )
        # extra_clearance: gripper is PID on real robot; slight clearance improves
        # contact force when grasping (mirrors WidowX250SSimpler from RL4VLA)
        extra_clearance = 0.001
        gripper_controller = PDJointPosMimicControllerConfig(
            joint_names=self.gripper_joint_names,
            lower=0.015 - extra_clearance,
            upper=0.037 + extra_clearance,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=True,
            drive_mode="force",
        )
        controller_configs.update(
            dict(
                arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos=dict(
                    arm=arm_controller,
                    gripper=gripper_controller,
                )
            )
        )
        return controller_configs
