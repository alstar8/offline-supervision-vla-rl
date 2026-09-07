"""
WidowX250SBridgeDatasetFlatTable for OpenReal2Sim.
Matches RL4VLA bridge_dataset_eval: same camera, gripper. Uses unique uid to avoid conflict
with RL4VLA's agent (which only has arm_pd_ee_target_delta_pose_align2, no pd_joint_pos).
"""
import numpy as np
import sapien
from mani_skill.agents.controllers import (
    PDEEPoseControllerConfig,
    PDJointPosMimicControllerConfig,
)
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig

from .widowx_openr2s import WidowX250SOpenR2S


@register_agent(asset_download_ids=["widowx250s"])
class WidowX250SBridgeDatasetFlatTableOpenR2S(WidowX250SOpenR2S):
    """WidowX250S for Bridge flat_table. pd_joint_pos for motion planning.
    RL4VLA widowx250s_bridgedataset_flat_table conflicts (only EE controller)."""

    uid = "widowx250s_bridgedataset_flat_table_openr2s"

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

    @property
    def _sensor_configs(self):
        """3rd_view_camera — same pose as RL4VLA Bridge dataset (logitech C920)."""
        return [
            CameraConfig(
                uid="3rd_view_camera",
                pose=sapien.Pose(
                    [0.00, -0.16, 0.36],
                    [0.8992917, -0.09263245, 0.35892478, 0.23209205],
                ),
                width=640,
                height=480,
                entity_uid="base_link",
                intrinsic=np.array(
                    [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
                ),
            ),
        ]
