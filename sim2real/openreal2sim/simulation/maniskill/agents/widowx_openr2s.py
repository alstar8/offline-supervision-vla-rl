"""
WidowX250S agent for OpenReal2Sim. Extends ManiSkill WidowX250S with tcp and build_grasp_pose.
Adapted from RL4VLA/ManiSkill agents/robots/widowx/widowx.py
"""
import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor


@register_agent(asset_download_ids=["widowx250s"])
class WidowX250SOpenR2S(BaseAgent):
    """WidowX250S with tcp and build_grasp_pose for OpenReal2Sim motion planning."""

    uid = "widowx250s_openr2s"
    urdf_path = f"{ASSET_DIR}/robots/widowx/wx250s.urdf"
    urdf_config = dict()

    arm_joint_names = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    ]
    gripper_joint_names = ["left_finger", "right_finger"]
    ee_link_name = "ee_gripper_link"

    # RL4VLA WidowX250SBridgeDataset: для плотного захвата
    arm_stiffness = [1169.79, 730.0, 808.46, 1229.13, 1272.28, 1056.33]
    arm_damping = [330.0, 180.0, 152.12, 309.62, 201.05, 269.51]
    arm_force_limit = [200, 200, 100, 100, 100, 100]
    gripper_stiffness = 1000
    gripper_damping = 200
    gripper_force_limit = 60

    @property
    def _controller_configs(self):
        joint_names = self.arm_joint_names + self.gripper_joint_names
        stiffness = list(self.arm_stiffness) + [self.gripper_stiffness] * 2
        damping = list(self.arm_damping) + [self.gripper_damping] * 2
        force_limit = list(self.arm_force_limit) + [self.gripper_force_limit] * 2
        return dict(
            pd_joint_pos=PDJointPosControllerConfig(
                joint_names=joint_names,
                lower=None,  # use URDF limits
                upper=None,
                stiffness=stiffness,
                damping=damping,
                force_limit=force_limit,
                normalize_action=False,
                drive_mode="force",
            ),
        )

    def _after_loading_articulation(self):
        self.finger1_link = self.robot.links_map["left_finger_link"]
        self.finger2_link = self.robot.links_map["right_finger_link"]

    def _after_init(self):
        if hasattr(self.robot, "find_link_by_name"):
            self.tcp = self.robot.find_link_by_name(self.ee_link_name)
        else:
            self.tcp = self.robot.links_map[self.ee_link_name]

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object."""
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (ee_gripper_link). Same convention as Panda."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)
