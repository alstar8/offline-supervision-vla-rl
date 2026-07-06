from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import sapien
import torch
from gymnasium import spaces

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import PDEEPoseControllerConfig
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosController, PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.utils.structs.actor import Actor


REPO_ROOT = Path(__file__).resolve().parents[5]
RC5_URDF_PATH = REPO_ROOT / "real2sim" / "assets" / "rc5" / "Robot _with_right_hand.urdf"


class RC5HandPoseController(PDJointPosController):
    config: "RC5HandPoseControllerConfig"

    def _initialize_action_space(self):
        self.single_action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def set_action(self, action):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos

        grasp_amount = action[:, :1]
        open_qpos = torch.as_tensor(
            self.config.open_qpos,
            dtype=self.qpos.dtype,
            device=self.qpos.device,
        ).view(1, -1)
        closed_qpos = torch.as_tensor(
            self.config.closed_qpos,
            dtype=self.qpos.dtype,
            device=self.qpos.device,
        ).view(1, -1)
        self._target_qpos = open_qpos + grasp_amount * (closed_qpos - open_qpos)
        self.set_drive_targets(self._target_qpos)


@dataclass
class RC5HandPoseControllerConfig(PDJointPosControllerConfig):
    open_qpos: Union[Sequence[float], np.ndarray, None] = None
    closed_qpos: Union[Sequence[float], np.ndarray, None] = None
    controller_cls = RC5HandPoseController


@register_agent()
class RC5AeroHandRight(BaseAgent):
    uid = "rc5_aero_hand_right"
    urdf_path = str(RC5_URDF_PATH)
    urdf_config = dict(
        _materials=dict(
            fingertip=dict(
                static_friction=500.0,
                dynamic_friction=500.0,
                restitution=0.0,
            )
        ),
        link=dict(
            right_thumb_tip_link=dict(
                material="fingertip",
                patch_radius=0.30,
                min_patch_radius=0.15,
            ),
            right_index_tip_link=dict(
                material="fingertip",
                patch_radius=0.30,
                min_patch_radius=0.15,
            ),
            right_middle_tip_link=dict(
                material="fingertip",
                patch_radius=0.30,
                min_patch_radius=0.15,
            ),
        ),
    )

    arm_joint_names = [
        "joint0",
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
    ]
    # Keep the PPO/OpenVLA action contract widowx-compatible for now:
    # one scalar drives a coarse "close/open hand" synergy across flexion joints.
    gripper_joint_names = [
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
    # Use the dedicated tool TCP frame rather than the physical prehand mount.
    ee_link_name = "right_tcp_link"

    arm_stiffness = [900.0, 900.0, 800.0, 450.0, 250.0, 180.0]
    arm_damping = [220.0, 220.0, 180.0, 100.0, 60.0, 40.0]
    arm_force_limit = [120.0, 120.0, 100.0, 70.0, 50.0, 30.0]
    arm_friction = 0.0

    gripper_stiffness = 900.0
    gripper_damping = 120.0
    gripper_force_limit = 200.0
    gripper_open_qpos = np.deg2rad(
        [
            100.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    ).astype(np.float32)
    gripper_closed_qpos = np.deg2rad(
        [
            100.0,
            55.0,
            30.0,
            30.0,
            60.0,
            60.0,
            60.0,
            60.0,
            60.0,
            60.0,
            60.0,
            60.0,
            60.0,
            60.0,
            60.0,
            60.0,
        ]
    ).astype(np.float32)

    def get_state(self):
        state = super().get_state()
        state["ee_pose"] = self.robot.find_link_by_name(self.ee_link_name).pose
        return state

    def _after_loading_articulation(self):
        self.thumb_tip_link = self.robot.links_map["right_thumb_tip_link"]
        self.index_tip_link = self.robot.links_map["right_index_tip_link"]
        self.finger_tip_links = [
            self.index_tip_link,
            self.robot.links_map["right_middle_tip_link"],
            self.robot.links_map["right_ring_tip_link"],
            self.robot.links_map["right_pinky_tip_link"],
        ]
        # Prevent the hand from fighting impossible self-collisions while closing.
        hand_links = [
            "right_base_link",
            "right_t_link",
            "right_thumb_mcp_link",
            "right_thumb_proximal_link",
            "right_thumb_distal_link",
            "right_thumb_tip_link",
            "right_index_proximal_link",
            "right_index_middle_link",
            "right_index_distal_link",
            "right_index_tip_link",
            "right_middle_proximal_link",
            "right_middle_middle_link",
            "right_middle_distal_link",
            "right_middle_tip_link",
            "right_ring_proximal_link",
            "right_ring_middle_link",
            "right_ring_distal_link",
            "right_ring_tip_link",
            "right_pinky_proximal_link",
            "right_pinky_middle_link",
            "right_pinky_distal_link",
            "right_pinky_tip_link",
        ]
        for link_name in hand_links:
            self.robot.links_map[link_name].set_collision_group_bit(group=2, bit_idx=31, bit=1)
        # Compatibility shim for existing WidowX-oriented task code that
        # expects a pinch center from two gripper links.
        self.finger1_link = self.thumb_tip_link
        self.finger2_link = self.index_tip_link

    def _after_init(self):
        self.tcp = self.robot.find_link_by_name(self.ee_link_name)

    @property
    def ee_pose_at_robot_base(self):
        to_base = self.robot.pose.inv()
        return to_base * self.tcp.pose

    def _contact_force_norm(self, link, obj: Actor):
        force = self.scene.get_pairwise_contact_forces(link, obj)
        return torch.linalg.norm(force, axis=1)

    def is_grasping(self, object: Actor, min_force=0.25, min_non_thumb_contacts=1):
        thumb_contact = self._contact_force_norm(self.thumb_tip_link, object) >= min_force

        other_contacts = [
            self._contact_force_norm(link, object) >= min_force
            for link in self.finger_tip_links
        ]
        other_contacts = torch.stack(other_contacts, dim=1)
        supported = other_contacts.sum(dim=1) >= min_non_thumb_contacts

        return thumb_contact & supported

    @property
    def _controller_configs(self):
        arm_joint_common_kwargs = dict(
            joint_names=self.arm_joint_names,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            normalize_action=False,
        )
        arm_ee_common_kwargs = dict(
            **arm_joint_common_kwargs,
            pos_lower=-1.0,
            pos_upper=1.0,
            rot_lower=-np.pi / 2,
            rot_upper=np.pi / 2,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=True,
        )
        arm_joint_pos = PDJointPosControllerConfig(
            **arm_joint_common_kwargs,
            lower=None,
            upper=None,
            use_delta=False,
        )
        arm_pd_ee_target_delta_pose_align2 = PDEEPoseControllerConfig(
            **arm_ee_common_kwargs, use_target=True
        )
        arm_pd_ee_delta_pose_align2 = PDEEPoseControllerConfig(
            **arm_ee_common_kwargs, use_target=False
        )

        gripper_pd_joint_pos = RC5HandPoseControllerConfig(
            joint_names=self.gripper_joint_names,
            lower=0.0,
            upper=1.0,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            open_qpos=self.gripper_open_qpos,
            closed_qpos=self.gripper_closed_qpos,
            normalize_action=True,
            drive_mode="force",
        )

        controller = dict(
            arm=arm_pd_ee_target_delta_pose_align2,
            gripper=gripper_pd_joint_pos,
        )
        controller_non_target = dict(
            arm=arm_pd_ee_delta_pose_align2,
            gripper=gripper_pd_joint_pos,
        )
        controller_joint = dict(
            arm=arm_joint_pos,
            gripper=gripper_pd_joint_pos,
        )
        return dict(
            arm_pd_joint_pos=arm_joint_pos,
            arm_pd_ee_delta_pose_align2=arm_pd_ee_delta_pose_align2,
            arm_pd_ee_target_delta_pose_align2=arm_pd_ee_target_delta_pose_align2,
            arm_pd_joint_pos_gripper_pd_joint_pos=controller_joint,
            arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos=controller_non_target,
            arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos=controller,
        )
