"""RC5 + Aero Hand agent for OpenReal2Sim / ManiSkill integration."""
import os
from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.utils.structs.actor import Actor

from openreal2sim.simulation.maniskill.rc5_grasp_heuristic import (
    rc5_finger_chain_contact_is_grasping,
)


def _default_asset_dir() -> str:
    repo_local = Path(__file__).resolve().parent.parent / 'robot_assets' / 'rc5_aero_hand' / 'urdf_rc5_right_hand'
    return os.environ.get('RC5_AERO_HAND_ASSET_DIR', str(repo_local))


def _default_urdf_filename() -> str:
    return os.environ.get(
        'RC5_AERO_HAND_URDF_FILENAME',
        'Robot _with_right_hand_colored_visual_continuous.urdf',
    )


@register_agent()
class RC5AeroHandOpenR2S(BaseAgent):
    """RC5 robot with Aero right hand.

    First-stage integration keeps the full articulation but treats the hand as a
    separate subsystem. The arm uses 6 joints and the palm proxy (`prehand`) is
    exposed as the end-effector link.
    """

    uid = 'rc5_aero_hand_openr2s'
    urdf_path = str(Path(_default_asset_dir()) / _default_urdf_filename())
    urdf_config = dict()
    disable_self_collisions = True

    canonical_arm_joint_names = [
        'joint0',
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
    ]
    canonical_hand_joint_names = [
        'right_thumb_cmc_abd',
        'right_thumb_cmc_flex',
        'right_thumb_mcp',
        'right_thumb_ip',
        'right_index_mcp_flex',
        'right_index_pip',
        'right_index_dip',
        'right_middle_mcp_flex',
        'right_middle_pip',
        'right_middle_dip',
        'right_ring_mcp_flex',
        'right_ring_pip',
        'right_ring_dip',
        'right_pinky_mcp_flex',
        'right_pinky_pip',
        'right_pinky_dip',
    ]
    arm_joint_names = list(canonical_arm_joint_names)
    hand_joint_names = list(canonical_hand_joint_names)
    ee_link_name = 'right_tcp_link'

    # Conservative defaults for a first integration pass.
    arm_stiffness = [900, 900, 800, 500, 400, 300]
    arm_damping = [160, 160, 140, 80, 60, 40]
    arm_force_limit = [300, 300, 200, 120, 80, 60]
    hand_stiffness = 40
    hand_damping = 8
    hand_force_limit = 20

    # Preset hand postures for the first OpenVLA-compatible adapter.
    canonical_hand_open_qpos = [
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ]
    canonical_hand_close_qpos = [
        0.45, 0.45, 0.70, 0.70,
        0.90, 1.00, 0.80,
        0.90, 1.00, 0.80,
        0.90, 1.00, 0.80,
        0.90, 1.00, 0.80,
    ]
    hand_open_qpos = list(canonical_hand_open_qpos)
    hand_close_qpos = list(canonical_hand_close_qpos)
    @property
    def _controller_configs(self):
        joint_names = self.arm_joint_names + self.hand_joint_names
        stiffness = list(self.arm_stiffness) + [self.hand_stiffness] * len(self.hand_joint_names)
        damping = list(self.arm_damping) + [self.hand_damping] * len(self.hand_joint_names)
        force_limit = list(self.arm_force_limit) + [self.hand_force_limit] * len(self.hand_joint_names)
        return dict(
            pd_joint_pos=PDJointPosControllerConfig(
                joint_names=joint_names,
                lower=None,
                upper=None,
                stiffness=stiffness,
                damping=damping,
                force_limit=force_limit,
                normalize_action=False,
                drive_mode='force',
            )
        )

    def _reorder_hand_qpos_from_canonical(self, canonical_qpos):
        if len(canonical_qpos) != len(self.canonical_hand_joint_names):
            raise RuntimeError(
                f"RC5 hand preset length mismatch: expected {len(self.canonical_hand_joint_names)}, "
                f"got {len(canonical_qpos)}"
            )
        value_by_name = dict(zip(self.canonical_hand_joint_names, canonical_qpos))
        try:
            return [float(value_by_name[name]) for name in self.hand_joint_names]
        except KeyError as exc:
            raise RuntimeError(f"RC5 active hand joint '{exc.args[0]}' is missing from canonical hand joint list") from exc

    def _after_loading_articulation(self):
        active_joint_names = [joint.name for joint in self.robot.get_active_joints()]
        expected_joint_names = self.canonical_arm_joint_names + self.canonical_hand_joint_names
        if len(active_joint_names) != len(expected_joint_names):
            raise RuntimeError(
                f"RC5 active joint count mismatch: expected {len(expected_joint_names)}, got {len(active_joint_names)}. "
                f"active_joint_names={active_joint_names}"
            )
        if len(set(active_joint_names)) != len(active_joint_names):
            raise RuntimeError(f"RC5 active joint names contain duplicates: {active_joint_names}")
        if set(active_joint_names) != set(expected_joint_names):
            raise RuntimeError(
                "RC5 active joint set mismatch.\n"
                f"expected={expected_joint_names}\n"
                f"actual={active_joint_names}"
            )
        if active_joint_names[: len(self.canonical_arm_joint_names)] != self.canonical_arm_joint_names:
            raise RuntimeError(
                "RC5 arm active joint order mismatch.\n"
                f"expected arm prefix={self.canonical_arm_joint_names}\n"
                f"actual active order={active_joint_names}"
            )

        self.active_joint_names = list(active_joint_names)
        self.arm_joint_names = list(active_joint_names[: len(self.canonical_arm_joint_names)])
        self.hand_joint_names = list(active_joint_names[len(self.canonical_arm_joint_names):])
        self.hand_open_qpos = self._reorder_hand_qpos_from_canonical(self.canonical_hand_open_qpos)
        thumb_joint = [joint for joint in self.robot.get_active_joints() if joint.name == "right_thumb_cmc_abd"][0]
        thumb_limits = thumb_joint.get_limits()
        thumb_upper = float(thumb_limits[0, 1].item() if hasattr(thumb_limits[0, 1], "item") else thumb_limits[0, 1])
        if thumb_upper <= 0:
            raise RuntimeError(f"RC5 right_thumb_cmc_abd upper limit must be positive, got {thumb_upper}")
        self.right_thumb_cmc_abd_upper = thumb_upper
        canonical_close_qpos = list(self.canonical_hand_close_qpos)
        canonical_close_qpos[0] = thumb_upper
        self.hand_close_qpos = self._reorder_hand_qpos_from_canonical(canonical_close_qpos)

        links_map = self.robot.links_map
        self.palm_link = links_map['prehand']
        self.thumb_tip_link = links_map['right_thumb_tip_link']
        self.index_tip_link = links_map['right_index_tip_link']
        self.middle_tip_link = links_map['right_middle_tip_link']
        self.ring_tip_link = links_map['right_ring_tip_link']
        self.pinky_tip_link = links_map['right_pinky_tip_link']
        self.thumb_chain_links = [
            links_map['right_thumb_proximal_link'],
            links_map['right_thumb_distal_link'],
            links_map['right_thumb_tip_link'],
        ]
        self.index_chain_links = [
            links_map['right_index_proximal_link'],
            links_map['right_index_middle_link'],
            links_map['right_index_distal_link'],
            links_map['right_index_tip_link'],
        ]
        self.middle_chain_links = [
            links_map['right_middle_proximal_link'],
            links_map['right_middle_middle_link'],
            links_map['right_middle_distal_link'],
            links_map['right_middle_tip_link'],
        ]
        self.ring_chain_links = [
            links_map['right_ring_proximal_link'],
            links_map['right_ring_middle_link'],
            links_map['right_ring_distal_link'],
            links_map['right_ring_tip_link'],
        ]
        self.pinky_chain_links = [
            links_map['right_pinky_proximal_link'],
            links_map['right_pinky_middle_link'],
            links_map['right_pinky_distal_link'],
            links_map['right_pinky_tip_link'],
        ]

    def _after_init(self):
        if hasattr(self.robot, 'find_link_by_name'):
            self.tcp = self.robot.find_link_by_name(self.ee_link_name)
        else:
            self.tcp = self.robot.links_map[self.ee_link_name]

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """First-stage grasp heuristic for the dexterous hand.

        We treat the object as grasped if any thumb-chain link has contact and
        any non-thumb finger chain also has contact.
        """
        thumb_chain_forces = [
            self.scene.get_pairwise_contact_forces(link, object) for link in self.thumb_chain_links
        ]
        finger_chain_forces = [
            [self.scene.get_pairwise_contact_forces(link, object) for link in self.index_chain_links],
            [self.scene.get_pairwise_contact_forces(link, object) for link in self.middle_chain_links],
            [self.scene.get_pairwise_contact_forces(link, object) for link in self.ring_chain_links],
            [self.scene.get_pairwise_contact_forces(link, object) for link in self.pinky_chain_links],
        ]
        return rc5_finger_chain_contact_is_grasping(
            thumb_chain_forces=thumb_chain_forces,
            finger_chain_forces=finger_chain_forces,
            min_force=min_force,
        )

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)
