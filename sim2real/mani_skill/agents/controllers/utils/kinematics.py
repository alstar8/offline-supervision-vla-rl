"""
Code for kinematics utilities on CPU/GPU
"""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from typing import List

try:
    import pytorch_kinematics as pk
except ImportError:
    raise ImportError(
        "pytorch_kinematics_ms not installed. Install with pip install pytorch_kinematics_ms"
    )
import torch
from sapien.wrapper.pinocchio_model import PinocchioModel

from mani_skill.utils import common
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.articulation_joint import ArticulationJoint
from mani_skill.utils.structs.pose import Pose

# currently fast_kinematics has some bugs on some systems so we use the slower pytorch kinematics package instead.
# try:
#     import fast_kinematics
# except:
#     # not all systems support the fast_kinematics package at the moment
#     fast_kinematics = None


class Kinematics:
    @staticmethod
    def _solve_damped_least_squares(
        jacobian: torch.Tensor,
        error: torch.Tensor,
        damping: float,
    ):
        batch_size = jacobian.shape[0]
        joint_dim = jacobian.shape[-1]
        j_t = jacobian.transpose(1, 2)
        eye = torch.eye(joint_dim, device=jacobian.device, dtype=jacobian.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        lhs = torch.bmm(j_t, jacobian) + float(damping) * eye
        rhs = torch.bmm(j_t, error.unsqueeze(-1))
        delta_joint_pos = torch.linalg.solve(lhs, rhs).squeeze(-1)
        pseudo_inverse = torch.linalg.solve(lhs, j_t)
        return delta_joint_pos, pseudo_inverse

    def __init__(
        self,
        urdf_path: str,
        end_link_name: str,
        articulation: Articulation,
        active_joint_indices: torch.Tensor,
    ):
        """
        Initialize the kinematics solver. It will be run on whichever device the articulation is on.

        Args:
            urdf_path (str): path to the URDF file
            end_link_name (str): name of the end-effector link
            articulation (Articulation): the articulation object
            active_joint_indices (torch.Tensor): indices of the active joints that can be controlled
        """
        self.urdf_path = urdf_path
        self.end_link = articulation.links_map[end_link_name]
        self.end_link_idx = articulation.links.index(self.end_link)
        self.active_joint_indices = active_joint_indices
        self.articulation = articulation
        self.device = articulation.device
        # note that everything past the end-link is ignored. Any joint whose ancestor is self.end_link is ignored
        cur_link = self.end_link
        active_ancestor_joints: List[ArticulationJoint] = []
        while cur_link is not None:
            if cur_link.joint.active_index is not None:
                active_ancestor_joints.append(cur_link.joint)
            cur_link = cur_link.joint.parent_link
        active_ancestor_joints = active_ancestor_joints[::-1]
        self.active_ancestor_joints = active_ancestor_joints

        # initially self.active_joint_indices references active joints that are controlled.
        # we also make the assumption that the active index is the same across all parallel managed joints
        self.active_ancestor_joint_idxs = [
            (x.active_index[0]).cpu().item() for x in self.active_ancestor_joints
        ]
        self.controlled_joints_idx_in_qmask = [
            self.active_ancestor_joint_idxs.index(idx)
            for idx in self.active_joint_indices
        ]
        if self.device.type == "cuda":
            self.use_gpu_ik = True
            self._setup_gpu()
        else:
            self.use_gpu_ik = False
            self._setup_cpu()

    def _setup_cpu(self):
        """setup the kinematics solvers on the CPU"""
        self.use_gpu_ik = False
        # NOTE (stao): currently using the pinnochio that comes packaged with SAPIEN
        self.qmask = torch.zeros(
            self.articulation.max_dof, dtype=bool, device=self.device
        )
        self.pmodel: PinocchioModel = self.articulation._objs[
            0
        ].create_pinocchio_model()
        self.qmask[self.active_joint_indices] = 1

    def _setup_gpu(self):
        """setup the kinematics solvers on the GPU"""
        self.use_gpu_ik = True
        with open(self.urdf_path, "rb") as f:
            urdf_str = f.read()

        # NOTE (stao): it seems that the pk library currently always outputs some complaints if there are unknown attributes in a URDF. Hide it with this contextmanager here
        @contextmanager
        def suppress_stdout_stderr():
            """A context manager that redirects stdout and stderr to devnull"""
            with open(devnull, "w") as fnull:
                with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
                    yield (err, out)

        with suppress_stdout_stderr():
            self.pk_chain = pk.build_serial_chain_from_urdf(
                urdf_str,
                end_link_name=self.end_link.name,
            ).to(device=self.device)
        lim = torch.tensor(self.pk_chain.get_joint_limits(), device=self.device)
        self.pik = pk.PseudoInverseIK(
            self.pk_chain,
            joint_limits=lim.T,
            early_stopping_any_converged=True,
            max_iterations=200,
            num_retries=1,
        )

        self.qmask = torch.zeros(
            len(self.active_ancestor_joints), dtype=bool, device=self.device
        )
        self.qmask[self.controlled_joints_idx_in_qmask] = 1

    def compute_ik(
        self,
        target_pose: Pose,
        q0: torch.Tensor,
        pos_only: bool = False,
        action=None,
        use_delta_ik_solver: bool = False,
        current_pose: Pose = None,
        solver_config: dict = None,
        preferred_qpos: torch.Tensor = None,
    ):
        """Given a target pose, via inverse kinematics compute the target joint positions that will achieve the target pose

        Args:
            target_pose (Pose): target pose of the end effector in the world frame. note this is not relative to the robot base frame!
            q0 (torch.Tensor): initial joint positions of every active joint in the articulation
            pos_only (bool): if True, only the position of the end link is considered in the IK computation
            action (torch.Tensor): delta action to be applied to the articulation. Used for fast delta IK solutions on the GPU.
            use_delta_ik_solver (bool): If true, returns the target joint positions that correspond with a delta IK solution. This is specifically
                used for GPU simulation to determine which GPU IK algorithm to use.
        """
        if self.use_gpu_ik:
            q0 = q0[:, self.active_ancestor_joint_idxs]
            if not use_delta_ik_solver:
                if current_pose is not None:
                    if solver_config is None:
                        solver_config = dict(type="levenberg_marquardt", alpha=1.0, damping=1e-4)
                    solver_type = str(solver_config.get("type", "levenberg_marquardt"))
                    solver_mode = str(solver_config.get("mode", "single_stage"))
                    alpha = float(solver_config.get("alpha", 1.0))
                    damping = float(solver_config.get("damping", 1e-4))
                    q0_controlled = q0[:, self.qmask]
                    jacobian = self.pk_chain.jacobian(q0)[:, :, self.qmask]
                    delta_pos = target_pose.p - current_pose.p
                    if preferred_qpos is not None:
                        preferred_qpos = preferred_qpos.to(device=self.device, dtype=q0_controlled.dtype)
                        if preferred_qpos.ndim == 1:
                            preferred_qpos = preferred_qpos.reshape(1, -1)
                        if preferred_qpos.shape[0] == 1 and q0_controlled.shape[0] > 1:
                            preferred_qpos = preferred_qpos.repeat(q0_controlled.shape[0], 1)
                        if preferred_qpos.shape != q0_controlled.shape:
                            raise RuntimeError(
                                f"preferred_qpos shape mismatch: expected {tuple(q0_controlled.shape)}, got {tuple(preferred_qpos.shape)}"
                            )
                    if pos_only:
                        pose_error = delta_pos
                        jacobian = jacobian[:, 0:3, :]
                    elif solver_mode == "hierarchical_position_first":
                        position_damping = float(solver_config.get("position_damping", damping))
                        orientation_damping = float(solver_config.get("orientation_damping", damping))
                        orientation_alpha = float(solver_config["orientation_alpha"])
                        orientation_alpha_near_target = float(solver_config["orientation_alpha_near_target"])
                        orientation_position_threshold = float(solver_config["orientation_position_threshold"])
                        posture_gain = float(solver_config["posture_gain"])
                        posture_gain_near_target = float(solver_config["posture_gain_near_target"])
                        posture_joint_weights = solver_config["posture_joint_weights"]
                        posture_joint_weights = torch.as_tensor(
                            posture_joint_weights,
                            device=self.device,
                            dtype=q0_controlled.dtype,
                        ).reshape(1, -1)
                        if posture_joint_weights.shape[-1] != q0_controlled.shape[-1]:
                            raise RuntimeError(
                                "posture_joint_weights length mismatch: "
                                f"expected {q0_controlled.shape[-1]}, got {posture_joint_weights.shape[-1]}"
                            )
                        if posture_joint_weights.shape[0] == 1 and q0_controlled.shape[0] > 1:
                            posture_joint_weights = posture_joint_weights.repeat(q0_controlled.shape[0], 1)
                        delta_quat = rotation_conversions.quaternion_multiply(
                            target_pose.q,
                            rotation_conversions.quaternion_invert(current_pose.q),
                        )
                        orientation_error = rotation_conversions.quaternion_to_axis_angle(delta_quat)
                        position_error_norm = torch.linalg.norm(delta_pos, dim=1, keepdim=True)
                        near_target_mask = position_error_norm <= orientation_position_threshold

                        jacobian_pos = jacobian[:, 0:3, :]
                        jacobian_rot = jacobian[:, 3:6, :]
                        delta_q_pos, pos_pinv = self._solve_damped_least_squares(
                            jacobian_pos,
                            delta_pos,
                            position_damping,
                        )
                        identity = torch.eye(
                            jacobian_pos.shape[-1],
                            device=self.device,
                            dtype=jacobian_pos.dtype,
                        ).unsqueeze(0).repeat(jacobian_pos.shape[0], 1, 1)
                        nullspace_pos = identity - torch.bmm(pos_pinv, jacobian_pos)
                        jacobian_rot_null = torch.bmm(jacobian_rot, nullspace_pos)
                        orientation_residual = orientation_error - torch.bmm(
                            jacobian_rot,
                            delta_q_pos.unsqueeze(-1),
                        ).squeeze(-1)
                        delta_q_rot, rot_pinv = self._solve_damped_least_squares(
                            jacobian_rot_null,
                            orientation_residual,
                            orientation_damping,
                        )
                        orientation_alpha_batch = torch.where(
                            near_target_mask,
                            torch.full_like(position_error_norm, orientation_alpha_near_target),
                            torch.full_like(position_error_norm, orientation_alpha),
                        )
                        delta_joint_pos = delta_q_pos + orientation_alpha_batch * delta_q_rot
                        if preferred_qpos is not None and posture_gain > 0.0:
                            nullspace_rot = nullspace_pos - torch.bmm(rot_pinv, jacobian_rot_null)
                            posture_error = posture_joint_weights * (preferred_qpos - q0_controlled)
                            posture_gain_batch = torch.where(
                                near_target_mask,
                                torch.full_like(position_error_norm, posture_gain_near_target),
                                torch.full_like(position_error_norm, posture_gain),
                            )
                            delta_joint_pos = delta_q_pos + orientation_alpha_batch * delta_q_rot + posture_gain_batch * torch.bmm(
                                nullspace_rot,
                                posture_error.unsqueeze(-1),
                            ).squeeze(-1)
                        return q0_controlled + alpha * delta_joint_pos
                    else:
                        delta_quat = rotation_conversions.quaternion_multiply(
                            target_pose.q,
                            rotation_conversions.quaternion_invert(current_pose.q),
                        )
                        delta_rot = rotation_conversions.matrix_to_euler_angles(
                            rotation_conversions.quaternion_to_matrix(delta_quat), "XYZ"
                        )
                        pose_error = torch.cat([delta_pos, delta_rot], dim=1)
                    if solver_type == "levenberg_marquardt":
                        delta_joint_pos, _ = self._solve_damped_least_squares(jacobian, pose_error, damping)
                    elif solver_type == "pseudo_inverse":
                        delta_joint_pos = (
                            torch.linalg.pinv(jacobian) @ pose_error.unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        raise ValueError(f"Unsupported IK solver type: {solver_type}")
                    return q0_controlled + alpha * delta_joint_pos
                tf = pk.Transform3d(
                    pos=target_pose.p,
                    rot=target_pose.q,
                    device=self.device,
                )
                self.pik.initial_config = q0  # shape (num_retries, active_ancestor_dof)
                result = self.pik.solve(
                    tf
                )  # produce solutions in shape (B, num_retries/initial_configs, active_ancestor_dof)
                # TODO return mask for invalid solutions. CPU returns None at the moment
                return result.solutions[:, 0, :]
            else:
                jacobian = self.pk_chain.jacobian(q0)
                # code commented out below is the fast kinematics method
                # jacobian = (
                #     self.fast_kinematics_model.jacobian_mixed_frame_pytorch(
                #         self.articulation.get_qpos()[:, self.active_ancestor_joint_idxs]
                #     )
                #     .view(-1, len(self.active_ancestor_joints), 6)
                #     .permute(0, 2, 1)
                # )
                # jacobian = jacobian[:, :, self.qmask]
                if pos_only:
                    jacobian = jacobian[:, 0:3]

                # NOTE (stao): this method of IK is from https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf by Samuel R. Buss
                delta_joint_pos = torch.linalg.pinv(jacobian) @ action.unsqueeze(-1)
                return q0 + delta_joint_pos.squeeze(-1)
        else:
            result, success, error = self.pmodel.compute_inverse_kinematics(
                self.end_link_idx,
                target_pose.sp,
                initial_qpos=q0.cpu().numpy()[0],
                active_qmask=self.qmask,
                max_iterations=100,
            )
            if success:
                return common.to_tensor(
                    [result[self.active_ancestor_joint_idxs]], device=self.device
                )
            else:
                return None
