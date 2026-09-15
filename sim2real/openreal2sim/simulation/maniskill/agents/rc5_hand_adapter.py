"""Controllers for mapping a 1-DOF OpenVLA gripper action to RC5 hand joint presets."""
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import torch
from gymnasium import spaces

from mani_skill.agents.controllers.base_controller import ControllerConfig
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosController
from mani_skill.utils.structs.types import Array, DriveMode


class RCPresetHandController(PDJointPosController):
    """A 1-DOF hand controller that switches between preset open/close qpos targets."""

    config: 'RCPresetHandControllerConfig'

    def _initialize_action_space(self):
        self.single_action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos

        open_qpos = torch.as_tensor(
            self.config.open_qpos, device=self.device, dtype=self.qpos.dtype
        ).view(1, -1)
        close_qpos = torch.as_tensor(
            self.config.close_qpos, device=self.device, dtype=self.qpos.dtype
        ).view(1, -1)
        hold_mask = (torch.abs(action[..., :1]) <= float(self.config.hold_epsilon)).expand(-1, open_qpos.shape[-1])
        open_mask = (action[..., :1] > float(self.config.hold_epsilon)).expand(-1, open_qpos.shape[-1])
        target_qpos = torch.where(open_mask, open_qpos, close_qpos)
        self._target_qpos = torch.where(hold_mask, self._start_qpos, target_qpos)

        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)


@dataclass
class RCPresetHandControllerConfig(ControllerConfig):
    open_qpos: Sequence[float]
    close_qpos: Sequence[float]
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    lower: Union[None, float, Sequence[float]] = None
    upper: Union[None, float, Sequence[float]] = None
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    use_delta: bool = False
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = False
    drive_mode: Union[Sequence[DriveMode], DriveMode] = 'force'
    hold_epsilon: float = 1e-4
    controller_cls = RCPresetHandController


class RCLevelHandController(PDJointPosController):
    """A 1-DOF hand controller mapping an absolute openness level to hand qpos.

    Action o in [0, 1]: 1 = fully open (open_qpos), 0 = fully closed (close_qpos).
    The target is the linear interpolation close_qpos + o * (open_qpos - close_qpos),
    so any level in between produces a proportional aperture. The pipeline quantizes
    commands to {0.0, 0.2, ..., 1.0}; the controller itself accepts any value in [0, 1].
    """

    config: 'RCLevelHandControllerConfig'

    def _initialize_action_space(self):
        self.single_action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos

        open_qpos = torch.as_tensor(
            self.config.open_qpos, device=self.device, dtype=self.qpos.dtype
        ).view(1, -1)
        close_qpos = torch.as_tensor(
            self.config.close_qpos, device=self.device, dtype=self.qpos.dtype
        ).view(1, -1)
        openness = action[..., :1].clamp(0.0, 1.0).to(self.qpos.dtype)
        self._target_qpos = close_qpos + openness * (open_qpos - close_qpos)

        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)


@dataclass
class RCLevelHandControllerConfig(ControllerConfig):
    open_qpos: Sequence[float]
    close_qpos: Sequence[float]
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    lower: Union[None, float, Sequence[float]] = None
    upper: Union[None, float, Sequence[float]] = None
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    use_delta: bool = False
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = False
    drive_mode: Union[Sequence[DriveMode], DriveMode] = 'force'
    controller_cls = RCLevelHandController
