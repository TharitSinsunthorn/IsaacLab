from __future__ import annotations
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

class CPGJointPositionAction(JointPositionAction):
    """Joint action modulated by CPG phase signals."""

    def apply_actions(self):
        # Example: modulate processed_actions with CPG output
        cpg_signal = self._compute_cpg_signal()
        modulated_actions = self.processed_actions + cpg_signal

        # Apply to joint position target
        self._asset.set_joint_position_target(modulated_actions, joint_ids=self._joint_ids)

    def _compute_cpg_signal(self):
        # Compute your CPG phase (e.g. sinusoids per leg/joint)
        # Return shape: (num_envs, num_joints)
        phase = torch.sin(torch.linspace(0, 2 * torch.pi, self._num_joints, device=self.device))
        return 0.1 * phase.unsqueeze(0).repeat(self.num_envs, 1)  # just an example

