from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
from pxr import UsdPhysics

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.operational_space import OperationalSpaceController
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sim.utils import find_matching_prims

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class QuadrupedDiffIKAction(ActionTerm):
    r"""Quadruped Differential IK Action that handles multiple legs using task-space control."""

    cfg: actions_cfg.QuadrupedDiffIKActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.QuadrupedDiffIKActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self.legs = {}
        self.controllers = {}
        for leg_name, leg_cfg in self.cfg.legs.items():
            # resolve the joints over which the action term is applied
            joint_ids, joint_names = self._asset.find_joints(leg_cfg.joint_names)
            num_joints = len(joint_ids)
            # parse the body index
            body_ids, body_names = self._asset.find_bodies(leg_cfg.body_name)
            if len(body_ids) != 1:
                raise ValueError(
                    f"Expected one match for the body name: {leg_cfg.body_name}. Found {len(body_ids)}: {body_names}."
                )
            # save only the first body index
            body_idx = body_ids[0]
            body_name = body_names[0]
            # check if articulation is fixed-base
            # if fixed-base then the jacobian for the base is not computed
            # this means that number of bodies is one less than the articulation's number of bodies
            if self._asset.is_fixed_base:
                jacobi_body_idx = body_idx - 1
                jacobi_joint_ids = joint_ids
            else:
                jacobi_body_idx = body_idx
                jacobi_joint_ids = [i + 6 for i in joint_ids]

            # log info for debugging
            omni.log.info(
                f"Resolved joint names for the action term {self.__class__.__name__}:"
                f" {joint_names} [{joint_ids}]"
            )
            omni.log.info(
                f"Resolved body name for the action term {self.__class__.__name__}: {body_name} [{body_idx}]"
            )
            # Avoid indexing across all joints for efficiency
            if num_joints == self._asset.num_joints:
                joint_ids = slice(None)

            # create the differential IK controller
            ik_controller = DifferentialIKController(
                cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device
            )

            self.legs[leg_name] = {
                "joint_ids": joint_ids,
                "jacobi_joint_ids": jacobi_joint_ids,
                "body_idx": body_idx,
                "jacobi_body_idx": jacobi_body_idx,
                "offset_pos": torch.tensor(leg_cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
                if leg_cfg.body_offset else None,
                "offset_rot": torch.tensor(leg_cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
                if leg_cfg.body_offset else None,
            }
            self.controllers[leg_name] = ik_controller

        # create tensors for raw and processed actions
        action_dim = sum([ctrl.action_dim for ctrl in self.controllers.values()])
        self._raw_actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        # save the scale as tensors
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        self.leg_bounds = {
            "FL": {"x": (0.15, 0.25), "y": (0.1, 0.20), "z": (-0.4, -0.1)},
            "FR": {"x": (0.15, 0.25), "y": (-0.20, -0.1), "z": (-0.4, -0.1)},
            "RL": {"x": (-0.25, -0.15), "y": (0.1, 0.20), "z": (-0.4, -0.1)},
            "RR": {"x": (-0.25, -0.15), "y": (-0.20, -0.1), "z": (-0.4, -0.1)}
        }

        # # parse clip
        # if self.cfg.clip is not None:
        #     if isinstance(cfg.clip, dict):
        #         self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
        #             self.num_envs, self.action_dim, 1
        #         )
        #         index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
        #         self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
        #     else:
        #         raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        leg_index = 0
        for leg_name, ctrl in self.controllers.items():
            leg_cfg = self.legs[leg_name]
            ee_pos_curr, ee_quat_curr = self._compute_frame_pose(leg_cfg)
            dim = ctrl.action_dim

            action_slice = self._processed_actions[:, leg_index:leg_index+dim]

            # Clamp action based on leg-specific bounds
            bounds = self.leg_bounds[leg_name]
            for i, axis in enumerate(["x", "y", "z"]):
                min_val, max_val = bounds[axis]
                action_slice[:, i] = torch.clamp(action_slice[:, i], min=min_val, max=max_val)

            ctrl.set_command(action_slice, ee_pos_curr, ee_quat_curr)
            leg_index += dim

    def apply_actions(self):
        all_joint_ids = []
        all_joint_targets = torch.zeros_like(self._asset.data.joint_pos)

        for leg_name, ctrl in self.controllers.items():
            leg_cfg = self.legs[leg_name]
            ee_pos_curr, ee_quat_curr = self._compute_frame_pose(leg_cfg)
            joint_pos = self._asset.data.joint_pos[:, leg_cfg["joint_ids"]]
            jacobian = self._compute_frame_jacobian(leg_cfg)
            joint_pos_des = ctrl.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)

            # Write the per-leg joint positions into the full array
            all_joint_targets[:, leg_cfg["joint_ids"]] = joint_pos_des
            all_joint_ids.extend(leg_cfg["joint_ids"])  # Optional if you want to update partial

        # Set all joint targets at once
        self._asset.set_joint_position_target(all_joint_targets)


    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    """
    Helper functions.
    """

    def _compute_frame_pose(self, leg_cfg) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pos_w = self._asset.data.body_pos_w[:, leg_cfg["body_idx"]]
        ee_quat_w = self._asset.data.body_quat_w[:, leg_cfg["body_idx"]]
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        # account for the offset
        if leg_cfg["offset_pos"] and leg_cfg["offset_rot"] is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b, ee_quat_b, leg_cfg["offset_pos"], leg_cfg["offset_rot"]
            )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self, leg_cfg):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        jacobian = self._asset.root_physx_view.get_jacobians()[:, leg_cfg["jacobi_body_idx"], :, leg_cfg["jacobi_joint_ids"]]
        base_rot = self._asset.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        if leg_cfg["offset_pos"] and leg_cfg["offset_rot"] is not None:
            jacobian[:, 0:3, :] += torch.bmm(
                -math_utils.skew_symmetric_matrix(leg_cfg["offset_pos"]), jacobian[:, 3:, :]
            )
            jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(leg_cfg["offset_rot"]), jacobian[:, 3:, :])

        return jacobian
