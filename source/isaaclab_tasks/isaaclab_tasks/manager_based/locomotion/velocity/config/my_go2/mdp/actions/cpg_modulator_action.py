from __future__ import annotations
import torch
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict

import omni.log

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sim.utils import find_matching_prims

# Import the base action class you're inheriting from
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class CPGQuadrupedAction(ActionTerm):
    """
    Quadruped Action that uses CPGs to generate foot trajectories.
    The RL agent modulates CPG parameters (mux, muy, omega) per leg.
    Inherits from QuadrupedDiffIKAction to utilize its IK setup.
    """

    cfg: actions_cfg.CPGQuadrupedActionCfg
    _asset: Articulation
    _scale: torch.Tensor
    _clip: torch.Tensor
    
    # CPG states for each leg, stored per environment
    _rx: Dict[str, torch.Tensor] = {}
    _rxdot: Dict[str, torch.Tensor] = {}
    _ry: Dict[str, torch.Tensor] = {}
    _rydot: Dict[str, torch.Tensor] = {}
    _theta: Dict[str, torch.Tensor] = {}

    # CPG parameters (modulated by RL agent) for each leg
    _mux: Dict[str, torch.Tensor] = {}
    _muy: Dict[str, torch.Tensor] = {}
    _omega: Dict[str, torch.Tensor] = {}
    _gp: Dict[str, torch.Tensor] = {}

    _hip_offsets: Dict[str, torch.Tensor] = {}


    def __init__(self, cfg: actions_cfg.CPGQuadrupedActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Store the environment reference for later use
        self.env = env

        # Store CPG dynamics constant (alpha from paper)
        self.cpg_alpha = self.cfg.cpg_alpha
        # Simulation timestep from environment
        self.sim_dt = self.env.physics_dt

        # Define the ranges for mapping RL actions to CPG parameters
        self._cpg_param_ranges = {
            "mu": self.cfg.mu_range,
            "omega": self.cfg.omega_range,
        }
        # Initialize CPG states and parameters for each leg, for each environment
        # These will be tensors of shape (num_envs,)
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

            self._rx[leg_name] = torch.full((self.num_envs,), leg_cfg.init_mux, device=self.device)
            self._rxdot[leg_name] = torch.zeros(self.num_envs, device=self.device)
            self._ry[leg_name] = torch.full((self.num_envs,), leg_cfg.init_muy, device=self.device)
            self._rydot[leg_name] = torch.zeros(self.num_envs, device=self.device)
            self._theta[leg_name] = torch.full((self.num_envs,), leg_cfg.init_theta, device=self.device)

            # Initialize CPG parameters with their default values from config
            self._mux[leg_name] = torch.full((self.num_envs,), leg_cfg.init_mux, device=self.device)
            self._muy[leg_name] = torch.full((self.num_envs,), leg_cfg.init_muy, device=self.device)
            self._omega[leg_name] = torch.full((self.num_envs,), leg_cfg.init_omega, device=self.device)
            self._gp[leg_name] = torch.zeros(self.num_envs, device=self.device)  # Default to 0.0

            # Reshape to (1, 3) so it can be broadcasted when added to (num_envs, 3)
            self.mu_min, self.mu_max = self._cpg_param_ranges["mu"]
            self.omega_min, self.omega_max = self._cpg_param_ranges["omega"]

            self._hip_offsets[leg_name] = torch.tensor(leg_cfg.hip_offset, device=self.device).view(1, 3)

        # Override action_dim inherited from parent:
        # RL agent outputs 3 parameters (mux, muy, omega) for each of the 4 legs
        action_dim = len(self.cfg.legs) * 4
        # Re-initialize _raw_actions with the new dimension
        self._raw_actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        # save the scale as tensors
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        # self.leg_bounds = {
        #     "FL": {"x": (0.15, 0.25), "y": (0.1, 0.20), "z": (-0.4, -0.1)},
        #     "FR": {"x": (0.15, 0.25), "y": (-0.20, -0.1), "z": (-0.4, -0.1)},
        #     "RL": {"x": (-0.25, -0.15), "y": (0.1, 0.20), "z": (-0.4, -0.1)},
        #     "RR": {"x": (-0.25, -0.15), "y": (-0.20, -0.1), "z": (-0.4, -0.1)}
        # }

        self.coupling_weights = self.cfg.coupling_weights
        self.phase_offsets = self.cfg.phase_offsets
        self._coupling_enable = self.cfg.coupling_enable

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
        """
        Processes the raw actions from the RL agent.
        Maps raw actions to CPG parameters, updates CPG states,
        and computes desired foot positions for the IK controllers.
        """
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale

        # Get contact forces from observation manager
        obs_flat = self.env.observation_manager._obs_buffer["policy"]  # shape: (num_envs, total_obs_dim)

        # Find index of the term in the flattened vector
        term_names = self.env.observation_manager._group_obs_term_names["policy"]
        term_shapes = self.env.observation_manager._group_obs_term_dim["policy"]

        # Compute offset
        start_idx = 0
        for name, shape in zip(term_names, term_shapes):
            if name == "contact_force_vector":
                break
            start_idx += int(np.prod(shape))

        end_idx = start_idx + int(np.prod((4, 3)))  # or use shape directly
        contact_force_flat = obs_flat[:, start_idx:end_idx]  # shape: (num_envs, 12)
        contact_force = contact_force_flat.view(-1, 4, 3)    # shape: (num_envs, 4, 3)


        # --- Map raw RL actions to CPG parameters ---
        # Assuming actions are ordered per leg: [delta_mux_FL, delta_muy_FL, delta_omega_FL, delta_mu_FR, ...]
        action_idx_offset = 0
        for leg_name in self.cfg.legs.keys():
            # Extract actions for current leg (3 values: delta_mux, delta_muy delta_omega)
            delta_mux = self._processed_actions[:, action_idx_offset]
            delta_muy = self._processed_actions[:, action_idx_offset + 1]
            delta_omega = self._processed_actions[:, action_idx_offset + 2]
            delta_gp = self._processed_actions[:, action_idx_offset + 3]

            # # Map mu
            # self._mux[leg_name] = torch.clamp(delta_mux, min=self.mu_min, max=self.mu_max)
            # self._muy[leg_name] = torch.clamp(delta_muy, min=self.mu_min, max=self.mu_max)

            # # Map omega
            # self._omega[leg_name] = torch.clamp(delta_omega, min=self.omega_min, max=self.omega_max)

            # self._gp[leg_name] = torch.clamp(delta_gp, min=0.0, max=0.1)


            # Map mu
            self._mux[leg_name] = self.mu_min + (delta_mux + 1.0) / 2.0 * (self.mu_max - self.mu_min)
            self._muy[leg_name] = self.mu_min + (delta_muy + 1.0) / 2.0 * (self.mu_max - self.mu_min)
            
            # Map omega
            self._omega[leg_name] = self.omega_min + (delta_omega + 1.0) / 2.0 * (self.omega_max - self.omega_min)
            
            self._gp[leg_name] = (delta_gp + 1.0) / 2.0 * (self.cfg.global_gp)

            action_idx_offset += 4 # Move to next leg's parameters

        # --- Update CPG states and compute desired foot positions for all legs ---
        for i, leg_name in enumerate(self.cfg.legs.keys()):
            # CPG Amplitude dynamics (Eq 1 from paper)
            a_val = self.cpg_alpha
            rx_val = self._rx[leg_name]
            ry_val = self._ry[leg_name]
            rxdot_val = self._rxdot[leg_name]
            rydot_val = self._rydot[leg_name]
            
            # Update the CPG amplitude states
            mux_val = self._mux[leg_name]
            muy_val = self._muy[leg_name]

            rxddot_val = a_val * ((a_val / 4) * (mux_val - rx_val) - rxdot_val)
            self._rxdot[leg_name] += rxddot_val * self.sim_dt
            self._rx[leg_name] += rxdot_val * self.sim_dt

            ryddot_val = a_val * ((a_val / 4) * (muy_val - ry_val) - rydot_val)
            self._rydot[leg_name] += ryddot_val * self.sim_dt
            self._ry[leg_name] += rydot_val * self.sim_dt

            # CPG Phase update (Eq 2 from paper)
            omega_val = 2*torch.pi*self._omega[leg_name]
            
            if self._coupling_enable:
                # Calculate coupling contribution
                coupling_contribution = torch.zeros_like(omega_val, device=self.device) # Initialize with zeros
                # for other_leg_name in self.cfg.legs.keys():
                #     if other_leg_name != leg_name:
                #         # Example of accessing (you'll need to define this indexing based on your setup):
                #         key = f"{leg_name}_{other_leg_name}"
                #         w_ij = self.coupling_weights.get(key, torch.zeros_like(omega_val, device=self.device))
                #         phi_ij = self.phase_offsets.get(key, torch.zeros_like(omega_val, device=self.device))

                #         theta_j = self._theta[other_leg_name] # Phase of the influencing leg
                        
                #         # Add to the total coupling contribution for the current leg
                #         coupling_contribution += 0.5 * (self._rx[other_leg_name] + self._ry[other_leg_name]) \
                #             * w_ij * torch.sin(theta_j - self._theta[leg_name] - phi_ij)
                Ni = contact_force[:, i, 2].unsqueeze(1)  # Normal force for the current leg
                cos_theta_expanded = torch.cos(self._theta[leg_name]).unsqueeze(1)
                coupling_contribution += (-0.8 * Ni * cos_theta_expanded).squeeze(1)
                self._theta[leg_name] += (omega_val + coupling_contribution) * self.sim_dt
            else:
                self._theta[leg_name] += (omega_val) * self.sim_dt
            
            self._theta[leg_name] %= (2 * torch.pi) # Ensure phase stays in [0, 2pi]

            cos_theta = torch.cos(self._theta[leg_name])
            sin_theta = torch.sin(self._theta[leg_name])

            gc = self.cfg.global_gc
            # gp = self.cfg.global_gp
            h = self.cfg.global_h
            d_step = self.cfg.global_d_step

            ee_x_des_relative = -d_step * (2*self._rx[leg_name] - 3) * cos_theta
            ee_y_des_relative = -d_step * (2*self._ry[leg_name] - 3) * cos_theta
            ee_z_des_relative = -h + torch.where(sin_theta > 0, gc, self._gp[leg_name]) * sin_theta

            # Stack the relative positions
            ee_pos_des_relative = torch.stack((ee_x_des_relative, ee_y_des_relative, ee_z_des_relative), dim=-1)

            # ADD THE HIP OFFSET TO GET THE DESIRED POSITION IN THE ROBOT'S BASE FRAME
            ee_pos_des = ee_pos_des_relative + self._hip_offsets[leg_name]

            leg_cfg_parent = self.legs[leg_name] # Using a different name to avoid conflict with leg_cfg from CPGQuadrupedActionCfg.legs
            ee_pos_curr, ee_quat_curr = self._compute_frame_pose(leg_cfg_parent) # Current EE pose in base frame

            # Clamp the desired position to the leg's operational space bounds
            # bounds = self.leg_bounds[leg_name]
            # ee_pos_des_clamped = torch.empty_like(ee_pos_des)
            # ee_pos_des_clamped[:, 0] = torch.clamp(ee_pos_des[:, 0], min=bounds["x"][0], max=bounds["x"][1])
            # ee_pos_des_clamped[:, 1] = torch.clamp(ee_pos_des[:, 1], min=bounds["y"][0], max=bounds["y"][1])
            # ee_pos_des_clamped[:, 2] = torch.clamp(ee_pos_des[:, 2], min=bounds["z"][0], max=bounds["z"][1])
            
            # Set the command for the differential IK controller
            self.controllers[leg_name].set_command(ee_pos_des, ee_pos_curr, ee_quat_curr)

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
        """Resets the CPG states and parameters for specified environments."""
        # Call the parent's reset to clear its internal states (e.g., raw_actions)
        self._raw_actions[env_ids] = 0.0

        # Reset CPG states for specified environments
        for leg_name in self.cfg.legs.keys():
            if env_ids is None: # Reset all environments
                self._rx[leg_name][:] = self.cfg.legs[leg_name].init_mux
                self._rxdot[leg_name][:] = 0.0
                self._ry[leg_name][:] = self.cfg.legs[leg_name].init_muy
                self._rydot[leg_name][:] = 0.0
                self._theta[leg_name][:] = self.cfg.legs[leg_name].init_theta # Reset to initial phase
                self._mux[leg_name][:] = self.cfg.legs[leg_name].init_mux
                self._muy[leg_name][:] = self.cfg.legs[leg_name].init_muy
                self._omega[leg_name][:] = self.cfg.legs[leg_name].init_omega
                self._gp[leg_name][:] = 0.0 
            else: # Reset specific environments
                self._rx[leg_name][env_ids] = self.cfg.legs[leg_name].init_mux
                self._rxdot[leg_name][env_ids] = 0.0
                self._ry[leg_name][env_ids] = self.cfg.legs[leg_name].init_muy
                self._rydot[leg_name][env_ids] = 0.0
                self._theta[leg_name][env_ids] = self.cfg.legs[leg_name].init_theta # Reset to initial phase
                self._mux[leg_name][env_ids] = self.cfg.legs[leg_name].init_mux
                self._muy[leg_name][env_ids] = self.cfg.legs[leg_name].init_muy
                self._omega[leg_name][env_ids] = self.cfg.legs[leg_name].init_omega
                self._gp[leg_name][env_ids] = 0.0


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

        if leg_cfg["offset_pos"] is not None and leg_cfg["offset_rot"] is not None:
            jacobian[:, 0:3, :] += torch.bmm(
                -math_utils.skew_symmetric_matrix(leg_cfg["offset_pos"]), jacobian[:, 3:, :]
            )
            jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(leg_cfg["offset_rot"]), jacobian[:, 3:, :])

        return jacobian