# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import FrameTransformerData, ContactSensor
from . import CPGQuadrupedAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
    from . import CPGQuadrupedAction


"""
contact state.
"""

def contact_bool(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 1e-3) -> torch.Tensor:
    """Boolean contact state of the specified bodies (1 if in contact, 0 otherwise)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_for_selected_bodies = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norm = torch.norm(forces_for_selected_bodies, dim=-1)
    return (force_norm > force_threshold).float()


def contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Contact force vector for the specified bodies (in world frame), flattened for observations."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_for_selected_bodies = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    
    # Flatten the (num_selected_bodies, 3) part into a single vector
    # If forces_for_selected_bodies is (N, B, 3), this makes it (N, B*3)
    return forces_for_selected_bodies.flatten(start_dim=1)


def local_contact_force_observation(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Contact force vector for the specified bodies (in local frame), flattened for observations."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_for_selected_bodies_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    sensor_orientation_w = contact_sensor.data.quat_w[:, sensor_cfg.body_ids, :]
    # Flatten forces and quaternions for the utility function call 
    # Both become (num_envs * num_selected_bodies, X).
    forces_flat = forces_for_selected_bodies_w.reshape(-1, 3)
    quats_flat = sensor_orientation_w.reshape(-1, 4)
    # Transform the world-frame forces to the LOCAL FRAME OF THE BODY.
    local_forces_flat = math_utils.quat_apply_inverse(quats_flat, forces_flat)
    # Reshape the transformed forces back to their original batch structure.
    # Shape: (num_envs, num_selected_bodies, 3)
    local_forces_reshaped = local_forces_flat.view(forces_for_selected_bodies_w.shape)
    # Flatten the (num_selected_bodies, 3) part into a single vector for observations.
    # If `local_forces_reshaped` is (N, B, 3), this makes it (N, B*3).
    return local_forces_reshaped.flatten(start_dim=1)


"""
ee state.
"""


def ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

    return ee_pos


def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    # make first element of quaternion positive
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat


"""
CPG state
"""

def get_cpg_internal_states(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Retrieves the internal CPG states:
    (r_x, r_dot_x, r_y, r_dot_y, theta, theta_dot) for each leg.
    """
    cpg_action_term: CPGQuadrupedAction | None = None
    for _, action_term in env.action_manager._terms.items():
        if isinstance(action_term, CPGQuadrupedAction):
            cpg_action_term = action_term
            break

    if cpg_action_term is None:
        raise RuntimeError("CPGQuadrupedAction not found.")

    all_legs_cpg_states = []

    for leg_name in cpg_action_term.cfg.legs.keys():
        current_rx = cpg_action_term._rx[leg_name]
        current_rxdot = cpg_action_term._rxdot[leg_name]
        current_ry = cpg_action_term._ry[leg_name]
        current_rydot = cpg_action_term._rydot[leg_name]
        current_theta = cpg_action_term._theta[leg_name]

        # Compute theta_dot = omega + coupling
        # If you already apply coupling during training, this is approximate:
        current_theta_dot = 2*torch.pi*cpg_action_term._omega[leg_name]

        leg_vector = torch.stack([
            current_rx,
            current_rxdot,
            current_ry,
            current_rydot,
            current_theta,
            current_theta_dot,

        ], dim=1)

        all_legs_cpg_states.append(leg_vector)

    return torch.cat(all_legs_cpg_states, dim=1)

