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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
