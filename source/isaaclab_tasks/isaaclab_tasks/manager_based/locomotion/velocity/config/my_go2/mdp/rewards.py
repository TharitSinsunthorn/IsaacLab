# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the reward functions that can be used for Spot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


##
# Task Rewards
##


def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.0, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        )

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


##
# Regularization Penalties
##


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("" \
        "Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=20.0), dim=1) + torch.var(
        torch.clip(last_contact_time, max=1.0), dim=1
    )


# ! look into simplifying the kernel here; it's a little oddly complex
def base_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.8 * torch.square(asset.data.root_lin_vel_b[:, 2]) + 0.2 * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )


def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.projected_gravity_b[:, :2]), dim=1)


"""
Joint Related
"""


def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc), dim=1)


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque), dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel), dim=1)


def energy_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Computes an energy penalty based on the absolute joint power (torque * velocity)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel # Shape: (num_envs, num_joints)
    joint_torques = asset.data.applied_torque # Shape: (num_envs, num_joints)
    joint_power = torch.sum(torch.abs(joint_torques * joint_vel), dim=1) # Shape: (num_envs,)
    return joint_power


"""
Contact Related
"""


def contact_force_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in net contact forces across the feet in each environment."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # get net contact forces: [num_envs, history, num_feet, 3]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    
    # compute force norms: [num_envs, history, num_feet]
    force_norms = net_forces.norm(dim=-1)
    
    # take the most recent timestep: [num_envs, num_feet]
    current_forces = force_norms[:, -1, :]
    
    # compute variance across feet: [num_envs]
    return torch.var(current_forces, dim=1)


def contact_force_z_penalty(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize excessive Z-axis contact forces on specified bodies."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history  # shape: (num_envs, T, num_bodies, 3)

    # select z-component of forces for target bodies
    z_forces = net_contact_forces[:, :, sensor_cfg.body_ids, 2]  # shape: (num_envs, T, num_bodies)

    # max absolute z-force over time for each env and body
    max_z_force = torch.max(torch.abs(z_forces), dim=1)[0]  # shape: (num_envs, num_bodies)

    # compute how much it exceeds the threshold
    violation = max_z_force - threshold  # shape: (num_envs, num_bodies)
    return torch.sum(violation.clip(min=0.0), dim=1)  # shape: (num_envs,)


def body_frame_contact_force_z_penalty(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize excessive Z-axis contact forces in the BODY FRAME on specified bodies."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name] # Access the robot/asset that owns these bodies
    net_contact_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :] # Shape: (num_envs, num_selected_bodies, 3)
    body_orientations_w = asset.data.body_quat_w[:, sensor_cfg.body_ids, :] # Shape: (num_envs, num_selected_bodies, 4)
    body_forces_flat = math_utils.quat_apply_inverse(
        body_orientations_w.reshape(-1, 4),
        net_contact_forces_w.reshape(-1, 3)
    )
    body_forces_flat_reshaped = body_forces_flat.view(net_contact_forces_w.shape)
    z_forces_local = body_forces_flat_reshaped[:, :, 2]
    max_z_force = torch.max(torch.abs(z_forces_local), dim=1)[0] # Shape: (num_envs,)
    violation = max_z_force - threshold # Shape: (num_envs,)
    return torch.sum(violation.clip(min=0.0)) # Shape: (num_envs,)


def local_contact_force_z_penalty(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize excessive Z-axis contact forces in the LOCAL FRAME on specified bodies."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    body_orientations_w = contact_sensor.data.quat_w[:, sensor_cfg.body_ids, :]
    # Flatten forces and quaternions for the utility function call.
    #    Both become (num_envs * num_selected_bodies, X).
    forces_flat = net_contact_forces_w.reshape(-1, 3)
    quats_flat = body_orientations_w.reshape(-1, 4)
    net_contact_forces_local = math_utils.quat_apply_inverse(quats_flat, forces_flat)
    # Reshape the transformed forces back to their original batch structure.
    # Shape: (num_envs, num_selected_bodies, 3)
    net_contact_forces_local = net_contact_forces_local.view(net_contact_forces_w.shape)
    z_forces_local = net_contact_forces_local[:, :, 2] # Shape: (num_envs, num_selected_bodies)
    max_z_force = torch.max(torch.abs(z_forces_local), dim=1)[0] # Shape: (num_envs,)
    violation = max_z_force - threshold # Shape: (num_envs,)
    return torch.sum(violation.clip(min=0.0)) # Shape: (num_envs,)


def crawl_balance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize when multiple feet are in the air at the same time."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_in_air = (contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0).float()

    # If more than one foot is in the air, apply a penalty
    penalty = torch.sum(feet_in_air, dim=1) - 1  # Only one foot should be in the air
    penalty = torch.clamp(penalty, min=0)  # Only apply when > 1 foot is in air

    return penalty  # Negative reward (penalty)


def crawl_stance_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, mode_time: float) -> torch.Tensor:
    """Encourage longer stance times."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    return torch.sum(torch.clip(current_contact_time, max=mode_time), dim=1)


def crawl_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target_stance_ratio: float = 0.75,
    tolerance: float = 0.05,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Compute stance and swing times
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]  # shape: (num_envs, num_feet)
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    # Avoid division by zero
    gait_cycle_time = contact_time + air_time + 1e-5
    stance_ratio_per_foot = contact_time / gait_cycle_time

    # Average stance ratio across feet
    stance_ratio = torch.mean(stance_ratio_per_foot, dim=1)

    # Reward based on how close stance ratio is to target
    stance_reward = torch.exp(-torch.square(stance_ratio - target_stance_ratio) / tolerance)

    return stance_reward


def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


def swing_impact_penalty(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize high contact forces when a foot lands on the ground (i.e., swing-to-stance transition)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get first contact indicator: 1 if just made contact this step, 0 otherwise
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    # Get current contact forces
    contact_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    contact_force_magnitudes = contact_forces_z.abs()

    # Penalize only on landing (first contact) and above threshold
    violation = (contact_force_magnitudes - threshold).clip(min=0.0)
    impact_penalty = violation * first_contact  # only penalize at first contact

    return torch.sum(impact_penalty, dim=1)


def swing_impact_b_penalty(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize high contact forces in the BODY FRAME when a foot lands on the ground (i.e., swing-to-stance transition)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]  # Access the robot/asset that owns these bodies
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    net_contact_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    body_orientations_w = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]
    body_forces_flat = math_utils.quat_apply_inverse(
        body_orientations_w.reshape(-1, 4),
        net_contact_forces_w.reshape(-1, 3)
    )
    body_forces_reshaped = body_forces_flat.view(net_contact_forces_w.shape)
    z_forces_local_abs = body_forces_reshaped[:, :, 2].abs()
    violation = (z_forces_local_abs - threshold).clip(min=0.0)
    impact_penalty_per_body = violation * first_contact
    return torch.sum(impact_penalty_per_body, dim=1)


def total_contact_force_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg,threshold: float) -> torch.Tensor:
    """
    Penalize if the sum of Z-axis contact forces (in world frame) from all specified bodies
    is significantly lower than the robot's total weight.
    This encourages the robot to maintain sufficient ground contact.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    total_z_force_sum = torch.sum(net_contact_forces_w[:, :, 2].abs(), dim=1)
    violation = total_z_force_sum - threshold # Shape: (num_envs,)
    return torch.sum(violation.clip(min=0.0))


"""
GIA
"""
def stability_margin_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    normalize_angle: float, # This should be math.pi / 2
) -> torch.Tensor:
    """
    Rewards the robot for maintaining a high stability margin based on the Gravito-Inertial Inclination Margin (GIIM).
    This encourages the Gravito-Inertial Acceleration (GIA) vector to point well inside the stability polyhedron.
    
    References:
    - "Towards Legged Locomotion on Steep Planetary Terrain" (Weibel et al., 2023) 
    - "Dynamics and Equilibrium of Legged-Climbing Robots" (Ribeiro, 2021) 
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    #----------- GIA vector calculation ---------#
    g_tensor = env.sim._gravity_tensor.clone().detach()

    # Expand body_mass for broadcasting with acceleration
    body_mass_on_device = asset.data.default_mass.to(env.device)
    body_mass_expanded = body_mass_on_device.unsqueeze(-1) # (num_envs, num_bodies, 1, 1) -> (num_envs, num_bodies, 1) after first squeeze
    # Calculate m_j * dot_r_j for all bodies
    mass_times_accel = body_mass_expanded * asset.data.body_lin_acc_w # (num_envs, num_bodies, 3)
    sum_mass_times_accel = torch.sum(mass_times_accel, dim=1) # (num_envs, 3)
    # Calculate total robot mass (w) for each environment
    total_robot_mass = torch.sum(asset.data.default_mass, dim=1) # (num_envs, 1)
    total_robot_mass_expanded = total_robot_mass.unsqueeze(-1) # Shape: (num_envs, 1)
    # Calculate dot_r_g (CoM acceleration)
    dot_r_g = sum_mass_times_accel / (total_robot_mass_expanded.to(env.device) + 1e-6) # (num_envs, 3)
    # Calculate Gravito-Inertial Acceleration (GIA) 
    a_gi = g_tensor - dot_r_g # (num_envs, 3)
    #----------- GIA vector calculation ---------#

    #----------- Define Stability Polyhedron and Tumbling Axes ----------#
    robot_cog_w = asset.data.root_com_pos_w # (num_envs, 3)
    # Get contact point positions for bodies specified in sensor_cfg (e.g., feet)
    potential_contact_points_w = contact_sensor.data.pos_w[:, sensor_cfg.body_ids, :] # (num_envs, num_feet, 3)
    contact_forces_on_feet = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :] # (num_envs, num_feet, 3)
    is_in_contact_feet = contact_forces_on_feet.norm(dim=-1) > 0.1 # (num_envs, num_feet)

    num_feet = potential_contact_points_w.shape[1]
    if num_feet < 2:
        return torch.zeros(env.num_envs, device=env.device, dtype=a_gi.dtype)

    # --- Vectorized Calculation of Stability for adjacent feet pairs ---

    # Define adjacent pairs of feet for a quadruped (assuming a standard order like FL, FR, RL, RR in sensor_cfg.body_ids)
    # This list of pairs corresponds to the edges of the support polygon, or "subsequent body parts".
    # This must match the order in which body_ids are stored in sensor_cfg.body_ids.
    # Example for (0=FL, 1=FR, 2=RL, 3=RR):
    # (FL, FR), (FR, RR), (RR, RL), (RL, FL) - forms the quadrilateral base
    # (FL, RR) and (FR, RL) would be diagonals if included, but paper implies only subsequent.
    # Make sure these indices are within the actual `num_feet`.
    if num_feet == 4: # Standard quadruped
        # These are the 4 edges of the support quadrilateral
        adjacent_pair_indices = torch.tensor([
            [0, 1], # FL-FR
            [1, 3], # FR-RR (assuming RR is index 3)
            [3, 2], # RR-RL (assuming RL is index 2)
            [2, 0]  # RL-FL (assuming FL is index 0)
        ], device=env.device, dtype=torch.long)
    else: # Fallback to all combinations if not a standard 4-leg, or if you explicitly want all pairs.
          # For a generic multi-legged robot or for debugging with variable leg count:
        adjacent_pair_indices = torch.combinations(torch.arange(num_feet, device=env.device), r=2) 

    num_relevant_pairs = adjacent_pair_indices.shape[0]

    # Expand A_GI and robot_cog_w to match the dimensions needed for pairs
    a_gi_expanded = a_gi.unsqueeze(1).expand(-1, num_relevant_pairs, -1) # (num_envs, num_pairs, 3)
    robot_cog_w_expanded = robot_cog_w.unsqueeze(1).expand(-1, num_relevant_pairs, -1) # (num_envs, num_pairs, 3)

    # Get positions of foot A and foot B for all relevant pairs in all environments
    p_e_a_all_pairs = potential_contact_points_w[:, adjacent_pair_indices[:, 0], :] # (num_envs, num_pairs, 3)
    p_e_b_all_pairs = potential_contact_points_w[:, adjacent_pair_indices[:, 1], :] # (num_envs, num_pairs, 3)
    # print(p_e_a_all_pairs.shape, p_e_b_all_pairs.shape)

    # Calculate vectors from CoG to foot A and foot B for all pairs
    vec_cog_to_pe_b = robot_cog_w_expanded - p_e_b_all_pairs 
    vec_cog_to_pe_a = robot_cog_w_expanded - p_e_a_all_pairs 
    
    # Calculate normal vectors (n_gab) for all faces
    n_gab_all_pairs = torch.linalg.cross(vec_cog_to_pe_b, vec_cog_to_pe_a, dim=-1) # (num_envs, num_pairs, 3)
    n_gab_norm = torch.norm(n_gab_all_pairs, dim=-1, keepdim=True) + 1e-6 # (num_envs, num_pairs, 1)
    n_gab_unit = n_gab_all_pairs / n_gab_norm # (num_envs, num_pairs, 3)

    # Calculate dot product of n_gab_unit with a_gi_expanded for all pairs
    dot_product_all_pairs = torch.sum(n_gab_unit * a_gi_expanded, dim=-1) # (num_envs, num_pairs)
    
    # Calculate norm of A_GI for all environments, expanded for pairs
    a_gi_norm_expanded_for_pairs = torch.norm(a_gi.unsqueeze(1), dim=-1, keepdim=True) + 1e-6 # (num_envs, 1, 1)
    a_gi_norm_expanded_for_pairs = a_gi_norm_expanded_for_pairs.expand(-1, num_relevant_pairs, -1) # (num_envs, num_pairs, 1)
    
    cosine_angle_all_pairs = dot_product_all_pairs / a_gi_norm_expanded_for_pairs.squeeze(-1) # (num_envs, num_pairs)
    cosine_angle_all_pairs = torch.clamp(cosine_angle_all_pairs, -1.0, 1.0) 
    
    angle_all_pairs = torch.acos(cosine_angle_all_pairs) # (num_envs, num_pairs)

    # Calculate reward term for each face
    term_reward_all_pairs = angle_all_pairs - normalize_angle # (num_envs, num_pairs)

    # Create a mask for active pairs: True if both feet in a pair are in contact
    # is_in_contact_feet is (num_envs, num_feet)
    # adjacent_pair_indices is (num_pairs, 2)
    # is_feet_contacted_for_pairs will be (num_envs, num_pairs, 2)
    is_feet_contacted_for_pairs = is_in_contact_feet[:, adjacent_pair_indices] 
    
    # A pair is active if ALL (both) feet in that pair are in contact
    pair_is_active_mask = is_feet_contacted_for_pairs.all(dim=-1) # (num_envs, num_pairs)

    # Apply mask to rewards (set inactive pairs' rewards to zero)
    masked_rewards_all_pairs = term_reward_all_pairs * pair_is_active_mask.float() # (num_envs, num_pairs)

    # Sum rewards for all active faces for each environment
    total_angle_reward_per_env = torch.sum(masked_rewards_all_pairs, dim=1) # (num_envs,)
    # print(total_angle_reward_per_env)
    # print(total_angle_reward_per_env.shape)

    # Handle environments with less than 2 active contacts (no valid polygon)
    num_active_contacts_per_env = is_in_contact_feet.sum(dim=-1) # (num_envs,)
    
    rewards_final = torch.where(
        num_active_contacts_per_env < 2,
        torch.zeros_like(total_angle_reward_per_env), # Assign 0.0 for invalid cases
        total_angle_reward_per_env
    )
    
    return rewards_final