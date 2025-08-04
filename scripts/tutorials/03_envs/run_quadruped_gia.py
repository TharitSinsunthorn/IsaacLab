# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the quadruped RL environment and visualize
the Gravito-Inertial Acceleration (GIA) vector using pre-defined markers.

Usage:
    ./isaaclab.sh -p scripts/tutorials/03_envs/run_quadruped_rl_env.py --num_envs 1

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the quadruped RL environment with GIA visualization.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn (set to 1 for clear visualization).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import math
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils 

# Import the specific pre-defined marker configurations
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG , FRAME_MARKER_CFG

# Your specific environment config imports
from isaaclab_tasks.manager_based.locomotion.velocity.config.my_go2.lowg_env_cfg import LowGravityUnitreeGo2RoughEnvCfg

# --- START: Custom quat_from_vectors implementation ---
# This function is needed because it's not directly available in your isaaclab.utils.math.py
# This computes a quaternion that rotates vector v0 to vector v1.
# It assumes v0 and v1 are normalized inputs and returns the quaternion in (w, x, y, z) format.
@torch.jit.script
def quat_from_vectors(v0: torch.Tensor, v1: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Explicitly specify p=2 for torch.norm when dim is also provided for jit compilation
    v0_norm = v0 / (v0.norm(p=2, dim=-1, keepdim=True) + eps)
    v1_norm = v1 / (v1.norm(p=2, dim=-1, keepdim=True) + eps)

    dot = torch.sum(v0_norm * v1_norm, dim=-1) # (...,) dot products
    cross = torch.cross(v0_norm, v1_norm, dim=-1) # (..., 3) cross products

    # Initialize quaternion result
    quat = torch.zeros(v0.shape[:-1] + (4,), device=v0.device, dtype=v0.dtype) # (..., 4)

    # Handle collinear vectors (dot product close to 1 or -1) for numerical stability
    straight_line_mask = (dot > (1.0 - eps)) # (...,) boolean mask for vectors aligned
    opposite_line_mask = (dot < (-1.0 + eps)) # (...,) boolean mask for vectors anti-aligned

    # Case 1: Vectors are almost collinear and in the same direction (identity quaternion: [1, 0, 0, 0])
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=v0.device, dtype=v0.dtype)
    if identity_quat.ndim < quat.ndim: # Handle broadcasting for batch dimensions
        identity_quat = identity_quat.expand(quat.shape[:-1] + (4,))
    quat = torch.where(straight_line_mask.unsqueeze(-1), identity_quat, quat)

    # Case 2: Vectors are almost collinear and in opposite directions (180 degree rotation)
    arbitrary_axis = torch.where(
        torch.abs(v0_norm[..., 2]) < 0.9, # If not aligned with Z-axis
        torch.tensor([0.0, 0.0, 1.0], device=v0.device, dtype=v0.dtype),
        torch.tensor([0.0, 1.0, 0.0], device=v0.device, dtype=v0.dtype)
    )
    rotation_axis_180 = torch.cross(v0_norm, arbitrary_axis, dim=-1)
    # Explicitly specify p=2 for torch.norm here too
    rotation_axis_180 = rotation_axis_180 / (rotation_axis_180.norm(p=2, dim=-1, keepdim=True) + eps) # Normalize axis

    # Quaternion for 180 deg rotation: w=0, (x,y,z) = axis
    quat_180 = torch.cat([torch.zeros(v0.shape[:-1] + (1,), device=v0.device, dtype=v0.dtype), rotation_axis_180], dim=-1)
    
    quat = torch.where(opposite_line_mask.unsqueeze(-1), quat_180, quat)

    # Case 3: General case (vectors not collinear)
    s = torch.sqrt((1.0 + dot) * 2.0) # (...,)
    inv_s = 1.0 / (s + eps) # (...,)

    w = 0.5 * s
    x = cross[..., 0] * inv_s
    y = cross[..., 1] * inv_s
    z = cross[..., 2] * inv_s

    general_quat = torch.stack([w, x, y, z], dim=-1) # (..., 4)

    quat = torch.where(straight_line_mask.unsqueeze(-1) | opposite_line_mask.unsqueeze(-1), quat, general_quat)
    
    return quat
# --- END: Custom quat_from_vectors implementation ---


def main():
    """Main function."""
    # create environment configuration
    env_cfg = LowGravityUnitreeGo2RoughEnvCfg()
    # Set num_envs to 1 for clear visualization
    env_cfg.scene.num_envs = 1 
    env_cfg.actions.quadruped_action_cfg.scale = 1.0
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # --- Setup VisualizationMarkers for GIA vectors using pre-defined configs ---
    
    # GIA Vector (Red Arrow)
    gia_arrow_visualizer = VisualizationMarkers(RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/GIA_Vector_Arrow"))
    # Gravity Vector (Blue Arrow)
    gravity_arrow_visualizer = VisualizationMarkers(BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Gravity_Vector_Arrow"))
    # CoM Acceleration Vector (Green Arrow)
    com_accel_arrow_visualizer = VisualizationMarkers(GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/CoM_Accel_Vector_Arrow"))

    # # CoM Position (Black Sphere) - Custom SphereCfg needed as no pre-defined black sphere
    # com_sphere_visualizer_cfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/CoM_Position_Sphere",
    #     markers={
    #         "sphere": sim_utils.SphereCfg(
    #             radius=0.03, # Radius of the sphere
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)), # Black color
    #         ),
    #     },
    # )
    # com_sphere_visualizer = VisualizationMarkers(com_sphere_visualizer_cfg)

    # --- Setup VisualizationMarkers for Stability Polygon Lines ---
    # Reusing FRAME_MARKER_CFG which has a "connecting_line" prototype (a cylinder) 
    polygon_arrow_visualizer = VisualizationMarkers(BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Stability_Polygon_Arrows"))
    # The "connecting_line" prototype is a CylinderCfg with radius=0.002, height=1.0. We will scale its height to match vector length.

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset environment periodically
            if count % 500 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            
            # sample random actions (using your existing action logic from the original script)
            ee_pos = torch.zeros_like(env.action_manager.action)
            target_ee = torch.tensor([
                0.3, 0.0, -.4, 1.0,    # leg 1
                0.3, 0.0, -.4, 1.0,    # leg 2
                0.3, 0.0, -.4, 1.0,    # leg 3
                0.3, 0.0, -.4, 1.0     # leg 4
                ], device=ee_pos.device)
            ee_pos[:] = target_ee.unsqueeze(0).repeat(ee_pos.shape[0], 1) 
            
            # step the environment forward
            obs, rew, terminated, truncated, info = env.step(ee_pos)
            
            # update counter
            count += 1

            # --- Calculate GIA and components for visualization ---
            # We'll calculate for the first environment (env_idx=0) for clear visualization
            env_idx = 0 

            # Get Gravitational Acceleration (g) in the world frame
            g_w_tensor_all_envs = torch.tensor(env.sim._gravity_tensor, device=env.device, dtype=torch.float32).view(1, 3).clone().detach()
            g_w_tensor_all_envs = g_w_tensor_all_envs.expand(env.num_envs, -1)
            # Calculate Total Robot Acceleration (dot_r_g_w) - Simplified to base acceleration
            # [cite_start]Using the root link's linear acceleration as a proxy for the whole robot's CoM acceleration
            base_body_idx = env.scene["robot"].find_bodies(["base"])[0] 
            dot_r_g_w_all_envs = env.scene["robot"].data.body_lin_acc_w[:, base_body_idx, :].squeeze(1) 
            # Calculate Gravito-Inertial Acceleration (GIA)
            a_gi_w_all_envs = g_w_tensor_all_envs - dot_r_g_w_all_envs # (num_envs, 3)
            # Robot CoM position in world frame (r_g) - Simplified to base position
            # [cite_start]Using the root link's position as a proxy for the whole robot's CoM position
            robot_cog_w_all_envs = env.scene["robot"].data.root_pos_w # (num_envs, 3)

            # --- Extract data for the single environment to visualize ---
            robot_cog_pos_single = robot_cog_w_all_envs[env_idx]
            gravity_vec_single = g_w_tensor_all_envs[env_idx]
            com_accel_vec_single = dot_r_g_w_all_envs[env_idx]
            gia_vec_single = a_gi_w_all_envs[env_idx]

            # --- Prepare Data for VisualizationMarkers ---
            vector_display_scale = 2.0 
            min_arrow_length = 0.05 # Minimum length to ensure very small vectors are still visible
            # Base direction for arrows (arrow points along X-axis by default)
            base_arrow_direction = torch.tensor([1.0, 0.0, 0.0], device=env.device, dtype=torch.float32) 
            
            # --- Visualize the markers! ---
            # GIA Vector (Red Arrow)
            gia_arrow_visualizer.visualize(
                translations=robot_cog_pos_single.unsqueeze(0),
                orientations=quat_from_vectors(base_arrow_direction, math_utils.normalize(gia_vec_single)).unsqueeze(0)
                    if gia_vec_single.norm() > 1e-6 else torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device, dtype=torch.float32),
                scales=torch.tensor([[max(gia_vec_single.norm() * vector_display_scale, min_arrow_length), 0.5, 0.05]], device=env.device),
            )

            # Gravity Vector (Blue Arrow)
            gravity_arrow_visualizer.visualize(
                translations=robot_cog_pos_single.unsqueeze(0),
                # Use math_utils.normalize()
                orientations=quat_from_vectors(base_arrow_direction, math_utils.normalize(gravity_vec_single)).unsqueeze(0)
                    if gravity_vec_single.norm() > 1e-6 else torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device, dtype=torch.float32),
                scales=torch.tensor([[max(gravity_vec_single.norm() * vector_display_scale, min_arrow_length), 0.5, 0.05]], device=env.device),
            )

            # CoM Acceleration Vector (Green Arrow)
            com_accel_arrow_visualizer.visualize(
                translations=robot_cog_pos_single.unsqueeze(0),
                # Use math_utils.normalize()
                orientations=quat_from_vectors(base_arrow_direction, math_utils.normalize(com_accel_vec_single)).unsqueeze(0)
                    if com_accel_vec_single.norm() > 1e-6 else torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device, dtype=torch.float32),
                scales=torch.tensor([[max(com_accel_vec_single.norm() * vector_display_scale, min_arrow_length), 0.05, 0.05]], device=env.device),
            )

            # # CoM Position (Black Sphere)
            # com_sphere_visualizer.visualize(
            #     translations=robot_cog_pos_single.unsqueeze(0),
            #     orientations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device, dtype=torch.float32), # Identity quaternion
            #     scales=torch.tensor([[1.0, 1.0, 1.0]], device=env.device), # Use base scale from cfg (radius=0.03)
            # )

            # --- Visualization for Stability Polygon Lines (using arrows) ---
            line_positions = []
            line_orientations = []
            line_scales = []
            
            vector_display_scale_lines = 10.0 # Scale for the length of these polygon lines
            min_arrow_length_lines = 0.01 # Smaller min length for these lines

            # Robot's foot body names (ensure these match your URDF exactly)
            contact_body_names_list = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"] 
            contact_body_ids = env.scene["robot"].find_bodies(contact_body_names_list)[0]
            
            # Get foot positions regardless of contact status
            foot_positions_all_envs = env.scene["robot"].data.body_pos_w[:, contact_body_ids, :]
            
            # Extract single environment data for feet
            # Assuming the order in contact_body_names_list is FL=0, FR=1, RL=2, RR=3 based on find_bodies result
            fl_foot_pos = foot_positions_all_envs[env_idx, 0, :]
            fr_foot_pos = foot_positions_all_envs[env_idx, 1, :]
            rl_foot_pos = foot_positions_all_envs[env_idx, 2, :]
            rr_foot_pos = foot_positions_all_envs[env_idx, 3, :]

            # --- Define the specific line segments for the support polygon and pyramid edges ---
            # These are ordered to form the base quadrilateral and then connect to CoG.

            # 1. Base quadrilateral segments (foot-to-foot lines)
            # Order: FL -> FR -> RR -> RL -> FL (closing the loop)
            foot_segments_to_draw = [
                (fl_foot_pos, fr_foot_pos), # FL to FR
                (fr_foot_pos, rr_foot_pos), # FR to RR
                (rr_foot_pos, rl_foot_pos), # RR to RL
                (rl_foot_pos, fl_foot_pos), # RL to FL
            ]

            # 2. Pyramid segments (CoG-to-foot lines)
            cog_segments_to_draw = [
                (robot_cog_pos_single, fl_foot_pos),
                (robot_cog_pos_single, fr_foot_pos),
                (robot_cog_pos_single, rl_foot_pos),
                (robot_cog_pos_single, rr_foot_pos),
            ]
            
            # Combine all segments for drawing
            all_segments_to_draw = foot_segments_to_draw + cog_segments_to_draw

            # --- Loop through segments and prepare marker data ---
            for start_point, end_point in all_segments_to_draw:
                vector = end_point - start_point
                length = vector.norm()
                
                # Only draw if segment has non-zero length to avoid normalization issues
                if length > 1e-6:
                    rotation_quat = quat_from_vectors(base_arrow_direction, math_utils.normalize(vector))
                    mid_point = (start_point + end_point) / 2.0 # Place arrow at midpoint of line segment

                    line_positions.append(mid_point)
                    line_orientations.append(rotation_quat)
                    # Scales for arrow: (length, width, height)
                    # Use a smaller width/height for these polygon lines to distinguish them (e.g., 0.05)
                    line_scales.append(torch.tensor([max(length * vector_display_scale_lines, min_arrow_length_lines), 0.1, 0.01], device=env.device))

            # --- Visualize the collected lines (arrows) ---
            # This visualizer call will always execute, drawing the defined segments.
            # Use marker_idx 0 as "arrow" is the first/only prototype in GREEN_ARROW_X_MARKER_CFG.
            polygon_arrow_visualizer.visualize(
                translations=torch.stack(line_positions),
                orientations=torch.stack(line_orientations),
                scales=torch.stack(line_scales),
                marker_indices=torch.full((len(line_positions),), 0, device=env.device, dtype=torch.long) 
            )


    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()