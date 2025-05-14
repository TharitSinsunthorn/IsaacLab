# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/quadruped_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Testing on using the differential IK controller with quadruped robot.")
parser.add_argument("--robot", type=str, default="unitree go2", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=5, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


YONAKA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.environ['HOME'] + "/ilab_tharit/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/yonaka/model/moonbotY3"
        ".usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            # kinematic_enabled=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": -0.0 * math.pi / 180.0,
            ".*_calf_joint": -0.0 * math.pi / 180.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=21.0,
            stiffness=200.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)

@configclass
class QuadrupedScenceCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]

    # Create a shared IK controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    FL_ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/FL_ee_current"))
    FL_goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/FL_ee_goal"))
    FR_ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/FR_ee_current"))
    FR_goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/FR_ee_goal"))
    RL_ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/RL_ee_current"))
    RL_goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/RL_ee_goal"))
    RR_ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/RR_ee_current"))
    RR_goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/RR_ee_goal"))

    # Leg config: joint names, body names, and IK goals
    legs_config = [
        ("FL", ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"], "FL_foot", [
            torch.tensor([0.2,  0.2, -0.1], device=sim.device),
            torch.tensor([0.2,  0.2, -0.2], device=sim.device),
            torch.tensor([0.15,  0.15, -0.1], device=sim.device),
        ]),
        ("FR", ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"], "FR_foot", [
            torch.tensor([0.2, -0.2, -0.1], device=sim.device),
            torch.tensor([0.2, -0.2, -0.2], device=sim.device),
            torch.tensor([0.15, -0.15, -0.1], device=sim.device),
        ]),
        ("RL", ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"], "RL_foot", [
            torch.tensor([-0.2,  0.2, -0.1], device=sim.device),
            torch.tensor([-0.2,  0.2, -0.2], device=sim.device),
            torch.tensor([-0.15,  0.15, -0.1], device=sim.device),
        ]),
        ("RR", ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"], "RR_foot", [
            torch.tensor([-0.2, -0.2, -0.1], device=sim.device),
            torch.tensor([-0.2, -0.2, -0.2], device=sim.device),
            torch.tensor([-0.15, -0.15, -0.1], device=sim.device),
        ]),
    ]

    sim_dt = sim.get_physics_dt()
    count = 0
    current_goal_idx = 0

    # Simulation loop
    while simulation_app.is_running():
        if count % 150 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset controller
            diff_ik_controller.reset()
            # change goal
            current_goal_idx = (current_goal_idx + 1) % 3  # Cycle goals

        full_joint_targets = robot.data.joint_pos.clone()

        for leg_name, joint_names, body_name, goal_list in legs_config:
            goal = goal_list[current_goal_idx]
            # Specify robot-specific parameters and resolve the scene entities
            entity_cfg = SceneEntityCfg("robot", joint_names=joint_names, body_names=[body_name])
            entity_cfg.resolve(scene)

            if robot.is_fixed_base:
                ee_jacobi_idx = entity_cfg.body_ids[0] - 1
                jacobi_joint_ids = entity_cfg.joint_ids
            else:
                ee_jacobi_idx = entity_cfg.body_ids[0]
                jacobi_joint_ids = [i + 6 for i in entity_cfg.joint_ids]

            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, jacobi_joint_ids]
            ee_pose_w = robot.data.body_state_w[:, entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos_leg = robot.data.joint_pos[:, entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            ik_command = goal.repeat(scene.num_envs, 1)
            diff_ik_controller.set_command(ik_command, ee_quat=ee_quat_b)
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_leg)

            full_joint_targets[:, entity_cfg.joint_ids] = joint_pos_des
            
            ee_pose_w = robot.data.body_state_w[:, entity_cfg.body_ids[0], 0:7]
            if leg_name == "FL":
                # visualize the markers
                FL_ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                FL_goal_marker.visualize(ik_command + robot.data.root_state_w[:, 0:3],
                                  torch.tensor([[0.0, 0.0, 0.0, 1.0]] * scene.num_envs, device=sim.device))
            elif leg_name == "FR":  
                FR_ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                FR_goal_marker.visualize(ik_command + robot.data.root_state_w[:, 0:3],
                                  torch.tensor([[0.0, 0.0, 0.0, 1.0]] * scene.num_envs, device=sim.device))
            elif leg_name == "RL":
                RL_ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                RL_goal_marker.visualize(ik_command + robot.data.root_state_w[:, 0:3],
                                  torch.tensor([[0.0, 0.0, 0.0, 1.0]] * scene.num_envs, device=sim.device))
            elif leg_name == "RR":
                RR_ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                RR_goal_marker.visualize(ik_command + robot.data.root_state_w[:, 0:3],
                                  torch.tensor([[0.0, 0.0, 0.0, 1.0]] * scene.num_envs, device=sim.device))

        # apply actions
        print(f"Joint targets: {full_joint_targets}")
        robot.set_joint_position_target(full_joint_targets)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = QuadrupedScenceCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
