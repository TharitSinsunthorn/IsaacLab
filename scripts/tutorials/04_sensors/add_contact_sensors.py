# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example on using the contact sensor.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
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
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

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
            ".*_thigh_joint": -40.0 * math.pi / 180.0,
            ".*_calf_joint": -120.0 * math.pi / 180.0,
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
class ContactSensorSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Rigid Object
    # cube = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Cube",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.5, 0.5, 0.1),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.5, 0.05)),
    # )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/defaultGroundPlane"],
    )

    contact_forces_LF = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        track_air_time=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )

    contact_forces_RF = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FR_foot",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )

    contact_forces_H = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/R[L,R]_foot",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():

        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands
        targets = scene["robot"].data.default_joint_pos
        # -- apply action to the robot
        scene["robot"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print information from the sensors
        # print("-------------------------------")
        # print(scene["contact_forces_LF"])
        # print("Received force matrix of: ", scene["contact_forces_LF"].data.force_matrix_w)
        # print("Received contact force of: ", scene["contact_forces_LF"].data.net_forces_w)
        # print("-------------------------------")
        # print(scene["contact_forces_RF"])
        # print("Received force matrix of: ", scene["contact_forces_RF"].data.force_matrix_w)
        # print("Received contact force of: ", scene["contact_forces_RF"].data.net_forces_w)
        # print("-------------------------------")
        # print(scene["contact_forces_H"])
        # print("Received force matrix of: ", scene["contact_forces_H"].data.force_matrix_w)
        # print("Received contact force of: ", scene["contact_forces_H"].data.net_forces_w)

        # net_contact_forces = scene["contact_forces_H"].data.net_forces_w_history
        # violation = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]
        # print(f"Violations: {violation}")
        last_air_time = scene["contact_forces_LF"].data.last_air_time[:,:]
        last_contact_time = scene["contact_forces_LF"].data.last_contact_time[:,:]
        print(f"Last air time: {last_air_time}")
        print(f"Last contact time: {last_contact_time}")

def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = ContactSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
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