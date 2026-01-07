# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_quadruped_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.my_go2.flat_env_cfg import LowGravityUnitreeGo2FlatEnvCfg_PLAY
from isaaclab_tasks.manager_based.locomotion.velocity.config.my_go2.flat_env_cfg import MyUnitreeGo2FlatEnvCfg_PLAY
from isaaclab_tasks.manager_based.locomotion.velocity.config.my_go2.lowg_env_cfg import LowGravityUnitreeGo2RoughEnvCfg


def main():
    """Main function."""
    # create environment configuration
    # env_cfg = MyUnitreeGo2FlatEnvCfg_PLAY()
    env_cfg = LowGravityUnitreeGo2FlatEnvCfg_PLAY()

    # -- Set a fixed spawn position and orientation
    # This overrides the randomization defined in the environment configuration file.
    # We set the min and max of the pose range to the same value.
    # Position: x=0, y=0, z=0.4
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.05, 0.05)
    }
    # Orientation: No rotation (roll=0, pitch=0, yaw=0)
    # Note: The key for rotation might be different in your config (e.g., "rot_range").
    # Assuming the reset function uses "pose_range" for rotation as well based on common practice.
    # If your reset function uses a separate "rot_range", you would set that instead.
    # For `reset_root_state_uniform`, rotation is part of `pose_range`.
    env_cfg.events.reset_base.params["pose_range"]["roll"] = (0.0, 0.0)
    env_cfg.events.reset_base.params["pose_range"]["pitch"] = (0.0, 0.0)
    env_cfg.events.reset_base.params["pose_range"]["yaw"] = (-math.pi/2, -math.pi/2)
    
    env_cfg.events.add_base_mass = None
    
    env_cfg.terminations.base_contact = None
    env_cfg.terminations.thigh_contact = None
    env_cfg.terminations.calf_contact = None

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.actions.quadruped_action_cfg.scale = 1.0
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # -- Define the two different CPG states for the robot's legs
    # Action to perform when the robot is on the ground (move)
    in_air_action = torch.tensor([
        0.0, 0.0, -1.0, -0.7,   # leg 1
        0.0, 0.0, -1.0, -0.7,   # leg 2
        0.0, 0.0, -1.0, 1.0,   # leg 3
        0.0, 0.0, -1.0, 1.0    # leg 4
    ], device=env.device)

    # Action to perform when the robot is in the air (not move)
    move_action = torch.tensor([
        -0., 0.0, -0.4, -0.6,  # leg 1
        -0., 0.0, -0.4, -0.6,  # leg 2
        -0., 0.0, -0.4, 1.0,  # leg 3
        -0., 0.0, -0.4, 1.0   # leg 4
    ], device=env.device)

    actions = in_air_action.repeat(env.num_envs, 1)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1000 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            # First, we need to know the current contact state to decide the action
            # So we get the observation before stepping
            obs, rew, terminated, truncated, info = env.step(actions)

            # -- Check contact forces to decide which action to take
            # The contact_force_vector observation starts at index 39 and has 12 dimensions.
            contact_forces = obs["policy"][:, 39:51]
            # A boolean tensor indicating if any foot is on the ground for each environment
            is_in_contact = torch.any(contact_forces > 1.0, dim=1)

            # -- Select action based on contact state for each environment
            # `torch.where` is a vectorized if/else statement.
            # It checks the condition `is_in_contact` for each environment.
            # If True, it selects `move_action`; otherwise, it selects `in_air_action`.
            # We use unsqueeze(-1) to make the condition broadcastable to the action tensor shape.
            actions = torch.where(is_in_contact.unsqueeze(-1), move_action, in_air_action)

            # step the environment with the chosen actions

            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
