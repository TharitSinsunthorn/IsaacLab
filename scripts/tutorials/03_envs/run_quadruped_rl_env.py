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
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.my_go2.flat_env_cfg import MyUnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.my_go2.lowg_env_cfg import LowGravityUnitreeGo2RoughEnvCfg


def main():
    """Main function."""
    # create environment configuration
    env_cfg = LowGravityUnitreeGo2RoughEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.actions.quadruped_action_cfg.scale = 0.5
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 500 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            ee_pos = torch.zeros_like(env.action_manager.action)

            # Assuming ee_pos is shaped [num_envs, 12]
            # Define the target EE positions for one environment
            target_ee = torch.tensor([
                1.0, 0.0, -1.0, 1.0,   # leg 1
                1.0, 0.0, -1.0, 1.0,   # leg 2
                1.0, 0.0, -1.0, 1.0,   # leg 3
                1.0, 0.0, -1.0, 1.0    # leg 4
                ], device=ee_pos.device)

            # Repeat this across all environments
            ee_pos[:] = target_ee.unsqueeze(0).repeat(ee_pos.shape[0], 1)  # Shape [num_envs, 12]

            # step the environment
            obs, rew, terminated, truncated, info = env.step(ee_pos)
            # print current observation
            # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
