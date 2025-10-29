# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.managers import SceneEntityCfg

import numpy as np
import csv
import os

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent with a structured, round-based evaluation protocol."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    # Ensure only one environment is run for simple evaluation logging
    # Note: If you want to run multiple envs, you must adapt the logging to handle the batch data.
    if env_cfg.scene.num_envs > 1:
        print("[WARNING] Setting number of environments to 1 for simplified evaluation logging.")
        env_cfg.scene.num_envs = 1

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # --- START Custom Evaluation Setup ---
    unwrapped_env = env.unwrapped
    if args_cli.video:
        unwrapped_env = unwrapped_env.unwrapped

    # asset_cfg = SceneEntityCfg(name="robot")
    # robot = unwrapped_env.unwrapped.scene[asset_cfg.name]
    robot = unwrapped_env.unwrapped.scene["robot"]

    # TOTAL MASS: Sum the mass of all individual bodies
    robot_mass = torch.sum(robot.data.default_mass[0]).item()
    gravity = abs(unwrapped_env.unwrapped.sim._gravity_tensor[2].item())
    # print(f"[EVAL] Robot mass: {robot_mass:.2f} kg")
    print(f"[EVAL] Using gravity: {gravity:.2f} m/s^2 for CoT calculation.")

    # Setup for round-based evaluation logging
    eval_log_path = os.path.join(log_dir, "locomotion_eval", f"policy_eval_var.csv")
    os.makedirs(os.path.dirname(eval_log_path), exist_ok=True)

    eval_log_file = open(eval_log_path, "w", newline="")
    csv_writer = csv.writer(eval_log_file)
    csv_writer.writerow(["round", "target_vel_x", "avg_velocity", "avg_power", "cot"])
    print(f"[EVAL] Logging round summaries to: {eval_log_path}")

    dt = env.unwrapped.step_dt
    stabilization_steps = 500  # Timesteps to ignore at the start of each round

    # Define the evaluation protocol
    target_velocities = np.arange(0.1, 1.11, 0.1)

    # --- END Custom Evaluation Setup ---

    # --- Main Evaluation Loop ---
    for round_idx, target_vel in enumerate(target_velocities):
        if not simulation_app.is_running():
            break

        print("-" * 80)
        print(f"[ROUND {round_idx + 1}/{len(target_velocities)}] Starting evaluation with target velocity: {target_vel:.2f} m/s")
        print(f"[EVAL] Robot mass: {robot_mass:.2f} kg")
        print("-" * 80)

        # Reset environment and metrics for the new round
        # WRAP RESET IN INFERENCE MODE TO PREVENT TENSOR STATE CONFLICT
        with torch.inference_mode():
            obs, _ = env.reset()
        
        round_step_count = 0
        round_velocities = []
        round_powers = []

        # Inner loop for a single episode (one round)
        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                # --- Set the constant command for this round ---
                command_term = unwrapped_env.command_manager.get_term("base_velocity")
                command_tensor = torch.tensor([target_vel, 0.0, 0.0], device=unwrapped_env.device)
                command_term.command[:] = command_tensor.repeat(unwrapped_env.num_envs, 1)

                # Agent and environment stepping
                actions = policy(obs)
                obs, rewards, dones, infos = env.step(actions)

            round_step_count += 1

            # --- Collect data only after stabilization period ---
            if round_step_count > stabilization_steps:
                v_actual = torch.linalg.norm(robot.data.root_lin_vel_b[:, 0:2], dim=1)[0].cpu().numpy()
                # tau = robot.data.applied_torque[:, asset_cfg.joint_ids].cpu().numpy()
                # qd = robot.data.joint_vel[:, asset_cfg.joint_ids].cpu().numpy()
                tau = robot.data.applied_torque.cpu().numpy()
                qd = robot.data.joint_vel.cpu().numpy()
                power = float(np.sum(np.abs(np.multiply(tau, qd))))

                round_velocities.append(v_actual)
                round_powers.append(power)

            # Check for episode termination (timeout or failure)
            if dones[0]:
                print(f"[ROUND {round_idx + 1}] Episode finished at step {round_step_count}. Calculating metrics...")
                break

            # Real-time delay
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

        # --- End of Round: Calculate and Log Metrics ---
        if round_velocities:  # Check if any data was collected
            avg_vel = np.mean(round_velocities)
            avg_power = np.mean(round_powers)
            cot = avg_power / (robot_mass * gravity * avg_vel) if avg_vel > 0.01 else float('inf')

            # Log to console
            print("\n[ROUND METRICS]")
            print(f"  Target Velocity:    {target_vel:.2f} m/s")
            print(f"  Avg. Actual Vel:    {avg_vel:.4f} m/s")
            print(f"  Avg. Power:         {avg_power:.4f} W")
            print(f"  Cost of Transport:  {cot:.4f}")

            # Log to CSV
            csv_writer.writerow([round_idx + 1, target_vel, avg_vel, avg_power, cot])
            eval_log_file.flush()  # Ensure data is written immediately
        else:
            print(f"[ROUND {round_idx + 1}] No metrics collected (episode ended before {stabilization_steps} steps). Skipping.")

    # --- Final Cleanup ---
    print("\n" + "=" * 80)
    print("Evaluation protocol finished.")
    print(f"Round summaries saved to: {eval_log_path}")
    print("=" * 80)

    if eval_log_file:
        eval_log_file.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
