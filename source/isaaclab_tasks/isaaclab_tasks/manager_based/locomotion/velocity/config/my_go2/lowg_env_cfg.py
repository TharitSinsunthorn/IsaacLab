# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.low_G_env_cfg import LowGravityLocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class LowGravityUnitreeGo2RoughEnvCfg(LowGravityLocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # override rewards
        # --task
        self.rewards.track_lin_vel_xy_exp.weight = 2.0 # default 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75 # default 0.75
        self.rewards.feet_air_time = None # default 0.125
        self.rewards.foot_clearance.weight = 0.1 # default 0.0
        self.rewards.feet_stance.weight = 0.1
        self.rewards.crawl_reward = None
        
        # -- penalties
        # body related
        self.rewards.lin_vel_z_l2 = None # default -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05 # default -0.05
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.body_lin_acc_l2.weight = -5.0e-4 # -5.0e-4
        # joint related
        self.rewards.dof_torques_l2 = None# default -1.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7 # default -2.5
        self.rewards.dof_vel_l2.weight = -0.001 # default -0.01
        self.rewards.dof_pos_limits.weight = -1.0 # default 0.0
        self.rewards.action_rate_l2.weight = -0.005 # default -0.01
        # foot related
        self.rewards.contact_force_var.weight = -0.001 # default -0.1
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.contact_forces = None # default -0.25s
        self.rewards.feet_contact_limit = None
        self.rewards.foot_slip = None
        self.rewards.swing_impact = None
        
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class LowGravityUnitreeGo2RoughEnvCfg_PLAY(LowGravityUnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.viewer.eye = [2.5, 2.5, 1.5]
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
