# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import MyUnitreeGo2RoughEnvCfg
from .lowg_env_cfg import LowGravityUnitreeGo2RoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

import math

@configclass
class MyUnitreeGo2FlatEnvCfg(MyUnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.7, 0.7), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi))
        # override rewards
        # rewards # default according to go2 setting
        self.rewards.track_lin_vel_xy_exp.weight = 3.0 # default 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75 # default 0.75
        self.rewards.feet_air_time.weight = 0.25 # default 0.125
        self.rewards.foot_clearance.weight = 0.1 # default 0.1
        self.rewards.feet_stance = None
        self.rewards.crawl_reward = None

        # -- penalties
        # body related
        self.rewards.lin_vel_z_l2.weight = -1.0 # default -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05 # default -0.05
        self.rewards.flat_orientation_l2.weight = -2.5 # default -2.5
        self.rewards.body_lin_acc_l2 = None # default -5.0e-4
        # joint related
        self.rewards.dof_torques_l2.weight = -1.0e-5 # default -1.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7 # default -2.5
        self.rewards.dof_vel_l2 = None # default -0.01
        self.rewards.dof_pos_limits.weight= -1.0 # default 0.0
        self.rewards.action_rate_l2.weight = -0.01 # default -0.01
        # foot related
        self.rewards.undesired_contacts.weight = -1.0 # default -1.0
        self.rewards.contact_forces = None # default -0.25s
        self.rewards.feet_contact_limit = None
        self.rewards.foot_slip = None ##EDITEDz
        self.rewards.feet_sync = None # default 10.0

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class MyUnitreeGo2FlatEnvCfg_PLAY(MyUnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class LowGravityUnitreeGo2FlatEnvCfg(LowGravityUnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        # --task
        self.rewards.track_lin_vel_xy_exp.weight = 1.5 # default 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75 # default 0.75
        
        # -- penalties
        self.rewards.ang_vel_xy_l2.weight = -0.05 # default -0.05
        self.rewards.body_lin_acc_l2.weight = -2.5e-4 # -5.0e-4
        self.rewards.dof_torques_l2 = None
        # self.rewards.dof_torques_l2.weight = -1.6e-4 # default -1.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-3 # default -2.5
        self.rewards.action_rate_l2.weight = -0.005 # default -0.01
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 1.0 # default 0.125
        self.rewards.undesired_contacts = None
        # self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh"
        # self.rewards.undesired_contacts.weight = -0.015 # default -1.0
        self.rewards.contact_forces = None
        # self.rewards.contact_forces.params["sensor_cfg"].body_names = ".*_foot"
        # self.rewards.contact_forces.weight = -0.25 # default -0.25
        self.rewards.flat_orientation_l2.weight = -0.05
        
        self.rewards.dof_pos_limits = None
        # self.rewards.dof_pos_limits.weight = -0.0 # default 0.0
        
        # self.rewards.feet_slides = None
        self.rewards.feet_slides.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = -0.1 # default -0.1

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        
        
class LowGravityUnitreeGo2FlatEnvCfg_PLAY(LowGravityUnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
