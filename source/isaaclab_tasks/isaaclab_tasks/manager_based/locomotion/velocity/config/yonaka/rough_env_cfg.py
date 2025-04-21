# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.my_velocity_env_cfg import MyLocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
import os
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

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
        pos=(0.0, 0.0, 0.35),
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
            stiffness=250.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)

@configclass
class MoonbotYonakaRoughEnvCfg(MyLocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = YONAKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.scene.height_scanner.pattern_cfg.size = [1.2, 1.2]
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

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

        # rewards
        # override rewards
        # --task
        self.rewards.track_lin_vel_xy_exp.weight = 1.5 # default 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75 # default 0.75
        self.rewards.feet_air_time = None # default 0.125
        self.rewards.foot_clearance.weight = 0.5
        self.rewards.feet_stance = None
        self.rewards.crawl_reward.weight = 0.3

        # -- penalties
        # body related
        self.rewards.lin_vel_z_l2.weight = -1.0 # default -2.0
        self.rewards.ang_vel_xy_l2 = None # default -0.05
        self.rewards.flat_orientation_l2.weight = -1.0 # default -2.5
        self.rewards.body_lin_acc_l2 = None # default -5.0e-4
        # joint related
        self.rewards.dof_torques_l2.weight = -1.0e-5 # default -1.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7 # default -2.5
        self.rewards.dof_pos_limits.weight = -1.0 # default 0.0
        self.rewards.action_rate_l2.weight = -0.01 # default -0.01
        # foot related
        self.rewards.undesired_contacts = None
        self.rewards.contact_forces = None # default -0.25s
        self.rewards.feet_contact_limit.weight = -0.3
        self.rewards.foot_slip.weight = -0.05

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class MoonbotYonakaRoughEnvCfg_PLAY(MoonbotYonakaRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
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
