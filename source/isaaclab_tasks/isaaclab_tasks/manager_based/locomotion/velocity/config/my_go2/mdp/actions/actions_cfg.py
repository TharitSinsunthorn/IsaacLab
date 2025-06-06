from dataclasses import MISSING

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import cpg_modulator_action, task_space_actions
from isaaclab.envs.mdp.actions import JointActionCfg

import torch
import numpy as np

##
# Joint actions.
##

@configclass
class QuadrupedDiffIKActionCfg(ActionTermCfg):
    """Configuration for inverse differential kinematics action term.

    See :class:`QuadrupedDiffIKAction` for more details.
    """
    @configclass
    class LegCfg:

        @configclass
        class OffsetCfg:
            """The offset pose from parent frame to child frame.

            On many robots, end-effector frames are fictitious frames that do not have a corresponding
            rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
            For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
            "panda_hand" frame.
            """

            pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
            """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
            rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
            """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

        joint_names: list[str] = MISSING
        """List of joint names or regex expressions that the action will be mapped to."""
        body_name: str = MISSING
        """Name of the body or frame for which IK is performed."""
        body_offset: OffsetCfg | None = None
        """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""


    class_type: type[ActionTerm] = task_space_actions.QuadrupedDiffIKAction

    legs: dict[str, LegCfg] = MISSING
    """Dictionary of leg name (e.g., 'FL', 'FR') to leg-specific configuration."""
    
    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""
    controller: DifferentialIKControllerCfg = MISSING
    """The configuration for the differential IK controller."""


@configclass
class CPGQuadrupedActionCfg(ActionTermCfg):
    """
    Configuration for a CPG-based action term for quadruped locomotion.
    RL agent modulates CPG parameters (mu, omega, psi) per leg.
    """
    @configclass
    class LegCfg:
        """Leg-specific configuration for the CPG and IK setup."""
        @configclass
        class OffsetCfg:
            pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
            rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

        joint_names: list[str] = MISSING
        body_name: str = MISSING
        body_offset: OffsetCfg | None = None

        # CPG specific initial/default values for each leg
        init_mux: float = 1.5 # Default target amplitude
        init_muy: float = 1.5 # Default target amplitude
        init_omega: float = 0.0 # Default 1 Hz frequency
        
        init_theta: float = 0.0 # Default phase offset
        hip_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    coupling_weights: dict[str, float] = {
        "FL_FR": 6.5, "FR_FL": 6.5, # Front legs anti-phase
        "RL_RR": 6.5, "RR_RL": 6.5, # Rear legs anti-phase
        "FL_RR": 6.5, "RR_FL": 6.5, # Diagonal coupling (FL-RR) anti-phase
        "FR_RL": 6.5, "RL_FR": 6.5, # Diagonal coupling (FR-RL) anti-phase
        "FL_RL": 0.0, "RL_FL": 0.0,  # No direct front-to-rear coupling (trot is more diagonal)
        "FR_RR": 0.0, "RR_FR": 0.0,
    }
    phase_offsets: dict[str, float] = {
        "FL_FR": torch.pi, "FR_FL": torch.pi,
        "RL_RR": torch.pi, "RR_RL": torch.pi,
        "FL_RR": 0.0, "RR_FL": 0.0,
        "FR_RL": 0.0, "RL_FR": 0.0,
        "FL_RL": torch.pi, "RL_FL": torch.pi,  # No direct front-to-rear coupling (trot is more diagonal)
        "FR_RR": torch.pi, "RR_FR": torch.pi,
    }
    coupling_enable: bool = False

    # Point to the actual class
    class_type: type[ActionTerm] = cpg_modulator_action.CPGQuadrupedAction # This will be set in env_cfg.py as CPGQuadrupedAction

    # Ranges for the RL agent's output for each CPG parameter
    mu_range: tuple[float, float] = (1.0, 2.0)
    omega_range: tuple[float, float] = (0.0, 30.0) # Hz

    # Global CPG parameters (if not modulated by RL per-leg)
    global_gc: float = 0.1 # Ground clearance
    global_gp: float = 0.05 # Ground penetration
    global_h: float = 0.22 # Robot height
    global_d_step: float = 0.2 # Step size scale
    cpg_alpha: float = 150.0 # 'a' parameter for CPG amplitude dynamics

    legs: dict[str, LegCfg] = MISSING
    """Dictionary of leg name (e.g., 'FL', 'FR') to leg-specific configuration."""

    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""
    controller: DifferentialIKControllerCfg = MISSING
    """The configuration for the differential IK controller."""

