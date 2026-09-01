"""Microduck Agile environment.

E0 motivates this delta:

* The deployed policy saturated at 0.229 m/s for a 0.8 m/s command, while
  0.4 m/s produced essentially the same speed, so the command curriculum is
  capability-matched instead of human-sized.
* Lateral speed was only 0.022 m/s at a 0.4 m/s command, and every failed push
  was lateral, so lateral commands start smaller and grow separately.
* The filtered gait measured 94.3% single support, 3.0% flight, and 1.70 Hz,
  so the fixed air-time window and strong upright prior are replaced with
  adaptive timing and lighter motion blockers.
* Unfiltered torque saturation was 9.3% versus 0% filtered, so training uses
  the runtime's per-joint target EMA (legs 0.7, head/neck 0.5). The raw action
  observation remains unchanged.

These hypotheses are falsified by a later E1 battery if capability-matched
commands still saturate, lateral tracking remains near zero, adaptive timing
does not change the gait structure, or runtime-filtered transfer still shows
the E0 saturation and impact-cost pattern.
"""

import math
from copy import deepcopy
from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg


def make_microduck_agile_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the Agile variant without mutating the velocity recipe."""
    cfg = deepcopy(make_microduck_velocity_env_cfg(play=play, rough=rough))

    old_action = cfg.actions["joint_pos"]
    assert isinstance(old_action, JointPositionActionCfg)
    action_kwargs = {f.name: getattr(old_action, f.name) for f in fields(old_action)}
    action_kwargs.update(
        alpha_map={r".*neck.*": 0.5, r".*head.*": 0.5},
        default_alpha=0.7,
    )
    cfg.actions["joint_pos"] = microduck_mdp.EmaJointPositionActionCfg(
        **action_kwargs
    )

    cfg.rewards["air_time"] = RewardTermCfg(
        func=microduck_mdp.air_time_adaptive,
        weight=3.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.01,
            "running_threshold": 0.30,
            "walk_threshold_min": 0.10,
            "walk_threshold_max": 0.30,
            "run_threshold_min": 0.06,
            "run_threshold_max": 0.22,
        },
    )
    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.1)
    cfg.rewards["body_ang_vel"].weight = -0.02
    cfg.rewards["angular_momentum"].weight = -0.01
    cfg.rewards["joint_torques"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2,
        weight=-0.01,
    )
    cfg.rewards["action_over_limit"] = RewardTermCfg(
        func=microduck_mdp.action_over_limit_penalty,
        weight=-0.5,
        params={"action_name": "joint_pos", "overshoot": 0.3},
    )
    # trunk_vertical_accel_penalty is self-negating (returns <= 0), hence
    # this positive weight makes impacts a cost rather than a reward.
    cfg.rewards["trunk_impact"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.002,
    )

    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-0.25, 0.25)
    twist.ranges.lin_vel_y = (-0.15, 0.15)
    twist.ranges.ang_vel_z = (-0.8, 0.8)
    twist.rel_turn_in_place_envs = 0.15
    cfg.curriculum["command_capability"] = CurriculumTermCfg(
        func=microduck_mdp.velocity_command_ranges_curriculum,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {"step": 0, "lin_vel_range": 0.25, "ang_vel_range": 0.8},
                {
                    "step": 1000 * NUM_STEPS_PER_ENV,
                    "lin_vel_range": 0.35,
                    "ang_vel_range": 1.2,
                },
                {
                    "step": 2000 * NUM_STEPS_PER_ENV,
                    "lin_vel_range": 0.45,
                    "ang_vel_range": 1.5,
                },
                {
                    "step": 3000 * NUM_STEPS_PER_ENV,
                    "lin_vel_range": 0.55,
                    "ang_vel_range": 1.8,
                },
            ],
            "lin_vel_y_scale": 0.6,
        },
    )

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.2},
                {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.3},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.45},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -0.6},
            ],
        },
    )

    return cfg


MicroduckAgileRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="agile",
    run_name="agile",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
