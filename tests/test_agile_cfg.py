import pytest
import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks
from mjlab.tasks.velocity import mdp as velocity_mdp

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.backlash import make_backlash_variant
from mjlab_microduck.tasks.microduck_agile_env_cfg import (
    make_microduck_agile_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)

WALK_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]


def test_ema_target_blend():
    alpha = torch.tensor(0.7)
    filtered = torch.tensor(0.0)
    first = microduck_mdp.ema_target_blend(torch.tensor(1.0), filtered, alpha)
    second = microduck_mdp.ema_target_blend(torch.tensor(1.0), first, alpha)
    assert first.item() == pytest.approx(0.7)
    assert second.item() == pytest.approx(0.91)


def test_resolve_ema_alphas():
    alphas = microduck_mdp.resolve_ema_alphas(
        WALK_JOINT_NAMES,
        {r".*neck.*": 0.5, r".*head.*": 0.5},
        0.7,
    )
    assert len(alphas) == 14
    assert alphas[5:9] == [0.5] * 4
    assert alphas[:5] == [0.7] * 5
    assert alphas[9:] == [0.7] * 5

    with pytest.raises(ValueError):
        microduck_mdp.resolve_ema_alphas(
            WALK_JOINT_NAMES, {r"nonexistent": 0.5}, 0.7
        )
    for invalid in (0.0, 1.5):
        with pytest.raises(ValueError):
            microduck_mdp.resolve_ema_alphas(WALK_JOINT_NAMES, {}, invalid)


def test_agile_action_wiring():
    agile = make_microduck_agile_env_cfg()
    velocity = make_microduck_velocity_env_cfg()
    action = agile.actions["joint_pos"]
    old_action = velocity.actions["joint_pos"]

    assert isinstance(action, microduck_mdp.EmaJointPositionActionCfg)
    assert action.alpha_map == {r".*neck.*": 0.5, r".*head.*": 0.5}
    assert action.default_alpha == 0.7
    assert action.scale == 1.0
    assert action.use_default_offset is True
    assert action.entity_name == old_action.entity_name
    assert action.actuator_names == old_action.actuator_names


def test_agile_raw_obs_and_command_layout():
    cfg = make_microduck_agile_env_cfg()
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        assert terms["actions"].func is velocity_mdp.last_action
        assert terms["actions"].params == {}
        assert list(terms).index("command") < list(terms).index("head_command")
        assert list(terms).index("head_command") < list(terms).index("body_command")
        assert terms["command"].params["command_name"] == "twist"
        assert terms["head_command"].params["command_name"] == "head_pose"
        assert terms["body_command"].params["command_name"] == "body_pose"


def test_agile_air_time_term():
    term = make_microduck_agile_env_cfg().rewards["air_time"]
    assert term.func is microduck_mdp.air_time_adaptive
    assert term.weight == 3.0
    assert "threshold_min" not in term.params
    assert "threshold_max" not in term.params


def test_agile_reward_signs_and_motion_blockers():
    rewards = make_microduck_agile_env_cfg().rewards
    assert rewards["joint_torques"].weight < 0
    assert rewards["action_over_limit"].weight < 0
    assert rewards["trunk_impact"].weight > 0
    assert rewards["upright"].weight == 1.0
    assert rewards["upright"].params["std"] ** 2 == pytest.approx(0.1)
    assert rewards["body_ang_vel"].weight == -0.02
    assert rewards["angular_momentum"].weight == -0.01


def test_agile_commands_and_curriculum():
    cfg = make_microduck_agile_env_cfg()
    twist = cfg.commands["twist"]
    assert twist.ranges.lin_vel_x == (-0.25, 0.25)
    assert twist.ranges.lin_vel_y == (-0.15, 0.15)
    assert twist.ranges.ang_vel_z == (-0.8, 0.8)
    assert twist.rel_turn_in_place_envs == 0.15

    capability = cfg.curriculum["command_capability"].params
    assert capability["lin_vel_y_scale"] == 0.6
    stages = capability["velocity_stages"]
    assert [s["step"] for s in stages] == sorted(s["step"] for s in stages)
    assert [s["lin_vel_range"] for s in stages] == sorted(
        s["lin_vel_range"] for s in stages
    )
    assert [s["ang_vel_range"] for s in stages] == sorted(
        s["ang_vel_range"] for s in stages
    )

    body_pose = cfg.commands["body_pose"]
    assert cfg.rewards["body_pose_tracking"].weight == 0
    assert any(lo != hi for lo, hi in body_pose.ranges)


def test_agile_standalone_invariants():
    cfg = make_microduck_agile_env_cfg()
    assert "expand_bam_friction_fields" in cfg.events
    assert cfg.events["expand_bam_friction_fields"].mode == "startup"
    assert "nan_state" in cfg.terminations
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[-1]["weight"] == -0.6


def test_agile_backlash_variant():
    cfg = make_backlash_variant(
        make_microduck_agile_env_cfg(), MICRODUCK_WALK_BACKLASH_ROBOT_CFG
    )
    assert isinstance(cfg.actions["joint_pos"], microduck_mdp.EmaJointPositionActionCfg)
    assert cfg.observations["actor"].terms["joint_pos"].func is (
        microduck_mdp.joint_pos_rel_backlash
    )
    assert cfg.observations["actor"].terms["joint_vel"].func is (
        microduck_mdp.joint_vel_rel_backlash
    )


def test_agile_registry():
    import mjlab_microduck.tasks  # noqa: F401

    task_ids = list_tasks()
    assert {
        "Mjlab-Agile-Flat-MicroDuck",
        "Mjlab-Agile-Rough-MicroDuck",
        "Mjlab-Agile-Flat-Backlash-MicroDuck",
        "Mjlab-Agile-Rough-Backlash-MicroDuck",
    } <= set(task_ids)


def test_velocity_non_regression():
    cfg = make_microduck_velocity_env_cfg()
    assert isinstance(cfg.actions["joint_pos"], JointPositionActionCfg)
    assert not isinstance(cfg.actions["joint_pos"], microduck_mdp.EmaJointPositionActionCfg)
    assert cfg.rewards["air_time"].params["threshold_min"] == 0.125
    assert cfg.rewards["upright"].weight == 2.0
