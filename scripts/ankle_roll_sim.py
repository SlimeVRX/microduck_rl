#!/usr/bin/env python3
"""Same physics tests, three robots: baseline Microduck vs the two ankle-roll builds.

Everything here is a MuJoCo rollout, not a formula. The models come from
`scripts/ankle_roll_variant.py`; the point is to answer three questions that
static geometry cannot:

  T0  Is the ankle-roll robot still a valid robot? (mass, CoM, sole height,
      and a 3 s settle test at STAND from noisy inits -- tilt, not just height.)
  T1  How hard a sideways shove survives, with *the best simple balance
      controller each robot can have*: gains are grid-searched per model, so the
      baseline is not handicapped by gains picked for the ankle-roll robot.
  T2  How far the CoM can travel sideways while both soles stay flat on the
      floor -- the quantity `hw_tradeoff.py` could only estimate kinematically.
  T3  What today's shipped policy does on the new hardware: `alpha_walking.onnx`
      drives the 14 original servos while the roll joints are held at 0. This is
      the cost side of the trade -- extra ankle mass and (for `serial`) 18 mm of
      extra leg length, with a policy that was never trained for either.

The balance controller is hand-written, not learned, so T1/T2 absolutes are a
lower bound for every robot. The comparison between columns is the result.

    uv run python scripts/ankle_roll_sim.py --walking policies/alpha_walking.onnx
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = ROOT / "src/mjlab_microduck/robot/microduck"

# 14 servos in runtime order (scripts/eval_battery.py, robotd/src/control.rs).
SERVO_ORDER = (
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
)
HOME = np.array(
    [
        0.0,
        -0.0873,
        -0.4579,
        -0.0049,
        0.4530,
        0.3491,
        0.3491,
        0.0,
        0.0,
        0.0,
        0.0873,
        0.4579,
        0.0049,
        -0.4530,
    ],
    dtype=np.float64,
)
HEAD_JOINTS = (5, 6, 7, 8)
CONTROL_HZ = 50.0
DECIMATION = 4
WALK_ACTION_SCALE = 0.9
TORQUE_LIMIT = 0.96
FALL_TILT_DEG = 45.0
FALL_Z = 0.06


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, xyz = quat[0], quat[1:4]
    t = np.cross(xyz, vec) * 2.0
    return vec - w * t + np.cross(xyz, t)


class Robot:
    """One compiled scene plus the index bookkeeping the tests need."""

    def __init__(self, scene: Path) -> None:
        self.name = scene.stem
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.key_stand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        self.trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.gyro = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        free = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.free_dof = int(m.jnt_dofadr[free])
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_geom = {
            side: mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision"
            )
            for side in ("left", "right")
        }
        self.act = {
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(m.nu)
        }
        self.servo_ctrl = [self.act[n] for n in SERVO_ORDER]
        self.roll_ctrl = [
            self.act[n]
            for n in ("left_ankle_roll", "right_ankle_roll")
            if n in self.act
        ]
        self.has_roll = bool(self.roll_ctrl)
        self.qpos_of = {
            n: int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for n, i in self.act.items()
        }
        self.qvel_of = {
            n: int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for n, i in self.act.items()
        }

    # -- state helpers -----------------------------------------------------
    def home_ctrl(self) -> np.ndarray:
        ctrl = np.zeros(self.model.nu)
        ctrl[self.servo_ctrl] = HOME
        return ctrl

    def projected_gravity(self) -> np.ndarray:
        quat = self.data.xquat[self.trunk]
        return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

    def lean(self) -> tuple[float, float]:
        """Sagittal and lateral trunk lean in radians, from projected gravity."""
        g = self.projected_gravity()
        return math.atan2(g[0], -g[2]), math.atan2(g[1], -g[2])

    def rates(self) -> np.ndarray:
        adr = self.model.sensor_adr[self.gyro]
        return self.data.sensordata[adr : adr + 3].copy()

    def tilt_deg(self) -> float:
        return math.degrees(
            math.acos(min(1.0, max(-1.0, -self.projected_gravity()[2])))
        )

    def fallen(self) -> bool:
        return self.tilt_deg() > FALL_TILT_DEG or self.data.xpos[self.trunk][2] < FALL_Z

    def sole_z(self) -> float:
        zs = []
        for gid in self.foot_geom.values():
            mujoco.mj_forward(self.model, self.data)
            zs.append(self._geom_z_min(gid))
        return min(zs)

    def _geom_z_min(self, gid: int) -> float:
        m, d = self.model, self.data
        dataid = m.geom_dataid[gid]
        verts = m.mesh_vert[
            m.mesh_vertadr[dataid] : m.mesh_vertadr[dataid] + m.mesh_vertnum[dataid]
        ].astype(np.float64)
        world = d.geom_xpos[gid] + verts @ d.geom_xmat[gid].reshape(3, 3).T
        return float(world[:, 2].min())

    def contacts(self) -> dict[str, bool]:
        out = {"left": False, "right": False}
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            for side, gid in self.foot_geom.items():
                if {con.geom1, con.geom2} == {self.floor, gid}:
                    out[side] = True
        return out

    def reset_stand(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_stand)

    def settle(self, seconds: float = 0.8, ctrl: np.ndarray | None = None) -> None:
        self.data.ctrl[:] = self.home_ctrl() if ctrl is None else ctrl
        for _ in range(int(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

    def clip(self, ctrl: np.ndarray) -> np.ndarray:
        return np.clip(
            ctrl,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )


# ---------------------------------------------------------------------------
# T0 -- is it a valid robot
# ---------------------------------------------------------------------------
def t0_static(robot: Robot, bal: Balancer, seeds: int = 5) -> dict:
    robot.reset_stand()
    mujoco.mj_forward(robot.model, robot.data)
    sole = robot.sole_z()
    com = float(robot.data.subtree_com[0][2])
    static = {
        "mass_kg": float(robot.model.body_subtreemass[1]),
        "com_above_sole_m": com - sole,
        "trunk_z_at_stand_m": float(robot.data.xpos[robot.trunk][2]),
        "nu": int(robot.model.nu),
    }

    tilts, heights, both_feet = [], [], 0
    rng = np.random.default_rng(0)
    for _ in range(seeds):
        robot.reset_stand()
        # Noisy init, as AGENTS.md demands of any pose claimed to be an equilibrium.
        qadr = [robot.qpos_of[n] for n in SERVO_ORDER]
        robot.data.qpos[qadr] += rng.normal(0.0, 0.02, len(qadr))
        bal.hold(3.0)
        tilts.append(robot.tilt_deg())
        heights.append(float(robot.data.xpos[robot.trunk][2]))
        both_feet += int(all(robot.contacts().values()))
    static |= {
        "settle_tilt_deg_max": max(tilts),
        "settle_trunk_z_m": float(np.mean(heights)),
        "settle_both_feet": f"{both_feet}/{seeds}",
    }
    return static


# ---------------------------------------------------------------------------
# balance controller (shared structure, per-model gains)
# ---------------------------------------------------------------------------
@dataclass
class Gains:
    kp_ankle_pitch: float = 1.2
    kd_ankle_pitch: float = 0.08
    kp_hip_roll: float = 1.2
    kd_hip_roll: float = 0.08
    kp_ankle_roll: float = 0.0
    kd_ankle_roll: float = 0.0


class Balancer:
    """Lean feedback into whichever joints the robot actually has.

    Identical law on every model: sagittal lean drives both ankle pitches,
    lateral lean drives both hip rolls, and -- only where the hardware exists --
    both ankle rolls. Channel signs are not guessed: `calibrate` measures which
    way each joint tips the trunk on the compiled model.
    """

    def __init__(self, robot: Robot, gains: Gains, signs: dict[str, float]) -> None:
        self.robot = robot
        self.g = gains
        self.signs = signs

    def hold(self, seconds: float, bias: float = 0.0) -> None:
        """Run the closed loop at 50 Hz, optionally with an open-loop lateral bias."""
        for _ in range(int(seconds * CONTROL_HZ)):
            self.robot.data.ctrl[:] = self.step_ctrl(bias)
            for _ in range(DECIMATION):
                mujoco.mj_step(self.robot.model, self.robot.data)
            if self.robot.fallen():
                return

    def step_ctrl(self, bias: float = 0.0) -> np.ndarray:
        r = self.robot
        theta, phi = r.lean()
        w = r.rates()
        u_pitch = self.g.kp_ankle_pitch * theta + self.g.kd_ankle_pitch * w[1]
        u_hip = self.g.kp_hip_roll * phi + self.g.kd_hip_roll * w[0] + bias
        u_ank = self.g.kp_ankle_roll * phi + self.g.kd_ankle_roll * w[0] + bias

        ctrl = r.home_ctrl()
        for side in ("left", "right"):
            ctrl[r.act[f"{side}_ankle"]] += self.signs[f"{side}_ankle"] * u_pitch
            ctrl[r.act[f"{side}_hip_roll"]] += self.signs[f"{side}_hip_roll"] * u_hip
            key = f"{side}_ankle_roll"
            if key in r.act:
                ctrl[r.act[key]] += self.signs[key] * u_ank
        return r.clip(ctrl)


def calibrate_signs(
    robot: Robot, probe: float = 0.2, window: float = 0.2
) -> dict[str, float]:
    """Which way does each joint tip the trunk, measured one joint at a time?

    Mirrored MJCF frames make the left and right joints of a pair need OPPOSITE
    control signs (`left_ankle` spins about +y, `right_ankle` about -y), so this
    probes each joint alone and keeps the sign that makes the feedback negative.
    The window is short on purpose: at HOME the servos alone cannot hold the pose,
    and both the reference and the probe rollout are sinking identically, so the
    difference is the joint's effect.
    """
    signs: dict[str, float] = {}
    for axis, names in (
        (0, ["left_ankle", "right_ankle"]),
        (1, ["left_hip_roll", "right_hip_roll", "left_ankle_roll", "right_ankle_roll"]),
    ):
        robot.reset_stand()
        robot.settle(window)
        reference = robot.lean()[axis]
        for name in names:
            if name not in robot.act:
                continue
            robot.reset_stand()
            ctrl = robot.home_ctrl()
            ctrl[robot.act[name]] += probe
            robot.settle(window, ctrl)
            signs[name] = -1.0 if robot.lean()[axis] - reference > 0 else 1.0
    return signs


def push_survives(
    robot: Robot, bal: Balancer, dv: float, axis: int, seconds: float = 2.5
) -> bool:
    robot.reset_stand()
    bal.hold(1.0)
    if robot.fallen():
        return False  # these gains cannot even stand
    robot.data.qvel[robot.free_dof + axis] += dv
    bal.hold(seconds)
    return not robot.fallen()


def max_push(
    robot: Robot, bal: Balancer, axis: int, hi: float = 2.0, iters: int = 7
) -> float:
    if not push_survives(robot, bal, 0.05, axis):
        return 0.0
    lo = 0.05
    if push_survives(robot, bal, hi, axis):
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if push_survives(robot, bal, mid, axis):
            lo = mid
        else:
            hi = mid
    return lo


# The servos alone cannot hold STAND (see docs): every gain below 1.0 rad/rad of
# lean falls over on its own, so the grid starts where standing is possible.
GAIN_GRID = (2.0, 4.0, 6.0)
DAMP_GRID = (0.06, 0.15)
ANKLE_ROLL_RATIO = (0.0, 0.5, 1.0, 2.0)


def t1_push(robot: Robot, signs: dict[str, float]) -> dict:
    """Best lateral/sagittal shove each robot survives, over its own gain grid."""
    best = {"lateral": (0.0, None), "sagittal": (0.0, None)}
    for kp in GAIN_GRID:
        for kd in DAMP_GRID:
            for ratio in ANKLE_ROLL_RATIO if robot.has_roll else (0.0,):
                gains = Gains(
                    kp_ankle_pitch=kp,
                    kd_ankle_pitch=kd,
                    kp_hip_roll=kp,
                    kd_hip_roll=kd,
                    kp_ankle_roll=ratio * kp,
                    kd_ankle_roll=ratio * kd,
                )
                bal = Balancer(robot, gains, signs)
                lat = max_push(robot, bal, axis=1)
                if lat > best["lateral"][0]:
                    best["lateral"] = (lat, gains)
                sag = max_push(robot, bal, axis=0)
                if sag > best["sagittal"][0]:
                    best["sagittal"] = (sag, gains)
    out = {}
    for key, (value, gains) in best.items():
        out[f"max_push_{key}_mps"] = value
        out[f"gains_{key}"] = None if gains is None else vars(gains)
    return out


# ---------------------------------------------------------------------------
# T2 -- how far the CoM travels sideways with both soles down
# ---------------------------------------------------------------------------
def t2_lateral_com(
    robot: Robot, bal: Balancer, step: float = 0.02, limit: float = 1.2
) -> dict:
    """Ramp an open-loop sideways bias on top of the balancer; how far does the CoM go?

    Both robots get the same bias on the same channel; the ankle-roll robot also
    feeds it to its roll joints, which is exactly the capability being priced.
    Only samples where BOTH soles still touch the floor count.
    """
    robot.reset_stand()
    bal.hold(1.0)
    mujoco.mj_forward(robot.model, robot.data)
    y0 = float(robot.data.subtree_com[0][1])
    best = 0.0
    bias = 0.0
    while bias < limit:
        bias += step
        bal.hold(0.15, bias=bias)
        if robot.fallen():
            break
        if all(robot.contacts().values()):
            best = max(best, abs(float(robot.data.subtree_com[0][1]) - y0))
    return {"flat_sole_com_travel_mm": 1000.0 * best, "flat_sole_bias_end_rad": bias}


# ---------------------------------------------------------------------------
# T3 -- today's shipped policy on the new hardware
# ---------------------------------------------------------------------------
def t3_walking(
    robot: Robot,
    onnx_path: Path,
    seeds: int = 5,
    duration: float = 6.0,
    legs_lowpass: float = 0.7,
    head_lowpass: float = 0.5,
) -> dict:
    """Several rollouts, because one 6 s rollout of a 1.7 Hz gait is noise."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    runs = [
        _walk_once(robot, sess, seed, duration, legs_lowpass, head_lowpass)
        for seed in range(seeds)
    ]
    speeds = [r["speed"] for r in runs]
    return {
        "walk_speed_mps": float(np.mean(speeds)),
        "walk_speed_min_max": [float(np.min(speeds)), float(np.max(speeds))],
        "walk_falls": f"{sum(r['fell'] is not None for r in runs)}/{seeds}",
        "walk_torque_sat_frac": float(np.mean([r["sat"] for r in runs])),
        "walk_torque_peak_nm": float(np.max([r["peak"] for r in runs])),
    }


def _walk_once(
    robot: Robot,
    sess,
    seed: int,
    duration: float,
    legs_lowpass: float,
    head_lowpass: float,
) -> dict:
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    qadr = [robot.qpos_of[n] for n in SERVO_ORDER]
    vadr = [robot.qvel_of[n] for n in SERVO_ORDER]
    command = np.zeros(13, dtype=np.float32)
    command[0] = 0.4  # forward, mid-range of what E0 measured as reachable

    robot.reset_stand()
    rng = np.random.default_rng(seed)
    robot.data.qpos[qadr] += rng.normal(0.0, 0.01, len(qadr))
    robot.settle(0.6)
    last_action = np.zeros(14, dtype=np.float32)
    filtered: np.ndarray | None = None
    p0 = robot.data.xpos[robot.trunk].copy()
    yaws, sat, torque_peak = [], [], []
    fell_at = None

    for step in range(int(duration * CONTROL_HZ)):
        gyro = robot.rates().astype(np.float32)
        obs = np.concatenate(
            [
                gyro,
                robot.projected_gravity().astype(np.float32),
                robot.data.qpos[qadr].astype(np.float32) - HOME.astype(np.float32),
                robot.data.qvel[vadr].astype(np.float32),
                last_action,
                command,
            ]
        ).astype(np.float32)
        action = sess.run([out_name], {in_name: obs.reshape(1, -1)})[0].squeeze(0)
        last_action = action.copy()
        target = HOME + WALK_ACTION_SCALE * action
        if filtered is not None:
            for j in range(14):
                a = head_lowpass if j in HEAD_JOINTS else legs_lowpass
                target[j] = a * target[j] + (1 - a) * filtered[j]
        filtered = target.copy()
        ctrl = np.zeros(robot.model.nu)
        ctrl[robot.servo_ctrl] = target
        # The roll joints exist but no trained policy commands them: hold them home.
        robot.data.ctrl[:] = ctrl
        for _ in range(DECIMATION):
            mujoco.mj_step(robot.model, robot.data)
        tq = np.abs(robot.data.actuator_force[: robot.model.nu])
        torque_peak.append(float(tq.max()))
        sat.append(float((tq > 0.98 * TORQUE_LIMIT).any()))
        quat = robot.data.xquat[robot.trunk]
        yaws.append(
            math.atan2(
                2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
            )
        )
        if robot.fallen():
            fell_at = step / CONTROL_HZ
            break

    disp = robot.data.xpos[robot.trunk] - p0
    elapsed = (len(sat)) / CONTROL_HZ
    heading = float(np.mean(yaws)) if yaws else 0.0
    forward = disp[0] * math.cos(heading) + disp[1] * math.sin(heading)
    return {
        "speed": forward / elapsed if elapsed > 0 else 0.0,
        "fell": fell_at,
        "sat": float(np.mean(sat)) if sat else 0.0,
        "peak": float(np.max(torque_peak)) if torque_peak else 0.0,
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scenes",
        nargs="*",
        default=[
            "scene_walk.xml",
            "scene_walk_ankleroll_coincident.xml",
            "scene_walk_ankleroll_serial.xml",
        ],
    )
    ap.add_argument("--walking", default=None, help="alpha_walking.onnx for T3")
    ap.add_argument(
        "--walk-scenes",
        nargs="*",
        default=[
            "scene.xml",
            "scene_ankleroll_coincident.xml",
            "scene_ankleroll_serial.xml",
        ],
        help="allcollisions scenes for T3 (the model E0 used)",
    )
    ap.add_argument("--only", nargs="*", default=None, choices=["t0", "t1", "t2", "t3"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    want = set(args.only) if args.only else {"t0", "t1", "t2", "t3"}

    results: dict[str, dict] = {}
    for scene in args.scenes:
        robot = Robot(ROBOT_DIR / scene)
        row: dict = {}
        signs = calibrate_signs(robot)
        row["signs"] = signs
        reference = Balancer(robot, Gains(4.0, 0.15, 4.0, 0.15, 0.0, 0.0), signs)
        if "t0" in want:
            row |= t0_static(robot, reference)
        if "t1" in want:
            row |= t1_push(robot, signs)
        if "t2" in want:
            best = row.get("gains_lateral")
            bal = Balancer(robot, Gains(**best) if best else reference.g, signs)
            row |= t2_lateral_com(robot, bal)
        results[robot.name] = row
        if len(row) > 1:
            print(
                f"{robot.name}: {json.dumps({k: v for k, v in row.items() if k != 'signs'})}"
            )

    if "t3" in want and args.walking:
        for scene in args.walk_scenes:
            robot = Robot(ROBOT_DIR / scene)
            row = t3_walking(robot, Path(args.walking))
            results.setdefault(robot.name, {}).update(row)
            print(f"{robot.name}: {json.dumps(row)}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
