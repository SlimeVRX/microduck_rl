#!/usr/bin/env python3
"""Headless agility battery for a deployed walking ONNX, with the runtime's
target low-pass as an A/B knob.

Why this exists: "more agile" is not measurable, so every architecture claim in
`docs/agility-breakthrough-plan.md` is gated on six numbers. This script produces
them from a checkpoint without training anything.

The control path mirrors the runtime (`microduck`'s `robotd/src/control.rs`) rather
than the training env:

    obs[1,61]  = gyro(3) | projected gravity(3) | q-home(14) | qdot(14) | last action(14)
                 | twist(3) head(4) body(6)
    action     = policy(obs)                       # fed back RAW, unfiltered
    target     = HOME + action_scale * action      # action_scale 0.9 for walking
    target'    = alpha * target + (1-alpha) * prev_target   # per-group EMA, prev = FILTERED

`--legs-lowpass` / `--head-lowpass` take the runtime defaults (0.7 / 0.5) or 1.0,
which is the unfiltered path training assumes ("Policies are UNFILTERED").
Absolute numbers are sim numbers on plain position actuators — the A/B delta
between two arms sharing one actuator model is the result, not the absolutes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field

import mujoco
import numpy as np
import onnxruntime as ort

SCENE_XML = "src/mjlab_microduck/robot/microduck/scene.xml"
CONTROL_HZ = 50.0
DECIMATION = 4
HEAD_JOINTS = (5, 6, 7, 8)  # neck_pitch, head_pitch, head_yaw, head_roll
TORQUE_LIMIT = 0.96  # Nm, `chosen_actuator` forcerange in robot_allcollisions.xml

# Runtime constants this control path is replicated from (repo `microduck`).
STANDING_THRESHOLD = 0.05   # duck-control/src/policy.rs: DEFAULT_STANDING_THRESHOLD
WALK_ACTION_SCALE = 0.9     # robotd/src/control.rs: Tuning::default().action_scale
STAND_ACTION_SCALE = 1.0    # ... standing_action_scale
STANDING_GAIN_RATIO = 0.8   # ... standing_gain_ratio

# HOME frame (STAND2), identical to scripts/infer_policy.py's DEFAULT_POSE.
HOME = np.array(
    [
        0.0, -0.0873, -0.4579, -0.0049, 0.4530,   # left leg
        0.3491, 0.3491, 0.0, 0.0,                 # neck + head
        0.0, 0.0873, 0.4579, 0.0049, -0.4530,     # right leg
    ],
    dtype=np.float32,
)


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, xyz = quat[0], quat[1:4]
    t = np.cross(xyz, vec) * 2.0
    return vec - w * t + np.cross(xyz, t)


@dataclass
class Trace:
    """Per-tick history of one rollout."""

    t: list[float] = field(default_factory=list)
    vx: list[float] = field(default_factory=list)
    vy: list[float] = field(default_factory=list)
    wz: list[float] = field(default_factory=list)
    px: list[float] = field(default_factory=list)
    py: list[float] = field(default_factory=list)
    yaw: list[float] = field(default_factory=list)
    z: list[float] = field(default_factory=list)
    tilt: list[float] = field(default_factory=list)
    az: list[float] = field(default_factory=list)
    torque_peak: list[float] = field(default_factory=list)
    saturated: list[float] = field(default_factory=list)
    action_rate: list[float] = field(default_factory=list)
    contacts: list[tuple[bool, bool]] = field(default_factory=list)
    foot_z: list[tuple[float, float]] = field(default_factory=list)
    slip: list[float] = field(default_factory=list)
    fell_at: float | None = None

    def arr(self, name: str) -> np.ndarray:
        return np.asarray(getattr(self, name), dtype=np.float64)

    def window(self, start_s: float) -> slice:
        t = self.arr("t")
        return slice(int(np.searchsorted(t, start_s)), len(t))


class Rollout:
    """One closed-loop episode of the walking policy in MuJoCo."""

    def __init__(
        self,
        model: mujoco.MjModel,
        session: ort.InferenceSession,
        action_scale: float,
        legs_lowpass: float | None,
        head_lowpass: float | None,
        stand_session: ort.InferenceSession | None = None,
        kp_nominal: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.session = session
        self.stand_session = stand_session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        self.action_scale = action_scale
        # Position-actuator gain, so the runtime's softer standing gain is reproducible:
        # for `position`, gainprm[0] = kp and biasprm[1] = -kp. `kp_nominal` must come from
        # the freshly compiled model — reading it here would compound the previous rollout's
        # standing ratio into the next rollout's nominal.
        self.kp_nominal = (
            model.actuator_gainprm[: model.nu, 0].copy() if kp_nominal is None else kp_nominal
        )
        self.legs_lowpass = legs_lowpass
        self.head_lowpass = head_lowpass
        self.nu = model.nu

        self.qpos_idx = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
        self.qvel_idx = [int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
        self.trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.accel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        trunk_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.trunk_qvel_adr = int(model.jnt_dofadr[trunk_jid])
        self.floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_geom = {
            side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
            for side in ("left", "right")
        }
        for side, gid in self.foot_geom.items():
            if gid < 0:
                raise ValueError(f"geom {side}_foot_collision not found in the scene")
        self.foot_body = {
            side: int(model.geom_bodyid[gid]) for side, gid in self.foot_geom.items()
        }
        self.key_stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")

    # -- observation -------------------------------------------------------
    def _projected_gravity(self) -> np.ndarray:
        quat = self.data.xquat[self.trunk_id].astype(np.float32)
        return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))

    def _sensor(self, sensor_id: int) -> np.ndarray:
        adr = self.model.sensor_adr[sensor_id]
        return self.data.sensordata[adr : adr + 3].copy().astype(np.float32)

    def _set_gain_ratio(self, ratio: float) -> None:
        kp = self.kp_nominal * ratio
        self.model.actuator_gainprm[: self.nu, 0] = kp
        self.model.actuator_biasprm[: self.nu, 1] = -kp

    def observation(self, command: np.ndarray) -> np.ndarray:
        q = self.data.qpos[self.qpos_idx].astype(np.float32) - HOME[: self.nu]
        qd = self.data.qvel[self.qvel_idx].astype(np.float32)
        return np.concatenate(
            [self._sensor(self.gyro_id), self._projected_gravity(), q, qd, self.last_action, command]
        ).astype(np.float32)

    # -- one rollout -------------------------------------------------------
    def run(
        self,
        duration: float,
        command_fn,
        push_fn=None,
        settle: float = 0.6,
        seed: int = 0,
    ) -> Trace:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_stand)
        self.last_action = np.zeros(self.nu, dtype=np.float32)
        self.filtered: np.ndarray | None = None
        self._set_gain_ratio(1.0)
        rng = np.random.default_rng(seed)
        trace = Trace()
        control_dt = DECIMATION * self.model.opt.timestep

        # Settle at HOME so every arm starts from the same standing state.
        self.data.ctrl[:] = HOME[: self.nu]
        for _ in range(int(settle / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

        prev_action = np.zeros(self.nu, dtype=np.float32)
        steps = round(duration * CONTROL_HZ)
        for step in range(steps):
            t = step / CONTROL_HZ
            command = np.asarray(command_fn(t), dtype=np.float32)
            # Runtime scheduler: below the standing threshold the twist command hands the
            # tick to the standing net, at its own action scale and a softer servo gain.
            standing = (
                self.stand_session is not None
                and float(np.linalg.norm(command[:3])) <= STANDING_THRESHOLD
            )
            session = self.stand_session if standing else self.session
            scale = STAND_ACTION_SCALE if standing else self.action_scale
            self._set_gain_ratio(STANDING_GAIN_RATIO if standing else 1.0)
            action = session.run(
                [session.get_outputs()[0].name],
                {session.get_inputs()[0].name: self.observation(command).reshape(1, -1)},
            )[0].squeeze(0).astype(np.float32)
            self.last_action = action.copy()  # RAW feedback, as the runtime does

            target = HOME[: self.nu] + scale * action
            if self.filtered is not None:
                if self.head_lowpass is not None:
                    for j in HEAD_JOINTS:
                        target[j] = self.head_lowpass * target[j] + (1 - self.head_lowpass) * self.filtered[j]
                if self.legs_lowpass is not None:
                    for j in range(self.nu):
                        if j in HEAD_JOINTS:
                            continue
                        target[j] = self.legs_lowpass * target[j] + (1 - self.legs_lowpass) * self.filtered[j]
            self.filtered = target.copy()
            self.data.ctrl[:] = target

            if push_fn is not None:
                push_fn(t, self.data, self.trunk_qvel_adr, rng)

            for _ in range(DECIMATION):
                mujoco.mj_step(self.model, self.data)

            self._record(trace, t, action, prev_action, control_dt)
            prev_action = action
            if trace.fell_at is not None:
                break
        return trace

    def _record(self, trace: Trace, t: float, action, prev_action, control_dt: float) -> None:
        d = self.data
        gravity = self._projected_gravity()
        lin_world = d.qvel[self.trunk_qvel_adr : self.trunk_qvel_adr + 3].copy()
        quat = d.xquat[self.trunk_id].astype(np.float32)
        lin_body = quat_rotate_inverse(quat, lin_world.astype(np.float32))
        torque = np.abs(d.actuator_force[: self.nu])

        trace.t.append(t)
        trace.vx.append(float(lin_body[0]))
        trace.vy.append(float(lin_body[1]))
        trace.wz.append(float(d.qvel[self.trunk_qvel_adr + 5]))
        trace.z.append(float(d.xpos[self.trunk_id][2]))
        trace.px.append(float(d.xpos[self.trunk_id][0]))
        trace.py.append(float(d.xpos[self.trunk_id][1]))
        trace.yaw.append(
            float(math.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                             1 - 2 * (quat[2] ** 2 + quat[3] ** 2)))
        )
        trace.tilt.append(float(math.degrees(math.acos(min(1.0, max(-1.0, -gravity[2]))))))
        trace.az.append(float(self._sensor(self.accel_id)[2]))
        trace.torque_peak.append(float(torque.max()))
        trace.saturated.append(float((torque > 0.98 * TORQUE_LIMIT).any()))
        trace.action_rate.append(float(np.abs(action - prev_action).mean() / control_dt))

        contacts, slip = self._foot_state()
        trace.contacts.append(contacts)
        trace.slip.append(slip)
        trace.foot_z.append(
            tuple(float(d.geom_xpos[self.foot_geom[s]][2]) for s in ("left", "right"))
        )

        # Fallen: trunk tilted past 45 deg or dropped below 60 mm (stand ~117 mm).
        if trace.tilt[-1] > 45.0 or trace.z[-1] < 0.06:
            trace.fell_at = t

    def _foot_state(self) -> tuple[tuple[bool, bool], float]:
        d = self.data
        contact = {"left": False, "right": False}
        slip = 0.0
        for i in range(d.ncon):
            con = d.contact[i]
            for side, gid in self.foot_geom.items():
                if {con.geom1, con.geom2} == {self.floor_id, gid}:
                    contact[side] = True
        for side, bid in self.foot_body.items():
            if contact[side]:
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
                slip = max(slip, float(np.linalg.norm(vel[3:5])))
        return (contact["left"], contact["right"]), slip


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def cmd(vx=0.0, vy=0.0, wz=0.0) -> np.ndarray:
    """61D-compatible 13D command block: twist(3) | head(4) | body(6), rest zero."""
    out = np.zeros(13, dtype=np.float32)
    out[0], out[1], out[2] = vx, vy, wz
    return out


def steady(trace: Trace, start_s: float) -> dict:
    """Achieved twist over the steady window.

    Linear speeds come from net displacement rotated into the robot's mean heading,
    not from averaging instantaneous body-frame velocity: at 1.7 Hz step frequency the
    instantaneous signal is dominated by per-step trunk sway, which averages to
    something that is not the travel speed.
    """
    w = trace.window(start_s)
    t = trace.arr("t")[w]
    if t.size < 2:
        return {"vx": float("nan"), "vy": float("nan"), "wz": float("nan")}
    span = float(t[-1] - t[0])
    dx = float(trace.arr("px")[w][-1] - trace.arr("px")[w][0])
    dy = float(trace.arr("py")[w][-1] - trace.arr("py")[w][0])
    yaw = np.unwrap(trace.arr("yaw")[w])
    heading = float(np.mean(yaw))
    fwd = (dx * math.cos(heading) + dy * math.sin(heading)) / span
    lat = (-dx * math.sin(heading) + dy * math.cos(heading)) / span
    return {
        "vx": fwd,
        "vy": lat,
        "wz": float((yaw[-1] - yaw[0]) / span),
    }


def cost_metrics(trace: Trace, start_s: float) -> dict:
    w = trace.window(start_s)
    az = trace.arr("az")[w]
    return {
        "az_peak": float(np.max(np.abs(az))) if az.size else float("nan"),
        "az_p95": float(np.percentile(np.abs(az), 95)) if az.size else float("nan"),
        "torque_peak": float(np.max(trace.arr("torque_peak")[w])) if az.size else float("nan"),
        "torque_sat_frac": float(np.mean(trace.arr("saturated")[w])) if az.size else float("nan"),
        "action_rate_mean": float(np.mean(trace.arr("action_rate")[w])) if az.size else float("nan"),
        "slip_p95": float(np.percentile(trace.arr("slip")[w], 95)) if az.size else float("nan"),
    }


def gait_metrics(trace: Trace, start_s: float) -> dict:
    w = trace.window(start_s)
    contacts = trace.contacts[w]
    if not contacts:
        return {}
    both = sum(1 for c in contacts if c[0] and c[1])
    none = sum(1 for c in contacts if not c[0] and not c[1])
    single = len(contacts) - both - none
    foot_z = np.asarray(trace.foot_z[w])
    left = np.asarray([c[0] for c in contacts])
    # step frequency from left-foot touchdown events
    touchdowns = np.flatnonzero((~left[:-1]) & left[1:])
    if touchdowns.size >= 2:
        step_hz = CONTROL_HZ / float(np.mean(np.diff(touchdowns)))
    else:
        step_hz = float("nan")
    return {
        "double_support_frac": both / len(contacts),
        "single_support_frac": single / len(contacts),
        "flight_frac": none / len(contacts),
        "step_freq_hz": step_hz,
        "foot_clearance_max_mm": float(1000.0 * (foot_z.max() - foot_z.min())),
    }


def m1_speed_envelope(make, hold: float = 6.0) -> dict:
    """M1 — commanded vs achieved twist across the envelope, with fall flags."""
    out = {"forward": [], "lateral": [], "yaw": []}
    for vx in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        tr = make().run(hold, lambda t, vx=vx: cmd(vx=vx))
        s = steady(tr, 2.0)
        out["forward"].append(
            {"cmd": vx, "achieved": s["vx"], "err": s["vx"] - vx, "fell_at": tr.fell_at}
        )
    for vy in (0.1, 0.2, 0.3, 0.4):
        tr = make().run(hold, lambda t, vy=vy: cmd(vy=vy))
        s = steady(tr, 2.0)
        out["lateral"].append(
            {"cmd": vy, "achieved": s["vy"], "err": s["vy"] - vy, "fell_at": tr.fell_at}
        )
    for wz in (0.5, 1.0, 1.5, 2.0):
        tr = make().run(hold, lambda t, wz=wz: cmd(wz=wz))
        s = steady(tr, 2.0)
        out["yaw"].append(
            {"cmd": wz, "achieved": s["wz"], "err": s["wz"] - wz, "fell_at": tr.fell_at}
        )
    return out


RISE_WINDOW_S = 0.5


def _smoothed_rate(trace: Trace, channel: str, window_s: float = RISE_WINDOW_S) -> tuple[np.ndarray, np.ndarray]:
    """Trailing sliding-window rate of a position channel (px/py/yaw).

    Instantaneous body velocity is unusable for rise times: at ~1.7 Hz step frequency
    the trunk sways past any 90% threshold within one tick, which reports 0 s rise for
    a reversal that visibly takes half a second. The window is TRAILING (rate at t uses
    [t-window, t]) so no pre-command motion can leak backwards across the step; the
    price is that every rise time carries up to +window of averaging lag, which makes
    `window_s` the resolution floor, not the measurement.
    """
    t = trace.arr("t")
    p = np.unwrap(trace.arr(channel)) if channel == "yaw" else trace.arr(channel)
    span = max(1, int(window_s * CONTROL_HZ))
    if t.size < span + 1:
        return t[:0], t[:0]
    rate = (p[span:] - p[:-span]) / (t[span:] - t[:-span])
    return t[span:], rate


def m2_step_response(make) -> dict:
    """M2 — how fast a twist command becomes motion (rise) and reverses."""

    def rise_time(trace: Trace, t0: float, channel: str) -> float:
        t, rate = _smoothed_rate(trace, channel)
        if t.size == 0:
            return float("nan")
        # 90% of the achieved steady value, not of the command: the ceiling is M1's job.
        tail = rate[t >= t0 + 2.0]
        steady_v = float(np.mean(tail)) if tail.size else float("nan")
        if not math.isfinite(steady_v) or abs(steady_v) < 0.02:
            return float("nan")
        target = 0.9 * abs(steady_v)
        # If the pre-command state already clears the threshold in the commanded
        # direction, the transition is not observable — the walking net does not stand
        # still under a zero command, it marches in place and drifts. Report nan rather
        # than a 0 s rise that reads as instantaneous response.
        pre = rate[(t >= t0 - RISE_WINDOW_S) & (t < t0)]
        if pre.size and float(np.max(np.sign(steady_v) * pre)) >= target:
            return float("nan")
        idx = np.flatnonzero((t >= t0) & (np.sign(steady_v) * rate >= target))
        return float(t[idx[0]] - t0) if idx.size else float("nan")

    fwd = make().run(6.0, lambda t: cmd(vx=0.4 if t >= 1.0 else 0.0))
    rev = make().run(9.0, lambda t: cmd(vx=0.4 if t < 4.0 else -0.4))
    turn = make().run(6.0, lambda t: cmd(wz=1.0 if t >= 1.0 else 0.0))
    return {
        "start_walk_rise_s": rise_time(fwd, 1.0, "px"),
        "reverse_rise_s": rise_time(rev, 4.0, "px"),
        "turn_rise_s": rise_time(turn, 1.0, "yaw"),
        "rise_window_s": RISE_WINDOW_S,
        "reverse_fell_at": rev.fell_at,
        "start_fell_at": fwd.fell_at,
        "turn_fell_at": turn.fell_at,
    }


def m3_disturbance(make, magnitudes=(0.2, 0.4, 0.6, 0.8, 1.0)) -> dict:
    """M3 — survival under a base-velocity impulse while walking at 0.3 m/s."""
    results = []
    for mag in magnitudes:
        for axis, name in ((0, "front"), (1, "side")):
            for sign in (1, -1):
                def push(t, data, adr, rng, axis=axis, sign=sign, mag=mag):
                    if abs(t - 3.0) < 1e-6:
                        data.qvel[adr + axis] += sign * mag

                tr = make().run(7.0, lambda t: cmd(vx=0.3), push_fn=push)
                results.append(
                    {
                        "magnitude": mag,
                        "axis": name,
                        "sign": sign,
                        "survived": tr.fell_at is None,
                        "recovery_tilt_max_deg": float(np.max(tr.arr("tilt")[tr.window(3.0)])),
                    }
                )
    failed = [r["magnitude"] for r in results if not r["survived"]]
    return {
        "trials": results,
        "survival_rate": sum(r["survived"] for r in results) / len(results),
        "max_all_survived": max([m for m in magnitudes if all(
            r["survived"] for r in results if r["magnitude"] == m)], default=0.0),
        "first_failure_magnitude": min(failed, default=None),
    }


def m4_gait_dynamics(make) -> dict:
    """M4 — is the gait dynamic or quasi-static, at three speeds."""
    out = {}
    for vx in (0.2, 0.4):
        tr = make().run(8.0, lambda t, vx=vx: cmd(vx=vx))
        out[f"vx_{vx}"] = {**gait_metrics(tr, 2.0), "fell_at": tr.fell_at}
    return out


def m5_cost(make) -> dict:
    """M5 — the honest cost side: impact, torque saturation, thrash, slip."""
    out = {}
    for label, command in (("walk_0.4", cmd(vx=0.4)), ("turn_1.0", cmd(wz=1.0)), ("idle", cmd())):
        tr = make().run(8.0, lambda t, c=command: c)
        out[label] = {**cost_metrics(tr, 2.0), "fell_at": tr.fell_at}
    return out


def m6_idle_stability(make) -> dict:
    """M6 — zero command: drift and jitter of the deployment idle state."""
    tr = make().run(10.0, lambda t: cmd())
    w = tr.window(2.0)
    s = steady(tr, 2.0)
    return {
        "drift_speed_mps": float(math.hypot(s["vx"], s["vy"])),
        "drift_yaw_rate": s["wz"],
        "tilt_mean_deg": float(np.mean(tr.arr("tilt")[w])),
        "tilt_max_deg": float(np.max(tr.arr("tilt")[w])),
        "action_rate_mean": float(np.mean(tr.arr("action_rate")[w])),
        "fell_at": tr.fell_at,
    }


def battery(model, session, action_scale, legs_lowpass, head_lowpass, only=None, stand_session=None) -> dict:
    kp_nominal = model.actuator_gainprm[: model.nu, 0].copy()

    def make():
        return Rollout(
            model, session, action_scale, legs_lowpass, head_lowpass, stand_session, kp_nominal
        )

    tests = {
        "m1_speed_envelope": m1_speed_envelope,
        "m2_step_response": m2_step_response,
        "m3_disturbance": m3_disturbance,
        "m4_gait_dynamics": m4_gait_dynamics,
        "m5_cost": m5_cost,
        "m6_idle_stability": m6_idle_stability,
    }
    out = {}
    for name, fn in tests.items():
        if only and name not in only:
            continue
        print(f"  … {name}", flush=True)
        out[name] = fn(make)
    return out


def parse_alpha(value: str) -> float | None:
    """Runtime semantics: alpha >= 1.0 means the filter is off (see robotd-params)."""
    alpha = float(value)
    return None if alpha >= 1.0 else alpha


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="walking ONNX (obs[1,61] -> actions[1,14])")
    ap.add_argument(
        "--stand-policy",
        help="standing ONNX; with it, ticks under the standing threshold run the stand net "
        "at its own action scale and gain, which is what the runtime scheduler does. "
        "Without it, rest and rise-from-rest measure the walking net idling.",
    )
    ap.add_argument("--scene", default=SCENE_XML)
    ap.add_argument("--action-scale", type=float, default=WALK_ACTION_SCALE, help="runtime walking action_scale")
    ap.add_argument("--legs-lowpass", type=parse_alpha, default="0.7")
    ap.add_argument("--head-lowpass", type=parse_alpha, default="0.5")
    ap.add_argument("--only", nargs="*", help="subset of metric ids to run")
    ap.add_argument("--out", help="write JSON results here")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)

    def load(path: str) -> ort.InferenceSession:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        in_shape = sess.get_inputs()[0].shape
        out_shape = sess.get_outputs()[0].shape
        if int(in_shape[-1]) != 61 or int(out_shape[-1]) != 14:
            raise SystemExit(
                f"{path}: contract must be obs[1,61] -> actions[1,14], got {in_shape} -> {out_shape}"
            )
        return sess

    session = load(args.policy)
    stand_session = load(args.stand_policy) if args.stand_policy else None

    label = f"legs_lowpass={args.legs_lowpass} head_lowpass={args.head_lowpass}"
    print(f"[{label}] action_scale={args.action_scale} scene={args.scene}", flush=True)
    results = {
        "policy": args.policy,
        "scene": args.scene,
        "action_scale": args.action_scale,
        "stand_policy": args.stand_policy,
        "legs_lowpass": args.legs_lowpass,
        "head_lowpass": args.head_lowpass,
        "metrics": battery(
            model,
            session,
            args.action_scale,
            args.legs_lowpass,
            args.head_lowpass,
            args.only,
            stand_session,
        ),
    }
    text = json.dumps(results, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
