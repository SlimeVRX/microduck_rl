"""E4 hardware trade-off calculator (CPU only, no training).

Every number is derived from the shipped walk MJCF plus the BAM `xl330/m6`
parameters, so a variant is described by *editing the model in memory* and
re-deriving the same eight quantities. Prints one row per variant.

    uv run python scripts/hw_tradeoff.py

Quantities, and where each comes from:

  z_com, w0   inverted-pendulum height / rate: w0 = sqrt(g / z_com); the fall
              time constant is 1/w0. Height is measured from the sole, not
              from the world origin, so crouching shows up correctly.
  a_fwd/back  CoP authority from the foot: a = g * lever / z_com, lever taken
  a_lat       from the actual foot box in the STAND keyframe (single support
              for lateral).
  reach_lat   forward kinematics: how far the swing foot moves sideways at the
              hip-roll limit, and how much the foot rises doing so (a rising
              foot cannot be planted, so the reach is unusable without an
              ankle-roll DOF).
  f_swing     torque-limited step frequency: sinusoidal hip swing of amplitude
              A gives peak torque I_swing * A * (2*pi*f)^2 <= tau_max.
  dv_lat      sideways push survivable = reach_lat * w0 + a_lat * (1 / (2*f)),
              i.e. step where the capture point is, plus what the foot absorbs
              during one half step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

G = 9.81
SCENE = Path(__file__).resolve().parents[1] / (
    "src/mjlab_microduck/robot/microduck/scene_walk.xml"
)
STAND_KEY = 1

# BAM xl330/m6 (bam.model.load_model) + microduck_constants kp_fw.
KT = 0.36601349688984386
R_OHM = 2.8113923539223227
MAX_CURRENT = 1.75  # A, XL330Actuator firmware limit
ERROR_GAIN = 0.0028773775022263564
KP_FW = 200.0
VIN_NOMINAL = 7.4

TAU_MAX = KT * MAX_CURRENT
STIFFNESS = KT * VIN_NOMINAL * KP_FW * ERROR_GAIN / R_OHM
SWING_AMPLITUDE = 0.3  # rad, half-range of hip pitch over a step

HEAD_BODIES = ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft")
SWING_BODIES = ("upper_leg_left", "leg", "ankle_left")
ANKLE_BODIES = ("ankle_left", "ankle_right")
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")
# Foot box half-sizes are ordered (thickness, half-width, half-length).
SIZE_WIDTH, SIZE_LENGTH = 1, 2


@dataclass
class Variant:
    name: str
    note: str
    add_ankle_mass: float = 0.0  # kg per ankle body
    head_mass_scale: float = 1.0
    foot_width_delta: float = 0.0  # m added to each half-width
    foot_length_delta: float = 0.0  # m added to each half-length
    hip_roll_limit: float | None = None  # rad, overrides both signs
    foot_stays_flat: bool = False  # an ankle-roll DOF keeps the sole planted
    crouch: float = 0.0  # rad of extra knee bend, lowers the CoM


def bid(m, name: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def jid(m, name: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)


def apply(m, v: Variant) -> None:
    for b in ANKLE_BODIES:
        m.body_mass[bid(m, b)] += v.add_ankle_mass
    for b in HEAD_BODIES:
        i = bid(m, b)
        m.body_mass[i] *= v.head_mass_scale
        m.body_inertia[i] *= v.head_mass_scale
    for g in FOOT_GEOMS:
        gi = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        m.geom_size[gi][SIZE_WIDTH] += v.foot_width_delta
        m.geom_size[gi][SIZE_LENGTH] += v.foot_length_delta
    if v.hip_roll_limit is not None:
        for j in ("left_hip_roll", "right_hip_roll"):
            m.jnt_range[jid(m, j)] = (-v.hip_roll_limit, v.hip_roll_limit)


def foot_extents(m, d, geom: str) -> dict[str, float]:
    gi = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom)
    c = d.geom_xpos[gi]
    rot = d.geom_xmat[gi].reshape(3, 3)
    s = m.geom_size[gi]
    corners = np.array(
        [
            [sx, sy, sz]
            for sx in (-s[0], s[0])
            for sy in (-s[1], s[1])
            for sz in (-s[2], s[2])
        ]
    )
    w = (rot @ corners.T).T + c
    ankle = d.xanchor[jid(m, "left_ankle" if "left" in geom else "right_ankle")]
    return {
        "front": float(w[:, 0].max() - ankle[0]),
        "back": float(ankle[0] - w[:, 0].min()),
        "half_width": float((w[:, 1].max() - w[:, 1].min()) / 2.0),
        "y_center": float((w[:, 1].max() + w[:, 1].min()) / 2.0),
        "z_min": float(w[:, 2].min()),
    }


def swing_inertia(m, d) -> float:
    axis = d.xaxis[jid(m, "left_hip_pitch")]
    anchor = d.xanchor[jid(m, "left_hip_pitch")]
    total = 0.0
    for b in SWING_BODIES:
        i = bid(m, b)
        rot = d.ximat[i].reshape(3, 3)
        inertia = rot @ np.diag(m.body_inertia[i]) @ rot.T
        r = d.xipos[i] - anchor
        r_perp = r - np.dot(r, axis) * axis
        total += axis @ inertia @ axis + m.body_mass[i] * float(r_perp @ r_perp)
    return total


def lateral_reach(m, d, flat: bool, crouch: float) -> tuple[float, float]:
    """Sideways swing-foot travel at the hip-roll limit, and the foot's rise."""
    limit = float(m.jnt_range[jid(m, "left_hip_roll")][1])
    adr = m.jnt_qposadr[jid(m, "left_hip_roll")]
    gi = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")

    mujoco.mj_resetDataKeyframe(m, d, STAND_KEY)
    crouch_pose(m, d, crouch)
    mujoco.mj_forward(m, d)
    y0, z0 = float(d.geom_xpos[gi][1]), float(d.geom_xpos[gi][2])

    mujoco.mj_resetDataKeyframe(m, d, STAND_KEY)
    crouch_pose(m, d, crouch)
    d.qpos[adr] = limit
    mujoco.mj_forward(m, d)
    y1, z1 = float(d.geom_xpos[gi][1]), float(d.geom_xpos[gi][2])

    reach, rise = y1 - y0, z1 - z0
    if not flat and rise > 0.0:
        # Without ankle roll the sole tilts off the ground as it swings out;
        # only the part of the sweep that keeps the foot plantable counts.
        reach *= max(0.0, 1.0 - rise / max(reach, 1e-9))
    return reach, rise


def crouch_pose(m, d, extra_bend: float) -> None:
    """Bend both legs in place: hip pitch back, knee in, ankle compensating."""
    for side in ("left", "right"):
        for joint, sign in (
            (f"{side}_hip_pitch", -1.0),
            (f"{side}_knee", 2.0),
            (f"{side}_ankle", -1.0),
        ):
            d.qpos[m.jnt_qposadr[jid(m, joint)]] += sign * extra_bend


def com_height(m, d) -> float:
    """CoM height above the soles (the pendulum length that actually matters)."""
    sole = min(foot_extents(m, d, g)["z_min"] for g in FOOT_GEOMS)
    return float(d.subtree_com[0][2]) - sole


def evaluate(v: Variant) -> dict:
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    apply(m, v)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, STAND_KEY)
    crouch_pose(m, d, v.crouch)
    mujoco.mj_forward(m, d)
    mujoco.mj_comPos(m, d)

    mass = float(m.body_mass.sum())
    z = com_height(m, d)
    w0 = (G / z) ** 0.5
    foot = foot_extents(m, d, "left_foot_collision")

    i_swing = swing_inertia(m, d)
    f_swing = (TAU_MAX / (i_swing * SWING_AMPLITUDE)) ** 0.5 / (2 * np.pi)
    reach, rise = lateral_reach(m, d, v.foot_stays_flat, v.crouch)
    a_lat = G * foot["half_width"] / z
    dv_lat = reach * w0 + a_lat / (2 * f_swing)

    return {
        "name": v.name,
        "note": v.note,
        "mass": mass,
        "z_com": z,
        "w0": w0,
        "a_fwd": G * foot["front"] / z,
        "a_back": G * foot["back"] / z,
        "a_lat": a_lat,
        "reach_lat": reach,
        "rise": rise,
        "i_swing": i_swing,
        "f_swing": f_swing,
        "dv_lat": dv_lat,
    }


VARIANTS = [
    Variant("baseline", "as shipped"),
    Variant(
        "ankle_roll",
        "+1 XL330 per ankle (20 g each), sole stays planted",
        add_ankle_mass=0.020,
        foot_stays_flat=True,
    ),
    Variant("foot_wider_10", "sole 10 mm wider each side", foot_width_delta=0.010),
    Variant(
        "foot_longer_5", "sole 5 mm longer front and back", foot_length_delta=0.005
    ),
    Variant("hip_roll_0p6", "hip-roll limit 0.384 -> 0.60 rad", hip_roll_limit=0.60),
    Variant("head_minus_30pct", "head chain 280 g -> 196 g", head_mass_scale=0.70),
    Variant(
        "combo_cheap",
        "hip-roll 0.60 rad + sole 10 mm wider",
        hip_roll_limit=0.60,
        foot_width_delta=0.010,
    ),
    # Crouching trades reaction time for CoP authority, but it also folds the
    # leg: the hip-roll sweep stops moving the foot sideways, so `reach`
    # collapses. Kinematics only — a rollout is needed to settle how much of
    # that a real stepping gait recovers.
    Variant("crouch_0p6", "knees bent 0.6 rad more (no hardware change)", crouch=0.60),
    Variant(
        "combo_full",
        "ankle roll + hip-roll 0.60 rad + sole 10 mm wider",
        add_ankle_mass=0.020,
        foot_stays_flat=True,
        hip_roll_limit=0.60,
        foot_width_delta=0.010,
    ),
]


def main() -> None:
    print(f"tau_max = {TAU_MAX:.3f} N.m   stiffness = {STIFFNESS:.3f} N.m/rad")
    print(f"full-CoP ankle torque needs {G:.2f}*lever*m / stiffness of error\n")
    head = "{:<17} {:>6} {:>7} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6} {:>7}"
    print(
        head.format(
            "variant",
            "mass",
            "z_com",
            "a_fwd",
            "a_back",
            "a_lat",
            "reach",
            "rise",
            "f_sw",
            "dv_lat",
        )
    )
    base = None
    for v in VARIANTS:
        r = evaluate(v)
        if base is None:
            base = r
        print(
            "{:<17} {:>6.3f} {:>7.4f} {:>6.2f} {:>6.2f} {:>6.2f} {:>6.1f} {:>7.1f}"
            " {:>6.2f} {:>7.3f}".format(
                r["name"],
                r["mass"],
                r["z_com"],
                r["a_fwd"],
                r["a_back"],
                r["a_lat"],
                r["reach_lat"] * 1000,
                r["rise"] * 1000,
                r["f_swing"],
                r["dv_lat"],
            )
        )
    print("\nunits: kg, m, m/s^2, mm, mm, Hz, m/s")
    print("(reach/rise = sideways swing-foot travel and lift at the hip-roll limit)")

    assert base is not None
    print(
        "\nbaseline fall time constant 1/w0 = {:.4f} s".format(1.0 / base["w0"])
        + "  (CoP accel scales as 1/z_com, reaction time as sqrt(z_com))"
    )


if __name__ == "__main__":
    main()
