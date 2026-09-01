#!/usr/bin/env python3
"""Offscreen renders of the ankle-roll builds, so the mechanism can be looked at.

Two kinds of frame per model:
  stand   settled at STAND, front view and an ankle close-up -- what the extra
          joint costs in stack height
  lean    the frame of maximum sideways CoM travel with BOTH soles still on the
          floor, found by the same ramped lateral bias T2 uses in
          `ankle_roll_sim.py`. This is the picture of the capability being
          priced: the baseline runs out at a few millimetres, the ankle-roll
          builds keep going.

    uv run python scripts/ankle_roll_render.py --out docs/img
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ankle_roll_sim import Balancer, Gains, Robot, calibrate_signs

ROBOT_DIR = ROOT / "src/mjlab_microduck/robot/microduck"
SCENES = (
    "scene_walk.xml",
    "scene_walk_ankleroll_coincident.xml",
    "scene_walk_ankleroll_serial.xml",
)


def camera(
    distance: float,
    azimuth: float,
    elevation: float,
    height: float,
    lookat: tuple[float, float, float] | None = None,
) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat if lookat is not None else (0.0, 0.0, height)
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def shoot(
    robot: Robot,
    cam: mujoco.MjvCamera,
    width: int,
    height: int,
    show_collision_group: bool = False,
):
    with mujoco.Renderer(robot.model, height, width) as renderer:
        if show_collision_group:
            renderer._scene_option.geomgroup[3] = 1
        renderer.update_scene(robot.data, camera=cam)
        return renderer.render()


def _side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    left_label: str = "baseline",
    right_label: str = "new build",
) -> np.ndarray:
    panel_height, panel_width = left.shape[:2]
    if right.shape[:2] != (panel_height, panel_width):
        raise ValueError("comparison panels must have identical dimensions")
    canvas = Image.new("RGB", (2 * panel_width, panel_height), "black")
    canvas.paste(Image.fromarray(left), (0, 0))
    canvas.paste(Image.fromarray(right), (panel_width, 0))
    draw = ImageDraw.Draw(canvas)
    for x, label in ((8, left_label), (panel_width + 8, right_label)):
        draw.rectangle((x - 4, 6, x + 10 * len(label), 30), fill="black")
        draw.text((x, 8), label, fill="white")
    return np.asarray(canvas)


def _set_visual_alpha(robot: Robot, alpha: float) -> None:
    """Make stock visual-only meshes translucent for a mechanism comparison."""
    visual = (robot.model.geom_contype == 0) & (robot.model.geom_conaffinity == 0)
    robot.model.geom_rgba[visual, 3] = alpha


def _sole_contact_counts(robot: Robot) -> dict[str, int]:
    counts = {"left": 0, "right": 0}
    for contact in robot.data.contact[: robot.data.ncon]:
        for side, geom_id in robot.foot_geom.items():
            if {int(contact.geom1), int(contact.geom2)} == {
                robot.floor,
                geom_id,
            }:
                counts[side] += 1
    return counts


def _trunk_roll_degrees(robot: Robot) -> float:
    return math.degrees(robot.lean()[1])


def _mechanism_pose(robot: Robot, roll: float = 0.0) -> None:
    robot.reset_stand()
    if "left_ankle_roll" in robot.qpos_of:
        robot.data.qpos[robot.qpos_of["left_ankle_roll"]] = roll
    robot.data.qvel[:] = 0.0
    mujoco.mj_forward(robot.model, robot.data)


def _comparison_robot(
    scene: str, width: int, height: int, translucent_visuals: bool = False
) -> Robot:
    robot = Robot(ROBOT_DIR / scene)
    robot.model.vis.global_.offwidth = width
    robot.model.vis.global_.offheight = height
    if translucent_visuals:
        _set_visual_alpha(robot, 0.25)
    # The mechanism meshes share the collision-only group with the stock
    # collision meshes.  Show the group so E5 parts render, but suppress the
    # stock collision shell so it cannot obscure the mechanism.
    e5 = np.zeros(robot.model.ngeom, dtype=bool)
    for geom_id in range(robot.model.ngeom):
        name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        e5[geom_id] = name.startswith("e5_") or "_e5_" in name
    stock_collision = (robot.model.geom_group == 3) & ~e5
    robot.model.geom_rgba[stock_collision, 3] = 0.0
    return robot


def _render_mechanism_comparison(
    out: Path, width: int, height: int
) -> tuple[dict[str, object], dict[str, object]]:
    baseline = _comparison_robot("scene_walk.xml", width, height, True)
    new = _comparison_robot("scene_walk_e5_remote_sole.xml", width, height, True)
    _mechanism_pose(baseline)
    _mechanism_pose(new)
    cam = camera(
        0.075,
        -45.0,
        -8.0,
        0.040,
        (-0.012, 0.052, 0.040),
    )
    mediapy.write_image(
        out / "e5_compare_mechanism.png",
        _side_by_side(
            shoot(baseline, cam, width, height, True),
            shoot(new, cam, width, height, True),
        ),
    )
    return (
        {
            "trunk_roll_deg": _trunk_roll_degrees(baseline),
            "sole_contacts": _sole_contact_counts(baseline),
        },
        {
            "trunk_roll_deg": _trunk_roll_degrees(new),
            "sole_contacts": _sole_contact_counts(new),
        },
    )


def _best_lean_frame(
    robot: Robot, bal: Balancer, cam: mujoco.MjvCamera, width: int, height: int
) -> tuple[np.ndarray, dict[str, object]]:
    robot.reset_stand()
    bal.hold(1.0)
    mujoco.mj_forward(robot.model, robot.data)
    y0 = float(robot.data.subtree_com[0][1])
    best: tuple[float, float, np.ndarray | None, dict[str, object] | None] = (
        0.0,
        0.0,
        None,
        None,
    )
    bias = 0.0
    while bias < 1.2:
        bias += 0.02
        bal.hold(0.15, bias=bias)
        if robot.fallen() or not all(robot.contacts().values()):
            continue
        travel = abs(float(robot.data.subtree_com[0][1]) - y0)
        if travel > best[0]:
            best = (
                travel,
                bias,
                shoot(robot, cam, width, height, True),
                {
                    "bias_rad": bias,
                    "travel_mm": 1000.0 * travel,
                    "trunk_roll_deg": _trunk_roll_degrees(robot),
                    "sole_contacts": _sole_contact_counts(robot),
                },
            )
    if best[2] is None or best[3] is None:
        raise RuntimeError(f"{robot.name} has no valid flat-sole lean frame")
    return best[2], best[3]


def _render_lean_comparison(
    out: Path, width: int, height: int
) -> tuple[dict[str, object], dict[str, object]]:
    baseline = _comparison_robot("scene_walk.xml", width, height)
    new = _comparison_robot("scene_walk_e5_remote_sole.xml", width, height)
    baseline_signs = calibrate_signs(baseline)
    new_signs = calibrate_signs(new)
    gains = Gains(2.0, 0.15, 2.0, 0.15, 0.0, 0.0)
    baseline_bal = Balancer(baseline, gains, baseline_signs)
    new_bal = Balancer(new, gains, new_signs)
    cam = camera(0.5, 0.0, -10.0, 0.09)
    baseline_image, baseline_metrics = _best_lean_frame(
        baseline, baseline_bal, cam, width, height
    )
    new_image, new_metrics = _best_lean_frame(new, new_bal, cam, width, height)
    mediapy.write_image(
        out / "e5_compare_lean.png",
        _side_by_side(baseline_image, new_image),
    )
    return baseline_metrics, new_metrics


def render_comparisons(out: Path, width: int, height: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    mechanism = _render_mechanism_comparison(out, width, height)
    lean = _render_lean_comparison(out, width, height)
    print(f"mechanism: baseline={mechanism[0]} new={mechanism[1]}")
    print(f"lean: baseline={lean[0]} new={lean[1]}")
    print(f"wrote comparisons to {out}")


def render_remote_left_closeups(
    robot: Robot, out: Path, width: int, height: int
) -> None:
    left_mechanism = (-0.018, 0.052, 0.050)
    side = camera(0.12, -90.0, -8.0, 0.050, left_mechanism)
    front = camera(0.12, -45.0, -8.0, 0.050, left_mechanism)
    for pose_name, roll in (("stand", 0.0), ("roll20", math.radians(20.0))):
        robot.reset_stand()
        if roll:
            robot.data.qpos[robot.qpos_of["left_ankle_roll"]] = roll
        robot.data.qvel[:] = 0.0
        mujoco.mj_forward(robot.model, robot.data)
        prefix = f"{robot.name}_left_mechanism"
        mediapy.write_image(
            out / f"{prefix}_side_{pose_name}.png",
            shoot(robot, side, width, height),
        )
        mediapy.write_image(
            out / f"{prefix}_front_{pose_name}.png",
            shoot(robot, front, width, height),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs/img"))
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--bias-step", type=float, default=0.02)
    ap.add_argument("--bias-limit", type=float, default=1.2)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=700)
    ap.add_argument(
        "--comparisons",
        action="store_true",
        help="render baseline/E5 comparison figures",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.comparisons:
        render_comparisons(out, args.width, args.height)
        return 0
    # azimuth 0 = looking at the robot head-on, which is the plane the lateral
    # lean and the sole-flatness question live in.
    front = camera(0.5, 0.0, -10.0, 0.09)
    ankle = camera(0.15, 8.0, -6.0, 0.025)

    for scene in args.scenes or SCENES:
        robot = Robot(ROBOT_DIR / scene)
        robot.model.vis.global_.offwidth = args.width
        robot.model.vis.global_.offheight = args.height
        signs = calibrate_signs(robot)
        # Same gains T2 uses, including zero roll FEEDBACK: lean feedback on the
        # roll joints fights the commanded lean, so the roll joints get only the
        # open-loop bias here, which is what the capability question asks.
        bal = Balancer(robot, Gains(2.0, 0.15, 2.0, 0.15, 0.0, 0.0), signs)
        robot.reset_stand()
        bal.hold(1.0)
        mujoco.mj_forward(robot.model, robot.data)
        y0 = float(robot.data.subtree_com[0][1])
        mediapy.write_image(
            out / f"{robot.name}_stand.png",
            shoot(robot, front, args.width, args.height),
        )
        mediapy.write_image(
            out / f"{robot.name}_ankle.png",
            shoot(robot, ankle, args.width, args.height),
        )

        best = (0.0, 0.0, None, None)
        bias = 0.0
        while bias < args.bias_limit:
            bias += args.bias_step
            bal.hold(0.15, bias=bias)
            if robot.fallen():
                break
            if not all(robot.contacts().values()):
                continue
            travel = abs(float(robot.data.subtree_com[0][1]) - y0)
            if travel > best[0]:
                best = (
                    travel,
                    bias,
                    shoot(robot, front, args.width, args.height),
                    shoot(robot, ankle, args.width, args.height),
                )
        if best[2] is not None:
            mediapy.write_image(out / f"{robot.name}_lean.png", best[2])
            mediapy.write_image(out / f"{robot.name}_lean_ankle.png", best[3])
        if scene == "scene_walk_e5_remote.xml":
            render_remote_left_closeups(robot, out, args.width, args.height)
        print(
            f"{robot.name}: best flat-sole CoM travel {1000 * best[0]:.1f} mm "
            f"at bias {best[1]:.2f} rad"
        )
    print(f"wrote to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
