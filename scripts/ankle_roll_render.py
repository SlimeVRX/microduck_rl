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
import sys
from pathlib import Path

import mediapy
import mujoco

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
    distance: float, azimuth: float, elevation: float, height: float
) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.0, 0.0, height)
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def shoot(robot: Robot, cam: mujoco.MjvCamera, width: int, height: int):
    with mujoco.Renderer(robot.model, height, width) as renderer:
        renderer.update_scene(robot.data, camera=cam)
        return renderer.render()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs/img"))
    ap.add_argument("--bias-step", type=float, default=0.02)
    ap.add_argument("--bias-limit", type=float, default=1.2)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=700)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # azimuth 0 = looking at the robot head-on, which is the plane the lateral
    # lean and the sole-flatness question live in.
    front = camera(0.5, 0.0, -10.0, 0.09)
    ankle = camera(0.15, 8.0, -6.0, 0.025)

    for scene in SCENES:
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
        print(
            f"{robot.name}: best flat-sole CoM travel {1000 * best[0]:.1f} mm "
            f"at bias {best[1]:.2f} rad"
        )
    print(f"wrote to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
