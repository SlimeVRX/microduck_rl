#!/usr/bin/env python3
"""Generate the E5 ankle-roll parts in MuJoCo parent-body coordinates.

The source measurements are deliberately kept here rather than hidden in the
MJCF builder.  Geometry is constructed in millimetres and exported at MuJoCo's
metre scale, matching the existing Microduck mesh assets.  ``e5_variant.py``
places those meshes with zero geom offsets and reads the same files back for
their mass properties.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = ROOT / "src/mjlab_microduck/robot/microduck"
ASSET_DIR = ROBOT_DIR / "assets"
SIDES = ("left", "right")
PLA_DENSITY_G_CM3 = 1.24
SERVO_MASS_KG = 0.018
HARDWARE_MASS_KG = 0.004
PIVOT_INBOARD_MM = 12.0
LEVER_ARM_MM = 5.0
ROD_RADIUS_MM = 1.5
BALL_RADIUS_MM = 2.5
SOLE_FWD_MM = 59.0
SOLE_LATERAL_MM = 51.0
SOLE_THICKNESS_MM = 12.9
SOLE_CORNER_RADIUS_MM = 6.0
ROCKER_LENGTH_MM = 8.0
ROCKER_ANGLE_DEG = 15.0
WALL_MM = 2.0
SERVO_OFFSET_LOCAL_MM = {
    # First zero-new-contact candidate from the compiled original-range sweep:
    # along-shank=12 mm, inboard=-18 mm, rear=12 mm.
    "left": np.array([12.0, 12.0, -18.0]),
    "right": np.array([-12.0, 12.0, -18.0]),
}
SERVO_OFFSET_CANDIDATES_MM = tuple(
    (rear, along, inboard)
    for along in (12.0, 18.0, 24.0, 30.0, 36.0)
    for inboard in (-18.0, -14.0, -10.0, -6.0)
    for rear in (-12.0, -6.0, 0.0, 6.0, 12.0)
)
MASS_CAPS_G = {
    "yoke": 4.0,
    "footplate": 4.0,
    "cradle": 6.0,
}


def _box(
    extents: tuple[float, float, float], center: tuple[float, float, float]
) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents)
    mesh.apply_translation(center)
    return mesh


def _cylinder(
    radius: float,
    height: float,
    center: tuple[float, float, float],
    axis: tuple[float, float, float],
) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius, height, sections=32)
    axis_vec = np.asarray(axis, dtype=float)
    axis_vec /= np.linalg.norm(axis_vec)
    rotation = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axis_vec)
    if rotation is not None:
        mesh.apply_transform(rotation)
    mesh.apply_translation(center)
    return mesh


def _sphere(center: np.ndarray, radius: float = BALL_RADIUS_MM) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    mesh.apply_translation(center)
    return mesh


def _rod(start: np.ndarray, end: np.ndarray) -> trimesh.Trimesh:
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 0:
        raise ValueError("push-rod endpoints must be distinct")
    mesh = _cylinder(
        ROD_RADIUS_MM,
        length,
        tuple((start + end) / 2.0),
        tuple(delta / length),
    )
    return trimesh.util.concatenate([mesh, _sphere(start), _sphere(end)])


def _bar(start: np.ndarray, end: np.ndarray, width: float = WALL_MM) -> trimesh.Trimesh:
    """A printable rib between two points, represented by a square-section rod."""
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 0:
        raise ValueError("rib endpoints must be distinct")
    return _cylinder(
        width / 2.0,
        length,
        tuple((start + end) / 2.0),
        tuple(delta / length),
    )


def _union(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Keep each printable wall/rib as a separate watertight component."""
    printable = []
    for mesh in meshes:
        component = mesh.copy()
        if component.volume < 0:
            component.invert()
        printable.append(component)
    fused = trimesh.util.concatenate(printable)
    fused.remove_unreferenced_vertices()
    return fused


def _rounded_rectangle_points(
    width: float, depth: float, radius: float, segments: int = 8
) -> np.ndarray:
    points: list[tuple[float, float]] = []
    half_w, half_d = width / 2.0, depth / 2.0
    for cx, cy, start in (
        (half_w - radius, half_d - radius, 0.0),
        (-half_w + radius, half_d - radius, math.pi / 2.0),
        (-half_w + radius, -half_d + radius, math.pi),
        (half_w - radius, -half_d + radius, 3.0 * math.pi / 2.0),
    ):
        for i in range(segments + 1):
            angle = start + i * (math.pi / 2.0) / segments
            points.append(
                (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
            )
    return np.asarray(points)


def _rounded_ring(
    width: float,
    depth: float,
    radius: float,
    wall: float,
    y_top: float | np.ndarray,
    y_bottom: float | np.ndarray,
    center_x: float,
    center_z: float,
) -> trimesh.Trimesh:
    outer = _rounded_rectangle_points(width, depth, radius)
    inner = _rounded_rectangle_points(
        width - 2.0 * wall, depth - 2.0 * wall, radius - wall
    )
    outer = outer + (center_x, center_z)
    inner = inner + (center_x, center_z)
    if np.isscalar(y_bottom):
        bottom_outer_y = np.full(len(outer), float(y_bottom))
        bottom_inner_y = np.full(len(inner), float(y_bottom))
    else:
        bottom_outer_y = np.asarray(y_bottom, dtype=float)
        bottom_inner_y = bottom_outer_y.copy()
    top_outer_y = (
        np.full(len(outer), float(y_top))
        if np.isscalar(y_top)
        else np.asarray(y_top, dtype=float)
    )
    top_inner_y = top_outer_y.copy()
    top_outer = np.column_stack([outer[:, 0], top_outer_y, outer[:, 1]])
    top_inner = np.column_stack([inner[:, 0], top_inner_y, inner[:, 1]])
    bottom_outer = np.column_stack([outer[:, 0], bottom_outer_y, outer[:, 1]])
    bottom_inner = np.column_stack([inner[:, 0], bottom_inner_y, inner[:, 1]])
    vertices = np.vstack([top_outer, top_inner, bottom_outer, bottom_inner])
    n = len(outer)
    faces: list[tuple[int, int, int]] = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend(
            [
                (i, j, n + j),
                (i, n + j, n + i),
                (2 * n + i, 3 * n + i, 3 * n + j),
                (2 * n + i, 3 * n + j, 2 * n + j),
                (i, 2 * n + i, 2 * n + j),
                (i, 2 * n + j, j),
                (n + i, n + j, 3 * n + j),
                (n + i, 3 * n + j, 3 * n + i),
            ]
        )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def _sole_mesh(center_x: float, top_y: float, center_z: float) -> trimesh.Trimesh:
    rise = math.tan(math.radians(ROCKER_ANGLE_DEG))
    outline = _rounded_rectangle_points(
        SOLE_FWD_MM, SOLE_LATERAL_MM, SOLE_CORNER_RADIUS_MM
    )
    bottom_y = top_y - SOLE_THICKNESS_MM
    profile = []
    for x, _ in outline:
        distance = max(0.0, abs(x) - (SOLE_FWD_MM / 2.0 - ROCKER_LENGTH_MM))
        profile.append(bottom_y + distance * rise)
    parts = [
        _rounded_ring(
            SOLE_FWD_MM,
            SOLE_LATERAL_MM,
            SOLE_CORNER_RADIUS_MM,
            WALL_MM,
            top_y,
            top_y - WALL_MM,
            center_x,
            center_z,
        ),
        _rounded_ring(
            SOLE_FWD_MM,
            SOLE_LATERAL_MM,
            SOLE_CORNER_RADIUS_MM,
            WALL_MM,
            np.asarray(profile) + 2.0,
            np.asarray(profile),
            center_x,
            center_z,
        ),
    ]
    for x in (-SOLE_FWD_MM / 2.0 + WALL_MM, 0.0, SOLE_FWD_MM / 2.0 - WALL_MM):
        parts.append(
            _box(
                (WALL_MM, WALL_MM, SOLE_LATERAL_MM - 2.0 * WALL_MM),
                (center_x + x, top_y - 1.0, center_z),
            )
        )
    for z in (-SOLE_LATERAL_MM / 2.0 + WALL_MM, 0.0, SOLE_LATERAL_MM / 2.0 - WALL_MM):
        parts.append(
            _box(
                (SOLE_FWD_MM - 2.0 * WALL_MM, WALL_MM, WALL_MM),
                (center_x, top_y - 1.0, center_z + z),
            )
        )
    for x, z in itertools.product(
        (-SOLE_FWD_MM / 2.0 + WALL_MM, SOLE_FWD_MM / 2.0 - WALL_MM),
        (-SOLE_LATERAL_MM / 2.0 + WALL_MM, SOLE_LATERAL_MM / 2.0 - WALL_MM),
    ):
        distance = max(0.0, abs(x) - (SOLE_FWD_MM / 2.0 - ROCKER_LENGTH_MM))
        parts.append(
            _bar(
                np.array([center_x + x, top_y - 2.0, center_z + z]),
                np.array(
                    [center_x + x, bottom_y + distance * rise + 2.0, center_z + z]
                ),
            )
        )
    return _union(parts)


def _body_data(robot_xml: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return model, data


def _mesh_local_to_body(
    mesh: trimesh.Trimesh, body_pos: np.ndarray, body_rot: np.ndarray
) -> trimesh.Trimesh:
    world = mesh.copy()
    world.apply_transform(
        np.block(
            [
                [body_rot, body_pos.reshape(3, 1)],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]
        )
    )
    return world


def _body_point(data: mujoco.MjData, body_id: int, point: np.ndarray) -> np.ndarray:
    return data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ point


def _ankle_local_point(
    data: mujoco.MjData, body_id: int, world_point: np.ndarray
) -> np.ndarray:
    rot = data.xmat[body_id].reshape(3, 3)
    return rot.T @ (world_point - data.xpos[body_id])


def sole_reference(
    model: mujoco.MjModel, data: mujoco.MjData, side: str
) -> tuple[np.ndarray, float]:
    ankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ankle_{side}")
    geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision"
    )
    mesh_id = model.geom_dataid[geom_id]
    verts = model.mesh_vert[
        model.mesh_vertadr[mesh_id] : model.mesh_vertadr[mesh_id]
        + model.mesh_vertnum[mesh_id]
    ]
    world = data.geom_xpos[geom_id] + verts @ data.geom_xmat[geom_id].reshape(3, 3).T
    ankle_rot = data.xmat[ankle_id].reshape(3, 3)
    local = (world - data.xpos[ankle_id]) @ ankle_rot
    center = np.array(
        [
            (local[:, 0].min() + local[:, 0].max()) / 2.0,
            0.0,
            (local[:, 2].min() + local[:, 2].max()) / 2.0,
        ]
    )
    top_y = float(local[:, 1].max())
    return center, top_y


def lever_point_local(side: str) -> np.ndarray:
    """Return the footplate ball centre in the foot body's local frame."""
    fwd_sign = -1.0 if side == "left" else 1.0
    # The arm sits on the forward side of the plate; its radial offset from the
    # roll axis is the specified 5 mm in the local lateral direction.
    return np.array([fwd_sign * LEVER_ARM_MM, 0.0, -PIVOT_INBOARD_MM - LEVER_ARM_MM])


def remote_pushrod_points(
    model: mujoco.MjModel, data: mujoco.MjData, side: str
) -> tuple[np.ndarray, np.ndarray, float]:
    ankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ankle_{side}")
    leg_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "leg" if side == "left" else "leg_2"
    )
    foot_point_world = _body_point(data, ankle_id, lever_point_local(side) / 1000.0)
    offset = SERVO_OFFSET_LOCAL_MM[side] / 1000.0
    servo_world = _body_point(data, leg_id, offset)
    direction = foot_point_world - servo_world
    horn_world = servo_world + direction / np.linalg.norm(direction) * 0.005
    rod_length = float(np.linalg.norm(foot_point_world - horn_world))
    if rod_length * 1000.0 < 25.0:
        raise AssertionError(
            f"{side} pushrod length {rod_length * 1000.0:.3f} mm is below 25 mm"
        )
    foot_point_local = _ankle_local_point(data, ankle_id, foot_point_world)
    horn_local = _ankle_local_point(data, ankle_id, horn_world)
    return (
        horn_local * 1000.0,
        foot_point_local * 1000.0,
        rod_length * 1000.0,
    )


def yoke(side: str) -> trimesh.Trimesh:
    del side
    frame_parts = [
        _box((2.0, WALL_MM, 30.0), (-13.0, -1.0, 0.0)),
        _box((2.0, WALL_MM, 30.0), (13.0, -1.0, 0.0)),
        _box((2.0, 2.0, 30.0), (-13.0, -11.0, 0.0)),
        _box((2.0, 2.0, 30.0), (-13.0, -21.0, 0.0)),
        _box((2.0, 2.0, 30.0), (13.0, -11.0, 0.0)),
        _box((2.0, 2.0, 30.0), (13.0, -21.0, 0.0)),
    ]
    parts = [
        _box((34.0, WALL_MM, WALL_MM), (0.0, -1.0, -14.0)),
        _box((34.0, WALL_MM, WALL_MM), (0.0, -1.0, 14.0)),
        _box((WALL_MM, WALL_MM, 26.0), (-16.0, -1.0, 0.0)),
        _box((WALL_MM, WALL_MM, 26.0), (16.0, -1.0, 0.0)),
        *frame_parts,
        _cylinder(4.0, 4.0, (-13.0, 0.0, -PIVOT_INBOARD_MM), (1.0, 0.0, 0.0)),
        _cylinder(4.0, 4.0, (13.0, 0.0, -PIVOT_INBOARD_MM), (1.0, 0.0, 0.0)),
        trimesh.creation.annulus(8.0, 11.0, 3.0, sections=32),
        _box((4.0, WALL_MM, PIVOT_INBOARD_MM), (0.0, 0.0, -PIVOT_INBOARD_MM / 2.0)),
    ]
    return _union(parts)


def footplate(side: str, center: np.ndarray, top_y: float) -> trimesh.Trimesh:
    fwd_sign = -1.0 if side == "left" else 1.0
    plate = [
        _box((54.0, WALL_MM, WALL_MM), (center[0], top_y - 1.0, center[2] - 19.5)),
        _box((54.0, WALL_MM, WALL_MM), (center[0], top_y - 1.0, center[2] + 19.5)),
        _box((WALL_MM, WALL_MM, 41.0), (center[0] - 26.0, top_y - 1.0, center[2])),
        _box((WALL_MM, WALL_MM, 41.0), (center[0] + 26.0, top_y - 1.0, center[2])),
        _box((WALL_MM, WALL_MM, 41.0), (center[0], top_y - 1.0, center[2])),
    ]
    arm = _box(
        (WALL_MM, 8.0, WALL_MM),
        (
            center[0] + fwd_sign * LEVER_ARM_MM,
            top_y - 4.0,
            center[2] - PIVOT_INBOARD_MM - LEVER_ARM_MM,
        ),
    )
    return _union([*plate, arm])


def cradle() -> trimesh.Trimesh:
    x, y, z = 29.0, 20.0, 34.0
    half_x, half_y, half_z = x / 2.0, y / 2.0, z / 2.0
    parts = []
    for sx, sy, sz in itertools.product((-1.0, 1.0), repeat=3):
        p = np.array([sx * half_x, sy * half_y, sz * half_z])
        for axis in range(3):
            q = p.copy()
            q[axis] *= -1.0
            if tuple(p) < tuple(q):
                parts.append(_bar(p, q))
    for sx, sz in itertools.product((-1.0, 1.0), repeat=2):
        parts.append(
            _bar(
                np.array([sx * half_x, -half_y, 0.0]),
                np.array([sx * half_x, half_y, 0.0]),
            )
        )
        parts.append(
            _bar(
                np.array([sx * half_x, 0.0, -half_z]),
                np.array([sx * half_x, 0.0, half_z]),
            )
        )
    return _union(parts)


def write_part(name: str, mesh: trimesh.Trimesh) -> dict[str, float]:
    path = ASSET_DIR / f"{name}.stl"
    export_mesh = mesh.copy()
    export_mesh.apply_scale(0.001)
    export_mesh.export(path)
    volume_cm3 = abs(float(mesh.volume)) / 1000.0
    mass_g = volume_cm3 * PLA_DENSITY_G_CM3
    return {"name": name, "volume_cm3": volume_cm3, "mass_g": mass_g}


def build_all(robot_xml: Path) -> list[dict[str, float]]:
    model, data = _body_data(robot_xml)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    original = trimesh.load_mesh(ASSET_DIR / "sole_left.stl", process=True)
    original_volume_cm3 = abs(float(original.volume)) * 1.0e6
    original_mass_g = original_volume_cm3 * PLA_DENSITY_G_CM3
    sole_mass_cap_g = original_mass_g * 1.3
    print(
        f"original sole_left.stl: volume={original_volume_cm3:.4f} cm3 "
        f"mass={original_mass_g:.4f} g cap={sole_mass_cap_g:.4f} g"
    )
    for side in SIDES:
        center, top_y = sole_reference(model, data, side)
        rows.append(write_part(f"e5_yoke_{side}", yoke(side)))
        rows.append(write_part(f"e5_footplate_{side}", footplate(side, center, top_y)))
        rows.append(write_part(f"e5_cradle_{side}", cradle()))
        horn, lever, length = remote_pushrod_points(model, data, side)
        rows.append(write_part(f"e5_pushrod_{side}", _rod(horn, lever)))
        center_mm = center * 1000.0
        top_y_mm = top_y * 1000.0
        sole = _sole_mesh(center_mm[0], top_y_mm, center_mm[2])
        footprint = np.ptp(sole.vertices[:, (0, 2)], axis=0)
        assert abs(float(sole.vertices[:, 1].max()) - top_y_mm) <= 0.2
        assert (
            abs(
                float((sole.vertices[:, 0].min() + sole.vertices[:, 0].max()) / 2)
                - center_mm[0]
            )
            <= 0.2
        )
        assert (
            abs(
                float((sole.vertices[:, 2].min() + sole.vertices[:, 2].max()) / 2)
                - center_mm[2]
            )
            <= 0.2
        )
        assert np.all(np.abs(footprint - (SOLE_FWD_MM, SOLE_LATERAL_MM)) <= 0.5)
        sole_row = write_part(f"e5_sole_{side}", sole)
        rows.append(sole_row)
        for row in rows[-5:]:
            cap = (
                sole_mass_cap_g
                if "sole" in row["name"]
                else MASS_CAPS_G.get(row["name"].split("_")[1])
            )
            if cap is not None:
                assert row["mass_g"] <= cap + 1.0e-6, (
                    f"{row['name']} mass {row['mass_g']:.4f} g exceeds cap {cap:.4f} g"
                )
        print(f"{side}: pushrod length {length:.3f} mm")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="robot_walk.xml")
    args = parser.parse_args()
    rows = build_all(ROBOT_DIR / args.robot)
    print("part                       volume_cm3  mass_g")
    for row in rows:
        print(f"{row['name']:<26} {row['volume_cm3']:10.4f} {row['mass_g']:8.4f}")
    by_part = {
        name: np.mean(
            [row["mass_g"] for row in rows if row["name"].split("_")[1] == name]
        )
        for name in ("yoke", "footplate", "cradle", "pushrod", "sole")
    }
    base_model = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / args.robot))
    base_mass_g = float(base_model.body_subtreemass[1]) * 1000.0
    added = {
        "baseline": 0.0,
        "direct": by_part["yoke"] + by_part["footplate"] + 4.0 + 18.0,
        "remote": (
            by_part["yoke"]
            + by_part["footplate"]
            + by_part["cradle"]
            + by_part["pushrod"]
            + 4.0
            + 18.0
        ),
        "remote_sole": (
            by_part["yoke"]
            + by_part["footplate"]
            + by_part["cradle"]
            + by_part["pushrod"]
            + by_part["sole"]
            + 4.0
            + 18.0
        ),
    }
    print("variant       added_mass_per_leg_g  total_robot_mass_g")
    for variant, per_leg in added.items():
        print(f"{variant:<13} {per_leg:21.4f} {base_mass_g + 2 * per_leg:18.4f}")


if __name__ == "__main__":
    main()
