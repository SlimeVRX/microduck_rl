#!/usr/bin/env python3
"""Sweep compiled all-collision E5 models for non-floor interference."""

from __future__ import annotations

import argparse
import copy
import itertools
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from e5_ankle_cad import SERVO_OFFSET_CANDIDATES_MM, SERVO_OFFSET_LOCAL_MM

ROBOT_DIR = Path(__file__).resolve().parents[1] / "src/mjlab_microduck/robot/microduck"
VARIANTS = ("baseline", "direct", "remote", "remote_sole")
ANKLE_PITCH = (-0.6, -0.3, 0.0, 0.3, 0.6)
ANKLE_ROLL = {
    # Baseline has no roll joint, but is evaluated on the same five-slice
    # comparable grid so its collision fraction has the same denominator.
    "baseline": tuple(np.linspace(-np.deg2rad(45.0), np.deg2rad(45.0), 5)),
    "direct": tuple(np.linspace(-np.deg2rad(30.0), np.deg2rad(30.0), 5)),
    "remote": tuple(np.linspace(-np.deg2rad(45.0), np.deg2rad(45.0), 5)),
    "remote_sole": tuple(np.linspace(-np.deg2rad(45.0), np.deg2rad(45.0), 5)),
}
HIP_ROLL = (-0.6, -0.384, 0.0, 0.384, 0.6)
KNEE = (-0.9, -0.45, 0.0)
HIP_YAW = (-0.4, 0.0, 0.4)
PLACEMENT_HIP_ROLL = tuple(np.linspace(-0.384, 0.384, 5))
PLACEMENT_ANKLE_PITCH = ANKLE_PITCH
PLACEMENT_HIP_YAW = HIP_YAW
ROLL_SCAN_DEGREES = tuple(range(31))
ROLL_SCAN_PITCHES = ANKLE_PITCH
FOOT_DROP_MM = (0.0, 3.0, 6.0, 9.0)
CHAMFER_DEPTH_MM = (2.0, 4.0, 6.0)


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"joint not found: {name}")
    return jid


def _is_floor(model: mujoco.MjModel, geom_id: int) -> bool:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    body_id = model.geom_bodyid[geom_id]
    return body_id == 0 or "floor" in name.lower() or "ground" in name.lower()


def _geom_label(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name:
        return name
    body_id = model.geom_bodyid[geom_id]
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "body"
    mesh_id = int(model.geom_dataid[geom_id])
    mesh = (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        if mesh_id >= 0
        else None
    )
    return f"{body}/{mesh or 'geom'}"


def _load_model(scene: Path, raw: bool) -> mujoco.MjModel:
    if not raw:
        return mujoco.MjModel.from_xml_path(str(scene))

    scene_text = scene.read_text()
    include_match = re.search(r"<include\s+file=\"([^\"]+)\"", scene_text)
    if include_match is None:
        raise ValueError(f"{scene} has no robot include")
    robot_path = scene.parent / include_match.group(1)
    robot_text = robot_path.read_text()
    robot_text = re.sub(
        r"\s*<contact>.*?</contact>\s*", "\n", robot_text, flags=re.DOTALL
    )
    robot_fd, robot_tmp = tempfile.mkstemp(
        prefix="e5_raw_robot_", suffix=".xml", dir=scene.parent
    )
    scene_fd, scene_tmp = tempfile.mkstemp(
        prefix="e5_raw_scene_", suffix=".xml", dir=scene.parent
    )
    os.close(robot_fd)
    os.close(scene_fd)
    robot_tmp_path = Path(robot_tmp)
    scene_tmp_path = Path(scene_tmp)
    try:
        robot_tmp_path.write_text(robot_text)
        scene_text = re.sub(
            r'(<include\s+file=")[^"]+(")',
            rf"\g<1>{robot_tmp_path.name}\g<2>",
            scene_text,
            count=1,
        )
        scene_tmp_path.write_text(scene_text)
        return mujoco.MjModel.from_xml_path(str(scene_tmp_path))
    finally:
        robot_tmp_path.unlink(missing_ok=True)
        scene_tmp_path.unlink(missing_ok=True)


def _pose_joint_ids(model: mujoco.MjModel, side: str, roll: bool) -> dict[str, int]:
    names = {
        "ankle_pitch": f"{side}_ankle",
        "hip_roll": f"{side}_hip_roll",
        "knee": f"{side}_knee",
        "hip_yaw": f"{side}_hip_yaw",
    }
    if roll:
        names["ankle_roll"] = f"{side}_ankle_roll"
    return {key: _joint_id(model, name) for key, name in names.items()}


def _placement_contact_labels(
    model: mujoco.MjModel, data: mujoco.MjData
) -> set[tuple[str, str]]:
    labels = set()
    for contact in data.contact[: data.ncon]:
        if _is_floor(model, contact.geom1) or _is_floor(model, contact.geom2):
            continue
        labels.add(
            tuple(
                sorted(
                    (
                        _geom_label(model, contact.geom1),
                        _geom_label(model, contact.geom2),
                    )
                )
            )
        )
    return labels


def _is_new_e5_part(label: str) -> bool:
    return "e5_" in label


def _filtered_contact_pairs(
    model: mujoco.MjModel, data: mujoco.MjData
) -> list[tuple[str, str]]:
    pairs = []
    for contact in data.contact[: data.ncon]:
        if _is_floor(model, contact.geom1) or _is_floor(model, contact.geom2):
            continue
        pairs.append(
            tuple(
                sorted(
                    (
                        _geom_label(model, contact.geom1),
                        _geom_label(model, contact.geom2),
                    )
                )
            )
        )
    return pairs


def search_remote_servo_placement() -> tuple[int, tuple[float, float, float]]:
    """Search shank placements using compiled cradle and servo geometry."""
    base_scene = ROBOT_DIR / "scene.xml"
    base_robot = ROBOT_DIR / "robot_allcollisions.xml"
    base_model = mujoco.MjModel.from_xml_path(str(base_scene))
    base_data = mujoco.MjData(base_model)
    base_joint_ids = {
        side: _pose_joint_ids(base_model, side, False) for side in ("left", "right")
    }
    key = mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    knee_id = base_joint_ids["left"]["knee"]
    knee_min, knee_max = base_model.jnt_range[knee_id]
    poses = itertools.product(
        PLACEMENT_ANKLE_PITCH,
        PLACEMENT_HIP_ROLL,
        (float(knee_min), float((knee_min + knee_max) / 2.0), float(knee_max)),
        PLACEMENT_HIP_YAW,
    )
    poses = list(poses)
    baseline_contacts = []
    for pitch, hip_roll, knee, hip_yaw in poses:
        mujoco.mj_resetDataKeyframe(base_model, base_data, key)
        for side in ("left", "right"):
            values = {
                "ankle_pitch": pitch,
                "hip_roll": hip_roll,
                "knee": knee,
                "hip_yaw": hip_yaw,
            }
            for label, jid in base_joint_ids[side].items():
                base_data.qpos[base_model.jnt_qposadr[jid]] = values[label]
        mujoco.mj_forward(base_model, base_data)
        baseline_contacts.append(_placement_contact_labels(base_model, base_data))

    robot_root = ET.parse(base_robot).getroot()
    asset = robot_root.find("asset")
    if asset is None:
        raise ValueError("robot XML has no asset section")
    for side in ("left", "right"):
        ET.SubElement(asset, "mesh", {"file": f"e5_cradle_{side}.stl"})
        leg_name = "leg" if side == "left" else "leg_2"
        leg = next(
            body for body in robot_root.iter("body") if body.get("name") == leg_name
        )
        for name, mesh in (
            (f"e5_search_cradle_{side}", f"e5_cradle_{side}"),
            (f"e5_search_servo_{side}", "xl330"),
        ):
            body = ET.SubElement(leg, "body", {"name": name, "pos": "0 0 0"})
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{name}_geom",
                    "type": "mesh",
                    "mesh": mesh,
                    "class": "self_collision_only",
                },
            )
    scene_root = ET.parse(base_scene).getroot()
    results = []
    for candidate in SERVO_OFFSET_CANDIDATES_MM:
        candidate_root = copy.deepcopy(robot_root)
        for side in ("left", "right"):
            leg_name = "leg" if side == "left" else "leg_2"
            leg = next(
                body
                for body in candidate_root.iter("body")
                if body.get("name") == leg_name
            )
            rear, along, inboard = candidate
            rear *= -1.0 if side == "left" else 1.0
            pos = f"{rear / 1000.0:.9g} {along / 1000.0:.9g} {inboard / 1000.0:.9g}"
            for name in (f"e5_search_cradle_{side}", f"e5_search_servo_{side}"):
                next(body for body in leg if body.get("name") == name).set("pos", pos)
        robot_fd, robot_path = tempfile.mkstemp(
            prefix="e5_search_robot_", suffix=".xml", dir=ROBOT_DIR
        )
        scene_fd, scene_path = tempfile.mkstemp(
            prefix="e5_search_scene_", suffix=".xml", dir=ROBOT_DIR
        )
        os.close(robot_fd)
        os.close(scene_fd)
        robot_file = Path(robot_path)
        scene_file = Path(scene_path)
        try:
            ET.ElementTree(candidate_root).write(
                robot_file, encoding="unicode", xml_declaration=True
            )
            candidate_scene = copy.deepcopy(scene_root)
            include = candidate_scene.find("include")
            if include is None:
                raise ValueError("scene XML has no include")
            include.set("file", robot_file.name)
            ET.ElementTree(candidate_scene).write(
                scene_file, encoding="unicode", xml_declaration=True
            )
            model = mujoco.MjModel.from_xml_path(str(scene_file))
            data = mujoco.MjData(model)
            joints = {
                side: _pose_joint_ids(model, side, False) for side in ("left", "right")
            }
            candidate_new = 0
            for pose_index, (pitch, hip_roll, knee, hip_yaw) in enumerate(poses):
                mujoco.mj_resetDataKeyframe(model, data, key)
                for side in ("left", "right"):
                    values = {
                        "ankle_pitch": pitch,
                        "hip_roll": hip_roll,
                        "knee": knee,
                        "hip_yaw": hip_yaw,
                    }
                    for label, jid in joints[side].items():
                        data.qpos[model.jnt_qposadr[jid]] = values[label]
                mujoco.mj_forward(model, data)
                current = _placement_contact_labels(model, data)
                new = {
                    pair
                    for pair in current - baseline_contacts[pose_index]
                    if not any("e5_search_" in label for label in pair)
                }
                candidate_new += bool(new)
            results.append((candidate_new, candidate))
            if candidate_new == 0:
                extended_new = 0
                extended_poses = list(
                    itertools.product(
                        PLACEMENT_ANKLE_PITCH,
                        (-0.6, 0.6),
                        (
                            float(knee_min),
                            float((knee_min + knee_max) / 2.0),
                            float(knee_max),
                        ),
                        PLACEMENT_HIP_YAW,
                    )
                )
                for pitch, hip_roll, knee, hip_yaw in extended_poses:
                    mujoco.mj_resetDataKeyframe(base_model, base_data, key)
                    for side in ("left", "right"):
                        values = {
                            "ankle_pitch": pitch,
                            "hip_roll": hip_roll,
                            "knee": knee,
                            "hip_yaw": hip_yaw,
                        }
                        for label, jid in base_joint_ids[side].items():
                            base_data.qpos[base_model.jnt_qposadr[jid]] = values[label]
                    mujoco.mj_forward(base_model, base_data)
                    baseline = _placement_contact_labels(base_model, base_data)
                    mujoco.mj_resetDataKeyframe(model, data, key)
                    for side in ("left", "right"):
                        values = {
                            "ankle_pitch": pitch,
                            "hip_roll": hip_roll,
                            "knee": knee,
                            "hip_yaw": hip_yaw,
                        }
                        for label, jid in joints[side].items():
                            data.qpos[model.jnt_qposadr[jid]] = values[label]
                    mujoco.mj_forward(model, data)
                    new = {
                        pair
                        for pair in _placement_contact_labels(model, data) - baseline
                        if not any("e5_search_" in label for label in pair)
                    }
                    extended_new += bool(new)
                print(
                    "placement search selected first zero-new-contact candidate: "
                    f"rear={candidate[0]:g} mm along_shank={candidate[1]:g} mm "
                    f"inboard={candidate[2]:g} mm over {len(poses)} poses; "
                    f"extended_hip_roll_new={extended_new}/{len(extended_poses)}"
                )
                return len(poses), candidate
        finally:
            robot_file.unlink(missing_ok=True)
            scene_file.unlink(missing_ok=True)
    raise AssertionError(f"no zero-new-contact placement found: {results[:5]}")


def sweep_variant(scene: Path, variant: str) -> dict[str, object]:
    model = _load_model(scene, raw=False)
    raw_model = _load_model(scene, raw=True)
    data = mujoco.MjData(model)
    raw_data = mujoco.MjData(raw_model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    raw_key = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if key < 0:
        raise ValueError(f"{scene} has no STAND keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_resetDataKeyframe(raw_model, raw_data, raw_key)
    roll_enabled = variant != "baseline"
    joint_ids = {
        side: _pose_joint_ids(model, side, roll_enabled) for side in ("left", "right")
    }
    raw_joint_ids = {
        side: _pose_joint_ids(raw_model, side, roll_enabled)
        for side in ("left", "right")
    }
    poses = itertools.product(
        ANKLE_PITCH,
        ANKLE_ROLL[variant],
        HIP_ROLL,
        KNEE,
        HIP_YAW,
    )
    raw_pair_counts: Counter[tuple[str, str]] = Counter()
    filtered_pair_counts: Counter[tuple[str, str]] = Counter()
    filtered_pair_pose_counts: Counter[tuple[str, str]] = Counter()
    filtered_pair_first_pose: dict[tuple[str, str], tuple[float, ...]] = {}
    raw_colliding_poses = 0
    filtered_colliding_poses = 0
    worst_contacts = -1
    worst_pose: tuple[float, ...] | None = None
    worst_pairs: list[tuple[str, str]] = []
    for pose in poses:
        pitch, roll, hip_roll, knee, hip_yaw = pose
        mujoco.mj_resetDataKeyframe(model, data, key)
        for side in ("left", "right"):
            values = {
                "ankle_pitch": pitch,
                "hip_roll": hip_roll,
                "knee": knee,
                "hip_yaw": hip_yaw,
            }
            if roll_enabled:
                values["ankle_roll"] = roll
            for label, jid in joint_ids[side].items():
                data.qpos[model.jnt_qposadr[jid]] = values[label]
            for label, jid in raw_joint_ids[side].items():
                raw_data.qpos[raw_model.jnt_qposadr[jid]] = values[label]
        data.qvel[:] = 0.0
        raw_data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        mujoco.mj_forward(raw_model, raw_data)
        raw_pairs: list[tuple[str, str]] = []
        for contact in raw_data.contact[: raw_data.ncon]:
            if _is_floor(raw_model, contact.geom1) or _is_floor(
                raw_model, contact.geom2
            ):
                continue
            pair = tuple(
                sorted(
                    (
                        _geom_label(raw_model, contact.geom1),
                        _geom_label(raw_model, contact.geom2),
                    )
                )
            )
            raw_pairs.append(pair)
            raw_pair_counts[pair] += 1
        filtered_pairs = _filtered_contact_pairs(model, data)
        for pair in filtered_pairs:
            filtered_pair_counts[pair] += 1
        for pair in set(filtered_pairs):
            filtered_pair_pose_counts[pair] += 1
            filtered_pair_first_pose.setdefault(
                pair, tuple(float(value) for value in pose)
            )
        if raw_pairs:
            raw_colliding_poses += 1
        if filtered_pairs:
            filtered_colliding_poses += 1
            if len(filtered_pairs) > worst_contacts:
                worst_contacts = len(filtered_pairs)
                worst_pose = tuple(float(value) for value in pose)
                worst_pairs = filtered_pairs
    poses = sum(
        1
        for _ in itertools.product(
            ANKLE_PITCH, ANKLE_ROLL[variant], HIP_ROLL, KNEE, HIP_YAW
        )
    )
    return {
        "scene": scene.name,
        "poses": poses,
        "raw_colliding_poses": raw_colliding_poses,
        "raw_colliding_fraction": raw_colliding_poses / poses,
        "filtered_colliding_poses": filtered_colliding_poses,
        "filtered_colliding_fraction": filtered_colliding_poses / poses,
        "raw_top_pairs": raw_pair_counts.most_common(10),
        "top_pairs": filtered_pair_pose_counts.most_common(10),
        "pair_pose_counts": filtered_pair_pose_counts,
        "pair_first_pose": filtered_pair_first_pose,
        "worst_pose": worst_pose,
        "worst_contacts": worst_contacts,
        "worst_pairs": worst_pairs,
    }


def _set_sweep_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    key: int,
    joint_ids: dict[str, dict[str, int]],
    *,
    pitch: float,
    roll: float,
    hip_roll: float | None,
    knee: float | None,
    hip_yaw: float | None,
    nominal_qpos: np.ndarray | None = None,
) -> None:
    mujoco.mj_resetDataKeyframe(model, data, key)
    if nominal_qpos is not None:
        data.qpos[:] = nominal_qpos
    for side in ("left", "right"):
        values = {
            "ankle_pitch": pitch,
            "ankle_roll": roll,
            "hip_roll": hip_roll,
            "knee": knee,
            "hip_yaw": hip_yaw,
        }
        for label, jid in joint_ids[side].items():
            if values.get(label) is not None:
                data.qpos[model.jnt_qposadr[jid]] = values[label]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _is_collision_free(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    return not _filtered_contact_pairs(model, data)


def _stand_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, np.ndarray]:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(model, data, key)
    return key, data.qpos.copy()


def _set_one_ankle_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    key: int,
    stand_qpos: np.ndarray,
    side: str,
    roll_rad: float,
    pitch_rad: float | None = None,
) -> None:
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.qpos[:] = stand_qpos
    roll_id = _joint_id(model, f"{side}_ankle_roll")
    data.qpos[model.jnt_qposadr[roll_id]] = roll_rad
    if pitch_rad is not None:
        pitch_id = _joint_id(model, f"{side}_ankle")
        data.qpos[model.jnt_qposadr[pitch_id]] = pitch_rad
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _drop_foot_below_pitch_axis(
    model: mujoco.MjModel, data: mujoco.MjData, side: str, drop_mm: float
) -> None:
    if drop_mm == 0.0:
        return
    _, stand_qpos = _stand_qpos(model, data)
    mujoco.mj_forward(model, data)
    ankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ankle_{side}")
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"foot_{side}")
    ankle_xmat = data.xmat[ankle_id].reshape(3, 3)
    local_down = ankle_xmat.T @ np.array([0.0, 0.0, -1.0])
    model.body_pos[foot_id] += local_down * (drop_mm / 1000.0)
    data.qpos[:] = stand_qpos
    mujoco.mj_forward(model, data)


def _first_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    key: int,
    stand_qpos: np.ndarray,
    side: str,
    sign: int,
    pitch_rad: float | None = None,
) -> tuple[int | None, tuple[str, str] | None]:
    for degree in ROLL_SCAN_DEGREES:
        _set_one_ankle_pose(
            model,
            data,
            key,
            stand_qpos,
            side,
            sign * np.deg2rad(degree),
            pitch_rad,
        )
        pairs = _filtered_contact_pairs(model, data)
        if pairs:
            return degree, pairs[0]
    return None, None


def _geometry_vertical_gap(
    model: mujoco.MjModel, data: mujoco.MjData, side: str
) -> float:
    sole_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision"
    )
    leg_body = "leg" if side == "left" else "leg_2"
    leg_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, leg_body)
    shank_id = next(
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == leg_id
        and mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_MESH,
            int(model.geom_dataid[geom_id]),
        )
        == "leg"
    )

    def world_vertices(geom_id: int) -> np.ndarray:
        mesh_id = int(model.geom_dataid[geom_id])
        vertices = model.mesh_vert[
            model.mesh_vertadr[mesh_id] : model.mesh_vertadr[mesh_id]
            + model.mesh_vertnum[mesh_id]
        ]
        return (
            data.geom_xpos[geom_id] + vertices @ data.geom_xmat[geom_id].reshape(3, 3).T
        )

    sole_vertices = world_vertices(sole_id)
    shank_vertices = world_vertices(shank_id)
    return float(shank_vertices[:, 2].min() - sole_vertices[:, 2].max())


def _scan_side(
    scene: Path,
    side: str,
    pitch_rad: float | None = None,
    drop_mm: float = 0.0,
) -> dict[str, object]:
    model = _load_model(scene, raw=False)
    data = mujoco.MjData(model)
    key, stand_qpos = _stand_qpos(model, data)
    _drop_foot_below_pitch_axis(model, data, side, drop_mm)
    return {
        "positive": _first_contact(model, data, key, stand_qpos, side, 1, pitch_rad),
        "negative": _first_contact(model, data, key, stand_qpos, side, -1, pitch_rad),
    }


def _gap_curve(scene: Path, side: str) -> list[tuple[int, float]]:
    model = _load_model(scene, raw=False)
    data = mujoco.MjData(model)
    key, stand_qpos = _stand_qpos(model, data)
    curve = []
    for sign in (1, -1):
        for degree in ROLL_SCAN_DEGREES:
            _set_one_ankle_pose(
                model,
                data,
                key,
                stand_qpos,
                side,
                sign * np.deg2rad(degree),
            )
            curve.append((sign * degree, _geometry_vertical_gap(model, data, side)))
    return curve


def _first_nonpositive_gap(curve: list[tuple[int, float]]) -> int | None:
    for degree, gap in curve:
        if gap <= 0.0:
            return degree
    return None


def _boundary_text(result: tuple[int | None, tuple[str, str] | None]) -> str:
    first, _ = result
    if first is None:
        return "first_contact=None, max_clear=30"
    return f"first_contact={first}, max_clear={first - 1}"


def _shank_contact_details(
    scene: Path, side: str, roll_degree: int
) -> dict[str, object]:
    model = _load_model(scene, raw=False)
    data = mujoco.MjData(model)
    key, stand_qpos = _stand_qpos(model, data)
    _set_one_ankle_pose(model, data, key, stand_qpos, side, np.deg2rad(roll_degree))
    leg_body_name = "leg" if side == "left" else "leg_2"
    leg_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, leg_body_name)
    foot_label = f"{side}_foot_collision"
    contact = next(
        contact
        for contact in data.contact[: data.ncon]
        if foot_label
        in (
            _geom_label(model, int(contact.geom1)),
            _geom_label(model, int(contact.geom2)),
        )
        and leg_body_id
        in (
            int(model.geom_bodyid[int(contact.geom1)]),
            int(model.geom_bodyid[int(contact.geom2)]),
        )
    )
    shank_geom_id = next(
        int(geom_id)
        for geom_id in (contact.geom1, contact.geom2)
        if int(model.geom_bodyid[int(geom_id)]) == leg_body_id
    )
    mesh_id = int(model.geom_dataid[shank_geom_id])
    vertices = model.mesh_vert[
        model.mesh_vertadr[mesh_id] : model.mesh_vertadr[mesh_id]
        + model.mesh_vertnum[mesh_id]
    ]
    local_point = (contact.pos - data.geom_xpos[shank_geom_id]) @ data.geom_xmat[
        shank_geom_id
    ].reshape(3, 3)
    distances = np.linalg.norm(vertices - local_point, axis=1)
    nearby_ids = np.flatnonzero(distances <= 0.001)
    bounds = (vertices.min(axis=0), vertices.max(axis=0))
    inboard = local_point[0] <= bounds[0][0] + 0.001
    lower = local_point[2] <= bounds[0][2] + 0.002
    if inboard and lower:
        region = "lower inboard corner/edge"
    elif lower:
        region = "lower edge, away from inboard side"
    else:
        region = "broad side or non-lower feature"
    return {
        "roll_degree": roll_degree,
        "contact_point_world": contact.pos.copy(),
        "contact_distance": float(contact.dist),
        "contact_pair": (
            _geom_label(model, int(contact.geom1)),
            _geom_label(model, int(contact.geom2)),
        ),
        "contact_point_local": local_point,
        "nearby_vertices_local": vertices[nearby_ids],
        "nearby_distances": distances[nearby_ids],
        "bounds": bounds,
        "region": region,
    }


def _make_chamfered_leg(depth_mm: float):
    import trimesh

    mesh = trimesh.load_mesh(ROBOT_DIR / "assets" / "leg.stl", process=True)
    trimesh.repair.fill_holes(mesh)
    mesh.remove_unreferenced_vertices()
    lower, upper = mesh.bounds
    depth = depth_mm / 1000.0
    y_min = lower[1] - 0.001
    y_max = upper[1] + 0.001
    x_min = lower[0]
    z_min = lower[2]
    vertices = np.array(
        [
            [x_min, y_min, z_min],
            [x_min + depth, y_min, z_min],
            [x_min, y_min, z_min + depth],
            [x_min, y_max, z_min],
            [x_min + depth, y_max, z_min],
            [x_min, y_max, z_min + depth],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [3, 5, 4],
            [0, 3, 4],
            [0, 4, 1],
            [1, 4, 5],
            [1, 5, 2],
            [2, 5, 3],
            [2, 3, 0],
        ]
    )
    cutter = trimesh.Trimesh(vertices, faces, process=True)
    result = trimesh.boolean.difference([mesh, cutter], engine="manifold")
    return result, float((upper[0] - lower[0]) * 1000.0 - depth_mm)


def _load_chamfered_model(scene: Path, depth_mm: float) -> mujoco.MjModel:
    mesh, _ = _make_chamfered_leg(depth_mm)
    with tempfile.TemporaryDirectory(prefix="e5_chamfer_") as directory:
        directory_path = Path(directory)
        stl_path = directory_path / "leg_chamfer.stl"
        mesh.export(stl_path)
        include_match = re.search(r"<include\s+file=\"([^\"]+)\"", scene.read_text())
        if include_match is None:
            raise ValueError(f"{scene} has no robot include")
        robot_path = scene.parent / include_match.group(1)
        robot_text = robot_path.read_text()
        robot_text = robot_text.replace(
            'meshdir="assets"', f'meshdir="{ROBOT_DIR / "assets"}"'
        )
        robot_text = robot_text.replace(
            '<mesh file="leg.stl" />',
            f'<mesh file="leg.stl" /><mesh name="leg_chamfer" file="{stl_path}" />',
        )
        robot_text = re.sub(
            r'(<geom[^>]*class="collision"[^>]*mesh=")leg(")',
            r"\1leg_chamfer\2",
            robot_text,
        )
        temporary_robot = directory_path / robot_path.name
        temporary_robot.write_text(robot_text)
        scene_text = scene.read_text()
        scene_text = re.sub(
            r'<include\s+file="[^"]+"',
            f'<include file="{temporary_robot}"',
            scene_text,
            count=1,
        )
        temporary_scene = directory_path / scene.name
        temporary_scene.write_text(scene_text)
        return mujoco.MjModel.from_xml_path(str(temporary_scene))


def _scan_compiled_model(
    model: mujoco.MjModel, side: str
) -> dict[str, tuple[int | None, tuple[str, str] | None]]:
    data = mujoco.MjData(model)
    key, stand_qpos = _stand_qpos(model, data)
    return {
        "positive": _first_contact(model, data, key, stand_qpos, side, 1),
        "negative": _first_contact(model, data, key, stand_qpos, side, -1),
    }


def _print_contact_detail(detail: dict[str, object]) -> None:
    point = np.asarray(detail["contact_point_world"]) * 1000.0
    local = np.asarray(detail["contact_point_local"]) * 1000.0
    vertices = np.asarray(detail["nearby_vertices_local"]) * 1000.0
    distances = np.asarray(detail["nearby_distances"]) * 1000.0
    print(
        f"    roll={detail['roll_degree']:+d} deg, "
        f"pair={detail['contact_pair']}, "
        f"dist={float(detail['contact_distance']) * 1000:.4f} mm"
    )
    print(f"      contact world mm={np.round(point, 3).tolist()}")
    print(f"      contact shank-local mm={np.round(local, 3).tolist()}")
    print(f"      region={detail['region']}")
    print("      shank vertices within 1 mm (local mm):")
    for vertex, distance in zip(vertices, distances):
        print(f"        d={distance:.3f}: {np.round(vertex, 3).tolist()}")


def print_chamfer_scan() -> None:
    direct = ROBOT_DIR / "scene_e5_direct.xml"
    remote = ROBOT_DIR / "scene_e5_remote.xml"
    print("\nSHANK CONTACT DETAILS: direct sole/shank limits")
    for side, roll in (("left", -13), ("right", 13)):
        _print_contact_detail(_shank_contact_details(direct, side, roll))

    print("\nSHANK CONTACT DETAILS: remote sole/shank limits")
    for side, roll in (("left", -22), ("right", 22)):
        _print_contact_detail(_shank_contact_details(remote, side, roll))

    print("\nDIRECT IN-MEMORY 45-DEGREE SHANK CHAMFER SCAN")
    for depth_mm in CHAMFER_DEPTH_MM:
        _, thickness_mm = _make_chamfered_leg(depth_mm)
        model = _load_chamfered_model(direct, depth_mm)
        print(
            f"  chamfer={depth_mm:.0f} mm, "
            f"remaining plate thickness={thickness_mm:.3f} mm"
        )
        for side in ("left", "right"):
            result = _scan_compiled_model(model, side)
            print(
                f"    {side}: + {_boundary_text(result['positive'])} deg "
                f"pair={result['positive'][1]}; "
                f"- {_boundary_text(result['negative'])} deg "
                f"pair={result['negative'][1]}"
            )

    best_depth = CHAMFER_DEPTH_MM[-1]
    model = _load_chamfered_model(remote, best_depth)
    _, thickness_mm = _make_chamfered_leg(best_depth)
    print(
        f"\nREMOTE IN-MEMORY SCAN: chamfer={best_depth:.0f} mm, "
        f"remaining plate thickness={thickness_mm:.3f} mm"
    )
    for side in ("left", "right"):
        result = _scan_compiled_model(model, side)
        print(
            f"  {side}: + {_boundary_text(result['positive'])} deg "
            f"pair={result['positive'][1]}; "
            f"- {_boundary_text(result['negative'])} deg "
            f"pair={result['negative'][1]}"
        )


def print_fine_scan() -> None:
    direct = ROBOT_DIR / "scene_e5_direct.xml"
    remote = ROBOT_DIR / "scene_e5_remote.xml"
    print("\nONE-ANKLE STAND SCAN: direct, 1-degree increments")
    for side in ("left", "right"):
        result = _scan_side(direct, side)
        print(
            f"  {side}: + {_boundary_text(result['positive'])} deg "
            f"pair={result['positive'][1]}; "
            f"- {_boundary_text(result['negative'])} deg "
            f"pair={result['negative'][1]}"
        )

    print("\nONE-ANKLE PITCH CURVE: direct, STAND other joints")
    for side in ("left", "right"):
        print(f"  {side}:")
        for pitch in ROLL_SCAN_PITCHES:
            result = _scan_side(direct, side, pitch)
            print(
                f"    pitch={pitch:+.3f} rad: "
                f"+ {_boundary_text(result['positive'])} deg "
                f"{result['positive'][1]}; "
                f"- {_boundary_text(result['negative'])} deg "
                f"{result['negative'][1]}"
            )

    print("\nZERO-ROLL VERTICAL CLEARANCE AND ROLL CLOSURE: direct")
    for side in ("left", "right"):
        print(f"  {side}:")
        curve = _gap_curve(direct, side)
        print(
            "    first non-positive gap: "
            f"{_first_nonpositive_gap(curve)} deg by signed roll"
        )
        for signed_degree, gap in curve:
            if abs(signed_degree) % 5 == 0:
                print(f"    roll={signed_degree:+d} deg: gap={gap * 1000:.3f} mm")

    print("\nDROPPED-FOOT SENSITIVITY: direct, STAND")
    for side in ("left", "right"):
        print(f"  {side}:")
        for drop_mm in FOOT_DROP_MM:
            result = _scan_side(direct, side, drop_mm=drop_mm)
            print(
                f"    drop={drop_mm:.0f} mm: "
                f"+ {_boundary_text(result['positive'])} deg; "
                f"- {_boundary_text(result['negative'])} deg"
            )

    print("\nONE-ANKLE STAND SCAN: remote, 1-degree increments")
    for side in ("left", "right"):
        result = _scan_side(remote, side)
        print(
            f"  {side}: + {_boundary_text(result['positive'])} deg "
            f"pair={result['positive'][1]}; "
            f"- {_boundary_text(result['negative'])} deg "
            f"pair={result['negative'][1]}"
        )


def usable_roll_envelope(scene: Path, variant: str) -> list[dict[str, object]]:
    if variant == "baseline":
        return [
            {
                "ankle_pitch": pitch,
                "nominal_max_abs_roll": None,
                "subgrid_max_abs_roll": None,
            }
            for pitch in ANKLE_PITCH
        ]

    model = _load_model(scene, raw=False)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    joint_ids = {side: _pose_joint_ids(model, side, True) for side in ("left", "right")}
    mujoco.mj_resetDataKeyframe(model, data, key)
    nominal_qpos = data.qpos.copy()
    rows = []
    for pitch in ANKLE_PITCH:
        nominal_clear = []
        subgrid_clear = []
        for roll in ANKLE_ROLL[variant]:
            _set_sweep_pose(
                model,
                data,
                key,
                joint_ids,
                pitch=pitch,
                roll=roll,
                hip_roll=None,
                knee=None,
                hip_yaw=None,
                nominal_qpos=nominal_qpos,
            )
            if _is_collision_free(model, data):
                nominal_clear.append(abs(roll))

            clear_everywhere = True
            for hip_roll, knee, hip_yaw in itertools.product(HIP_ROLL, KNEE, HIP_YAW):
                _set_sweep_pose(
                    model,
                    data,
                    key,
                    joint_ids,
                    pitch=pitch,
                    roll=roll,
                    hip_roll=hip_roll,
                    knee=knee,
                    hip_yaw=hip_yaw,
                )
                if not _is_collision_free(model, data):
                    clear_everywhere = False
                    break
            if clear_everywhere:
                subgrid_clear.append(abs(roll))
        rows.append(
            {
                "ankle_pitch": pitch,
                "nominal_max_abs_roll": max(nominal_clear, default=None),
                "subgrid_max_abs_roll": max(subgrid_clear, default=None),
            }
        )
    return rows


def print_analysis(scene: Path, variant: str, result: dict[str, object]) -> None:
    poses = int(result["poses"])
    pose_counts = result["pair_pose_counts"]
    print(f"\n{variant} analysis:")
    print("  filtered top geom pairs:")
    for pair, _ in result["top_pairs"]:
        count = pose_counts[pair]
        new = _is_new_e5_part(pair[0]) or _is_new_e5_part(pair[1])
        print(
            f"    {pair[0]} / {pair[1]}: {count}/{poses} "
            f"({count / poses:.6f}), new_e5_part={new}"
        )

    print("  usable roll envelope (largest collision-free |roll|):")
    if variant == "baseline":
        print("    baseline has no ankle-roll DOF: N/A")
    else:
        for row in usable_roll_envelope(scene, variant):
            nominal = row["nominal_max_abs_roll"]
            subgrid = row["subgrid_max_abs_roll"]
            nominal_text = (
                "N/A"
                if nominal is None
                else f"{float(nominal):.6f} rad ({np.rad2deg(nominal):.1f} deg)"
            )
            subgrid_text = (
                "N/A"
                if subgrid is None
                else f"{float(subgrid):.6f} rad ({np.rad2deg(subgrid):.1f} deg)"
            )
            print(
                f"    pitch={row['ankle_pitch']:+.3f} rad: "
                f"nominal={nominal_text}, "
                f"full_subgrid={subgrid_text}"
            )

    new_pairs = [
        pair
        for pair in pose_counts
        if _is_new_e5_part(pair[0]) or _is_new_e5_part(pair[1])
    ]
    if not new_pairs:
        print("  worst new-part pair: none in filtered sweep")
        return
    worst = max(new_pairs, key=lambda pair: pose_counts[pair])
    print(
        "  worst new-part pair: "
        f"{worst[0]} / {worst[1]}: {pose_counts[worst]}/{poses} "
        f"({pose_counts[worst] / poses:.6f})"
    )
    print(
        "  first pose (ankle_pitch, ankle_roll, hip_roll, knee, hip_yaw): "
        f"{result['pair_first_pose'][worst]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROBOT_DIR)
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="print filtered pair frequencies and usable roll envelopes",
    )
    parser.add_argument(
        "--fine-scan",
        action="store_true",
        help="measure one-ankle roll boundaries and geometric clearance",
    )
    parser.add_argument(
        "--chamfer-scan",
        action="store_true",
        help="measure shank contact locations and in-memory chamfers",
    )
    args = parser.parse_args()
    scenes = {
        "baseline": args.directory / "scene.xml",
        "direct": args.directory / "scene_e5_direct.xml",
        "remote": args.directory / "scene_e5_remote.xml",
        "remote_sole": args.directory / "scene_e5_remote_sole.xml",
    }
    if args.fine_scan:
        print_fine_scan()
        return
    if args.chamfer_scan:
        print_chamfer_scan()
        return
    search_remote_servo_placement()
    print(
        "remote placement constants: "
        f"left={SERVO_OFFSET_LOCAL_MM['left'].tolist()} mm "
        f"right={SERVO_OFFSET_LOCAL_MM['right'].tolist()} mm"
    )
    for variant in VARIANTS:
        result = sweep_variant(scenes[variant], variant)
        print(f"\n{variant}: {result['scene']}")
        print(f"  poses: {result['poses']}")
        print(
            f"  raw colliding poses: {result['raw_colliding_poses']} "
            f"({result['raw_colliding_fraction']:.6f})"
        )
        print(
            f"  filtered colliding poses: {result['filtered_colliding_poses']} "
            f"({result['filtered_colliding_fraction']:.6f})"
        )
        print("  filtered top geom pairs:")
        for pair, count in result["top_pairs"]:
            print(f"    {count:4d} {pair[0]} / {pair[1]}")
        print(
            f"  worst pose (pitch, roll, hip_roll, knee, hip_yaw): {result['worst_pose']}"
        )
        print(f"  worst contact count: {result['worst_contacts']}")
        if args.analysis:
            print_analysis(scenes[variant], variant, result)


if __name__ == "__main__":
    main()
