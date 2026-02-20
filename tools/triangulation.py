from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class TriangulationCalibration:
    k: np.ndarray
    dist: np.ndarray
    plane1: np.ndarray
    plane2: np.ndarray
    schema_version: int
    updated_at: str
    calibration_id: str
    source_path: str


def _to_float_array(value: Any, shape: tuple[int, ...], field: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{field} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field} must contain only finite numeric values")
    return arr


def _to_dist_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 4:
        raise ValueError("intrinsics.dist_coeffs must have at least 4 coefficients")
    if not np.all(np.isfinite(arr)):
        raise ValueError("intrinsics.dist_coeffs must contain only finite values")
    return arr


def load_triangulation_calibration(path: Path) -> TriangulationCalibration:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("calibration payload must be an object")

    intr = payload.get("intrinsics")
    l1 = payload.get("laser1")
    l2 = payload.get("laser2")
    if not isinstance(intr, dict):
        raise ValueError("missing intrinsics in calibration")
    if not isinstance(l1, dict) or not isinstance(l2, dict):
        raise ValueError("missing laser1/laser2 planes in calibration")

    k = _to_float_array(intr.get("camera_matrix"), (3, 3), "intrinsics.camera_matrix")
    dist = _to_dist_array(intr.get("dist_coeffs"))
    plane1 = _to_float_array(l1.get("plane"), (4,), "laser1.plane")
    plane2 = _to_float_array(l2.get("plane"), (4,), "laser2.plane")

    schema_version = int(payload.get("schema_version", 0))
    updated_at = str(payload.get("updated_at", ""))
    calibration_id = f"sv{schema_version}:{updated_at}" if updated_at else f"sv{schema_version}:unknown"

    return TriangulationCalibration(
        k=k,
        dist=dist,
        plane1=plane1,
        plane2=plane2,
        schema_version=schema_version,
        updated_at=updated_at,
        calibration_id=calibration_id,
        source_path=str(path),
    )


def triangulate_pixels_to_world(
    points_xy: np.ndarray,
    *,
    angle_deg: float,
    k: np.ndarray,
    dist: np.ndarray,
    plane_abcd: np.ndarray,
    eps: float = 1e-9,
) -> tuple[list[list[float]], dict[str, int]]:
    if points_xy is None:
        return [], {"input": 0, "finite": 0, "intersected": 0, "output": 0}

    arr = np.asarray(points_xy, dtype=np.float64)
    if arr.size == 0:
        return [], {"input": 0, "finite": 0, "intersected": 0, "output": 0}
    if arr.ndim != 2 or arr.shape[1] < 2:
        return [], {"input": int(arr.shape[0]) if arr.ndim > 0 else 0, "finite": 0, "intersected": 0, "output": 0}

    pts = arr[:, :2]
    input_count = int(pts.shape[0])

    finite_mask = np.isfinite(pts).all(axis=1)
    pts_finite = pts[finite_mask]
    finite_count = int(pts_finite.shape[0])
    if finite_count == 0:
        return [], {"input": input_count, "finite": 0, "intersected": 0, "output": 0}

    undist = cv2.undistortPoints(pts_finite.reshape(-1, 1, 2), k, dist).reshape(-1, 2)
    rays = np.concatenate([undist, np.ones((undist.shape[0], 1), dtype=np.float64)], axis=1)

    n = plane_abcd[:3].reshape(3)
    d = float(plane_abcd[3])
    denom = rays @ n
    valid = np.abs(denom) > float(eps)

    t = np.zeros_like(denom)
    t[valid] = -d / denom[valid]
    valid &= np.isfinite(t)
    valid &= t > float(eps)

    intersect_count = int(np.count_nonzero(valid))
    if intersect_count == 0:
        return [], {"input": input_count, "finite": finite_count, "intersected": 0, "output": 0}

    p_cam = rays[valid] * t[valid, None]

    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))

    x_cam = p_cam[:, 0]
    y_cam = p_cam[:, 1]
    z_cam = p_cam[:, 2]

    x_w = c * x_cam - s * z_cam
    y_w = y_cam
    z_w = s * x_cam + c * z_cam

    world = np.stack([x_w, y_w, z_w], axis=1)
    world = world[np.isfinite(world).all(axis=1)]
    output_count = int(world.shape[0])

    xyz = [[float(p[0]), float(p[1]), float(p[2])] for p in world]
    return xyz, {
        "input": input_count,
        "finite": finite_count,
        "intersected": intersect_count,
        "output": output_count,
    }

