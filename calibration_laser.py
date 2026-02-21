from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .calibration_models import LaserPlaneCalibration
    from .scan_algo import capture_pair_details
except ImportError:
    from calibration_models import LaserPlaneCalibration
    from scan_algo import capture_pair_details


def plane_fit_svd(points_xyz: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be Nx3")
    if points_xyz.shape[0] < 3:
        raise ValueError("at least 3 points are required to fit a plane")

    centroid = points_xyz.mean(axis=0)
    centered = points_xyz - centroid
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / max(1e-12, np.linalg.norm(normal))
    d = -float(np.dot(normal, centroid))
    plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)

    denom = max(1e-12, np.linalg.norm(plane[:3]))
    dist = np.abs(points_xyz @ plane[:3] + plane[3]) / denom
    metrics = {
        "rmse": float(np.sqrt(np.mean(dist ** 2))),
        "mean_abs_error": float(np.mean(dist)),
        "max_abs_error": float(np.max(dist)),
    }
    return plane, metrics


def pixel_to_camera_ray(k: np.ndarray, uv: np.ndarray) -> np.ndarray:
    inv_k = np.linalg.inv(k)
    ones = np.ones((uv.shape[0], 1), dtype=np.float64)
    homog = np.concatenate([uv.astype(np.float64), ones], axis=1)
    rays = (inv_k @ homog.T).T
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    rays = rays / np.maximum(norms, 1e-12)
    return rays


def intersect_rays_with_plane(rays: np.ndarray, plane: np.ndarray) -> np.ndarray:
    n = plane[:3]
    d = float(plane[3])
    denom = rays @ n
    valid = np.abs(denom) > 1e-9
    t = np.full((rays.shape[0],), np.nan, dtype=np.float64)
    t[valid] = -d / denom[valid]
    valid = valid & (t > 0)
    pts = np.full_like(rays, np.nan)
    pts[valid] = rays[valid] * t[valid][:, None]
    return pts


@dataclass
class LaserCaptureSample:
    laser: int
    plane: List[float]
    points_px: List[List[float]]
    confidence: float = 0.0
    telemetry: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LaserSession:
    board_plane: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, -0.30])
    min_points_per_laser: int = 200
    samples: List[LaserCaptureSample] = field(default_factory=list)
    captures: int = 0


class LaserPlaneCalibrationService:
    def __init__(self):
        self.session: Optional[LaserSession] = None
        self.last_result: Dict[int, LaserPlaneCalibration] = {}

    def start(self, board_plane: Optional[List[float]] = None, min_points_per_laser: int = 200) -> Dict[str, Any]:
        bp = board_plane if board_plane is not None else [0.0, 0.0, 1.0, -0.30]
        if len(bp) != 4:
            raise ValueError("board_plane must be [a,b,c,d]")
        self.session = LaserSession(board_plane=[float(x) for x in bp], min_points_per_laser=int(min_points_per_laser))
        return self.status()

    def status(self) -> Dict[str, Any]:
        if self.session is None:
            return {"running": False, "has_result": bool(self.last_result)}
        s = self.session
        c1 = 0
        c2 = 0
        p1 = 0
        p2 = 0
        for smp in s.samples:
            n = len(smp.points_px)
            if smp.laser == 1:
                c1 += 1
                p1 += n
            else:
                c2 += 1
                p2 += n
        return {
            "running": True,
            "captures": int(s.captures),
            "samples_laser1": int(c1),
            "samples_laser2": int(c2),
            "points_laser1": int(p1),
            "points_laser2": int(p2),
            "min_points_per_laser": int(s.min_points_per_laser),
            "board_plane": list(s.board_plane),
            "has_result": bool(self.last_result),
        }

    def capture(
        self,
        *,
        laser_index: int,
        camera,
        gpio,
        laser,
        detector,
        intrinsics_k: np.ndarray,
        settle_s: float = 0.06,
        drop_n: int = 2,
        board_plane: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        if self.session is None:
            raise RuntimeError("laser calibration not started")
        s = self.session
        plane = np.array(board_plane if board_plane is not None else s.board_plane, dtype=np.float64)
        if plane.shape[0] != 4:
            raise ValueError("board_plane must have 4 values")

        pair = capture_pair_details(
            camera,
            gpio,
            laser,
            settle_s=settle_s,
            drop_n=drop_n,
            retry_n=2,
            max_pair_retries=2,
        )
        ambient, laser_frame = pair.ambient, pair.laser_frame
        if ambient is None or laser_frame is None:
            return {"ok": False, "accepted": False, "reason": "camera frame missing"}

        pts, _mask = detector.detect(ambient, laser_frame, object_mask=None)
        det = detector.detect_details(ambient, laser_frame, object_mask=None, mode="calibration")
        pts = det.points
        if not pts:
            return {"ok": True, "accepted": False, "reason": "no stripe points"}

        uv = np.array(pts, dtype=np.float64)
        rays = pixel_to_camera_ray(intrinsics_k, uv)
        xyz = intersect_rays_with_plane(rays, plane)
        finite = np.isfinite(xyz).all(axis=1)
        xyz_good = xyz[finite]
        uv_good = uv[finite]
        if xyz_good.shape[0] < 10:
            return {"ok": True, "accepted": False, "reason": "insufficient intersected points"}

        which = int(laser_index)
        if which not in (1, 2):
            raise ValueError("laser_index must be 1 or 2")
        s.samples.append(
            LaserCaptureSample(
                laser=which,
                plane=plane.tolist(),
                points_px=uv_good.astype(np.float32).tolist(),
                confidence=float(det.confidence),
                telemetry=dict(det.telemetry),
            )
        )
        s.captures += 1
        return {
            "ok": True,
            "accepted": True,
            "laser": which,
            "points": int(uv_good.shape[0]),
            "confidence": float(det.confidence),
            "telemetry": dict(det.telemetry),
            "pair": {
                "stable": bool(pair.stable),
                "attempts": int(pair.attempts),
                "drift": dict(pair.drift or {}),
            },
            "captures": int(s.captures),
        }

    def _collect_xyz_for_laser(self, laser: int, intrinsics_k: np.ndarray) -> Tuple[np.ndarray, int]:
        if self.session is None:
            raise RuntimeError("laser calibration not started")
        all_xyz: List[np.ndarray] = []
        sample_count = 0
        for smp in self.session.samples:
            if smp.laser != laser:
                continue
            sample_count += 1
            uv = np.array(smp.points_px, dtype=np.float64)
            if uv.size == 0:
                continue
            rays = pixel_to_camera_ray(intrinsics_k, uv)
            plane = np.array(smp.plane, dtype=np.float64)
            xyz = intersect_rays_with_plane(rays, plane)
            finite = np.isfinite(xyz).all(axis=1)
            xyz = xyz[finite]
            if xyz.shape[0] > 0:
                all_xyz.append(xyz)
        if not all_xyz:
            return np.zeros((0, 3), dtype=np.float64), sample_count
        return np.concatenate(all_xyz, axis=0), sample_count

    def solve(self, intrinsics_k: np.ndarray) -> Dict[int, LaserPlaneCalibration]:
        if self.session is None:
            raise RuntimeError("laser calibration not started")
        s = self.session
        out: Dict[int, LaserPlaneCalibration] = {}
        for laser in (1, 2):
            xyz, sample_count = self._collect_xyz_for_laser(laser, intrinsics_k)
            if xyz.shape[0] < int(s.min_points_per_laser):
                raise RuntimeError(
                    f"laser {laser} not enough points ({xyz.shape[0]} < {s.min_points_per_laser})"
                )
            plane, metrics = plane_fit_svd(xyz)
            out[laser] = LaserPlaneCalibration(
                laser=laser,
                plane=plane.astype(np.float64).tolist(),
                rmse=float(metrics["rmse"]),
                mean_abs_error=float(metrics["mean_abs_error"]),
                max_abs_error=float(metrics["max_abs_error"]),
                points_used=int(xyz.shape[0]),
                samples_used=int(sample_count),
                metadata={"board_plane_assumption": list(s.board_plane)},
            )
        self.last_result = out
        return out

