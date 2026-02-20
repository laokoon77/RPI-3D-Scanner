from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .calibration_models import IntrinsicsCalibration


def checkerboard_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid
    objp *= float(square_size)
    return objp


def mean_reprojection_error(
    object_points: List[np.ndarray],
    image_points: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    if not object_points:
        return float("inf")
    err_sum = 0.0
    n = 0
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        reproj, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        reproj = reproj.reshape(-1, 2)
        img2 = img.reshape(-1, 2)
        d = np.linalg.norm(reproj - img2, axis=1)
        err_sum += float(np.sum(d))
        n += int(d.shape[0])
    if n == 0:
        return float("inf")
    return err_sum / float(n)


def has_charuco_support() -> bool:
    return bool(
        hasattr(cv2, "aruco")
        and hasattr(cv2.aruco, "CharucoBoard")
        and hasattr(cv2.aruco, "interpolateCornersCharuco")
        and hasattr(cv2.aruco, "calibrateCameraCharuco")
    )


@dataclass
class IntrinsicsSession:
    board_type: str = "checkerboard"  # checkerboard|charuco
    checkerboard_cols: int = 9
    checkerboard_rows: int = 6
    square_size_m: float = 0.01
    min_frames: int = 12
    charuco_squares_x: int = 7
    charuco_squares_y: int = 5
    charuco_square_length_m: float = 0.02
    charuco_marker_length_m: float = 0.015
    aruco_dict_name: str = "DICT_4X4_50"

    frames_captured: int = 0
    frames_used: int = 0
    image_size: Optional[Tuple[int, int]] = None

    object_points: List[np.ndarray] = field(default_factory=list)
    image_points: List[np.ndarray] = field(default_factory=list)
    charuco_corners: List[np.ndarray] = field(default_factory=list)
    charuco_ids: List[np.ndarray] = field(default_factory=list)


class IntrinsicsCalibrationService:
    def __init__(self):
        self.session: Optional[IntrinsicsSession] = None
        self.last_result: Optional[IntrinsicsCalibration] = None

    def start(self, **kwargs) -> Dict[str, Any]:
        sess = IntrinsicsSession(**kwargs)
        if sess.board_type not in ("checkerboard", "charuco"):
            raise ValueError("board_type must be checkerboard or charuco")
        if sess.board_type == "charuco" and not has_charuco_support():
            raise RuntimeError("OpenCV aruco/charuco not available in this build")
        self.session = sess
        return self.status()

    def status(self) -> Dict[str, Any]:
        if self.session is None:
            return {"running": False, "has_result": self.last_result is not None}
        s = self.session
        return {
            "running": True,
            "board_type": s.board_type,
            "frames_captured": s.frames_captured,
            "frames_used": s.frames_used,
            "min_frames": s.min_frames,
            "image_size": list(s.image_size) if s.image_size else None,
            "has_result": self.last_result is not None,
        }

    def _capture_checkerboard(self, rgb: np.ndarray, s: IntrinsicsSession) -> Tuple[bool, str]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        pattern = (int(s.checkerboard_cols), int(s.checkerboard_rows))
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if not found or corners is None:
            return False, "checkerboard not found"

        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

        s.object_points.append(checkerboard_object_points(s.checkerboard_cols, s.checkerboard_rows, s.square_size_m))
        s.image_points.append(refined.reshape(-1, 2).astype(np.float32))
        s.frames_used += 1
        return True, "checkerboard captured"

    def _capture_charuco(self, rgb: np.ndarray, s: IntrinsicsSession) -> Tuple[bool, str]:
        if not has_charuco_support():
            return False, "charuco not available"
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        dict_id = getattr(cv2.aruco, s.aruco_dict_name, cv2.aruco.DICT_4X4_50)
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        board = cv2.aruco.CharucoBoard(
            (int(s.charuco_squares_x), int(s.charuco_squares_y)),
            float(s.charuco_square_length_m),
            float(s.charuco_marker_length_m),
            dictionary,
        )
        corners, ids, _rej = cv2.aruco.detectMarkers(gray, dictionary)
        if ids is None or len(ids) < 4:
            return False, "not enough aruco markers"
        n, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if n is None or int(n) < 6 or ch_corners is None or ch_ids is None:
            return False, "not enough charuco corners"
        s.charuco_corners.append(ch_corners)
        s.charuco_ids.append(ch_ids)
        s.frames_used += 1
        return True, "charuco captured"

    def capture(self, rgb: np.ndarray) -> Dict[str, Any]:
        if self.session is None:
            raise RuntimeError("intrinsics calibration not started")
        if rgb is None:
            return {"ok": False, "accepted": False, "reason": "empty frame"}

        s = self.session
        h, w = rgb.shape[:2]
        s.frames_captured += 1
        if s.image_size is None:
            s.image_size = (int(w), int(h))

        if s.board_type == "checkerboard":
            accepted, reason = self._capture_checkerboard(rgb, s)
        else:
            accepted, reason = self._capture_charuco(rgb, s)

        return {
            "ok": True,
            "accepted": bool(accepted),
            "reason": reason,
            "frames_captured": s.frames_captured,
            "frames_used": s.frames_used,
            "min_frames": s.min_frames,
        }

    def solve(self) -> IntrinsicsCalibration:
        if self.session is None:
            raise RuntimeError("intrinsics calibration not started")
        s = self.session
        if s.image_size is None:
            raise RuntimeError("no image size available")
        if s.frames_used < int(s.min_frames):
            raise RuntimeError(f"not enough valid frames ({s.frames_used} < {s.min_frames})")

        if s.board_type == "checkerboard":
            rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
                s.object_points,
                s.image_points,
                s.image_size,
                None,
                None,
            )
            mean_err = mean_reprojection_error(s.object_points, s.image_points, rvecs, tvecs, k, dist)
            board_desc: Dict[str, Any] = {
                "type": "checkerboard",
                "cols": int(s.checkerboard_cols),
                "rows": int(s.checkerboard_rows),
                "square_size_m": float(s.square_size_m),
            }
        else:
            dict_id = getattr(cv2.aruco, s.aruco_dict_name, cv2.aruco.DICT_4X4_50)
            dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            board = cv2.aruco.CharucoBoard(
                (int(s.charuco_squares_x), int(s.charuco_squares_y)),
                float(s.charuco_square_length_m),
                float(s.charuco_marker_length_m),
                dictionary,
            )
            rms, k, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                s.charuco_corners,
                s.charuco_ids,
                board,
                s.image_size,
                None,
                None,
            )
            mean_err = float(rms)
            board_desc = {
                "type": "charuco",
                "squares_x": int(s.charuco_squares_x),
                "squares_y": int(s.charuco_squares_y),
                "square_length_m": float(s.charuco_square_length_m),
                "marker_length_m": float(s.charuco_marker_length_m),
                "aruco_dict": s.aruco_dict_name,
            }

        rms_f = float(rms)
        quality = {
            "ok": bool(s.frames_used >= s.min_frames and np.isfinite(rms_f)),
            "rms_threshold": 1.2,
            "rms_ok": bool(rms_f <= 1.2),
            "mean_error_threshold_px": 1.0,
            "mean_error_ok": bool(float(mean_err) <= 1.0),
            "min_frames_required": int(s.min_frames),
        }

        result = IntrinsicsCalibration(
            method=s.board_type,
            image_size=[int(s.image_size[0]), int(s.image_size[1])],
            board=board_desc,
            camera_matrix=np.asarray(k, dtype=np.float64).tolist(),
            dist_coeffs=np.asarray(dist, dtype=np.float64).reshape(-1).tolist(),
            rms_reprojection_error=rms_f,
            mean_reprojection_error=float(mean_err),
            frames_used=int(s.frames_used),
            frames_captured=int(s.frames_captured),
            quality=quality,
        )
        self.last_result = result
        return result

