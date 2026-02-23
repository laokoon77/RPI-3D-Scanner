from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from .calibration_models import IntrinsicsCalibration
except ImportError:
    from calibration_models import IntrinsicsCalibration


log = logging.getLogger(__name__)

_CHECKERBOARD_DEBUG_DIR = Path("calibration") / "debug"


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
    if not hasattr(cv2, "aruco"):
        return False
    a = cv2.aruco
    has_board = hasattr(a, "CharucoBoard")
    has_detect = hasattr(a, "detectMarkers") or hasattr(a, "ArucoDetector")
    has_charuco_detect = hasattr(a, "interpolateCornersCharuco") or hasattr(a, "CharucoDetector")
    has_charuco_solve = hasattr(a, "calibrateCameraCharuco") or hasattr(cv2, "calibrateCamera")
    return bool(has_board and has_detect and has_charuco_detect and has_charuco_solve)


def charuco_runtime_diagnostics() -> Dict[str, Any]:
    a = getattr(cv2, "aruco", None)
    return {
        "opencv_version": str(getattr(cv2, "__version__", "unknown")),
        "has_aruco": bool(a is not None),
        "has_charuco_board": bool(a is not None and hasattr(a, "CharucoBoard")),
        "has_detectMarkers": bool(a is not None and hasattr(a, "detectMarkers")),
        "has_ArucoDetector": bool(a is not None and hasattr(a, "ArucoDetector")),
        "has_interpolateCornersCharuco": bool(a is not None and hasattr(a, "interpolateCornersCharuco")),
        "has_CharucoDetector": bool(a is not None and hasattr(a, "CharucoDetector")),
        "has_calibrateCameraCharuco": bool(a is not None and hasattr(a, "calibrateCameraCharuco")),
    }


_CANDIDATE_ARUCO_DICTS = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_7X7_50",
    "DICT_7X7_100",
]

_CANDIDATE_CHARUCO_DIMS = [
    (8, 8),
    (7, 5),
    (5, 7),
    (8, 6),
    (6, 8),
    (11, 8),
    (8, 11),
]


def _get_aruco_dictionary(dict_name: str):
    dict_id = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_4X4_1000)
    return cv2.aruco.getPredefinedDictionary(dict_id)


def _apply_charuco_legacy_pattern(board, enabled: bool = True) -> None:
    """Enable OpenCV legacy ChArUco pattern compatibility when supported."""
    if board is None:
        return
    if not enabled:
        return
    try:
        if hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(True)
    except Exception:
        pass


def _make_lenient_detector_params():
    """Build lenient ArucoDetector parameters for real-world camera images."""
    a = cv2.aruco
    if not hasattr(a, "DetectorParameters"):
        return None
    p = a.DetectorParameters()
    try:
        # Accept smaller markers (default 0.03 is too strict at moderate distances)
        p.minMarkerPerimeterRate = 0.01
        p.maxMarkerPerimeterRate = 10.0
        # Larger adaptive threshold window range handles varying lighting
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 53
        p.adaptiveThreshWinSizeStep = 10
        # Relax corner quality requirements
        p.minCornerDistanceRate = 0.01
        p.minOtsuStdDev = 3.0
        # Sub-pixel corner refinement
        p.cornerRefinementMethod = 1  # CORNER_REFINE_SUBPIX
        p.cornerRefinementWinSize = 5
        p.cornerRefinementMaxIterations = 30
        p.cornerRefinementMinAccuracy = 0.01
    except Exception:
        pass
    return p


def _detect_aruco_markers(gray: np.ndarray, dictionary):
    a = cv2.aruco
    lenient = _make_lenient_detector_params()

    variants = [gray]
    try:
        variants.append(cv2.equalizeHist(gray))
    except Exception:
        pass
    try:
        variants.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5))
    except Exception:
        pass

    best = ([], None, [])
    for img in variants:
        if hasattr(a, "detectMarkers"):
            corners, ids, rej = a.detectMarkers(img, dictionary)
        elif hasattr(a, "ArucoDetector"):
            # Try lenient first, then default params
            det_lenient = a.ArucoDetector(dictionary, lenient) if lenient is not None else a.ArucoDetector(dictionary)
            corners, ids, rej = det_lenient.detectMarkers(img)
        else:
            raise RuntimeError("OpenCV aruco marker detector not available")

        n = 0 if ids is None else int(len(ids))
        b = 0 if best[1] is None else int(len(best[1]))
        if n > b:
            best = (corners, ids, rej)
        if n >= 6:
            return corners, ids, rej
    corners, ids, rej = best
    return corners, ids, rej


def _charuco_detect_board(gray: np.ndarray, board) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Try CharucoDetector.detectBoard with lenient params, then legacy interpolateCornersCharuco."""
    a = cv2.aruco
    ch_corners: Optional[np.ndarray] = None
    ch_ids: Optional[np.ndarray] = None

    # Primary: new API CharucoDetector with lenient params (OpenCV 4.7+)
    if hasattr(a, "CharucoDetector"):
        for img_variant in [gray, cv2.equalizeHist(gray) if gray is not None else gray]:
            try:
                det_params = _make_lenient_detector_params()
                # Try to set CharucoParameters.minMarkers = 0 if available
                charuco_params = None
                if hasattr(a, "CharucoParameters"):
                    try:
                        charuco_params = a.CharucoParameters()
                        charuco_params.minMarkers = 0  # accept corner with any # of surrounding markers
                        try:
                            charuco_params.tryRefineMarkers = True
                        except Exception:
                            pass
                    except Exception:
                        charuco_params = None

                if charuco_params is not None and det_params is not None:
                    det = a.CharucoDetector(board, charuco_params, det_params)
                elif det_params is not None:
                    det = a.CharucoDetector(board, a.CharucoParameters() if hasattr(a, "CharucoParameters") else None, det_params) if hasattr(a, "CharucoParameters") else a.CharucoDetector(board)
                else:
                    det = a.CharucoDetector(board)

                result = det.detectBoard(img_variant)
                if result is not None:
                    if len(result) == 4:
                        cc, ci, _mc, _mi = result
                    elif len(result) == 2:
                        cc, ci = result
                    else:
                        cc, ci = None, None
                    n = 0 if ci is None else int(len(ci))
                    cur = 0 if ch_ids is None else int(len(ch_ids))
                    if n > cur:
                        ch_corners, ch_ids = cc, ci
                    if n >= 4:
                        return ch_corners, ch_ids
            except Exception:
                pass

    # Fallback: legacy interpolateCornersCharuco
    if hasattr(a, "interpolateCornersCharuco"):
        try:
            # We need marker corners/ids for this — detect them first
            det_params = _make_lenient_detector_params()
            if hasattr(a, "ArucoDetector"):
                det = a.ArucoDetector(board.getDictionary() if hasattr(board, "getDictionary") else cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000), det_params) if det_params is not None else a.ArucoDetector(board.getDictionary() if hasattr(board, "getDictionary") else cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000))
                mc, mi, _ = det.detectMarkers(gray)
            elif hasattr(a, "detectMarkers"):
                mc, mi, _ = a.detectMarkers(gray, board.getDictionary() if hasattr(board, "getDictionary") else cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000))
            else:
                mc, mi = None, None
            if mi is not None and len(mi) >= 1:
                n, cc, ci = a.interpolateCornersCharuco(mc, mi, gray, board)
                if n is not None and int(n) > 0 and cc is not None and ci is not None:
                    cur = 0 if ch_ids is None else int(len(ch_ids))
                    if int(len(ci)) > cur:
                        ch_corners, ch_ids = cc, ci
        except Exception:
            pass

    return ch_corners, ch_ids


def _frame_diagnostics(gray: np.ndarray) -> Dict[str, Any]:
    lap_var = 0.0
    try:
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        lap_var = 0.0
    try:
        p10, p50, p90 = np.percentile(gray, [10, 50, 90])
    except Exception:
        p10, p50, p90 = 0.0, 0.0, 0.0
    return {
        "shape": [int(gray.shape[1]), int(gray.shape[0])],
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "min": int(np.min(gray)),
        "max": int(np.max(gray)),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "laplacian_var": lap_var,
    }


def _array_basic_stats(img: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(img)
    return {
        "shape": [int(x) for x in arr.shape],
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
        "mean": float(np.mean(arr)) if arr.size else 0.0,
    }


def _to_gray_for_detection(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        raise ValueError("frame is None")
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] >= 3:
        # CameraService provides RGB888 frames; keep this conversion consistent.
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"unsupported frame shape for grayscale conversion: {tuple(frame.shape)}")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _save_checkerboard_debug_frames(rgb: np.ndarray, gray: np.ndarray, frame_index: int) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    try:
        _CHECKERBOARD_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
        rgb_path = _CHECKERBOARD_DEBUG_DIR / "checkerboard_capture_last_rgb.png"
        gray_path = _CHECKERBOARD_DEBUG_DIR / "checkerboard_capture_last_gray.png"
        idx_path = _CHECKERBOARD_DEBUG_DIR / f"checkerboard_capture_{int(frame_index):04d}.png"
        cv2.imwrite(str(rgb_path), rgb_bgr)
        cv2.imwrite(str(gray_path), gray)
        cv2.imwrite(str(idx_path), gray)
        paths = {
            "last_rgb": str(rgb_path),
            "last_gray": str(gray_path),
            "indexed_gray": str(idx_path),
        }
    except Exception:
        log.exception("checkerboard.capture.debug_save_failed")
    return paths


@dataclass
class IntrinsicsSession:
    board_type: str = "checkerboard"  # checkerboard|charuco
    checkerboard_cols: int = 9
    checkerboard_rows: int = 6
    square_size_m: float = 0.01
    min_frames: int = 12
    charuco_squares_x: int = 8
    charuco_squares_y: int = 8
    charuco_square_length_m: float = 0.015
    charuco_marker_length_m: float = 0.011
    aruco_dict_name: str = "DICT_4X4_1000"

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
            "checkerboard_cols": int(s.checkerboard_cols),
            "checkerboard_rows": int(s.checkerboard_rows),
            "frames_captured": s.frames_captured,
            "frames_used": s.frames_used,
            "min_frames": s.min_frames,
            "image_size": list(s.image_size) if s.image_size else None,
            "has_result": self.last_result is not None,
        }

    def _capture_checkerboard(self, rgb: np.ndarray, s: IntrinsicsSession, capture_meta: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        gray = _to_gray_for_detection(rgb)
        rgb_stats = _array_basic_stats(rgb)
        gray_stats = _array_basic_stats(gray)
        diag = _frame_diagnostics(gray)
        debug_paths = _save_checkerboard_debug_frames(rgb, gray, int(s.frames_captured))
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        configured = (int(s.checkerboard_cols), int(s.checkerboard_rows))
        transform_path = "camera.grab_fresh_frame -> CameraService._apply_frame_transform -> intrinsics.capture"
        if isinstance(capture_meta, dict) and capture_meta:
            transform_path = str(capture_meta.get("transform_path", transform_path))
        log.info(
            "checkerboard.capture.begin configured=%sx%s frames_captured=%d frames_used=%d transform_path=%s rgb_shape=%s rgb_dtype=%s rgb_min=%.1f rgb_max=%.1f rgb_mean=%.2f gray_shape=%s gray_dtype=%s gray_min=%.1f gray_max=%.1f gray_mean=%.2f p10=%.1f p50=%.1f p90=%.1f lap_var=%.2f debug_paths=%s",
            int(configured[0]),
            int(configured[1]),
            int(s.frames_captured),
            int(s.frames_used),
            transform_path,
            rgb_stats["shape"],
            rgb_stats["dtype"],
            float(rgb_stats["min"]),
            float(rgb_stats["max"]),
            float(rgb_stats["mean"]),
            gray_stats["shape"],
            gray_stats["dtype"],
            float(gray_stats["min"]),
            float(gray_stats["max"]),
            float(gray_stats["mean"]),
            float(diag["p10"]),
            float(diag["p50"]),
            float(diag["p90"]),
            float(diag["laplacian_var"]),
            debug_paths,
        )

        # Try configured pattern first, then an inner-corner interpretation fallback.
        # Example: physical 8x8 squares => 7x7 inner corners for OpenCV.
        attempted_patterns: list[tuple[int, int]] = [configured]
        if configured[0] > 2 and configured[1] > 2:
            fallback = (configured[0] - 1, configured[1] - 1)
            if fallback not in attempted_patterns:
                attempted_patterns.append(fallback)

        found = False
        corners = None
        used_pattern = configured
        used_detector = "none"
        attempt_details: list[dict[str, Any]] = []

        for pattern in attempted_patterns:
            # 1) classic findChessboardCorners with adaptive+normalize flags
            found_std, corners_std = cv2.findChessboardCorners(gray, pattern, flags=flags)
            std_corners = int(len(corners_std)) if corners_std is not None else 0
            found_sb = False
            corners_sb = None
            # 2) run SB detector on the same exact grayscale frame for diagnostics/comparison
            if hasattr(cv2, "findChessboardCornersSB"):
                try:
                    found_sb, corners_sb = cv2.findChessboardCornersSB(gray, pattern)
                except Exception:
                    found_sb, corners_sb = False, None
            sb_corners = int(len(corners_sb)) if corners_sb is not None else 0

            if found_std and corners_std is not None:
                found, corners = True, corners_std
                used_pattern = pattern
                used_detector = "classic"
            elif found_sb and corners_sb is not None:
                found, corners = True, corners_sb
                used_pattern = pattern
                used_detector = "sb"
            attempt_details.append(
                {
                    "pattern": [int(pattern[0]), int(pattern[1])],
                    "classic_found": bool(found_std),
                    "classic_corners": std_corners,
                    "sb_found": bool(found_sb),
                    "sb_corners": sb_corners,
                }
            )
            log.info(
                "checkerboard.capture.try pattern=%sx%s classic_found=%s classic_corners=%d sb_found=%s sb_corners=%d",
                int(pattern[0]),
                int(pattern[1]),
                bool(found_std),
                int(std_corners),
                bool(found_sb),
                int(sb_corners),
            )
            if found:
                break

        if not found or corners is None:
            log.warning(
                "checkerboard.capture.reject frame=%sx%s configured=%sx%s attempted=%s details=%s mean=%.2f std=%.2f lap_var=%.2f reason=checkerboard_not_found",
                int(gray.shape[1]),
                int(gray.shape[0]),
                int(configured[0]),
                int(configured[1]),
                attempted_patterns,
                attempt_details,
                float(diag["mean"]),
                float(diag["std"]),
                float(diag["laplacian_var"]),
            )
            return False, "checkerboard not found"

        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

        if used_pattern != configured:
            log.warning(
                "checkerboard.capture.auto_adjust configured=%sx%s detected=%sx%s detector=%s",
                int(configured[0]),
                int(configured[1]),
                int(used_pattern[0]),
                int(used_pattern[1]),
                used_detector,
            )
            # Persist detected inner-corner interpretation for next frames.
            s.checkerboard_cols = int(used_pattern[0])
            s.checkerboard_rows = int(used_pattern[1])

        log.info(
            "checkerboard.capture.accept frame=%sx%s configured=%sx%s used=%sx%s detector=%s details=%s",
            int(gray.shape[1]),
            int(gray.shape[0]),
            int(configured[0]),
            int(configured[1]),
            int(used_pattern[0]),
            int(used_pattern[1]),
            used_detector,
            attempt_details,
        )

        s.object_points.append(checkerboard_object_points(int(used_pattern[0]), int(used_pattern[1]), s.square_size_m))
        s.image_points.append(refined.reshape(-1, 2).astype(np.float32))
        s.frames_used += 1
        if used_pattern != configured:
            return True, f"checkerboard captured (auto-adjusted to inner corners {used_pattern[0]}x{used_pattern[1]})"
        return True, "checkerboard captured"

    def _capture_charuco(self, rgb: np.ndarray, s: IntrinsicsSession) -> Tuple[bool, str]:
        if not has_charuco_support():
            return False, "charuco not available"
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        diag = _frame_diagnostics(gray)
        log.info(
            "charuco.capture.begin dict=%s dims=%sx%s square=%.6f marker=%.6f frame=%sx%s mean=%.2f std=%.2f p10=%.1f p50=%.1f p90=%.1f lap_var=%.2f",
            s.aruco_dict_name,
            int(s.charuco_squares_x),
            int(s.charuco_squares_y),
            float(s.charuco_square_length_m),
            float(s.charuco_marker_length_m),
            int(diag["shape"][0]),
            int(diag["shape"][1]),
            float(diag["mean"]),
            float(diag["std"]),
            float(diag["p10"]),
            float(diag["p50"]),
            float(diag["p90"]),
            float(diag["laplacian_var"]),
        )
        dictionary = _get_aruco_dictionary(s.aruco_dict_name)
        board = cv2.aruco.CharucoBoard(
            (int(s.charuco_squares_x), int(s.charuco_squares_y)),
            float(s.charuco_square_length_m),
            float(s.charuco_marker_length_m),
            dictionary,
        )
        _apply_charuco_legacy_pattern(board, True)
        corners, ids, _rej = _detect_aruco_markers(gray, dictionary)
        marker_count = 0 if ids is None else int(len(ids))
        log.info("charuco.capture.markers dict=%s marker_count=%d", s.aruco_dict_name, marker_count)

        # If primary dict finds < 2 markers, immediately try auto-tune to discover
        # the correct ArUco dictionary (board might use DICT_5X5_50, DICT_6X6_50 etc.)
        if marker_count < 2:
            auto = self._auto_tune_charuco_from_frame(gray, s)
            log.info(
                "charuco.capture.autotune_early retuned=%s dict=%s dims=%s score=%s",
                bool(auto.get("retuned", False)),
                auto.get("dict_name"),
                auto.get("dims"),
                auto.get("score"),
            )
            if auto.get("retuned", False):
                dictionary = _get_aruco_dictionary(s.aruco_dict_name)
                board = cv2.aruco.CharucoBoard(
                    (int(s.charuco_squares_x), int(s.charuco_squares_y)),
                    float(s.charuco_square_length_m),
                    float(s.charuco_marker_length_m),
                    dictionary,
                )
                _apply_charuco_legacy_pattern(board, True)
                corners, ids, _rej = _detect_aruco_markers(gray, dictionary)
                marker_count = 0 if ids is None else int(len(ids))
                log.info("charuco.capture.markers_after_autotune dict=%s marker_count=%d", s.aruco_dict_name, marker_count)
            if marker_count < 2:
                log.warning("charuco.capture.reject reason=not_enough_aruco_markers marker_count=%d dict=%s", marker_count, s.aruco_dict_name)
                return False, "not enough aruco markers"

        ch_corners, ch_ids = _charuco_detect_board(gray, board)
        if ch_corners is None or ch_ids is None or len(ch_ids) < 4:
            auto = self._auto_tune_charuco_from_frame(gray, s)
            log.info(
                "charuco.capture.autotune retuned=%s dict=%s dims=%s score=%s",
                bool(auto.get("retuned", False)),
                auto.get("dict_name"),
                auto.get("dims"),
                auto.get("score"),
            )
            if auto.get("retuned", False):
                dictionary = _get_aruco_dictionary(s.aruco_dict_name)
                board = cv2.aruco.CharucoBoard(
                    (int(s.charuco_squares_x), int(s.charuco_squares_y)),
                    float(s.charuco_square_length_m),
                    float(s.charuco_marker_length_m),
                    dictionary,
                )
                _apply_charuco_legacy_pattern(board, True)
                corners, ids, _rej = _detect_aruco_markers(gray, dictionary)
                if ids is not None and len(ids) >= 2:
                    ch_corners, ch_ids = _charuco_detect_board(gray, board)

        if ch_corners is None or ch_ids is None or len(ch_ids) < 4:
            corner_count = 0 if ch_ids is None else int(len(ch_ids))
            log.warning(
                "charuco.capture.reject reason=not_enough_charuco_corners dict=%s dims=%sx%s marker_count=%d corner_count=%d lap_var=%.2f mean=%.2f std=%.2f",
                s.aruco_dict_name,
                int(s.charuco_squares_x),
                int(s.charuco_squares_y),
                marker_count,
                corner_count,
                float(diag["laplacian_var"]),
                float(diag["mean"]),
                float(diag["std"]),
            )
            return False, "not enough charuco corners"

        s.charuco_corners.append(ch_corners)
        s.charuco_ids.append(ch_ids)
        s.frames_used += 1
        log.info(
            "charuco.capture.accepted dict=%s dims=%sx%s marker_count=%d corner_count=%d frames_used=%d",
            s.aruco_dict_name,
            int(s.charuco_squares_x),
            int(s.charuco_squares_y),
            marker_count,
            int(len(ch_ids)),
            int(s.frames_used),
        )
        return True, "charuco captured"

    def _auto_tune_charuco_from_frame(self, gray: np.ndarray, s: IntrinsicsSession) -> Dict[str, Any]:
        best = {
            "score": -1,
            "dict_name": s.aruco_dict_name,
            "dims": (int(s.charuco_squares_x), int(s.charuco_squares_y)),
        }

        dicts = [s.aruco_dict_name] + [d for d in _CANDIDATE_ARUCO_DICTS if d != s.aruco_dict_name]
        dims_list = [(int(s.charuco_squares_x), int(s.charuco_squares_y))] + [d for d in _CANDIDATE_CHARUCO_DIMS if d != (int(s.charuco_squares_x), int(s.charuco_squares_y))]

        for dn in dicts:
            try:
                dictionary = _get_aruco_dictionary(dn)
                corners, ids, _rej = _detect_aruco_markers(gray, dictionary)
            except Exception:
                continue
            if ids is None or len(ids) < 2:
                continue

            marker_count = int(len(ids))
            for dx, dy in dims_list:
                try:
                    board = cv2.aruco.CharucoBoard(
                        (int(dx), int(dy)),
                        float(s.charuco_square_length_m),
                        float(s.charuco_marker_length_m),
                        dictionary,
                    )
                    _apply_charuco_legacy_pattern(board, True)
                    corner_count = 0
                    if hasattr(cv2.aruco, "interpolateCornersCharuco"):
                        n, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                        if n is not None and ch_corners is not None and ch_ids is not None:
                            corner_count = int(len(ch_ids))
                    if corner_count < 1 and hasattr(cv2.aruco, "CharucoDetector"):
                        try:
                            det = cv2.aruco.CharucoDetector(board)
                            ch_corners, ch_ids, _mc, _mi = det.detectBoard(gray)
                            if ch_ids is not None:
                                corner_count = int(len(ch_ids))
                        except Exception:
                            pass
                    score = corner_count * 100 + marker_count
                    if score > int(best["score"]):
                        best = {"score": score, "dict_name": dn, "dims": (int(dx), int(dy))}
                except Exception:
                    continue

        retuned = False
        if int(best["score"]) > 0:
            new_dict = str(best["dict_name"])
            new_dims = best["dims"]
            if new_dict != s.aruco_dict_name or tuple(new_dims) != (int(s.charuco_squares_x), int(s.charuco_squares_y)):
                s.aruco_dict_name = new_dict
                s.charuco_squares_x = int(new_dims[0])
                s.charuco_squares_y = int(new_dims[1])
                retuned = True
        log.info(
            "charuco.autotune.result score=%d dict=%s dims=%sx%s retuned=%s",
            int(best["score"]),
            str(s.aruco_dict_name),
            int(s.charuco_squares_x),
            int(s.charuco_squares_y),
            bool(retuned),
        )
        return {
            "retuned": retuned,
            "dict_name": s.aruco_dict_name,
            "dims": [int(s.charuco_squares_x), int(s.charuco_squares_y)],
            "score": int(best["score"]),
        }

    def capture(self, rgb: np.ndarray, capture_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            accepted, reason = self._capture_checkerboard(rgb, s, capture_meta=capture_meta)
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
            dictionary = _get_aruco_dictionary(s.aruco_dict_name)
            board = cv2.aruco.CharucoBoard(
                (int(s.charuco_squares_x), int(s.charuco_squares_y)),
                float(s.charuco_square_length_m),
                float(s.charuco_marker_length_m),
                dictionary,
            )
            _apply_charuco_legacy_pattern(board, True)
            if hasattr(cv2.aruco, "calibrateCameraCharuco"):
                rms, k, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                    s.charuco_corners,
                    s.charuco_ids,
                    board,
                    s.image_size,
                    None,
                    None,
                )
            else:
                obj_pts: List[np.ndarray] = []
                img_pts: List[np.ndarray] = []
                for cc, ci in zip(s.charuco_corners, s.charuco_ids):
                    if cc is None or ci is None:
                        continue
                    if hasattr(board, "matchImagePoints"):
                        op, ip = board.matchImagePoints(cc, ci)
                        if op is None or ip is None:
                            continue
                        op = np.asarray(op, dtype=np.float32).reshape(-1, 3)
                        ip = np.asarray(ip, dtype=np.float32).reshape(-1, 2)
                        if len(op) >= 6 and len(ip) >= 6:
                            obj_pts.append(op)
                            img_pts.append(ip)
                if not obj_pts:
                    raise RuntimeError("charuco solve fallback could not build matched points")
                rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
                    obj_pts,
                    img_pts,
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
                "legacy_pattern": True,
            }

        rms_f = float(rms)
        rms_ok = bool(rms_f <= 1.2)
        mean_ok = bool(float(mean_err) <= 1.0)
        quality = {
            "ok": bool(s.frames_used >= s.min_frames and np.isfinite(rms_f) and rms_ok and mean_ok),
            "rms_threshold": 1.2,
            "rms_ok": rms_ok,
            "mean_error_threshold_px": 1.0,
            "mean_error_ok": mean_ok,
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


GUIDED_CHARUCO_INSTRUCTION = "Move the board to a new angle, keep it fully visible, then press Capture Step."


@dataclass
class GuidedCharucoSession:
    total_steps: int = 10
    current_step: int = 1
    min_frames_required: int = 20
    running: bool = False
    solved: bool = False
    calibration_ok: bool = False
    captures_attempted: int = 0
    accepted_frames: int = 0
    last_detection_result: str = "session not started"
    last_step_ok: bool = False
    last_step_message: str = "Workflow not started yet."
    instruction: str = GUIDED_CHARUCO_INSTRUCTION
    quality_summary: Dict[str, Any] = field(default_factory=dict)


class GuidedCharucoWorkflowService:
    def __init__(self, intrinsics_service: IntrinsicsCalibrationService):
        self._intrinsics = intrinsics_service
        self.session: Optional[GuidedCharucoSession] = None
        self.last_intrinsics: Optional[IntrinsicsCalibration] = None

    def start(
        self,
        *,
        total_steps: int = 20,
        min_frames_required: int = 20,
        charuco_squares_x: int = 8,
        charuco_squares_y: int = 8,
        charuco_square_length_m: float = 0.015,
        charuco_marker_length_m: float = 0.011,
        aruco_dict_name: str = "DICT_4X4_1000",
    ) -> Dict[str, Any]:
        steps = max(1, int(total_steps))
        min_frames = max(1, min(int(min_frames_required), steps))
        self._intrinsics.start(
            board_type="charuco",
            min_frames=min_frames,
            charuco_squares_x=int(charuco_squares_x),
            charuco_squares_y=int(charuco_squares_y),
            charuco_square_length_m=float(charuco_square_length_m),
            charuco_marker_length_m=float(charuco_marker_length_m),
            aruco_dict_name=str(aruco_dict_name),
        )
        self.session = GuidedCharucoSession(
            total_steps=steps,
            current_step=1,
            min_frames_required=min_frames,
            running=True,
            solved=False,
            calibration_ok=False,
            captures_attempted=0,
            accepted_frames=0,
            last_detection_result="session started",
            last_step_ok=False,
            last_step_message="Workflow started. Hold the board fully in view and press Capture Step.",
            instruction=GUIDED_CHARUCO_INSTRUCTION,
            quality_summary={},
        )
        self.last_intrinsics = None
        return self.status()

    @staticmethod
    def _step_text(current_step: int, total_steps: int) -> str:
        step = max(0, int(current_step))
        total = max(1, int(total_steps))
        if step > total:
            step = total
        return f"Step {step} of {total}"

    @staticmethod
    def _friendly_step_message(accepted: bool, reason: str) -> str:
        if accepted:
            return "Step accepted. Move the board to a new angle and press Capture Step."
        reason_l = str(reason).strip().lower()
        if "aruco" in reason_l or "charuco" in reason_l:
            return "Step not accepted. Keep the full board visible, move a bit closer, and retry."
        if "checkerboard" in reason_l:
            return "Step not accepted. Keep the full board in frame and retry."
        if "empty frame" in reason_l:
            return "Step not accepted. Camera frame was unavailable. Please retry."
        return "Step not accepted. Move the board, keep it fully visible, and try again."

    def status(self) -> Dict[str, Any]:
        s = self.session
        if s is None:
            return {
                "running": False,
                "has_result": self.last_intrinsics is not None,
                "step_text": "Step 0 of 0",
                "last_step_ok": False,
                "last_step_message": "Workflow not started yet.",
                "instruction": GUIDED_CHARUCO_INSTRUCTION,
            }
        step_text = self._step_text(s.current_step, s.total_steps)
        return {
            "running": bool(s.running),
            "solved": bool(s.solved),
            "calibration_ok": bool(s.calibration_ok),
            "current_step": int(s.current_step),
            "total_steps": int(s.total_steps),
            "step_text": step_text,
            "captures_attempted": int(s.captures_attempted),
            "accepted_frames": int(s.accepted_frames),
            "min_frames_required": int(s.min_frames_required),
            "last_detection_result": str(s.last_detection_result),
            "last_step_ok": bool(s.last_step_ok),
            "last_step_message": str(s.last_step_message),
            "instruction": str(s.instruction),
            "quality_summary": dict(s.quality_summary),
            "has_result": self.last_intrinsics is not None,
        }

    def capture_step(self, rgb: np.ndarray) -> Dict[str, Any]:
        s = self.session
        if s is None or not s.running:
            raise RuntimeError("guided charuco workflow not running")
        if s.solved:
            s.last_step_ok = bool(s.calibration_ok)
            s.last_step_message = "Calibration already finished."
            return {
                "ok": True,
                "step_capture": False,
                "reason": "already solved",
                "status": self.status(),
            }
        if s.accepted_frames >= s.total_steps:
            s.last_step_ok = False
            s.last_step_message = "All steps are already captured. Press Finish & Check."
            return {
                "ok": False,
                "step_capture": False,
                "reason": "all steps already captured",
                "status": self.status(),
            }

        capture = self._intrinsics.capture(rgb)
        accepted = bool(capture.get("accepted", False))
        reason = str(capture.get("reason", ""))
        s.captures_attempted += 1
        s.accepted_frames = int(capture.get("frames_used", s.accepted_frames))
        s.last_detection_result = reason
        s.last_step_ok = bool(accepted)
        s.last_step_message = self._friendly_step_message(accepted, reason)

        if s.accepted_frames < s.total_steps:
            s.current_step = min(s.total_steps, s.accepted_frames + 1)
            return {
                "ok": True,
                "step_capture": True,
                "accepted": accepted,
                "reason": reason,
                "status": self.status(),
            }

        s.current_step = s.total_steps
        solved_payload = self.solve_final()
        solved_payload.update(
            {
                "step_capture": True,
                "accepted": accepted,
                "reason": reason,
            }
        )
        return solved_payload

    def solve_final(self) -> Dict[str, Any]:
        s = self.session
        if s is None:
            raise RuntimeError("guided charuco workflow not started")
        if s.solved and self.last_intrinsics is not None:
            return {
                "ok": True,
                "solved": True,
                "intrinsics": self.last_intrinsics,
                "status": self.status(),
            }

        try:
            result = self._intrinsics.solve()
            quality = dict(result.quality or {})
            min_ok = bool(int(result.frames_used) >= int(s.min_frames_required))
            rms_ok = bool(quality.get("rms_ok", False))
            mean_ok = bool(quality.get("mean_error_ok", False))
            calibration_ok = bool(min_ok and rms_ok and mean_ok)
            quality_summary = {
                "ok": calibration_ok,
                "frames_used": int(result.frames_used),
                "captures_attempted": int(s.captures_attempted),
                "min_frames_required": int(s.min_frames_required),
                "rms_reprojection_error": float(result.rms_reprojection_error),
                "rms_threshold": float(quality.get("rms_threshold", 1.2)),
                "rms_ok": rms_ok,
                "mean_reprojection_error": float(result.mean_reprojection_error),
                "mean_error_threshold_px": float(quality.get("mean_error_threshold_px", 1.0)),
                "mean_error_ok": mean_ok,
            }
            s.quality_summary = quality_summary
            s.calibration_ok = calibration_ok
            s.running = False
            s.solved = True
            s.last_step_ok = bool(calibration_ok)
            s.last_step_message = (
                "Finished. Calibration quality looks good."
                if calibration_ok
                else "Finished, but quality is not good enough yet. Capture more varied board views and try again."
            )
            self.last_intrinsics = result
            return {
                "ok": True,
                "solved": True,
                "calibration_ok": calibration_ok,
                "quality": quality_summary,
                "intrinsics": result,
                "status": self.status(),
            }
        except Exception as e:
            s.running = False
            s.solved = True
            s.calibration_ok = False
            s.last_step_ok = False
            s.last_step_message = "Could not finish calibration. Please retry with clear and varied board views."
            s.quality_summary = {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "frames_used": int(s.accepted_frames),
                "captures_attempted": int(s.captures_attempted),
                "min_frames_required": int(s.min_frames_required),
            }
            return {
                "ok": False,
                "solved": False,
                "calibration_ok": False,
                "error": f"{type(e).__name__}: {e}",
                "quality": dict(s.quality_summary),
                "status": self.status(),
            }

