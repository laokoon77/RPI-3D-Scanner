from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from .camera_service import CameraService
    from .hardware_io import laser_set
    from .laser_detector_core import (
        DetectionProfile,
        DetectionResult,
        LaserDetectorCore,
        overlay_detection_jpeg,
    )
except ImportError:
    from camera_service import CameraService
    from hardware_io import laser_set
    from laser_detector_core import (
        DetectionProfile,
        DetectionResult,
        LaserDetectorCore,
        overlay_detection_jpeg,
    )


@dataclass
class StripeParams:
    threshold: Optional[int] = None
    blur_ksize: int = 3
    morph_ksize: int = 3
    window_half_width_px: int = 25
    min_rows: int = 24
    max_row_jump_px: float = 28.0
    max_gap_rows: int = 8
    adaptive_floor: int = 10
    adaptive_scale: float = 0.45


class StripeDetector:
    def __init__(self, params: StripeParams = StripeParams()):
        self.p = params
        self._core = LaserDetectorCore()
        self._scan_profile_name = "scan"
        self._cal_profile_name = "calibration"
        self._rebuild_profiles()

    def _rebuild_profiles(self) -> None:
        scan_prof = DetectionProfile(
            name="scan",
            threshold=self.p.threshold,
            adaptive_floor=int(self.p.adaptive_floor),
            adaptive_scale=float(self.p.adaptive_scale),
            blur_ksize=int(self.p.blur_ksize),
            morph_open_ksize=int(self.p.morph_ksize),
            morph_close_ksize=int(self.p.morph_ksize),
            window_half_width_px=int(self.p.window_half_width_px),
            min_rows=int(self.p.min_rows),
            max_row_jump_px=float(self.p.max_row_jump_px),
            max_gap_rows=int(self.p.max_gap_rows),
            roi_top_frac=0.02,
            roi_bottom_frac=0.98,
            roi_left_frac=0.02,
            roi_right_frac=0.98,
        )
        cal_prof = DetectionProfile(
            name="calibration",
            threshold=self.p.threshold,
            adaptive_floor=max(1, int(self.p.adaptive_floor) - 2),
            adaptive_scale=max(0.1, float(self.p.adaptive_scale) - 0.1),
            blur_ksize=int(self.p.blur_ksize),
            morph_open_ksize=int(self.p.morph_ksize),
            morph_close_ksize=int(self.p.morph_ksize),
            window_half_width_px=max(16, int(self.p.window_half_width_px)),
            min_rows=max(10, int(self.p.min_rows) - 8),
            max_row_jump_px=float(self.p.max_row_jump_px) + 8.0,
            max_gap_rows=int(self.p.max_gap_rows) + 4,
            roi_top_frac=0.0,
            roi_bottom_frac=1.0,
            roi_left_frac=0.0,
            roi_right_frac=1.0,
        )
        self._core.set_profile(self._scan_profile_name, scan_prof)
        self._core.set_profile(self._cal_profile_name, cal_prof)

    def _profile_for_mode(self, mode: str) -> str:
        if str(mode).strip().lower() in {"cal", "calibration"}:
            return self._cal_profile_name
        return self._scan_profile_name

    def detect(
        self,
        ambient_rgb: np.ndarray,
        laser_rgb: np.ndarray,
        object_mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[Tuple[float, int]], np.ndarray]:
        result = self.detect_details(ambient_rgb, laser_rgb, object_mask=object_mask)
        return result.points, result.mask

    def detect_details(
        self,
        ambient_rgb: np.ndarray,
        laser_rgb: np.ndarray,
        *,
        object_mask: Optional[np.ndarray] = None,
        roi_mask: Optional[np.ndarray] = None,
        mode: str = "scan",
    ) -> DetectionResult:
        self._rebuild_profiles()
        return self._core.detect(
            ambient_rgb,
            laser_rgb,
            profile_name=self._profile_for_mode(mode),
            object_mask=object_mask,
            roi_mask=roi_mask,
        )

    def overlay_jpeg(
        self,
        rgb: np.ndarray,
        points: List[Tuple[float, int]],
        quality: int = 80,
        object_mask: Optional[np.ndarray] = None,
    ) -> bytes:
        det = DetectionResult(points=list(points), mask=np.zeros(rgb.shape[:2], dtype=np.uint8), score=np.zeros(rgb.shape[:2], dtype=np.uint8), confidence=0.0, telemetry={})
        return overlay_detection_jpeg(rgb, det, quality=quality, object_mask=object_mask)

    def overlay_result_jpeg(
        self,
        rgb: np.ndarray,
        result: DetectionResult,
        quality: int = 80,
        object_mask: Optional[np.ndarray] = None,
        roi_mask: Optional[np.ndarray] = None,
        title: Optional[str] = None,
    ) -> bytes:
        return overlay_detection_jpeg(
            rgb,
            result,
            quality=quality,
            object_mask=object_mask,
            roi_mask=roi_mask,
            title=title,
        )


def _drop_frames(camera: CameraService, n: int):
    for _ in range(max(0, int(n))):
        camera.grab_fresh_frame(settle_s=0.0)


def _capture_with_retry(
    camera: CameraService,
    *,
    settle_s: float,
    retries: int,
    validator: Optional[Callable[[np.ndarray], bool]] = None,
) -> Optional[np.ndarray]:
    tries = max(1, int(retries) + 1)
    last: Optional[np.ndarray] = None
    for _ in range(tries):
        if hasattr(camera, "grab_stabilized_frame"):
            frame = camera.grab_stabilized_frame(settle_s=settle_s, retries=1, min_luma_delta=0.35)
        else:
            frame = camera.grab_fresh_frame(settle_s=settle_s)
        if frame is None:
            continue
        last = frame
        if validator is None or bool(validator(frame)):
            return frame
    return last


def capture_pair(
    camera: CameraService,
    gpio,
    laser,
    settle_s: float = 0.06,
    drop_n: int = 2,
    off_controls: Optional[Dict[str, Any]] = None,
    on_controls: Optional[Dict[str, Any]] = None,
    retry_n: int = 1,
):
    if off_controls:
        camera.set_controls(off_controls)
    laser_set(gpio, laser, False)
    ambient = _capture_with_retry(camera, settle_s=settle_s, retries=retry_n)
    _drop_frames(camera, drop_n)

    if on_controls:
        camera.set_controls(on_controls)
    laser_set(gpio, laser, True)
    laser_frame = _capture_with_retry(camera, settle_s=settle_s, retries=retry_n)
    _drop_frames(camera, drop_n)
    laser_set(gpio, laser, False)

    if off_controls:
        camera.set_controls(off_controls)

    return ambient, laser_frame


def capture_triplet(camera: CameraService, gpio, laser1, laser2, settle_s: float = 0.06, drop_n: int = 2):
    laser_set(gpio, laser1, False)
    laser_set(gpio, laser2, False)
    ambient = camera.grab_fresh_frame(settle_s=settle_s)
    _drop_frames(camera, drop_n)

    laser_set(gpio, laser1, True)
    l1 = camera.grab_fresh_frame(settle_s=settle_s)
    _drop_frames(camera, drop_n)
    laser_set(gpio, laser1, False)

    laser_set(gpio, laser2, True)
    l2 = camera.grab_fresh_frame(settle_s=settle_s)
    _drop_frames(camera, drop_n)
    laser_set(gpio, laser2, False)

    return ambient, l1, l2


def jpeg_with_text(rgb: np.ndarray, text: str, quality: int = 80) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else b""
