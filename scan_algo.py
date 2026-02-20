from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from .camera_service import CameraService
    from .hardware_io import laser_set
except ImportError:
    from camera_service import CameraService
    from hardware_io import laser_set


@dataclass
class StripeParams:
    threshold: Optional[int] = None
    blur_ksize: int = 3
    morph_ksize: int = 3
    window_half_width_px: int = 25


class StripeDetector:
    def __init__(self, params: StripeParams = StripeParams()):
        self.p = params

    def _laser_score(self, ambient_rgb: np.ndarray, laser_rgb: np.ndarray) -> np.ndarray:
        a = ambient_rgb.astype(np.int16)
        l = laser_rgb.astype(np.int16)
        diff = l - a
        score = diff[:, :, 0] - ((diff[:, :, 1] + diff[:, :, 2]) // 2)
        return np.clip(score, 0, 255).astype(np.uint8)

    def detect(
        self,
        ambient_rgb: np.ndarray,
        laser_rgb: np.ndarray,
        object_mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[Tuple[float, int]], np.ndarray]:
        score = self._laser_score(ambient_rgb, laser_rgb)

        k = int(self.p.blur_ksize)
        if k > 1:
            if k % 2 == 0:
                k += 1
            score = cv2.GaussianBlur(score, (k, k), 0)

        if self.p.threshold is None:
            _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(score, int(self.p.threshold), 255, cv2.THRESH_BINARY)

        mk = int(self.p.morph_ksize)
        if mk > 1:
            if mk % 2 == 0:
                mk += 1
            kernel = np.ones((mk, mk), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if object_mask is not None:
            mask = cv2.bitwise_and(mask, object_mask.astype(np.uint8))

        h, w = mask.shape
        points: List[Tuple[float, int]] = []

        for y in range(h):
            xs = np.where(mask[y] > 0)[0]
            if xs.size == 0:
                continue

            row_scores = score[y, xs].astype(np.float32)
            peak_idx = int(np.argmax(row_scores))
            peak_x = int(xs[peak_idx])

            lo = max(0, peak_x - int(self.p.window_half_width_px))
            hi = min(w - 1, peak_x + int(self.p.window_half_width_px))
            xs2 = xs[(xs >= lo) & (xs <= hi)]
            if xs2.size == 0:
                continue

            weights = score[y, xs2].astype(np.float32)
            s = float(weights.sum())
            x_sub = float(xs2.mean()) if s <= 1e-6 else float((xs2.astype(np.float32) * weights).sum() / s)
            points.append((x_sub, y))

        return points, mask

    def overlay_jpeg(
        self,
        rgb: np.ndarray,
        points: List[Tuple[float, int]],
        quality: int = 80,
        object_mask: Optional[np.ndarray] = None,
    ) -> bytes:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if object_mask is not None:
            m = object_mask.astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(bgr, contours, -1, (255, 0, 0), 2)

        for x, y in points:
            cv2.circle(bgr, (int(round(x)), int(y)), 1, (0, 255, 0), -1)

        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else b""


def _drop_frames(camera: CameraService, n: int):
    for _ in range(max(0, int(n))):
        camera.grab_fresh_frame(settle_s=0.0)


def capture_pair(camera: CameraService, gpio, laser, settle_s: float = 0.06, drop_n: int = 2):
    laser_set(gpio, laser, False)
    ambient = camera.grab_fresh_frame(settle_s=settle_s)
    _drop_frames(camera, drop_n)

    laser_set(gpio, laser, True)
    laser_frame = camera.grab_fresh_frame(settle_s=settle_s)
    _drop_frames(camera, drop_n)
    laser_set(gpio, laser, False)

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
