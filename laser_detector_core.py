from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectionProfile:
    name: str
    threshold: Optional[int] = None
    adaptive_floor: int = 10
    adaptive_scale: float = 0.45
    blur_ksize: int = 3
    morph_open_ksize: int = 3
    morph_close_ksize: int = 3
    min_component_area: int = 8
    max_component_area: int = 30000
    min_component_height: int = 2
    window_half_width_px: int = 25
    min_rows: int = 24
    max_row_jump_px: float = 28.0
    max_gap_rows: int = 8
    roi_top_frac: float = 0.0
    roi_bottom_frac: float = 1.0
    roi_left_frac: float = 0.0
    roi_right_frac: float = 1.0
    channel_mode: str = "r_minus_gb"


@dataclass
class DetectionResult:
    points: List[Tuple[float, int]]
    mask: np.ndarray
    score: np.ndarray
    confidence: float
    telemetry: Dict[str, Any]


def _odd(v: int) -> int:
    i = max(1, int(v))
    return i if i % 2 == 1 else i + 1


def _build_roi_mask(
    shape: Tuple[int, int],
    profile: DetectionProfile,
    roi_mask: Optional[np.ndarray],
) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    y0 = max(0, min(h, int(round(float(profile.roi_top_frac) * h))))
    y1 = max(0, min(h, int(round(float(profile.roi_bottom_frac) * h))))
    x0 = max(0, min(w, int(round(float(profile.roi_left_frac) * w))))
    x1 = max(0, min(w, int(round(float(profile.roi_right_frac) * w))))

    if y1 <= y0:
        y0, y1 = 0, h
    if x1 <= x0:
        x0, x1 = 0, w

    out = np.zeros((h, w), dtype=np.uint8)
    out[y0:y1, x0:x1] = 255

    if roi_mask is not None:
        out = cv2.bitwise_and(out, roi_mask.astype(np.uint8))
    return out


def default_detection_profiles() -> Dict[str, DetectionProfile]:
    return {
        "scan": DetectionProfile(
            name="scan",
            adaptive_floor=10,
            adaptive_scale=0.45,
            blur_ksize=3,
            morph_open_ksize=3,
            morph_close_ksize=3,
            min_component_area=10,
            min_rows=24,
            max_row_jump_px=28.0,
            max_gap_rows=8,
            window_half_width_px=25,
            roi_top_frac=0.02,
            roi_bottom_frac=0.98,
            roi_left_frac=0.02,
            roi_right_frac=0.98,
        ),
        "calibration": DetectionProfile(
            name="calibration",
            adaptive_floor=8,
            adaptive_scale=0.35,
            blur_ksize=3,
            morph_open_ksize=3,
            morph_close_ksize=3,
            min_component_area=6,
            min_rows=18,
            max_row_jump_px=36.0,
            max_gap_rows=12,
            window_half_width_px=32,
            roi_top_frac=0.0,
            roi_bottom_frac=1.0,
            roi_left_frac=0.0,
            roi_right_frac=1.0,
        ),
    }


class LaserDetectorCore:
    def __init__(self, profiles: Optional[Dict[str, DetectionProfile]] = None):
        p = profiles or default_detection_profiles()
        self._profiles: Dict[str, DetectionProfile] = dict(p)

    def set_profile(self, name: str, profile: DetectionProfile) -> None:
        self._profiles[str(name)] = profile

    def get_profile(self, name: str) -> DetectionProfile:
        if name not in self._profiles:
            raise ValueError(f"unknown profile: {name}")
        return self._profiles[name]

    def _laser_score(self, off_rgb: np.ndarray, on_rgb: np.ndarray, profile: DetectionProfile) -> np.ndarray:
        off_i = off_rgb.astype(np.int16)
        on_i = on_rgb.astype(np.int16)
        diff = on_i - off_i

        mode = str(profile.channel_mode)
        if mode == "r_minus_max_gb":
            score = diff[:, :, 0] - np.maximum(diff[:, :, 1], diff[:, :, 2])
        elif mode == "r_only":
            score = diff[:, :, 0]
        else:
            score = diff[:, :, 0] - ((diff[:, :, 1] + diff[:, :, 2]) // 2)

        return np.clip(score, 0, 255).astype(np.uint8)

    def detect(
        self,
        off_rgb: np.ndarray,
        on_rgb: np.ndarray,
        *,
        profile_name: str = "scan",
        object_mask: Optional[np.ndarray] = None,
        roi_mask: Optional[np.ndarray] = None,
    ) -> DetectionResult:
        profile = self.get_profile(profile_name)
        score = self._laser_score(off_rgb, on_rgb, profile)

        blur_k = _odd(profile.blur_ksize)
        if blur_k > 1:
            score = cv2.GaussianBlur(score, (blur_k, blur_k), 0)

        positive = score[score > 0]
        if profile.threshold is None:
            if positive.size > 0:
                p50 = float(np.percentile(positive, 50))
                p90 = float(np.percentile(positive, 90))
                threshold = int(round(max(profile.adaptive_floor, p50 + profile.adaptive_scale * (p90 - p50))))
            else:
                threshold = int(profile.adaptive_floor)
        else:
            threshold = int(profile.threshold)

        _t, mask = cv2.threshold(score, int(max(0, min(255, threshold))), 255, cv2.THRESH_BINARY)

        open_k = _odd(profile.morph_open_ksize)
        if open_k > 1:
            kernel = np.ones((open_k, open_k), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        close_k = _odd(profile.morph_close_ksize)
        if close_k > 1:
            kernel = np.ones((close_k, close_k), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        roi = _build_roi_mask(mask.shape, profile, roi_mask=roi_mask)
        mask = cv2.bitwise_and(mask, roi)

        if object_mask is not None:
            mask = cv2.bitwise_and(mask, object_mask.astype(np.uint8))

        kept_mask = np.zeros_like(mask)
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        kept_components = 0
        for label in range(1, int(num_labels)):
            area = int(stats[label, cv2.CC_STAT_AREA])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < int(profile.min_component_area):
                continue
            if area > int(profile.max_component_area):
                continue
            if h < int(profile.min_component_height):
                continue
            kept_mask[labels == label] = 255
            kept_components += 1

        h, w = kept_mask.shape
        points: List[Tuple[float, int]] = []
        prev_x: Optional[float] = None
        continuity_hits = 0
        gap_rows = 0
        considered_rows = 0

        for y in range(h):
            xs = np.where(kept_mask[y] > 0)[0]
            if xs.size == 0:
                gap_rows += 1
                if gap_rows > int(profile.max_gap_rows):
                    prev_x = None
                continue

            considered_rows += 1
            row_scores = score[y, xs].astype(np.float32)
            peak_idx = int(np.argmax(row_scores))
            peak_x = int(xs[peak_idx])

            lo = max(0, peak_x - int(profile.window_half_width_px))
            hi = min(w - 1, peak_x + int(profile.window_half_width_px))
            xs2 = xs[(xs >= lo) & (xs <= hi)]
            if xs2.size == 0:
                continue

            weights = score[y, xs2].astype(np.float32)
            sw = float(weights.sum())
            x_sub = float(xs2.mean()) if sw <= 1e-6 else float((xs2.astype(np.float32) * weights).sum() / sw)

            if prev_x is not None:
                if abs(x_sub - prev_x) > float(profile.max_row_jump_px):
                    continue
                continuity_hits += 1

            points.append((x_sub, y))
            prev_x = x_sub
            gap_rows = 0

        mean_score = float(score[kept_mask > 0].mean()) if int(np.count_nonzero(kept_mask)) > 0 else 0.0
        rows_factor = min(1.0, float(len(points)) / max(1.0, float(profile.min_rows)))
        continuity_factor = float(continuity_hits) / max(1.0, float(len(points) - 1)) if len(points) > 1 else 0.0
        strength_factor = float(np.clip((mean_score - float(threshold)) / max(1.0, float(threshold)), 0.0, 1.0))
        density_factor = float(len(points)) / max(1.0, float(h))
        confidence = float(np.clip(0.40 * rows_factor + 0.30 * continuity_factor + 0.20 * strength_factor + 0.10 * density_factor, 0.0, 1.0))

        telemetry = {
            "profile": asdict(profile),
            "threshold": int(threshold),
            "rows_total": int(h),
            "rows_considered": int(considered_rows),
            "rows_kept": int(len(points)),
            "continuity_hits": int(continuity_hits),
            "kept_components": int(kept_components),
            "mask_pixels": int(np.count_nonzero(kept_mask)),
            "mean_score": float(mean_score),
            "confidence": float(confidence),
        }

        return DetectionResult(
            points=points,
            mask=kept_mask,
            score=score,
            confidence=confidence,
            telemetry=telemetry,
        )


def overlay_detection_jpeg(
    rgb: np.ndarray,
    result: DetectionResult,
    *,
    quality: int = 80,
    object_mask: Optional[np.ndarray] = None,
    roi_mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if roi_mask is not None:
        contours, _ = cv2.findContours(roi_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, (0, 255, 255), 1)

    if object_mask is not None:
        contours, _ = cv2.findContours(object_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, (255, 0, 0), 1)

    for x, y in result.points:
        cv2.circle(bgr, (int(round(x)), int(y)), 1, (0, 255, 0), -1)

    txt = title or f"conf={result.confidence:.3f} rows={len(result.points)} th={result.telemetry.get('threshold', 0)}"
    cv2.putText(bgr, txt, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else b""
