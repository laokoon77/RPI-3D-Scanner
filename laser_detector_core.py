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
    channel_mode: str = "r_weighted_magenta"


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
            adaptive_scale=0.40,
            blur_ksize=3,
            morph_open_ksize=3,
            morph_close_ksize=3,
            min_component_area=6,
            min_rows=14,
            max_row_jump_px=40.0,
            max_gap_rows=14,
            window_half_width_px=30,
            roi_top_frac=0.02,
            roi_bottom_frac=0.98,
            roi_left_frac=0.02,
            roi_right_frac=0.98,
            channel_mode="r_weighted_magenta",
        ),
        "calibration": DetectionProfile(
            name="calibration",
            adaptive_floor=8,
            adaptive_scale=0.30,
            blur_ksize=3,
            morph_open_ksize=3,
            morph_close_ksize=3,
            min_component_area=4,
            min_rows=12,
            max_row_jump_px=48.0,
            max_gap_rows=18,
            window_half_width_px=32,
            roi_top_frac=0.0,
            roi_bottom_frac=1.0,
            roi_left_frac=0.0,
            roi_right_frac=1.0,
            channel_mode="r_weighted_magenta",
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
        r = diff[:, :, 0]
        g = diff[:, :, 1]
        b = diff[:, :, 2]

        mode = str(profile.channel_mode)
        if mode == "r_minus_max_gb":
            score = r - np.maximum(g, b)
        elif mode == "r_only":
            score = r
        elif mode == "magenta_minus_g":
            score = ((r + b) // 2) - g
        elif mode == "r_weighted_magenta":
            red_lift = r - ((g + b) // 2)
            magenta_lift = ((r + b) // 2) - g
            red_only_scaled = (r * 11) // 20
            score = np.maximum.reduce([red_lift, magenta_lift, red_only_scaled])
        else:
            score = r - ((g + b) // 2)

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
        components_total = max(0, int(num_labels) - 1)
        kept_components = 0
        rejected_area_small = 0
        rejected_area_large = 0
        rejected_height = 0
        mask_pixels_raw = int(np.count_nonzero(mask))
        for label in range(1, int(num_labels)):
            area = int(stats[label, cv2.CC_STAT_AREA])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < int(profile.min_component_area):
                rejected_area_small += 1
                continue
            if area > int(profile.max_component_area):
                rejected_area_large += 1
                continue
            if h < int(profile.min_component_height):
                rejected_height += 1
                continue
            kept_mask[labels == label] = 255
            kept_components += 1

        h, w = kept_mask.shape
        points: List[Tuple[float, int]] = []
        prev_x: Optional[float] = None
        continuity_hits = 0
        rejected_jump_rows = 0
        gap_resets = 0
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
                    rejected_jump_rows += 1
                    gap_rows += 1
                    if gap_rows > int(profile.max_gap_rows):
                        prev_x = None
                        gap_resets += 1
                    continue
                continuity_hits += 1

            points.append((x_sub, y))
            prev_x = x_sub
            gap_rows = 0

        mean_score = float(score[kept_mask > 0].mean()) if int(np.count_nonzero(kept_mask)) > 0 else 0.0
        positive = score[score > 0]
        p50 = float(np.percentile(positive, 50)) if positive.size else 0.0
        p90 = float(np.percentile(positive, 90)) if positive.size else 0.0
        p99 = float(np.percentile(positive, 99)) if positive.size else 0.0
        rows_factor = min(1.0, float(len(points)) / max(1.0, float(profile.min_rows)))
        continuity_factor = float(continuity_hits) / max(1.0, float(len(points) - 1)) if len(points) > 1 else 0.0
        strength_factor = float(np.clip((mean_score - float(threshold)) / max(1.0, float(threshold)), 0.0, 1.0))
        density_factor = float(len(points)) / max(1.0, float(h))
        confidence = float(np.clip(0.40 * rows_factor + 0.30 * continuity_factor + 0.20 * strength_factor + 0.10 * density_factor, 0.0, 1.0))

        telemetry = {
            "profile": asdict(profile),
            "channel_mode": str(profile.channel_mode),
            "threshold_source": "fixed" if profile.threshold is not None else "adaptive",
            "threshold": int(threshold),
            "rows_total": int(h),
            "rows_considered": int(considered_rows),
            "rows_kept": int(len(points)),
            "continuity_hits": int(continuity_hits),
            "rejected_jump_rows": int(rejected_jump_rows),
            "gap_resets": int(gap_resets),
            "components_total": int(components_total),
            "kept_components": int(kept_components),
            "rejected_components_area_small": int(rejected_area_small),
            "rejected_components_area_large": int(rejected_area_large),
            "rejected_components_height": int(rejected_height),
            "mask_pixels_raw": int(mask_pixels_raw),
            "mask_pixels": int(np.count_nonzero(kept_mask)),
            "mean_score": float(mean_score),
            "score_p50": float(p50),
            "score_p90": float(p90),
            "score_p99": float(p99),
            "positive_pixels": int(positive.size),
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
