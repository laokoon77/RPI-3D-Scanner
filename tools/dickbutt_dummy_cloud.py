from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class DickButtCloudConfig:
    steps: int = 180
    y_samples: int = 220
    image_width: int = 1280
    image_height: int = 720
    scale_r: float = 0.01
    scale_y: float = 0.01
    x_center: float = 640.0
    y_center_px: float = 360.0
    y_span_px: float = 420.0


class DickButtDummyPointCloudGenerator:
    """
    Generates a synthetic export JSON compatible with viewer/app.js.

    Output schema matches tools/export_run_to_json.py:
    {
      "version": 1,
      "format": "rpi_scanner_run_export",
      "mapping_defaults": {...},
      "steps": [
        {"index": 0, "angle_deg": ..., "laser1": [[x,y], ...], "laser2": [[x,y], ...]},
        ...
      ]
    }

    Geometry is intentionally stylized as a humorous "DickButt"-like volume
    for viewer testing.
    """

    def __init__(self, cfg: DickButtCloudConfig | None = None) -> None:
        self.cfg = cfg or DickButtCloudConfig()

    @staticmethod
    def _gaussian2(y: float, theta: float, y0: float, t0: float, sy: float, st: float, amp: float) -> float:
        dy = (y - y0) / max(1e-6, sy)
        dt = (theta - t0 + math.pi) % (2.0 * math.pi) - math.pi
        dt /= max(1e-6, st)
        return amp * math.exp(-(dy * dy + dt * dt))

    def _radius_profile(self, y_norm: float, theta: float) -> float:
        # Base torso / butt body
        body = 0.62 * math.exp(-((y_norm + 0.04) / 0.78) ** 2)

        # Butt cheeks (rear hemisphere around theta ~ pi)
        left_cheek = self._gaussian2(y_norm, theta, -0.05, math.pi - 0.30, 0.24, 0.33, 0.62)
        right_cheek = self._gaussian2(y_norm, theta, -0.05, -math.pi + 0.30, 0.24, 0.33, 0.62)

        # Head / face lobe (front, upper)
        head = self._gaussian2(y_norm, theta, 0.56, 0.0, 0.22, 0.50, 0.46)

        # Nose-like protrusion
        nose = self._gaussian2(y_norm, theta, 0.52, 0.02, 0.10, 0.18, 0.18)

        # Phallus shaft protruding from rear
        shaft = self._gaussian2(y_norm, theta, -0.03, math.pi, 0.08, 0.10, 0.95)

        # Tip / glans
        tip = self._gaussian2(y_norm, theta, -0.03, math.pi, 0.11, 0.06, 0.30)

        # Balls below shaft
        ball_l = self._gaussian2(y_norm, theta, -0.26, math.pi - 0.06, 0.10, 0.09, 0.42)
        ball_r = self._gaussian2(y_norm, theta, -0.26, -math.pi + 0.06, 0.10, 0.09, 0.42)

        return max(0.0, body + left_cheek + right_cheek + head + nose + shaft + tip + ball_l + ball_r)

    def _generate_step_points(self, angle_deg: float) -> list[list[float]]:
        cfg = self.cfg
        theta = math.radians(angle_deg)

        points: list[list[float]] = []
        for i in range(cfg.y_samples):
            t = i / max(1, cfg.y_samples - 1)
            y_norm = -1.0 + 2.0 * t

            radius_world = self._radius_profile(y_norm, theta)
            if radius_world < 0.015:
                continue

            # Inverse mapping used by viewer: radius = (x - x_center) * scale_r
            x = cfg.x_center + (radius_world / max(cfg.scale_r, 1e-6))
            y = cfg.y_center_px + y_norm * (cfg.y_span_px * 0.5)

            if 0.0 <= x < cfg.image_width and 0.0 <= y < cfg.image_height:
                points.append([float(x), float(y)])

        return points

    def generate_export_payload(self) -> dict[str, Any]:
        cfg = self.cfg
        steps: list[dict[str, Any]] = []

        for i in range(cfg.steps):
            angle_deg = (360.0 * i) / float(cfg.steps)
            pts = self._generate_step_points(angle_deg)

            # Slight offset between "lasers" so both can be toggled distinctly.
            pts2 = [[p[0] + 1.5, p[1]] for p in pts]

            steps.append(
                {
                    "index": i,
                    "file": f"synthetic_step_{i:04d}.npz",
                    "angle_deg": float(angle_deg),
                    "laser1": pts,
                    "laser2": pts2,
                }
            )

        return {
            "version": 1,
            "format": "rpi_scanner_run_export",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "run_dir": "synthetic://dickbutt",
                "points_dir": "synthetic://dickbutt/points",
                "step_count": cfg.steps,
            },
            "mapping_defaults": {
                "scale_y": float(cfg.scale_y),
                "scale_r": float(cfg.scale_r),
                "x_center": float(cfg.x_center),
            },
            "image_size": {"width": int(cfg.image_width), "height": int(cfg.image_height)},
            "steps": steps,
        }

    def write_json(self, output_path: Path) -> Path:
        payload = self.generate_export_payload()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic DickButt-like point cloud export JSON for viewer testing.")
    p.add_argument("--output", type=Path, default=Path("viewer") / "dickbutt_dummy_export.json")
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--y-samples", type=int, default=220)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--scale-y", type=float, default=0.01)
    p.add_argument("--scale-r", type=float, default=0.01)
    p.add_argument("--x-center", type=float, default=640.0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = DickButtCloudConfig(
        steps=max(12, int(args.steps)),
        y_samples=max(40, int(args.y_samples)),
        image_width=max(320, int(args.width)),
        image_height=max(240, int(args.height)),
        scale_y=float(args.scale_y),
        scale_r=float(args.scale_r),
        x_center=float(args.x_center),
        y_center_px=float(int(args.height) / 2.0),
        y_span_px=float(int(args.height) * 0.8),
    )

    out = DickButtDummyPointCloudGenerator(cfg).write_json(args.output)
    print(f"Synthetic export written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

