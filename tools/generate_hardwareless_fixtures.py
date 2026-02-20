from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mk_calibration(width: int, height: int) -> dict:
    fx = 950.0
    fy = 950.0
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return {
        "schema_version": 1,
        "updated_at": _iso_now(),
        "intrinsics": {
            "method": "synthetic",
            "image_size": [int(width), int(height)],
            "board": {"type": "synthetic"},
            "camera_matrix": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            "rms_reprojection_error": 0.0,
            "mean_reprojection_error": 0.0,
            "frames_used": 24,
            "frames_captured": 24,
            "quality": {"synthetic": True},
            "solved_at": _iso_now(),
        },
        "laser1": {
            "laser": 1,
            "plane": [0.22, 0.0, 1.0, -240.0],
            "rmse": 0.1,
            "mean_abs_error": 0.08,
            "max_abs_error": 0.2,
            "points_used": 5000,
            "samples_used": 30,
            "solved_at": _iso_now(),
            "metadata": {"synthetic": True},
        },
        "laser2": {
            "laser": 2,
            "plane": [-0.22, 0.0, 1.0, -240.0],
            "rmse": 0.1,
            "mean_abs_error": 0.08,
            "max_abs_error": 0.2,
            "points_used": 5000,
            "samples_used": 30,
            "solved_at": _iso_now(),
            "metadata": {"synthetic": True},
        },
    }


def _laser_points(angle_deg: float, width: int, height: int, shift: float) -> np.ndarray:
    n_rows = min(height - 40, 420)
    ys = np.linspace(20, min(height - 20, 20 + n_rows), n_rows, dtype=np.float32)
    theta = math.radians(angle_deg)
    wave = 75.0 * np.sin((ys / 70.0) + theta)
    radius_mod = 25.0 * float(np.cos(theta))
    x_center = (width / 2.0) + radius_mod + shift
    xs = x_center + wave
    pts = np.stack([xs, ys], axis=1)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(width - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(height - 1))
    return pts.astype(np.float32)


def generate_run(run_dir: Path, steps: int, width: int, height: int) -> None:
    points_dir = run_dir / "points"
    points_dir.mkdir(parents=True, exist_ok=True)

    for i in range(int(steps)):
        angle_deg = (360.0 * i) / float(steps)
        l1 = _laser_points(angle_deg, width, height, shift=-18.0)
        l2 = _laser_points(angle_deg, width, height, shift=18.0)
        np.savez_compressed(
            points_dir / f"step_{i:04d}.npz",
            angle_deg=np.float32(angle_deg),
            laser1=l1,
            laser2=l2,
        )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "requested_step_deg": 360.0 / float(steps),
                "span_deg": 360.0,
                "save_debug_images": False,
                "synthetic": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "plan.json").write_text(
        json.dumps(
            {
                "steps_count": int(steps),
                "captures_count": int(steps),
                "actual_step_deg": 360.0 / float(steps),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate hardwareless synthetic run and calibration fixtures")
    p.add_argument("--run-id", default=f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--calibration", type=Path, default=Path("calibration/calibration.json"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.runs_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    generate_run(run_dir=run_dir, steps=max(8, int(args.steps)), width=max(320, int(args.width)), height=max(240, int(args.height)))

    args.calibration.parent.mkdir(parents=True, exist_ok=True)
    args.calibration.write_text(json.dumps(_mk_calibration(int(args.width), int(args.height)), indent=2), encoding="utf-8")

    print(f"run_dir={run_dir.as_posix()}")
    print(f"calibration={args.calibration.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

