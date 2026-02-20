from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_points(arr: np.ndarray) -> list[list[float]]:
    if arr is None:
        return []
    if not isinstance(arr, np.ndarray):
        return []
    if arr.ndim != 2 or arr.shape[1] < 2:
        return []

    out: list[list[float]] = []
    for row in arr:
        out.append([float(row[0]), float(row[1])])
    return out


def _infer_image_size(steps: list[dict[str, Any]]) -> dict[str, int] | None:
    max_x = -1.0
    max_y = -1.0
    for step in steps:
        for key in ("laser1", "laser2"):
            for pt in step.get(key, []):
                if len(pt) < 2:
                    continue
                x, y = float(pt[0]), float(pt[1])
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y

    if max_x < 0 or max_y < 0:
        return None

    return {
        "width": int(np.ceil(max_x + 1.0)),
        "height": int(np.ceil(max_y + 1.0)),
    }


def export_run(run_dir: Path, output_path: Path, scale_y: float, scale_r: float, x_center: float | None) -> None:
    points_dir = run_dir / "points"
    if not points_dir.exists() or not points_dir.is_dir():
        raise FileNotFoundError(f"missing points directory: {points_dir}")

    step_files = sorted(points_dir.glob("step_*.npz"))
    if not step_files:
        raise FileNotFoundError(f"no step_*.npz files found in: {points_dir}")

    steps: list[dict[str, Any]] = []
    for idx, npz_path in enumerate(step_files):
        with np.load(npz_path, allow_pickle=False) as data:
            angle_deg = _safe_float(data["angle_deg"][()] if "angle_deg" in data else 0.0)
            laser1 = _safe_points(data["laser1"] if "laser1" in data else np.zeros((0, 2), dtype=np.float32))
            laser2 = _safe_points(data["laser2"] if "laser2" in data else np.zeros((0, 2), dtype=np.float32))

        steps.append(
            {
                "index": idx,
                "file": npz_path.name,
                "angle_deg": angle_deg,
                "laser1": laser1,
                "laser2": laser2,
            }
        )

    inferred_size = _infer_image_size(steps)
    if x_center is None:
        if inferred_size is not None:
            x_center = 0.5 * float(inferred_size["width"] - 1)
        else:
            x_center = 640.0

    payload: dict[str, Any] = {
        "version": 1,
        "format": "rpi_scanner_run_export",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "run_dir": str(run_dir),
            "points_dir": str(points_dir),
            "step_count": len(steps),
        },
        "mapping_defaults": {
            "scale_y": float(scale_y),
            "scale_r": float(scale_r),
            "x_center": float(x_center),
        },
        "image_size": inferred_size,
        "steps": steps,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a scanner run folder (points/step_*.npz) into one viewer JSON file."
    )
    p.add_argument("run_dir", type=Path, help="Path to run folder, e.g. runs/20260220_120000")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <run_dir>/viewer_export.json)",
    )
    p.add_argument("--scale-y", type=float, default=0.01, help="Default pseudo mapping scale for image row -> world Y")
    p.add_argument("--scale-r", type=float, default=0.01, help="Default pseudo mapping scale for image x -> radius")
    p.add_argument(
        "--x-center",
        type=float,
        default=None,
        help="Optional image x-center used in pseudo radius mapping (default: inferred from points)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    output = args.output if args.output is not None else (run_dir / "viewer_export.json")

    export_run(
        run_dir=run_dir,
        output_path=output,
        scale_y=float(args.scale_y),
        scale_r=float(args.scale_r),
        x_center=args.x_center,
    )

    print(f"Exported {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

