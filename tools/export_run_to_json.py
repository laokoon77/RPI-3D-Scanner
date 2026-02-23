from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from triangulation import (
    TriangulationCalibration,
    load_triangulation_calibration,
    triangulate_pixels_to_world,
)


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


def _as_xy_array(points: list[list[float]]) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    return arr[:, :2]


def _try_load_calibration(calibration_path: Path) -> tuple[TriangulationCalibration | None, str | None]:
    if not calibration_path.exists():
        return None, f"calibration file not found: {calibration_path}"
    try:
        calib = load_triangulation_calibration(calibration_path)
        return calib, None
    except Exception as e:
        return None, f"invalid calibration file ({calibration_path}): {type(e).__name__}: {e}"


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


def export_run(
    run_dir: Path,
    output_path: Path,
    scale_y: float,
    scale_r: float,
    x_center: float | None,
    calibration_path: Path,
) -> dict[str, Any]:
    points_dir = run_dir / "points"
    if not points_dir.exists() or not points_dir.is_dir():
        raise FileNotFoundError(f"missing points directory: {points_dir}")

    step_files = sorted(points_dir.glob("step_*.npz"))
    if not step_files:
        raise FileNotFoundError(f"no step_*.npz files found in: {points_dir}")

    calibration, calibration_error = _try_load_calibration(calibration_path)

    tri_stats = {
        "laser1": {"input": 0, "finite": 0, "intersected": 0, "output": 0},
        "laser2": {"input": 0, "finite": 0, "intersected": 0, "output": 0},
    }

    steps: list[dict[str, Any]] = []
    for idx, npz_path in enumerate(step_files):
        with np.load(npz_path, allow_pickle=False) as data:
            angle_deg = _safe_float(data["angle_deg"][()] if "angle_deg" in data else 0.0)
            laser1 = _safe_points(data["laser1"] if "laser1" in data else np.zeros((0, 2), dtype=np.float32))
            laser2 = _safe_points(data["laser2"] if "laser2" in data else np.zeros((0, 2), dtype=np.float32))

        laser1_xyz: list[list[float]] = []
        laser2_xyz: list[list[float]] = []

        if calibration is not None:
            l1_points, l1_s = triangulate_pixels_to_world(
                _as_xy_array(laser1),
                angle_deg=angle_deg,
                k=calibration.k,
                dist=calibration.dist,
                plane_abcd=calibration.plane1,
            )
            l2_points, l2_s = triangulate_pixels_to_world(
                _as_xy_array(laser2),
                angle_deg=angle_deg,
                k=calibration.k,
                dist=calibration.dist,
                plane_abcd=calibration.plane2,
            )
            laser1_xyz = l1_points
            laser2_xyz = l2_points

            for k in tri_stats["laser1"].keys():
                tri_stats["laser1"][k] += int(l1_s[k])
                tri_stats["laser2"][k] += int(l2_s[k])

        steps.append(
            {
                "index": idx,
                "file": npz_path.name,
                "angle_deg": angle_deg,
                "laser1": laser1,
                "laser2": laser2,
                "laser1_xyz": laser1_xyz,
                "laser2_xyz": laser2_xyz,
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
        "triangulation": {
            "enabled": bool(calibration is not None),
            "units": "mm",
            "coordinate_frame": {
                "camera": "OpenCV camera coordinates; +Z forward",
                "world": "camera points rotated by +angle_deg about Y axis",
            },
            "calibration": {
                "path": str(calibration_path),
                "id": calibration.calibration_id if calibration is not None else None,
                "schema_version": calibration.schema_version if calibration is not None else None,
                "updated_at": calibration.updated_at if calibration is not None else None,
                "valid": bool(calibration is not None),
                "error": calibration_error,
            },
            "stats": tri_stats,
        },
        "image_size": inferred_size,
        "steps": steps,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    xyz_points: list[list[float]] = []
    for step in steps:
        xyz_points.extend(step.get("laser1_xyz", []))
        xyz_points.extend(step.get("laser2_xyz", []))

    xyz_path = output_path.with_suffix(".xyz")
    xyz_written = False
    xyz_reason: str | None = None

    if calibration is None:
        xyz_reason = "triangulation unavailable (invalid or missing calibration)"
    elif len(xyz_points) <= 0:
        xyz_reason = "no triangulated points"
    else:
        xyz_path.parent.mkdir(parents=True, exist_ok=True)
        with xyz_path.open("w", encoding="utf-8", newline="\n") as f:
            for p in xyz_points:
                if len(p) < 3:
                    continue
                f.write(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}\n")
        xyz_written = True

    artifacts: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "json": {
            "path": output_path.as_posix(),
            "exists": output_path.exists(),
        },
        "xyz": {
            "path": xyz_path.as_posix(),
            "exists": xyz_path.exists(),
            "written": xyz_written,
            "points": int(len(xyz_points)),
            "skipped": not xyz_written,
            "skip_reason": xyz_reason,
        },
        "triangulation": {
            "enabled": bool(calibration is not None),
            "calibration_error": calibration_error,
            "stats": tri_stats,
        },
    }

    payload["artifacts"] = artifacts
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    artifacts_path = run_dir / "export_artifacts.json"
    artifacts_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    return artifacts


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
    p.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration/calibration.json"),
        help="Path to calibration JSON with intrinsics and laser planes (default: calibration/calibration.json)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    output = args.output if args.output is not None else (run_dir / "viewer_export.json")

    artifacts = export_run(
        run_dir=run_dir,
        output_path=output,
        scale_y=float(args.scale_y),
        scale_r=float(args.scale_r),
        x_center=args.x_center,
        calibration_path=args.calibration,
    )

    print(f"Exported {output}")
    if artifacts.get("xyz", {}).get("written"):
        print(f"XYZ {artifacts['xyz']['path']}")
    else:
        print(f"XYZ skipped: {artifacts.get('xyz', {}).get('skip_reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

