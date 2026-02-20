from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .calibration_models import CalibrationData, SCHEMA_VERSION, utc_now_iso


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"{field} must be numeric")


def _require_list(value: Any, field: str) -> list:
    if isinstance(value, list):
        return value
    raise ValueError(f"{field} must be a list")


def _validate_intrinsics(payload: Dict[str, Any]) -> None:
    for key in (
        "method",
        "image_size",
        "board",
        "camera_matrix",
        "dist_coeffs",
        "rms_reprojection_error",
        "mean_reprojection_error",
        "frames_used",
        "frames_captured",
    ):
        if key not in payload:
            raise ValueError(f"intrinsics.{key} is required")

    image_size = _require_list(payload["image_size"], "intrinsics.image_size")
    if len(image_size) != 2:
        raise ValueError("intrinsics.image_size must be [w, h]")
    _require_number(image_size[0], "intrinsics.image_size[0]")
    _require_number(image_size[1], "intrinsics.image_size[1]")

    k = _require_list(payload["camera_matrix"], "intrinsics.camera_matrix")
    if len(k) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in k):
        raise ValueError("intrinsics.camera_matrix must be 3x3")
    for r in range(3):
        for c in range(3):
            _require_number(k[r][c], f"intrinsics.camera_matrix[{r}][{c}]")

    dist = _require_list(payload["dist_coeffs"], "intrinsics.dist_coeffs")
    if len(dist) < 4:
        raise ValueError("intrinsics.dist_coeffs must have at least 4 coefficients")
    for i, v in enumerate(dist):
        _require_number(v, f"intrinsics.dist_coeffs[{i}]")

    _require_number(payload["rms_reprojection_error"], "intrinsics.rms_reprojection_error")
    _require_number(payload["mean_reprojection_error"], "intrinsics.mean_reprojection_error")
    _require_number(payload["frames_used"], "intrinsics.frames_used")
    _require_number(payload["frames_captured"], "intrinsics.frames_captured")


def _validate_laser_plane(payload: Dict[str, Any], key: str) -> None:
    for field in ("laser", "plane", "rmse", "mean_abs_error", "max_abs_error", "points_used", "samples_used"):
        if field not in payload:
            raise ValueError(f"{key}.{field} is required")

    laser = int(_require_number(payload["laser"], f"{key}.laser"))
    if laser not in (1, 2):
        raise ValueError(f"{key}.laser must be 1 or 2")

    plane = _require_list(payload["plane"], f"{key}.plane")
    if len(plane) != 4:
        raise ValueError(f"{key}.plane must be [a,b,c,d]")
    for i, v in enumerate(plane):
        _require_number(v, f"{key}.plane[{i}]")

    _require_number(payload["rmse"], f"{key}.rmse")
    _require_number(payload["mean_abs_error"], f"{key}.mean_abs_error")
    _require_number(payload["max_abs_error"], f"{key}.max_abs_error")
    _require_number(payload["points_used"], f"{key}.points_used")
    _require_number(payload["samples_used"], f"{key}.samples_used")


def validate_calibration_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("calibration payload must be an object")

    schema_version = int(payload.get("schema_version", SCHEMA_VERSION))
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version={schema_version}, expected={SCHEMA_VERSION}")

    updated_at = str(payload.get("updated_at") or utc_now_iso())

    intr = payload.get("intrinsics")
    if intr is not None:
        if not isinstance(intr, dict):
            raise ValueError("intrinsics must be object or null")
        _validate_intrinsics(intr)

    l1 = payload.get("laser1")
    if l1 is not None:
        if not isinstance(l1, dict):
            raise ValueError("laser1 must be object or null")
        _validate_laser_plane(l1, "laser1")

    l2 = payload.get("laser2")
    if l2 is not None:
        if not isinstance(l2, dict):
            raise ValueError("laser2 must be object or null")
        _validate_laser_plane(l2, "laser2")

    out = {
        "schema_version": schema_version,
        "updated_at": updated_at,
        "intrinsics": intr,
        "laser1": l1,
        "laser2": l2,
    }
    return out


class CalibrationStore:
    def __init__(self, path: str = "calibration/calibration.json"):
        self.path = Path(path)

    def default(self) -> CalibrationData:
        return CalibrationData()

    def load(self) -> CalibrationData:
        if not self.path.exists():
            return self.default()

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        validated = validate_calibration_payload(raw)
        return CalibrationData.from_dict(validated)

    def save(self, data: CalibrationData) -> CalibrationData:
        payload = validate_calibration_payload(data.to_dict())
        payload["updated_at"] = utc_now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return CalibrationData.from_dict(payload)

    def reset(self) -> CalibrationData:
        data = self.default()
        return self.save(data)

