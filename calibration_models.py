from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IntrinsicsCalibration:
    method: str
    image_size: List[int]  # [w, h]
    board: Dict[str, Any]
    camera_matrix: List[List[float]]
    dist_coeffs: List[float]
    rms_reprojection_error: float
    mean_reprojection_error: float
    frames_used: int
    frames_captured: int
    quality: Dict[str, Any] = field(default_factory=dict)
    solved_at: str = field(default_factory=utc_now_iso)


@dataclass
class LaserPlaneCalibration:
    laser: int
    plane: List[float]  # ax + by + cz + d = 0
    rmse: float
    mean_abs_error: float
    max_abs_error: float
    points_used: int
    samples_used: int
    solved_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationData:
    schema_version: int = SCHEMA_VERSION
    updated_at: str = field(default_factory=utc_now_iso)
    intrinsics: Optional[IntrinsicsCalibration] = None
    laser1: Optional[LaserPlaneCalibration] = None
    laser2: Optional[LaserPlaneCalibration] = None

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["schema_version"] = int(self.schema_version)
        out["updated_at"] = self.updated_at
        return out

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "CalibrationData":
        intr = payload.get("intrinsics")
        l1 = payload.get("laser1")
        l2 = payload.get("laser2")
        return CalibrationData(
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            updated_at=str(payload.get("updated_at", utc_now_iso())),
            intrinsics=IntrinsicsCalibration(**intr) if isinstance(intr, dict) else None,
            laser1=LaserPlaneCalibration(**l1) if isinstance(l1, dict) else None,
            laser2=LaserPlaneCalibration(**l2) if isinstance(l2, dict) else None,
        )

