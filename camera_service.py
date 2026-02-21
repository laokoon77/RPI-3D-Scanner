from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None  # type: ignore

log = logging.getLogger(__name__)

try:
    from libcamera import Transform
except Exception:
    Transform = None  # type: ignore

try:
    from libcamera import controls as lc_controls
except Exception:
    lc_controls = None  # type: ignore


@dataclass
class CameraSettings:
    size: Tuple[int, int] = (1280, 720)
    fps: Optional[float] = 25.0
    jpeg_quality: int = 80
    rotation_degrees: int = 180
    hflip: bool = False
    vflip: bool = False
    mock: bool = False


class CameraService:
    def __init__(self, settings: CameraSettings = CameraSettings()):
        self.settings = settings
        self._mock = bool(settings.mock)
        self._mock_controls: Dict[str, Any] = {}

        if not self._mock and Picamera2 is None:
            raise RuntimeError(
                "picamera2 is unavailable in this environment. "
                "Set SCANNER_MOCK_HW=1 to run in hardwareless mode."
            )

        self._picam: Any = None
        if not self._mock:
            self._picam = Picamera2()

        if not self._mock:
            transform = None
            if Transform is not None and (settings.hflip or settings.vflip):
                transform = Transform(
                    hflip=1 if settings.hflip else 0,
                    vflip=1 if settings.vflip else 0,
                )

            kwargs = {}
            if transform is not None:
                kwargs["transform"] = transform

            cfg = self._picam.create_video_configuration(  # type: ignore[union-attr]
                main={"size": settings.size, "format": "RGB888"},
                **kwargs,
            )
            self._picam.configure(cfg)  # type: ignore[union-attr]

            if settings.fps:
                us = int(round(1_000_000 / float(settings.fps)))
                self._picam.set_controls({"FrameDurationLimits": (us, us)})  # type: ignore[union-attr]

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_meta: Optional[Dict[str, Any]] = None
        self._latest_ts: float = 0.0

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _apply_frame_transform(self, frame: np.ndarray) -> np.ndarray:
        deg = int(self.settings.rotation_degrees) % 360
        if deg == 0:
            return frame
        if deg == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if deg == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if deg == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        raise ValueError(f"Unsupported camera rotation_degrees={self.settings.rotation_degrees}; expected one of 0,90,180,270")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        if not self._mock:
            self._picam.start()  # type: ignore[union-attr]
            target = self._capture_loop
            name = "camera-capture"
        else:
            target = self._mock_capture_loop
            name = "mock-camera-capture"
        self._thread = threading.Thread(target=target, daemon=True, name=name)
        self._thread.start()
        log.info("CameraService started (mock=%s)", self._mock)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if not self._mock:
            try:
                self._picam.stop()  # type: ignore[union-attr]
            except Exception:
                log.exception("Error stopping camera")
        log.info("CameraService stopped")

    def _generate_mock_frame(self, t0: float) -> tuple[np.ndarray, Dict[str, Any]]:
        w, h = int(self.settings.size[0]), int(self.settings.size[1])
        phase = int((t0 * 1000.0) % 255)

        x = np.linspace(0, 255, w, dtype=np.uint8)
        y = np.linspace(0, 255, h, dtype=np.uint8)
        xx = np.tile(x, (h, 1))
        yy = np.tile(y[:, None], (1, w))
        blue = ((xx.astype(np.uint16) + yy.astype(np.uint16) + phase) % 256).astype(np.uint8)

        frame = np.stack([xx, yy, blue], axis=2)

        meta: Dict[str, Any] = {
            "MockMode": True,
            "ExposureTime": int(10_000 + (phase * 5)),
            "AnalogueGain": 1.5,
            "ColourGains": (1.0, 1.0),
            "LensPosition": 1.5,
        }
        meta.update(self._mock_controls)
        return frame, meta

    def _mock_capture_loop(self) -> None:
        min_dt = 0.0
        if self.settings.fps:
            min_dt = 1.0 / max(1.0, float(self.settings.fps))

        while not self._stop.is_set():
            t0 = time.time()
            frame, meta = self._generate_mock_frame(t0)
            frame = self._apply_frame_transform(frame)
            with self._lock:
                self._latest_frame = frame
                self._latest_meta = meta
                self._latest_ts = t0

            dt = time.time() - t0
            if min_dt and dt < min_dt:
                time.sleep(min_dt - dt)

    def _capture_loop(self) -> None:
        min_dt = 0.0
        if self.settings.fps:
            min_dt = 1.0 / max(1.0, float(self.settings.fps))

        while not self._stop.is_set():
            t0 = time.time()
            try:
                frame = self._picam.capture_array()
                frame = self._apply_frame_transform(frame)
                meta = self._picam.capture_metadata()
                with self._lock:
                    self._latest_frame = frame
                    self._latest_meta = meta
                    self._latest_ts = t0
            except Exception:
                log.exception("Camera capture failed (will retry)")
                time.sleep(0.2)

            dt = time.time() - t0
            if min_dt and dt < min_dt:
                time.sleep(min_dt - dt)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def get_latest_metadata(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return None if self._latest_meta is None else dict(self._latest_meta)

    @staticmethod
    def _meta_float(meta: Optional[Dict[str, Any]], key: str) -> Optional[float]:
        if not meta:
            return None
        v = meta.get(key)
        if isinstance(v, (tuple, list)):
            if not v:
                return None
            v = v[0]
        try:
            return float(v)
        except Exception:
            return None

    def pair_metadata_stable(
        self,
        off_meta: Optional[Dict[str, Any]],
        on_meta: Optional[Dict[str, Any]],
        *,
        max_exposure_drift_rel: float = 0.25,
        max_gain_drift_rel: float = 0.25,
    ) -> tuple[bool, Dict[str, Any]]:
        off_exp = self._meta_float(off_meta, "ExposureTime")
        on_exp = self._meta_float(on_meta, "ExposureTime")
        off_gain = self._meta_float(off_meta, "AnalogueGain")
        on_gain = self._meta_float(on_meta, "AnalogueGain")

        exp_drift = 0.0
        gain_drift = 0.0
        if off_exp is not None and on_exp is not None and abs(off_exp) > 1e-6:
            exp_drift = abs(on_exp - off_exp) / abs(off_exp)
        if off_gain is not None and on_gain is not None and abs(off_gain) > 1e-6:
            gain_drift = abs(on_gain - off_gain) / abs(off_gain)

        stable = bool(exp_drift <= float(max_exposure_drift_rel) and gain_drift <= float(max_gain_drift_rel))
        details: Dict[str, Any] = {
            "off_exposure": off_exp,
            "on_exposure": on_exp,
            "off_gain": off_gain,
            "on_gain": on_gain,
            "exposure_drift_rel": float(exp_drift),
            "gain_drift_rel": float(gain_drift),
            "max_exposure_drift_rel": float(max_exposure_drift_rel),
            "max_gain_drift_rel": float(max_gain_drift_rel),
            "stable": bool(stable),
        }
        return stable, details

    def get_latest_timestamp(self) -> float:
        with self._lock:
            return self._latest_ts

    def get_latest_jpeg(self) -> Optional[bytes]:
        frame = self.get_latest_frame()
        if frame is None:
            return None
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.settings.jpeg_quality)])
        return buf.tobytes() if ok else None

    def set_controls(self, controls: Dict[str, Any]) -> None:
        if self._mock:
            self._mock_controls.update(dict(controls))
            return
        try:
            self._picam.set_controls(controls)
        except Exception:
            log.exception("set_controls failed: %s", controls)

    def apply_locked_controls(self, controls: Dict[str, Any], settle_s: float = 0.05, drop_n: int = 1) -> None:
        self.set_controls(dict(controls))
        _ = self.grab_fresh_frame(settle_s=max(0.0, float(settle_s)))
        for _i in range(max(0, int(drop_n))):
            _ = self.grab_fresh_frame(settle_s=0.0)

    def get_camera_controls(self) -> Dict[str, Any]:
        if self._mock:
            return dict(self._mock_controls)
        return dict(getattr(self._picam, "camera_controls", {}) or {})

    # ---- AF / AE / AWB helpers (best-effort; depends on stack) ----

    def enable_auto(self) -> None:
        c: Dict[str, Any] = {"AeEnable": True, "AwbEnable": True}
        if lc_controls is not None:
            c["AfMode"] = lc_controls.AfModeEnum.Continuous
        else:
            c["AfMode"] = 2  # fallback guess
        self.set_controls(c)

    def trigger_autofocus(self) -> None:
        c: Dict[str, Any] = {}
        if lc_controls is not None:
            c["AfMode"] = lc_controls.AfModeEnum.Auto
            c["AfTrigger"] = lc_controls.AfTriggerEnum.Start
        else:
            c["AfMode"] = 1
            c["AfTrigger"] = 0
        self.set_controls(c)

    def set_manual_focus(self, pos: float) -> None:
        c: Dict[str, Any] = {}
        if lc_controls is not None:
            c["AfMode"] = lc_controls.AfModeEnum.Manual
        else:
            c["AfMode"] = 0
        c["LensPosition"] = float(pos)
        self.set_controls(c)

    def freeze_from_current(self) -> None:
        meta = self.get_latest_metadata() or {}

        controls: Dict[str, Any] = {"AeEnable": False, "AwbEnable": False}

        if "ExposureTime" in meta:
            controls["ExposureTime"] = int(meta["ExposureTime"])
        if "AnalogueGain" in meta:
            controls["AnalogueGain"] = float(meta["AnalogueGain"])
        if "ColourGains" in meta:
            controls["ColourGains"] = tuple(meta["ColourGains"])

        # lock focus if possible
        if "LensPosition" in meta:
            if lc_controls is not None:
                controls["AfMode"] = lc_controls.AfModeEnum.Manual
            else:
                controls["AfMode"] = 0
            controls["LensPosition"] = float(meta["LensPosition"])

        self.set_controls(controls)

    def grab_fresh_frame(self, settle_s: float = 0.05, timeout_s: float = 1.0) -> Optional[np.ndarray]:
        if self._mock:
            if settle_s > 0:
                time.sleep(settle_s)
            t0 = time.time()
            frame, meta = self._generate_mock_frame(t0)
            frame = self._apply_frame_transform(frame)
            with self._lock:
                self._latest_frame = frame
                self._latest_meta = meta
                self._latest_ts = t0
            return frame.copy()

        if settle_s > 0:
            time.sleep(settle_s)

        t_start = time.time()
        ts0 = self.get_latest_timestamp()

        while time.time() - t_start < timeout_s:
            frame = self.get_latest_frame()
            if frame is not None and self.get_latest_timestamp() > ts0:
                return frame
            time.sleep(0.005)

        return self.get_latest_frame()

    def grab_stabilized_frame(
        self,
        settle_s: float = 0.05,
        retries: int = 2,
        min_luma_delta: float = 0.0,
    ) -> Optional[np.ndarray]:
        tries = max(1, int(retries) + 1)
        prev: Optional[np.ndarray] = None
        best: Optional[np.ndarray] = None

        for _i in range(tries):
            frame = self.grab_fresh_frame(settle_s=settle_s)
            if frame is None:
                continue
            best = frame
            if prev is None or float(min_luma_delta) <= 0.0:
                prev = frame
                continue

            g0 = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
            g1 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            delta = float(np.mean(np.abs(g1.astype(np.float32) - g0.astype(np.float32))))
            if delta <= float(min_luma_delta):
                return frame
            prev = frame

        return best
