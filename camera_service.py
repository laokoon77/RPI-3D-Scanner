from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from picamera2 import Picamera2

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
    hflip: bool = False
    vflip: bool = False


class CameraService:
    def __init__(self, settings: CameraSettings = CameraSettings()):
        self.settings = settings
        self._picam = Picamera2()

        transform = None
        if Transform is not None and (settings.hflip or settings.vflip):
            transform = Transform(
                hflip=1 if settings.hflip else 0,
                vflip=1 if settings.vflip else 0,
            )

        kwargs = {}
        if transform is not None:
            kwargs["transform"] = transform

        cfg = self._picam.create_video_configuration(
            main={"size": settings.size, "format": "RGB888"},
            **kwargs,
        )
        self._picam.configure(cfg)

        if settings.fps:
            us = int(round(1_000_000 / float(settings.fps)))
            self._picam.set_controls({"FrameDurationLimits": (us, us)})

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_meta: Optional[Dict[str, Any]] = None
        self._latest_ts: float = 0.0

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._picam.start()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="camera-capture")
        self._thread.start()
        log.info("CameraService started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self._picam.stop()
        except Exception:
            log.exception("Error stopping camera")
        log.info("CameraService stopped")

    def _capture_loop(self) -> None:
        min_dt = 0.0
        if self.settings.fps:
            min_dt = 1.0 / max(1.0, float(self.settings.fps))

        while not self._stop.is_set():
            t0 = time.time()
            try:
                frame = self._picam.capture_array()
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
        try:
            self._picam.set_controls(controls)
        except Exception:
            log.exception("set_controls failed: %s", controls)

    def get_camera_controls(self) -> Dict[str, Any]:
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
