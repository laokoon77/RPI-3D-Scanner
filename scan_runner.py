from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from .scan_algo import capture_pair
    from .turntable import Turntable
    from .background import BackgroundModel
    from .scan_algo import StripeDetector
except ImportError:
    from scan_algo import capture_pair
    from turntable import Turntable
    from background import BackgroundModel
    from scan_algo import StripeDetector


@dataclass
class ScanConfig:
    # user-facing request
    requested_step_deg: float = 2.0
    span_deg: float = 360.0

    # motion/capture pacing
    move_speed_sps: float = 320.0         # your suggested slow preset
    settle_after_move_s: float = 0.25     # allow vibrations to die out
    capture_settle_s: float = 0.06        # camera settle per frame
    drop_n: int = 2                       # drop a couple frames after toggles

    # storage
    out_root: str = "runs"
    save_debug_images: bool = False
    save_every_n_steps: int = 10          # only if save_debug_images True

    # if span is not 360, include last capture at end angle
    include_end_capture: bool = True

    # safety
    require_background: bool = True


@dataclass
class ScanPlan:
    steps_count: int          # number of moves
    captures_count: int       # number of capture positions
    actual_step_deg: float    # adjusted step so we end exactly at span


@dataclass
class ScanStatus:
    running: bool = False
    run_id: Optional[str] = None
    out_dir: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    current_index: int = 0
    total_captures: int = 0
    message: str = ""
    error: Optional[str] = None


def _now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def compute_plan(cfg: ScanConfig) -> ScanPlan:
    span = float(cfg.span_deg)
    req = max(1e-6, float(cfg.requested_step_deg))

    # number of moves to cover span at roughly requested step
    moves = max(1, int(round(span / req)))
    actual_step = span / float(moves)

    if span >= 359.999:
        # full circle: do NOT capture at the end angle (would duplicate start)
        captures = moves
    else:
        captures = moves + 1 if cfg.include_end_capture else moves

    return ScanPlan(steps_count=moves, captures_count=captures, actual_step_deg=actual_step)


def _write_jpg_rgb(path: Path, rgb: np.ndarray, quality: int = 85) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if ok:
        path.write_bytes(buf.tobytes())


class ScanController:
    """
    Runs a scan in a background thread.

    You provide:
      - camera (has grab_fresh_frame + set_controls)
      - gpio + laser objects
      - detector + bg model
      - turntable
      - normal_controls, laser_controls (dict or None)
      - a capture_lock (shared with overlay to avoid concurrent laser blinking)
    """

    def __init__(
        self,
        *,
        camera,
        gpio,
        laser1,
        laser2,
        detector: StripeDetector,
        bg: BackgroundModel,
        turntable: Turntable,
        capture_lock: threading.Lock,
        get_profiles_callable,  # function returning (normal_controls, laser_controls)
    ):
        self.camera = camera
        self.gpio = gpio
        self.laser1 = laser1
        self.laser2 = laser2
        self.detector = detector
        self.bg = bg
        self.turntable = turntable
        self.capture_lock = capture_lock
        self.get_profiles = get_profiles_callable

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._status = ScanStatus()
        self._status_lock = threading.Lock()

    # ---------- public API ----------

    def status(self) -> Dict[str, Any]:
        with self._status_lock:
            return asdict(self._status)

    def is_running(self) -> bool:
        with self._status_lock:
            return bool(self._status.running)

    def start(self, cfg: ScanConfig) -> Dict[str, Any]:
        if self.is_running():
            return {"ok": False, "error": "scan already running"}

        plan = compute_plan(cfg)

        if cfg.require_background and not self.bg.is_ready():
            return {"ok": False, "error": "background not captured"}

        run_id = _now_run_id()
        out_dir = Path(cfg.out_root) / run_id
        (out_dir / "points").mkdir(parents=True, exist_ok=True)
        if cfg.save_debug_images:
            (out_dir / "debug").mkdir(parents=True, exist_ok=True)

        # Save config + plan
        (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        (out_dir / "plan.json").write_text(json.dumps(asdict(plan), indent=2))

        with self._status_lock:
            self._status = ScanStatus(
                running=True,
                run_id=run_id,
                out_dir=str(out_dir),
                plan=asdict(plan),
                current_index=0,
                total_captures=plan.captures_count,
                message="Starting...",
                error=None,
            )

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(cfg, plan, out_dir),
            daemon=True,
            name="scan-runner",
        )
        self._thread.start()

        return {"ok": True, "run_id": run_id, "out_dir": str(out_dir), "plan": asdict(plan)}

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        return {"ok": True}

    # ---------- internal ----------

    def _set_status(self, **kwargs):
        with self._status_lock:
            for k, v in kwargs.items():
                setattr(self._status, k, v)

    def _run(self, cfg: ScanConfig, plan: ScanPlan, out_dir: Path):
        try:
            # Always rotate in one direction for backlash reasons
            step_deg = plan.actual_step_deg

            for i in range(plan.captures_count):
                if self._stop.is_set():
                    self._set_status(message="Stopped by user")
                    break

                angle_deg = i * step_deg
                self._set_status(current_index=i, message=f"Capturing {i+1}/{plan.captures_count} @ {angle_deg:.3f}°")

                # ---------- capture section (lock so overlay can't blink lasers at the same time) ----------
                with self.capture_lock:
                    normal_controls, laser_controls = self.get_profiles()

                    # 1) NORMAL frame (for object mask)
                    if normal_controls:
                        self.camera.set_controls(normal_controls)
                    ambient_normal = self.camera.grab_fresh_frame(settle_s=cfg.capture_settle_s)

                    if ambient_normal is None:
                        raise RuntimeError("camera returned None (ambient_normal)")

                    obj_mask = self.bg.foreground_mask(ambient_normal) if self.bg.is_ready() else None

                    # 2) LASER1 pair (laser profile)
                    if laser_controls:
                        self.camera.set_controls(laser_controls)
                    a1, l1 = capture_pair(
                        self.camera, self.gpio, self.laser1,
                        settle_s=cfg.capture_settle_s, drop_n=cfg.drop_n
                    )
                    if a1 is None or l1 is None:
                        raise RuntimeError("camera returned None (laser1 pair)")
                    pts1, _ = self.detector.detect(a1, l1, object_mask=obj_mask)

                    # 3) LASER2 pair
                    a2, l2 = capture_pair(
                        self.camera, self.gpio, self.laser2,
                        settle_s=cfg.capture_settle_s, drop_n=cfg.drop_n
                    )
                    if a2 is None or l2 is None:
                        raise RuntimeError("camera returned None (laser2 pair)")
                    pts2, _ = self.detector.detect(a2, l2, object_mask=obj_mask)

                # ---------- store results ----------
                # Save as numpy arrays (x,y in image coords)
                arr1 = np.array(pts1, dtype=np.float32) if pts1 else np.zeros((0, 2), dtype=np.float32)
                arr2 = np.array(pts2, dtype=np.float32) if pts2 else np.zeros((0, 2), dtype=np.float32)

                np.savez_compressed(
                    out_dir / "points" / f"step_{i:04d}.npz",
                    angle_deg=np.float32(angle_deg),
                    laser1=arr1,
                    laser2=arr2,
                )

                # Optional debug images every N steps
                if cfg.save_debug_images and (i % max(1, int(cfg.save_every_n_steps)) == 0):
                    _write_jpg_rgb(out_dir / "debug" / f"ambient_{i:04d}.jpg", ambient_normal, quality=85)
                    if obj_mask is not None:
                        cv2.imwrite(str(out_dir / "debug" / f"mask_{i:04d}.png"), obj_mask)

                    _write_jpg_rgb(out_dir / "debug" / f"laser1_{i:04d}.jpg", l1, quality=85)
                    _write_jpg_rgb(out_dir / "debug" / f"laser2_{i:04d}.jpg", l2, quality=85)

                # ---------- move to next position ----------
                if i < plan.captures_count - 1:
                    # Move turntable
                    self.turntable.move_deg(step_deg, speed_sps=cfg.move_speed_sps, hold=True)
                    time.sleep(cfg.settle_after_move_s)

            self._set_status(running=False, message="Done")

        except Exception as e:
            self._set_status(running=False, error=f"{type(e).__name__}: {e}", message="Error")
