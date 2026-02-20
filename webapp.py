import time
import threading
import logging
from dataclasses import asdict

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi import Response

from .scan_runner import ScanController, ScanConfig

from .camera_service import CameraService, CameraSettings
from .hardware_io import (
    gpio_open,
    gpio_close,
    stepper_init,
    stepper_enable,
    stepper_step,
    laser_init,
    laser_set,
)
from .scan_algo import StripeDetector, StripeParams, capture_pair, jpeg_with_text
from .background import BackgroundModel, BackgroundParams

from .turntable import Turntable, TurntableConfig
from .calibration_models import CalibrationData
from .calibration_store import CalibrationStore
from .calibration_intrinsics import IntrinsicsCalibrationService
from .calibration_laser import LaserPlaneCalibrationService

log = logging.getLogger(__name__)
app = FastAPI()

# Physical pin 16 = BCM23, physical pin 22 = BCM25
LASER1_PIN = 23
LASER2_PIN = 25

# IMPORTANT: your earlier behavior suggests enable is active-high on your setup
STEPPER_EN_ACTIVE_LOW = False
STEPPER_HOLD_ON_START = True

camera = CameraService(CameraSettings(size=(1280, 720), jpeg_quality=80, fps=25.0))
detector = StripeDetector(StripeParams())

bg = BackgroundModel(BackgroundParams(fg_threshold=25, morph_ksize=5, dilate_iters=1, use_otsu=False))

gpio = None
stepper = None
laser1 = None
laser2 = None
turntable = None

normal_controls = None  # dict or None
laser_controls = None   # dict or None

capture_lock = threading.Lock()
scan_ctl = None

calibration_lock = threading.Lock()
calibration_store = CalibrationStore("calibration/calibration.json")
calibration_data = CalibrationData()
intrinsics_cal = IntrinsicsCalibrationService()
laser_cal = LaserPlaneCalibrationService()

state_lock = threading.Lock()
view_mode = "live"       # "live" or "overlay"
active_laser = 1         # 1 or 2
lasers_enabled = False   # if False => ALWAYS show "LASERS OFF" overlay


@app.on_event("startup")
def startup():
    global gpio, stepper, laser1, laser2, turntable, calibration_data
    
    camera.start()

    gpio = gpio_open()
    stepper = stepper_init(gpio, en_active_low=STEPPER_EN_ACTIVE_LOW)
    laser1 = laser_init(gpio, LASER1_PIN)
    laser2 = laser_init(gpio, LASER2_PIN)

    laser_set(gpio, laser1, False)
    laser_set(gpio, laser2, False)
    
    turntable = Turntable(gpio, stepper, TurntableConfig(
        motor_steps_per_rev=200,
        microsteps=32,
        gear_driver=10,
        gear_driven=66,
    ))

    def _get_profiles():
    # grabs the latest globals
        return normal_controls, laser_controls

    global scan_ctl
    scan_ctl = ScanController(
        camera=camera,
        gpio=gpio,
        laser1=laser1,
        laser2=laser2,
        detector=detector,
        bg=bg,
        turntable=turntable,
        capture_lock=capture_lock,
        get_profiles_callable=_get_profiles,
    )

    try:
        calibration_data = calibration_store.load()
        log.info("Loaded calibration from %s", calibration_store.path)
    except Exception as e:
        calibration_data = calibration_store.default()
        log.exception("Failed to load calibration (%s), using defaults", e)



    if STEPPER_HOLD_ON_START:
        stepper_enable(gpio, stepper, True)


@app.on_event("shutdown")
def shutdown():
    try:
        if gpio and laser1 and laser2:
            laser_set(gpio, laser1, False)
            laser_set(gpio, laser2, False)
        if gpio and stepper:
            stepper_enable(gpio, stepper, False)
        if gpio:
            gpio_close(gpio)
    finally:
        camera.stop()


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Pi Scanner</title></head>
<body style="font-family:sans-serif;margin:16px">
  <h2>Pi Scanner</h2>
  <img src="/view.mjpg" style="max-width:100%;border:1px solid #ccc"/>

  <h3>View</h3>
  <button onclick="post('/api/view/mode?mode=live')">Live</button>
  <button onclick="post('/api/view/mode?mode=overlay')">Overlay</button>

  <h3>Safety</h3>
  <button onclick="post('/api/lasers/enabled?enabled=1')">Allow lasers (overlay blinking)</button>
  <button onclick="post('/api/lasers/enabled?enabled=0')">Lasers OFF (always)</button>

  <h3>Background</h3>
  <button onclick="post('/api/background/capture?n=15')">Capture background (remove object!)</button>
  <a href="/debug/objectmask.jpg" target="_blank">Open debug object mask</a>

  <h3>Camera</h3>
  <button onclick="post('/api/camera/auto')">Auto (AE/AWB + Continuous AF)</button>
  <button onclick="post('/api/camera/af')">Autofocus Now</button>
  <button onclick="post('/api/camera/freeze')">Freeze From Current</button>
  <input id="focus" type="number" step="0.1" min="0" max="10" value="1.5"/>
  <button onclick="post('/api/camera/focus?pos='+document.getElementById('focus').value)">Manual Focus</button>

  <h3>Camera profiles</h3>
  <button onclick="post('/api/camera/profile/save?name=normal')">Save NORMAL profile</button>
  <button onclick="post('/api/camera/profile/save?name=laser')">Save LASER profile</button>
  <button onclick="post('/api/camera/profile/apply?name=normal')">Apply NORMAL</button>
  <button onclick="post('/api/camera/profile/apply?name=laser')">Apply LASER</button>

  <h3>Stepper</h3>
  <button onclick="post('/api/stepper/enable?enabled=1')">Hold ON</button>
  <button onclick="post('/api/stepper/enable?enabled=0')">Hold OFF</button>
  <button onclick="post('/api/step?steps=1600&speed=500&hold=1')">Move ~90° (test)</button>
  <button onclick="post('/api/step?steps=6400&speed=500&hold=1')">Move 360° (test)</button>
  <button onclick="post('/api/step?steps=50&speed=500&hold=1')">Step +50</button>
  <button onclick="post('/api/step?steps=-50&speed=500&hold=1')">Step -50</button>



<script>
async function post(url){
  await fetch(url, {method:'POST'});
}
</script>
</body>
</html>
"""


def run_server(host="0.0.0.0", port=8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


@app.get("/view.mjpg")
def view_stream():
    boundary = b"--frame"

    def gen():
        while True:
            try:
                with state_lock:
                    mode = view_mode
                    laser_sel = active_laser
                    enabled = lasers_enabled

                frame = camera.get_latest_frame()

                if frame is None:
                    time.sleep(0.05)
                    continue

                if not enabled:
                    jpeg = jpeg_with_text(frame, "LASERS OFF", quality=camera.settings.jpeg_quality)
                    delay = 0.05
                else:
                    if mode == "live":
                        jpeg = camera.get_latest_jpeg()
                        delay = 0.02
                    else:
                        # Overlay mode: compute object mask from a normal frame,
                        # then blink laser and detect stripe on laser frames.
                        if normal_controls:
                            camera.set_controls(normal_controls)
                        ambient_normal = camera.grab_fresh_frame(settle_s=0.05)

                        obj_mask = bg.foreground_mask(ambient_normal) if (bg.is_ready() and ambient_normal is not None) else None

                        if laser_controls:
                            camera.set_controls(laser_controls)

                        laser_obj = laser1 if laser_sel == 1 else laser2
                        ambient_laser, laser_frame = capture_pair(camera, gpio, laser_obj, settle_s=0.06, drop_n=2)

                        if ambient_laser is None or laser_frame is None:
                            jpeg = jpeg_with_text(frame, "NO CAMERA FRAME", quality=camera.settings.jpeg_quality)
                        else:
                            pts, _ = detector.detect(ambient_laser, laser_frame, object_mask=obj_mask)
                            if not pts:
                                jpeg = jpeg_with_text(laser_frame, "NO STRIPE", quality=camera.settings.jpeg_quality)
                            else:
                                jpeg = detector.overlay_jpeg(
                                    laser_frame, pts,
                                    quality=camera.settings.jpeg_quality,
                                    object_mask=obj_mask
                                )
                        delay = 0.25

                if not jpeg:
                    time.sleep(delay)
                    continue

                yield boundary + b"\r\n"
                yield b"Content-Type: image/jpeg\r\n"
                yield f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                yield jpeg + b"\r\n"
                time.sleep(delay)

            except Exception as e:
                log.exception("view stream error: %s", e)
                f = camera.get_latest_frame()
                if f is not None:
                    jpeg = jpeg_with_text(f, f"STREAM ERROR: {type(e).__name__}", quality=camera.settings.jpeg_quality)
                    yield boundary + b"\r\n"
                    yield b"Content-Type: image/jpeg\r\n"
                    yield f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    yield jpeg + b"\r\n"
                time.sleep(0.5)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store"})


@app.post("/api/view/mode")
def api_view_mode(mode: str):
    global view_mode
    if mode not in ("live", "overlay"):
        return JSONResponse({"ok": False, "error": "mode must be live or overlay"}, status_code=400)
    with state_lock:
        view_mode = mode
    return {"ok": True, "mode": mode}


@app.post("/api/view/laser")
def api_view_laser(laser: int):
    global active_laser
    if laser not in (1, 2):
        return JSONResponse({"ok": False, "error": "laser must be 1 or 2"}, status_code=400)
    with state_lock:
        active_laser = laser
    return {"ok": True, "laser": laser}


@app.post("/api/lasers/enabled")
def api_lasers_enabled(enabled: int):
    global lasers_enabled
    en = bool(enabled)

    if not en:
        laser_set(gpio, laser1, False)
        laser_set(gpio, laser2, False)

    with state_lock:
        lasers_enabled = en

    return {"ok": True, "enabled": en}


@app.post("/api/background/capture")
def api_background_capture(n: int = 15, settle_s: float = 0.05):
    laser_set(gpio, laser1, False)
    laser_set(gpio, laser2, False)

    frames = []
    n = max(3, min(int(n), 60))
    for _ in range(n):
        f = camera.grab_fresh_frame(settle_s=settle_s)
        if f is not None:
            frames.append(f)

    if len(frames) < 3:
        return JSONResponse({"ok": False, "error": "not enough frames"}, status_code=500)

    bg.build_from_frames(frames)
    return {"ok": True, "frames": len(frames)}


@app.get("/debug/objectmask.jpg")
def debug_objectmask():
    frame = camera.get_latest_frame()
    if frame is None:
        return Response(content=b"", media_type="image/jpeg")

    if not bg.is_ready():
        return Response(
            content=jpeg_with_text(frame, "NO BACKGROUND CAPTURED", quality=camera.settings.jpeg_quality),
            media_type="image/jpeg",
        )

    mask = bg.foreground_mask(frame)
    if mask is None:
        return Response(
            content=jpeg_with_text(frame, "MASK=None", quality=camera.settings.jpeg_quality),
            media_type="image/jpeg",
        )

    nz = int(np.count_nonzero(mask))
    frac = nz / float(mask.shape[0] * mask.shape[1])

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, contours, -1, (255, 0, 0), 2)

    red = np.zeros_like(bgr)
    red[:, :, 2] = mask
    out = cv2.addWeighted(bgr, 1.0, red, 0.4, 0)

    cv2.putText(out, f"mask={nz} ({frac*100:.2f}%)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(camera.settings.jpeg_quality)])
    return Response(content=buf.tobytes() if ok else b"", media_type="image/jpeg")


@app.post("/api/step")
def api_step(steps: int, speed: float = 800.0, hold: int = 1):
    stepper_step(gpio, stepper, int(steps), speed_sps=float(speed), hold=bool(hold))
    return {"ok": True, "position_steps": stepper["position_steps"], "speed_sps": float(speed), "hold": bool(hold)}


@app.post("/api/stepper/enable")
def api_stepper_enable(enabled: int):
    stepper_enable(gpio, stepper, bool(enabled))
    return {"ok": True, "enabled": bool(enabled), "en_active_low": stepper.get("en_active_low")}


@app.post("/api/move_deg")
def api_move_deg(deg: float, speed: float = 800.0, hold: int = 1):
    steps = turntable.move_deg(float(deg), speed_sps=float(speed), hold=bool(hold))
    return {"ok": True, "deg": float(deg), "steps": int(steps), "angle_deg": float(turntable.angle_deg)}



@app.post("/api/laser/{which}/{state}")
def api_laser(which: int, state: str):
    target = laser1 if which == 1 else laser2
    laser_set(gpio, target, state == "on")
    return {"ok": True}


@app.post("/api/camera/auto")
def cam_auto():
    camera.enable_auto()
    return {"ok": True}

@app.post("/api/camera/af")
def cam_af():
    camera.trigger_autofocus()
    return {"ok": True}

@app.post("/api/camera/freeze")
def cam_freeze():
    camera.freeze_from_current()
    return {"ok": True}

@app.post("/api/camera/focus")
def cam_focus(pos: float):
    camera.set_manual_focus(pos)
    return {"ok": True}

@app.get("/api/camera/controls")
def cam_controls():
    return {"ok": True, "controls": camera.get_camera_controls()}

@app.get("/api/camera/meta")
def cam_meta():
    return {"ok": True, "meta": camera.get_latest_metadata() or {}}


@app.post("/api/camera/profile/save")
def cam_save_profile(name: str):
    global normal_controls, laser_controls
    meta = camera.get_latest_metadata() or {}

    # Force frozen behavior when applying this profile later
    prof = {"AeEnable": False, "AwbEnable": False}

    if "ExposureTime" in meta:
        prof["ExposureTime"] = int(meta["ExposureTime"])
    if "AnalogueGain" in meta:
        prof["AnalogueGain"] = float(meta["AnalogueGain"])
    if "ColourGains" in meta:
        prof["ColourGains"] = tuple(meta["ColourGains"])

    # lock focus if available
    if "LensPosition" in meta:
        prof["LensPosition"] = float(meta["LensPosition"])
        prof["AfMode"] = 0  # "manual" in our fallback scheme

    if name == "normal":
        normal_controls = prof
    elif name == "laser":
        laser_controls = prof
    else:
        return JSONResponse({"ok": False, "error": "name must be normal or laser"}, status_code=400)

    return {"ok": True, "name": name, "profile": prof}



def _scan_preflight_errors() -> str | None:
    if not bg.is_ready():
        return "No background captured. Use 'Capture background (remove object!)' first."

    if normal_controls is None:
        return "NORMAL camera profile not saved. Save NORMAL profile first."
    if laser_controls is None:
        return "LASER camera profile not saved. Save LASER profile first."

    # require explicit frozen flags (we enforce when saving above)
    for nm, prof in (("NORMAL", normal_controls), ("LASER", laser_controls)):
        if prof.get("AeEnable", None) is not False:
            return f"{nm} profile missing AeEnable=False"
        if prof.get("AwbEnable", None) is not False:
            return f"{nm} profile missing AwbEnable=False"

    if not lasers_enabled:
        return "Lasers are disabled. Click 'Allow lasers (overlay blinking)' before scanning."

    return None




@app.post("/api/camera/profile/apply")
def cam_apply_profile(name: str):
    prof = normal_controls if name == "normal" else laser_controls if name == "laser" else None
    if prof is None:
        return JSONResponse({"ok": False, "error": f"profile {name} not set"}, status_code=400)
    camera.set_controls(prof)
    return {"ok": True, "name": name}


@app.post("/api/scan/start")
def api_scan_start(
    step_deg: float = 2.0,
    span_deg: float = 360.0,
    speed: float = 320.0,
    save_images: int = 0,
):
    err = _scan_preflight_errors()
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)

    # Optional: force live view while scanning so overlay doesn't compete
    global view_mode
    with state_lock:
        view_mode = "live"

    cfg = ScanConfig(
        requested_step_deg=float(step_deg),
        span_deg=float(span_deg),
        move_speed_sps=float(speed),
        save_debug_images=bool(save_images),
        include_end_capture=True,
        require_background=True,
    )
    return scan_ctl.start(cfg)



@app.post("/api/scan/stop")
def api_scan_stop():
    return scan_ctl.stop()

@app.get("/api/scan/status")
def api_scan_status():
    return scan_ctl.status()


def _calibration_snapshot() -> dict:
    with calibration_lock:
        data = calibration_data
    payload = data.to_dict()
    payload["intrinsics_ready"] = data.intrinsics is not None
    payload["lasers_ready"] = data.laser1 is not None and data.laser2 is not None
    return payload


def _get_intrinsics_matrix() -> np.ndarray:
    with calibration_lock:
        intr = calibration_data.intrinsics
    if intr is None:
        raise RuntimeError("intrinsics not solved or loaded")
    return np.array(intr.camera_matrix, dtype=np.float64)


@app.get("/api/calibration/status")
def api_calibration_status():
    return {
        "ok": True,
        "store_path": str(calibration_store.path),
        "data": _calibration_snapshot(),
        "intrinsics_session": intrinsics_cal.status(),
        "laser_session": laser_cal.status(),
    }


@app.post("/api/calibration/intrinsics/start")
def api_calibration_intrinsics_start(
    board_type: str = "checkerboard",
    checkerboard_cols: int = 9,
    checkerboard_rows: int = 6,
    square_size_m: float = 0.01,
    min_frames: int = 12,
    charuco_squares_x: int = 7,
    charuco_squares_y: int = 5,
    charuco_square_length_m: float = 0.02,
    charuco_marker_length_m: float = 0.015,
    aruco_dict_name: str = "DICT_4X4_50",
):
    try:
        status = intrinsics_cal.start(
            board_type=board_type,
            checkerboard_cols=int(checkerboard_cols),
            checkerboard_rows=int(checkerboard_rows),
            square_size_m=float(square_size_m),
            min_frames=int(min_frames),
            charuco_squares_x=int(charuco_squares_x),
            charuco_squares_y=int(charuco_squares_y),
            charuco_square_length_m=float(charuco_square_length_m),
            charuco_marker_length_m=float(charuco_marker_length_m),
            aruco_dict_name=str(aruco_dict_name),
        )
        return {"ok": True, "status": status}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=400)


@app.post("/api/calibration/intrinsics/capture")
def api_calibration_intrinsics_capture(settle_s: float = 0.05):
    with capture_lock:
        frame = camera.grab_fresh_frame(settle_s=float(settle_s))
    if frame is None:
        return JSONResponse({"ok": False, "error": "camera frame unavailable"}, status_code=500)
    try:
        return intrinsics_cal.capture(frame)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=400)


@app.post("/api/calibration/intrinsics/solve")
def api_calibration_intrinsics_solve():
    global calibration_data
    try:
        result = intrinsics_cal.solve()
        with calibration_lock:
            calibration_data.intrinsics = result
        return {"ok": True, "intrinsics": asdict(result), "quality": result.quality}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=400)


@app.post("/api/calibration/laser/start")
def api_calibration_laser_start(
    board_a: float = 0.0,
    board_b: float = 0.0,
    board_c: float = 1.0,
    board_d: float = -0.30,
    min_points_per_laser: int = 200,
):
    try:
        status = laser_cal.start(
            board_plane=[float(board_a), float(board_b), float(board_c), float(board_d)],
            min_points_per_laser=int(min_points_per_laser),
        )
        return {"ok": True, "status": status}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=400)


@app.post("/api/calibration/laser/capture")
def api_calibration_laser_capture(
    laser: int,
    settle_s: float = 0.06,
    drop_n: int = 2,
):
    if laser not in (1, 2):
        return JSONResponse({"ok": False, "error": "laser must be 1 or 2"}, status_code=400)
    try:
        intrinsics_k = _get_intrinsics_matrix()
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=400)

    target = laser1 if int(laser) == 1 else laser2
    with capture_lock:
        try:
            return laser_cal.capture(
                laser_index=int(laser),
                camera=camera,
                gpio=gpio,
                laser=target,
                detector=detector,
                intrinsics_k=intrinsics_k,
                settle_s=float(settle_s),
                drop_n=int(drop_n),
            )
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=400)


@app.post("/api/calibration/laser/solve")
def api_calibration_laser_solve():
    global calibration_data
    try:
        intrinsics_k = _get_intrinsics_matrix()
        solved = laser_cal.solve(intrinsics_k)
        with calibration_lock:
            calibration_data.laser1 = solved.get(1)
            calibration_data.laser2 = solved.get(2)
        return {
            "ok": True,
            "laser1": asdict(solved[1]),
            "laser2": asdict(solved[2]),
        }
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=400)


@app.post("/api/calibration/save")
def api_calibration_save():
    global calibration_data
    try:
        with calibration_lock:
            saved = calibration_store.save(calibration_data)
            calibration_data = saved
        return {"ok": True, "store_path": str(calibration_store.path), "data": saved.to_dict()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.post("/api/calibration/load")
def api_calibration_load():
    global calibration_data
    try:
        loaded = calibration_store.load()
        with calibration_lock:
            calibration_data = loaded
        return {"ok": True, "store_path": str(calibration_store.path), "data": loaded.to_dict()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.post("/api/calibration/reset")
def api_calibration_reset():
    global calibration_data
    try:
        reset_data = calibration_store.reset()
        with calibration_lock:
            calibration_data = reset_data
        intrinsics_cal.session = None
        intrinsics_cal.last_result = None
        laser_cal.session = None
        laser_cal.last_result = {}
        return {"ok": True, "store_path": str(calibration_store.path), "data": reset_data.to_dict()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)
