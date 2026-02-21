import time
import threading
import logging
import subprocess
import sys
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi import Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

try:
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
    from .scan_algo import StripeDetector, StripeParams, capture_pair_details, jpeg_with_text
    from .background import BackgroundModel, BackgroundParams
    from .turntable import Turntable, TurntableConfig
    from .calibration_models import CalibrationData
    from .calibration_store import CalibrationStore
    from .calibration_intrinsics import IntrinsicsCalibrationService, GuidedCharucoWorkflowService
    from .calibration_laser import LaserPlaneCalibrationService
except ImportError:
    from scan_runner import ScanController, ScanConfig
    from camera_service import CameraService, CameraSettings
    from hardware_io import (
        gpio_open,
        gpio_close,
        stepper_init,
        stepper_enable,
        stepper_step,
        laser_init,
        laser_set,
    )
    from scan_algo import StripeDetector, StripeParams, capture_pair_details, jpeg_with_text
    from background import BackgroundModel, BackgroundParams
    from turntable import Turntable, TurntableConfig
    from calibration_models import CalibrationData
    from calibration_store import CalibrationStore
    from calibration_intrinsics import IntrinsicsCalibrationService, GuidedCharucoWorkflowService
    from calibration_laser import LaserPlaneCalibrationService

log = logging.getLogger(__name__)
app = FastAPI()

RUNS_ROOT = Path("runs")
VIEWER_ROOT = Path("viewer")
EXPORT_TOOL_PATH = Path("tools/export_run_to_json.py")

app.mount("/viewer", StaticFiles(directory=str(VIEWER_ROOT), html=True), name="viewer")

# Serve run export JSON files so the viewer "open export" links work
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/runs", StaticFiles(directory=str(RUNS_ROOT)), name="runs")

# Physical pin 16 = BCM23, physical pin 22 = BCM25
LASER1_PIN = 23
LASER2_PIN = 25

# IMPORTANT: your earlier behavior suggests enable is active-high on your setup
STEPPER_EN_ACTIVE_LOW = False
STEPPER_HOLD_ON_START = True

MOCK_HW = str(os.getenv("SCANNER_MOCK_HW", "")).strip().lower() in {"1", "true", "yes", "on"}
ENABLE_LEGACY_BACKGROUND_PATH = str(os.getenv("SCANNER_ENABLE_LEGACY_BACKGROUND_PATH", "")).strip().lower() in {"1", "true", "yes", "on"}

camera = CameraService(CameraSettings(size=(1280, 720), jpeg_quality=80, fps=25.0, rotation_degrees=180, mock=MOCK_HW))
detector = StripeDetector(StripeParams())

bg = BackgroundModel(BackgroundParams(fg_threshold=25, morph_ksize=5, dilate_iters=1, use_otsu=False))

gpio = None
stepper = None
laser1 = None
laser2 = None
turntable = None

normal_controls = None  # dict or None
laser_controls = None   # dict or None
camera_profiles: dict[str, dict[str, Any]] = {
    "normal": {},
    "laser": {},
}


def _normalize_controls_dict(src: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (src or {}).items():
        if v is None:
            continue
        if k == "ColourGains" and isinstance(v, (list, tuple)) and len(v) >= 2:
            out[k] = (float(v[0]), float(v[1]))
            continue
        if isinstance(v, (int, float, bool, tuple, list, str)):
            out[k] = v
    return out


def _sync_profile_aliases() -> None:
    global normal_controls, laser_controls
    normal = dict(camera_profiles.get("normal", {}) or {})
    laser = dict(camera_profiles.get("laser", {}) or {})
    normal_controls = normal if normal else None
    laser_controls = laser if laser else None


def _build_profile_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    prof: dict[str, Any] = {"AeEnable": False, "AwbEnable": False}
    if "ExposureTime" in meta:
        prof["ExposureTime"] = int(meta["ExposureTime"])
    if "AnalogueGain" in meta:
        prof["AnalogueGain"] = float(meta["AnalogueGain"])
    if "ColourGains" in meta:
        cg = meta["ColourGains"]
        if isinstance(cg, (list, tuple)) and len(cg) >= 2:
            prof["ColourGains"] = (float(cg[0]), float(cg[1]))
    if "LensPosition" in meta:
        prof["LensPosition"] = float(meta["LensPosition"])
        prof["AfMode"] = 0
    return prof


def _camera_state_payload() -> dict[str, Any]:
    meta = camera.get_latest_metadata() or {}
    return {
        "ok": True,
        "controls": camera.get_camera_controls(),
        "meta": meta,
        "profiles": {
            "normal": dict(camera_profiles.get("normal", {}) or {}),
            "laser": dict(camera_profiles.get("laser", {}) or {}),
        },
        "active_profiles": {
            "normal_set": bool(camera_profiles.get("normal")),
            "laser_set": bool(camera_profiles.get("laser")),
        },
    }

capture_lock = threading.Lock()
scan_ctl = None

calibration_lock = threading.Lock()
calibration_store = CalibrationStore("calibration/calibration.json")
calibration_data = CalibrationData()
intrinsics_cal = IntrinsicsCalibrationService()
guided_charuco = GuidedCharucoWorkflowService(intrinsics_cal)
laser_cal = LaserPlaneCalibrationService()

state_lock = threading.Lock()
view_mode = "live"       # "live" or "overlay"
active_laser = 1         # 1 or 2
lasers_enabled = False   # if False => ALWAYS show "LASERS OFF" overlay
latest_detector_telemetry: dict[str, Any] = {}


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
        _sync_profile_aliases()
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

    log.info("startup complete (mock_hw=%s)", MOCK_HW)


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
  <p><a href="/viewer/" target="_blank">Open viewer</a></p>

  <h3>View</h3>
  <button onclick="post('/api/view/mode?mode=live')">Live</button>
  <button onclick="post('/api/view/mode?mode=overlay')">Overlay</button>

  <h3>Safety</h3>
  <button onclick="post('/api/lasers/enabled?enabled=1')">Allow lasers (overlay blinking)</button>
  <button onclick="post('/api/lasers/enabled?enabled=0')">Lasers OFF (always)</button>

  <h3>Background (optional legacy path)</h3>
  <button onclick="post('/api/background/capture?n=15')">Capture legacy background (optional)</button>
  <a href="/debug/objectmask.jpg" target="_blank">Open debug object mask</a>

  <h3>Camera controls</h3>
  <div style="display:grid;grid-template-columns:repeat(4,minmax(140px,1fr));gap:8px;max-width:1100px">
    <label>ExposureTime (us)<input id="cam_exposure" type="number" step="100" min="1" value="10000"/></label>
    <label>AnalogueGain<input id="cam_gain" type="number" step="0.01" min="0" value="1.0"/></label>
    <label>AE Enable<select id="cam_ae"><option value="1">true</option><option value="0">false</option></select></label>
    <label>AWB Enable<select id="cam_awb"><option value="1">true</option><option value="0">false</option></select></label>

    <label>ColourGain Red<input id="cam_cg_r" type="number" step="0.01" min="0" value="1.0"/></label>
    <label>ColourGain Blue<input id="cam_cg_b" type="number" step="0.01" min="0" value="1.0"/></label>
    <label>Brightness<input id="cam_brightness" type="number" step="0.01" value="0.0"/></label>
    <label>Contrast<input id="cam_contrast" type="number" step="0.01" value="1.0"/></label>

    <label>Saturation<input id="cam_saturation" type="number" step="0.01" value="1.0"/></label>
    <label>Sharpness<input id="cam_sharpness" type="number" step="0.01" value="1.0"/></label>
    <label>LensPosition<input id="cam_lens" type="number" step="0.01" min="0" max="10" value="1.5"/></label>
    <label>Focus (legacy)<input id="focus" type="number" step="0.1" min="0" max="10" value="1.5"/></label>
  </div>
  <p>
    <button onclick="readCameraState()">Read current</button>
    <button onclick="applyCameraControls()">Apply controls</button>
    <button onclick="post('/api/camera/auto')">Auto (AE/AWB + Continuous AF)</button>
    <button onclick="post('/api/camera/freeze')">Freeze from current</button>
    <button onclick="post('/api/camera/af')">Autofocus now</button>
    <button onclick="post('/api/camera/focus?pos='+q('focus').value)">Manual focus (legacy)</button>
  </p>

  <h3>Camera profiles</h3>
  <button onclick="saveProfile('normal')">Save NORMAL profile</button>
  <button onclick="saveProfile('laser')">Save LASER profile</button>
  <button onclick="loadProfile('normal')">Load NORMAL</button>
  <button onclick="loadProfile('laser')">Load LASER</button>
  <pre id="cam_state" style="background:#f6f6f6;padding:8px;border:1px solid #ddd;max-width:980px;white-space:pre-wrap"></pre>

  <h3>Stepper</h3>
  <button onclick="post('/api/stepper/enable?enabled=1')">Hold ON</button>
  <button onclick="post('/api/stepper/enable?enabled=0')">Hold OFF</button>
  <button onclick="post('/api/stepper/polarity?en_active_low=0')">Polarity: active HIGH</button>
  <button onclick="post('/api/stepper/polarity?en_active_low=1')">Polarity: active LOW</button>
  <button onclick="post('/api/step?steps=1600&speed=500&hold=1')">Move ~90° (test)</button>
  <button onclick="post('/api/step?steps=6400&speed=500&hold=1')">Move 360° (test)</button>
  <button onclick="post('/api/step?steps=50&speed=500&hold=1')">Step +50</button>
  <button onclick="post('/api/step?steps=-50&speed=500&hold=1')">Step -50</button>

  <h3>Scan</h3>
  <label>Step deg <input id="scan_step_deg" type="number" step="0.1" min="0.1" value="2.0"/></label>
  <label>Span deg <input id="scan_span_deg" type="number" step="1" min="1" max="360" value="360"/></label>
  <label>Speed <input id="scan_speed" type="number" step="10" min="10" value="320"/></label>
  <label>Save images <input id="scan_save_images" type="checkbox"/></label>
  <button onclick="startScan()">Start scan</button>
  <button onclick="post('/api/scan/stop')">Stop scan</button>
  <pre id="scan_status" style="background:#f6f6f6;padding:8px;border:1px solid #ddd;max-width:980px;white-space:pre-wrap"></pre>

  <h3>Detector diagnostics</h3>
  <pre id="detector_diag" style="background:#f6f6f6;padding:8px;border:1px solid #ddd;max-width:980px;white-space:pre-wrap"></pre>

  <h3>Guided Camera Calibration</h3>
  <p style="max-width:900px">Follow the guided steps. Move the board to a new angle each time, keep the full board visible, then press <b>Capture Step</b>.</p>
  <p>
    <label for="charuco_total_steps"><b>Number of steps</b></label>
    <input id="charuco_total_steps" type="number" min="5" max="60" step="1" value="10" style="width:90px"/>
    <button onclick="startManualCharucoWorkflow()">Start</button>
    <button onclick="captureManualCharucoStep()">Capture Step</button>
    <button onclick="solveManualCharucoWorkflow()">Finish &amp; Check</button>
  </p>
  <p id="charuco_progress" style="font-size:1.2em;font-weight:700;margin:10px 0">Step 0 of 0</p>
  <p id="charuco_instruction">Move the board to a new angle, keep it fully visible, then press Capture Step.</p>
  <p id="charuco_step_feedback" style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #ccc;font-weight:700;background:#f3f4f6;color:#111827">Not started</p>

  <h3>Runs</h3>
  <button onclick="loadRuns()">Refresh runs</button>
  <div id="runs_list"></div>



<script>
async function post(url){
  const r = await fetch(url, {method:'POST'});
  const j = await r.json().catch(()=>({ok:false,error:'non-json response'}));
  if(!j.ok){
    console.error('POST failed', url, j);
  }
  await refreshScanStatus();
  await loadRuns();
  return j;
}

function q(id){ return document.getElementById(id); }

async function startScan(){
  const stepDeg = encodeURIComponent(q('scan_step_deg').value);
  const spanDeg = encodeURIComponent(q('scan_span_deg').value);
  const speed = encodeURIComponent(q('scan_speed').value);
  const saveImages = q('scan_save_images').checked ? 1 : 0;
  await post(`/api/scan/start?step_deg=${stepDeg}&span_deg=${spanDeg}&speed=${speed}&save_images=${saveImages}`);
}

function toNum(id){
  const v = Number(q(id).value);
  return Number.isFinite(v) ? v : null;
}

function fillIfFinite(id, v){
  if(typeof v === 'number' && Number.isFinite(v)) q(id).value = String(v);
}

async function readCameraState(){
  const target = q('cam_state');
  try{
    const r = await fetch('/api/camera/state');
    const j = await r.json();
    const meta = j.meta || {};
    const controls = j.controls || {};
    const cg = meta.ColourGains || controls.ColourGains || [1.0, 1.0];

    fillIfFinite('cam_exposure', Number(meta.ExposureTime ?? controls.ExposureTime));
    fillIfFinite('cam_gain', Number(meta.AnalogueGain ?? controls.AnalogueGain));
    fillIfFinite('cam_cg_r', Number(Array.isArray(cg) ? cg[0] : 1.0));
    fillIfFinite('cam_cg_b', Number(Array.isArray(cg) ? cg[1] : 1.0));
    fillIfFinite('cam_lens', Number(meta.LensPosition ?? controls.LensPosition));
    fillIfFinite('focus', Number(meta.LensPosition ?? controls.LensPosition));
    fillIfFinite('cam_brightness', Number(meta.Brightness ?? controls.Brightness));
    fillIfFinite('cam_contrast', Number(meta.Contrast ?? controls.Contrast));
    fillIfFinite('cam_saturation', Number(meta.Saturation ?? controls.Saturation));
    fillIfFinite('cam_sharpness', Number(meta.Sharpness ?? controls.Sharpness));

    const ae = meta.AeEnable ?? controls.AeEnable;
    const awb = meta.AwbEnable ?? controls.AwbEnable;
    q('cam_ae').value = (ae === false || ae === 0) ? '0' : '1';
    q('cam_awb').value = (awb === false || awb === 0) ? '0' : '1';

    target.textContent = JSON.stringify(j, null, 2);
  }catch(e){
    target.textContent = 'camera state fetch failed: ' + e;
  }
}

async function applyCameraControls(){
  const payload = {
    ExposureTime: toNum('cam_exposure'),
    AnalogueGain: toNum('cam_gain'),
    AeEnable: q('cam_ae').value === '1',
    AwbEnable: q('cam_awb').value === '1',
    ColourGains: [toNum('cam_cg_r') ?? 1.0, toNum('cam_cg_b') ?? 1.0],
    Brightness: toNum('cam_brightness'),
    Contrast: toNum('cam_contrast'),
    Saturation: toNum('cam_saturation'),
    Sharpness: toNum('cam_sharpness'),
    LensPosition: toNum('cam_lens'),
  };
  Object.keys(payload).forEach((k) => {
    if(payload[k] === null || payload[k] === undefined) delete payload[k];
  });

  const r = await fetch('/api/camera/controls/apply', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(payload),
  });
  const j = await r.json().catch(()=>({ok:false,error:'non-json response'}));
  if(!j.ok) console.error('apply controls failed', j);
  await readCameraState();
  return j;
}

async function saveProfile(name){
  await post('/api/camera/profile/save?name=' + encodeURIComponent(name));
  await readCameraState();
}

async function loadProfile(name){
  await post('/api/camera/profile/load?name=' + encodeURIComponent(name));
  await readCameraState();
}

async function refreshScanStatus(){
  const target = q('scan_status');
  try{
    const r = await fetch('/api/scan/status');
    const j = await r.json();
    target.textContent = JSON.stringify(j, null, 2);
  }catch(e){
    target.textContent = 'status fetch failed: ' + e;
  }
}

async function refreshDetectorDiag(){
  const target = q('detector_diag');
  try{
    const r = await fetch('/api/detector/telemetry');
    const j = await r.json();
    target.textContent = JSON.stringify(j, null, 2);
  }catch(e){
    target.textContent = 'detector telemetry fetch failed: ' + e;
  }
}

async function loadRuns(){
  const target = q('runs_list');
  try{
    const r = await fetch('/api/runs');
    const j = await r.json();
    if(!j.ok){
      target.textContent = j.error || 'failed to load runs';
      return;
    }
    if(!j.runs || !j.runs.length){
      target.textContent = 'No runs found.';
      return;
    }
    target.innerHTML = j.runs.map(run => {
      const exportFile = run.export_exists ? `<a href="/${run.export_relpath}" target="_blank">open export</a>` : 'no export';
      return `
        <div style="margin:6px 0;padding:6px;border:1px solid #ddd">
          <b>${run.run_id}</b> | steps: ${run.step_files} | ${exportFile}
          <button onclick="post('/api/runs/${run.run_id}/export')">Export</button>
        </div>
      `;
    }).join('');
  }catch(e){
    target.textContent = 'runs fetch failed: ' + e;
  }
}

function renderManualCharucoStatus(payload){
  const status = payload?.status || payload || {};
  const current = Number(status.current_step || 0);
  const total = Number(status.total_steps || 0);
  const defaultStepText = `Step ${Math.max(0, current)} of ${Math.max(0, total)}`;
  q('charuco_progress').textContent = String(status.step_text || payload?.step_text || defaultStepText);
  q('charuco_instruction').textContent = String(
    status.instruction ||
    'Move the board to a new angle, keep it fully visible, then press Capture Step.'
  );

  const feedback = q('charuco_step_feedback');
  const msg = String(status.last_step_message || payload?.last_step_message || 'Not started');
  const ok = Boolean(status.last_step_ok ?? payload?.last_step_ok);
  feedback.textContent = msg;
  if(msg.toLowerCase().includes('not started')){
    feedback.style.background = '#f3f4f6';
    feedback.style.color = '#111827';
    feedback.style.borderColor = '#d1d5db';
  }else if(ok){
    feedback.style.background = '#dcfce7';
    feedback.style.color = '#166534';
    feedback.style.borderColor = '#22c55e';
  }else{
    feedback.style.background = '#fee2e2';
    feedback.style.color = '#991b1b';
    feedback.style.borderColor = '#ef4444';
  }
}

async function refreshManualCharucoStatus(){
  try{
    const r = await fetch('/api/calibration/intrinsics/charuco-manual/status');
    const j = await r.json();
    renderManualCharucoStatus(j);
  }catch(e){
    q('charuco_step_feedback').textContent = 'Could not refresh guided status. Please retry.';
    q('charuco_step_feedback').style.background = '#fee2e2';
    q('charuco_step_feedback').style.color = '#991b1b';
    q('charuco_step_feedback').style.borderColor = '#ef4444';
  }
}

async function startManualCharucoWorkflow(){
  const rawSteps = Number(q('charuco_total_steps').value);
  const safeSteps = Number.isFinite(rawSteps) ? Math.max(5, Math.min(60, Math.round(rawSteps))) : 10;
  q('charuco_total_steps').value = String(safeSteps);
  const r = await fetch(`/api/calibration/intrinsics/charuco-manual/start?total_steps=${encodeURIComponent(safeSteps)}`, {method:'POST'});
  const j = await r.json().catch(()=>({ok:false,error:'non-json response'}));
  renderManualCharucoStatus(j);
  return j;
}

async function captureManualCharucoStep(){
  const r = await fetch('/api/calibration/intrinsics/charuco-manual/capture', {method:'POST'});
  const j = await r.json().catch(()=>({ok:false,error:'non-json response'}));
  renderManualCharucoStatus(j);
  return j;
}

async function solveManualCharucoWorkflow(){
  const r = await fetch('/api/calibration/intrinsics/charuco-manual/solve', {method:'POST'});
  const j = await r.json().catch(()=>({ok:false,error:'non-json response'}));
  renderManualCharucoStatus(j);
  return j;
}

refreshScanStatus();
refreshDetectorDiag();
loadRuns();
readCameraState();
refreshManualCharucoStatus();
setInterval(refreshScanStatus, 1000);
setInterval(refreshDetectorDiag, 1000);
setInterval(readCameraState, 2000);
setInterval(refreshManualCharucoStatus, 1000);
</script>
</body>
</html>
"""


@app.get("/viewer-home")
def viewer_home_redirect():
    return RedirectResponse(url="/viewer/")


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
                        pair = capture_pair_details(
                            camera,
                            gpio,
                            laser_obj,
                            settle_s=0.06,
                            drop_n=2,
                            off_controls=normal_controls,
                            on_controls=laser_controls,
                            retry_n=1,
                            max_pair_retries=2,
                        )
                        ambient_laser, laser_frame = pair.ambient, pair.laser_frame

                        if ambient_laser is None or laser_frame is None:
                            jpeg = jpeg_with_text(frame, "NO CAMERA FRAME", quality=camera.settings.jpeg_quality)
                        else:
                            det = detector.detect_details(ambient_laser, laser_frame, object_mask=obj_mask, mode="scan")
                            with state_lock:
                                latest_detector_telemetry.clear()
                                latest_detector_telemetry.update({
                                    "source": "overlay_preview",
                                    "active_laser": int(laser_sel),
                                    "confidence": float(det.confidence),
                                    "pair": {
                                        "stable": bool(pair.stable),
                                        "attempts": int(pair.attempts),
                                        "drift": dict(pair.drift or {}),
                                    },
                                    "telemetry": dict(det.telemetry),
                                })
                            if not det.points:
                                jpeg = jpeg_with_text(laser_frame, "NO STRIPE", quality=camera.settings.jpeg_quality)
                            else:
                                jpeg = detector.overlay_result_jpeg(
                                    laser_frame,
                                    det,
                                    quality=camera.settings.jpeg_quality,
                                    object_mask=obj_mask,
                                    title=f"L{laser_sel} conf={det.confidence:.3f}",
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
    if not ENABLE_LEGACY_BACKGROUND_PATH:
        return JSONResponse({"ok": False, "error": "legacy background path disabled by feature flag"}, status_code=404)
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
    if not ENABLE_LEGACY_BACKGROUND_PATH:
        f = camera.get_latest_frame()
        if f is None:
            return Response(content=b"", media_type="image/jpeg")
        return Response(
            content=jpeg_with_text(f, "LEGACY BACKGROUND PATH DISABLED", quality=camera.settings.jpeg_quality),
            media_type="image/jpeg",
        )
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


@app.get("/api/stepper/polarity")
def api_stepper_polarity_get():
    active_low = bool(stepper.get("en_active_low", False))
    return {
        "ok": True,
        "en_active_low": active_low,
        "enabled_level": 0 if active_low else 1,
        "disabled_level": 1 if active_low else 0,
    }


@app.post("/api/stepper/polarity")
def api_stepper_polarity_set(en_active_low: int):
    new_value = bool(en_active_low)
    old_value = bool(stepper.get("en_active_low", False))
    stepper["en_active_low"] = new_value
    return {
        "ok": True,
        "changed": old_value != new_value,
        "en_active_low": new_value,
        "enabled_level": 0 if new_value else 1,
        "disabled_level": 1 if new_value else 0,
    }


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
    return _camera_state_payload()


@app.get("/api/camera/state")
def cam_state():
    return _camera_state_payload()

@app.get("/api/camera/meta")
def cam_meta():
    return {"ok": True, "meta": camera.get_latest_metadata() or {}}


@app.post("/api/camera/ae")
def cam_ae(enabled: int):
    camera.set_controls({"AeEnable": bool(enabled)})
    return {"ok": True, "AeEnable": bool(enabled)}


@app.post("/api/camera/awb")
def cam_awb(enabled: int):
    camera.set_controls({"AwbEnable": bool(enabled)})
    return {"ok": True, "AwbEnable": bool(enabled)}


@app.post("/api/camera/controls/apply")
def cam_apply_controls(payload: dict[str, Any] = Body(default_factory=dict)):
    if not isinstance(payload, dict):
        return JSONResponse({"ok": False, "error": "JSON object payload expected"}, status_code=400)
    controls = _normalize_controls_dict(payload)
    if not controls:
        return JSONResponse({"ok": False, "error": "no supported controls supplied"}, status_code=400)
    camera.set_controls(controls)
    return {"ok": True, "applied": controls, "state": _camera_state_payload()}


@app.get("/api/camera/profiles")
def cam_profiles():
    return {
        "ok": True,
        "profiles": {
            "normal": dict(camera_profiles.get("normal", {}) or {}),
            "laser": dict(camera_profiles.get("laser", {}) or {}),
        },
    }


@app.post("/api/camera/profile/save")
def cam_save_profile(name: str):
    global normal_controls, laser_controls
    key = str(name).strip().lower()
    if key not in {"normal", "laser"}:
        return JSONResponse({"ok": False, "error": "name must be normal or laser"}, status_code=400)
    meta = camera.get_latest_metadata() or {}
    prof = _build_profile_from_meta(meta)
    camera_profiles[key] = _normalize_controls_dict(prof)
    _sync_profile_aliases()
    return {"ok": True, "name": key, "profile": dict(camera_profiles[key])}


@app.post("/api/camera/profile/load")
def cam_load_profile(name: str):
    key = str(name).strip().lower()
    if key not in {"normal", "laser"}:
        return JSONResponse({"ok": False, "error": "name must be normal or laser"}, status_code=400)
    prof = dict(camera_profiles.get(key, {}) or {})
    if not prof:
        return JSONResponse({"ok": False, "error": f"profile {key} not set"}, status_code=400)
    camera.set_controls(prof)
    return {"ok": True, "name": key, "profile": prof}


@app.get("/api/system/mode")
def api_system_mode():
    return {
        "ok": True,
        "mock_hw": bool(MOCK_HW),
        "legacy_background_path_enabled": bool(ENABLE_LEGACY_BACKGROUND_PATH),
    }



def _scan_preflight_errors() -> str | None:
    if ENABLE_LEGACY_BACKGROUND_PATH and not bg.is_ready():
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


@app.get("/api/detector/telemetry")
def api_detector_telemetry():
    payload = scan_ctl.detector_telemetry() if scan_ctl is not None else {}
    with state_lock:
        preview = dict(latest_detector_telemetry)
    return {
        "ok": True,
        "scan": payload,
        "preview": preview,
    }




@app.post("/api/camera/profile/apply")
def cam_apply_profile(name: str):
    key = str(name).strip().lower()
    prof = dict(camera_profiles.get(key, {}) or {})
    if not prof:
        return JSONResponse({"ok": False, "error": f"profile {key} not set"}, status_code=400)
    camera.set_controls(prof)
    return {"ok": True, "name": key, "profile": prof}


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
        require_background=bool(ENABLE_LEGACY_BACKGROUND_PATH),
        min_detection_confidence=0.12,
    )
    return scan_ctl.start(cfg)



@app.post("/api/scan/stop")
def api_scan_stop():
    return scan_ctl.stop()

@app.get("/api/scan/status")
def api_scan_status():
    return scan_ctl.status()


def _iter_run_dirs(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    runs = [p for p in root.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs


def _run_summary(run_dir: Path) -> dict:
    points_dir = run_dir / "points"
    step_files = list(points_dir.glob("step_*.npz")) if points_dir.exists() else []
    export_path = run_dir / "viewer_export.json"
    return {
        "run_id": run_dir.name,
        "run_relpath": run_dir.as_posix(),
        "points_relpath": points_dir.as_posix(),
        "step_files": len(step_files),
        "export_exists": export_path.exists(),
        "export_relpath": export_path.as_posix(),
    }


@app.get("/api/runs")
def api_runs_list(limit: int = 100):
    lim = max(1, min(int(limit), 1000))
    runs = _iter_run_dirs(RUNS_ROOT)
    payload = []
    for rd in runs:
        summary = _run_summary(rd)
        if int(summary["step_files"]) > 0:
            payload.append(summary)
        if len(payload) >= lim:
            break
    return {"ok": True, "count": len(payload), "runs": payload}


@app.post("/api/runs/{run_id}/export")
def api_runs_export(run_id: str, output_name: str = "viewer_export.json"):
    if not run_id or "/" in run_id or "\\" in run_id:
        return JSONResponse({"ok": False, "error": "invalid run_id"}, status_code=400)
    if not output_name or "/" in output_name or "\\" in output_name:
        return JSONResponse({"ok": False, "error": "invalid output_name"}, status_code=400)
    if not output_name.endswith(".json"):
        return JSONResponse({"ok": False, "error": "output_name must end with .json"}, status_code=400)

    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        return JSONResponse({"ok": False, "error": f"run not found: {run_id}"}, status_code=404)

    output_path = run_dir / output_name
    cmd = [
        sys.executable,
        str(EXPORT_TOOL_PATH),
        str(run_dir),
        "--output",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        return JSONResponse(
            {
                "ok": False,
                "run_id": run_id,
                "out_path": output_path.as_posix(),
                "error": (proc.stderr or proc.stdout or "export failed").strip(),
                "returncode": proc.returncode,
            },
            status_code=500,
        )

    return {
        "ok": True,
        "run_id": run_id,
        "out_path": output_path.as_posix(),
        "exists": output_path.exists(),
        "stdout": (proc.stdout or "").strip(),
    }


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
        "guided_charuco_session": guided_charuco.status(),
        "laser_session": laser_cal.status(),
    }


@app.post("/api/calibration/intrinsics/charuco-manual/start")
def api_calibration_intrinsics_charuco_manual_start(
    total_steps: int = 10,
    min_frames: int = 20,
    charuco_squares_x: int = 8,
    charuco_squares_y: int = 8,
    charuco_square_length_m: float = 0.015,
    charuco_marker_length_m: float = 0.011,
    aruco_dict_name: str = "DICT_4X4_50",
):
    try:
        safe_total_steps = max(5, min(int(total_steps), 60))
        status = guided_charuco.start(
            total_steps=safe_total_steps,
            min_frames_required=int(min_frames),
            charuco_squares_x=int(charuco_squares_x),
            charuco_squares_y=int(charuco_squares_y),
            charuco_square_length_m=float(charuco_square_length_m),
            charuco_marker_length_m=float(charuco_marker_length_m),
            aruco_dict_name=str(aruco_dict_name),
        )
        return {
            "ok": True,
            "status": status,
            "step_text": str(status.get("step_text", "Step 0 of 0")),
            "last_step_ok": bool(status.get("last_step_ok", False)),
            "last_step_message": str(status.get("last_step_message", "Workflow started.")),
        }
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=400)


@app.get("/api/calibration/intrinsics/charuco-manual/status")
def api_calibration_intrinsics_charuco_manual_status():
    status = guided_charuco.status()
    return {
        "ok": True,
        "status": status,
        "step_text": str(status.get("step_text", "Step 0 of 0")),
        "last_step_ok": bool(status.get("last_step_ok", False)),
        "last_step_message": str(status.get("last_step_message", "Workflow not started yet.")),
    }


@app.post("/api/calibration/intrinsics/charuco-manual/capture")
def api_calibration_intrinsics_charuco_manual_capture(settle_s: float = 0.05):
    global calibration_data
    with capture_lock:
        frame = camera.grab_fresh_frame(settle_s=float(settle_s))
    if frame is None:
        return JSONResponse({"ok": False, "error": "camera frame unavailable"}, status_code=500)
    try:
        payload = guided_charuco.capture_step(frame)
        status_dbg = payload.get("status") if isinstance(payload, dict) else {}
        if isinstance(status_dbg, dict):
            log.info(
                "guided.charuco.capture accepted=%s reason=%s step=%s/%s attempted=%s used=%s",
                payload.get("accepted"),
                payload.get("reason"),
                status_dbg.get("current_step"),
                status_dbg.get("total_steps"),
                status_dbg.get("captures_attempted"),
                status_dbg.get("accepted_frames"),
            )
        intrinsics_obj = payload.get("intrinsics")
        if intrinsics_obj is not None:
            with calibration_lock:
                calibration_data.intrinsics = intrinsics_obj
                calibration_data = calibration_store.save(calibration_data)
            payload["intrinsics"] = asdict(intrinsics_obj)
            payload["persisted"] = True
            payload["store_path"] = str(calibration_store.path)
        status = payload.get("status") if isinstance(payload, dict) else {}
        if isinstance(status, dict):
            payload["step_text"] = str(status.get("step_text", "Step 0 of 0"))
            payload["last_step_ok"] = bool(status.get("last_step_ok", False))
            payload["last_step_message"] = str(status.get("last_step_message", "Capture complete."))
        return payload
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=400)


@app.post("/api/calibration/intrinsics/charuco-manual/solve")
def api_calibration_intrinsics_charuco_manual_solve():
    global calibration_data
    try:
        payload = guided_charuco.solve_final()
        intrinsics_obj = payload.get("intrinsics")
        if intrinsics_obj is not None:
            with calibration_lock:
                calibration_data.intrinsics = intrinsics_obj
                calibration_data = calibration_store.save(calibration_data)
            payload["intrinsics"] = asdict(intrinsics_obj)
            payload["persisted"] = True
            payload["store_path"] = str(calibration_store.path)
        status = payload.get("status") if isinstance(payload, dict) else {}
        if isinstance(status, dict):
            payload["step_text"] = str(status.get("step_text", "Step 0 of 0"))
            payload["last_step_ok"] = bool(status.get("last_step_ok", False))
            payload["last_step_message"] = str(status.get("last_step_message", "Finished."))
        return payload
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=400)


@app.post("/api/calibration/intrinsics/start")
def api_calibration_intrinsics_start(
    board_type: str = "checkerboard",
    checkerboard_cols: int = 9,
    checkerboard_rows: int = 6,
    square_size_m: float = 0.01,
    min_frames: int = 12,
    charuco_squares_x: int = 8,
    charuco_squares_y: int = 8,
    charuco_square_length_m: float = 0.015,
    charuco_marker_length_m: float = 0.011,
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
        guided_charuco.session = None
        guided_charuco.last_intrinsics = None
        laser_cal.session = None
        laser_cal.last_result = {}
        return {"ok": True, "store_path": str(calibration_store.path), "data": reset_data.to_dict()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)
