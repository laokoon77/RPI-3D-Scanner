# RPi Laser 3D Scanner (Headless Ops Guide)

This project runs a Raspberry Pi laser-line scanner with:

- capture + turntable control
- browser control UI
- calibration persistence
- scan run management
- run export for the included WebGL viewer

Primary server module: [`webapp.py`](webapp.py)

---

## 1) Install dependencies

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependency manifest: [`requirements.txt`](requirements.txt)

Notes for Raspberry Pi:

- `picamera2` requires a Pi camera/libcamera environment.
- `lgpio` requires GPIO access (typically run with permissions that allow `/dev/gpiochip*`).

---

## 2) Start the scanner web server

### 2.1 One-command launcher (recommended)

From project root, run:

```bash
python start_scanner.py
```

Launcher script: [`start_scanner.py`](start_scanner.py)

Defaults:

- host: `0.0.0.0`
- port: `8000`

Optional environment overrides:

- `SCANNER_HOST` (example: `127.0.0.1`)
- `SCANNER_PORT` (example: `9000`)
- `SCANNER_MOCK_HW` (optional, enables hardwareless mode when set to `1/true/yes/on`)
- `SCANNER_ENABLE_LEGACY_BACKGROUND_PATH` (optional, enables legacy background flow when set to `1/true/yes/on`)

Camera orientation behavior:

- Live preview/capture/calibration/scan frames are rotated by 180° by default via [`CameraSettings.rotation_degrees`](camera_service.py:35).
- This keeps stream and saved/calibration frames orientation-consistent.

Windows `cmd.exe` example with overrides:

```bash
set SCANNER_MOCK_HW=1 && set SCANNER_ENABLE_LEGACY_BACKGROUND_PATH=0 && set SCANNER_HOST=127.0.0.1 && set SCANNER_PORT=8000 && python start_scanner.py
```

Linux/macOS example with overrides:

```bash
SCANNER_MOCK_HW=1 SCANNER_ENABLE_LEGACY_BACKGROUND_PATH=0 SCANNER_HOST=127.0.0.1 SCANNER_PORT=8000 python start_scanner.py
```

### 2.2 Existing startup commands (still supported)

Use the CLI entry in [`cli.py`](cli.py):

```bash
python -m scanner serve --host 0.0.0.0 --port 8000
```

Then open:

- scanner UI: `http://<pi-ip>:8000/`
- viewer app: `http://<pi-ip>:8000/viewer/`

If your package name is not `scanner`, run with your actual package/module name.

---

## 3) First-time calibration walkthrough

Calibration state is stored at `calibration/calibration.json` via [`CalibrationStore`](calibration_store.py:10).

### 3.1 Check calibration status

```bash
curl "http://<pi-ip>:8000/api/calibration/status"
```

### 3.2 Camera intrinsics (checkerboard/charuco)

1. Start intrinsics session:

```bash
curl -X POST "http://<pi-ip>:8000/api/calibration/intrinsics/start?board_type=checkerboard&checkerboard_cols=9&checkerboard_rows=6&square_size_m=0.01&min_frames=12"
```

2. Capture multiple frames with board at varied position/orientation:

```bash
curl -X POST "http://<pi-ip>:8000/api/calibration/intrinsics/capture"
```

Repeat until enough detections are collected.

3. Solve intrinsics:

```bash
curl -X POST "http://<pi-ip>:8000/api/calibration/intrinsics/solve"
```

### 3.2b Manual ChArUco intrinsics (guided 40-step workflow)

Use this when you want explicit user-guided repositioning of the board for each capture.

1. Start guided workflow (defaults to 40 steps):

```bash
curl -X POST "http://<pi-ip>:8000/api/calibration/intrinsics/charuco-manual/start?total_steps=40&min_frames=20"
```

2. For each step, manually move board to a **new pose**, then capture:

```bash
curl -X POST "http://<pi-ip>:8000/api/calibration/intrinsics/charuco-manual/capture"
```

Repeat until `40/40` is reached. Instruction text in API/UI is:

- `Move board to a new pose manually, then capture.`

3. Check progress/status at any time:

```bash
curl "http://<pi-ip>:8000/api/calibration/intrinsics/charuco-manual/status"
```

4. Final solve:

- auto-runs on capture step 40, or
- run manually:

```bash
curl -X POST "http://<pi-ip>:8000/api/calibration/intrinsics/charuco-manual/solve"
```

5. Pass/fail criteria (`quality_summary.ok`):

- accepted frames >= `min_frames` (default 20)
- RMS reprojection error <= 1.2
- mean reprojection error <= 1.0 px

When solve succeeds, intrinsics are persisted immediately to `calibration/calibration.json`.

### 3.3 Laser plane calibration

1. Start laser calibration session (board plane coefficients in camera frame):

```bash
curl -X POST "http://<pi-ip>:8000/api/calibration/laser/start?board_a=0&board_b=0&board_c=1&board_d=-0.30&min_points_per_laser=200"
```

2. Capture for each laser (repeat for quality):

```bash
curl -X POST "http://<pi-ip>:8000/api/calibration/laser/capture?laser=1"
curl -X POST "http://<pi-ip>:8000/api/calibration/laser/capture?laser=2"
```

3. Solve laser planes:

```bash
curl -X POST "http://<pi-ip>:8000/api/calibration/laser/solve"
```

### 3.4 Persist calibration

```bash
curl -X POST "http://<pi-ip>:8000/api/calibration/save"
```

Optional maintenance:

- reload from disk: `POST /api/calibration/load`
- reset to defaults: `POST /api/calibration/reset`

---

## 4) Scan workflow (OFF/ON-only default)

Default scan path no longer requires pure empty-background capture.

Before scan, preflight checks in [`_scan_preflight_errors()`](webapp.py) require:

1. NORMAL profile saved
2. LASER profile saved
3. lasers explicitly enabled

Legacy background path is now optional and disabled by default via env flag:

- `SCANNER_ENABLE_LEGACY_BACKGROUND_PATH=0` (default): no background preflight requirement.
- `SCANNER_ENABLE_LEGACY_BACKGROUND_PATH=1`: re-enables legacy background capture flow and background preflight gate.

### 4.1 Prepare camera from UI

In `http://<pi-ip>:8000/`:

1. set camera and focus (Auto / AF / Freeze / Manual)
2. save profiles:
   - **Save NORMAL profile**
   - **Save LASER profile**
3. click **Allow lasers (overlay blinking)**

Optional rollback flow (legacy only):

- enable `SCANNER_ENABLE_LEGACY_BACKGROUND_PATH=1`
- remove object and click **Capture legacy background (optional)**

### 4.2 Start scan

API example:

```bash
curl -X POST "http://<pi-ip>:8000/api/scan/start?step_deg=2.0&span_deg=360&speed=320&save_images=0"
```

Monitor status:

```bash
curl "http://<pi-ip>:8000/api/scan/status"
```

Detector telemetry (scan + preview diagnostics):

```bash
curl "http://<pi-ip>:8000/api/detector/telemetry"
```

Telemetry now includes additional tuning fields from [`LaserDetectorCore.detect()`](laser_detector_core.py:138), including:

- `channel_mode`, `threshold_source`, `threshold`
- `score_p50`, `score_p90`, `score_p99`, `positive_pixels`
- `components_total`, `kept_components`, component rejection counters
- `rejected_jump_rows`, `gap_resets`
- pair drift/stability telemetry in scan/preview payloads (`exposure_drift_rel`, `gain_drift_rel`, `stable`, `attempts`)

Stop scan:

```bash
curl -X POST "http://<pi-ip>:8000/api/scan/stop"
```

Runs are stored under `runs/<run_id>/` with point files in `runs/<run_id>/points/step_*.npz`.

---

## 5) Export + viewer workflow

### 5.1 List runs

```bash
curl "http://<pi-ip>:8000/api/runs"
```

### 5.2 Export run JSON (API)

```bash
curl -X POST "http://<pi-ip>:8000/api/runs/<run_id>/export"
```

Default output: `runs/<run_id>/viewer_export.json`

### 5.3 Export run JSON (CLI)

```bash
python tools/export_run_to_json.py runs/<run_id>
```

### 5.4 Open viewer

- integrated route: `http://<pi-ip>:8000/viewer/`
- or open [`viewer/index.html`](viewer/index.html) directly

Viewer details: [`README_VIEWER.md`](README_VIEWER.md)

---

## 8) Real-time camera tuning workflow (for violet-looking laser stripes)

When the red line appears violet/pink due to AWB/AE, start with manual controls and save two profiles.

### 8.1 Use the new UI panel on `/`

The main page now includes a real-time camera control panel with:

- exposure time (`ExposureTime`)
- analog gain (`AnalogueGain`)
- AWB toggle + colour gains (`ColourGains`)
- AE toggle (`AeEnable`)
- brightness/contrast/saturation/sharpness
- lens position (if the camera stack exposes it)
- actions: read current, apply controls, save/load `normal` and `laser` profiles

### 8.2 Recommended starting values

Starting point for laser capture profile (adjust per setup):

- `AeEnable=false`
- `AwbEnable=false`
- `ExposureTime=8000..14000` us
- `AnalogueGain=1.0..2.5`
- `ColourGains=(1.6, 0.8)` (slightly red-biased)
- `Brightness=0.0`, `Contrast=1.0`, `Saturation=1.0`, `Sharpness=1.0`

Then:

1. tune until overlay keeps a continuous stripe,
2. save as `laser`,
3. capture normal lighting and save as `normal`.

### 8.3 Camera control API

Core endpoints added in [`webapp.py`](webapp.py):

- `GET /api/camera/state`
- `GET /api/camera/controls`
- `GET /api/camera/meta`
- `POST /api/camera/controls/apply` (JSON controls)
- `POST /api/camera/ae?enabled=0|1`
- `POST /api/camera/awb?enabled=0|1`
- `GET /api/camera/profiles`
- `POST /api/camera/profile/save?name=normal|laser`
- `POST /api/camera/profile/load?name=normal|laser`
- `POST /api/camera/profile/apply?name=normal|laser` (backward-compatible)

Guided ChArUco workflow endpoints:

- `POST /api/calibration/intrinsics/charuco-manual/start`
- `POST /api/calibration/intrinsics/charuco-manual/capture`
- `GET /api/calibration/intrinsics/charuco-manual/status`
- `POST /api/calibration/intrinsics/charuco-manual/solve`

These are safe in mock mode as well (`SCANNER_MOCK_HW=1`).

---

## 6) Pseudo fallback vs calibrated real XYZ

Exporter logic in [`export_run()`](tools/export_run_to_json.py:81):

- If calibration is valid, output includes triangulated `laser1_xyz` / `laser2_xyz` in millimeters.
- If calibration is missing/invalid, export still succeeds with 2D stripe points (`laser1`, `laser2`) and mapping defaults (`scale_y`, `scale_r`, `x_center`) for pseudo visualization.

Interpretation:

- **Pseudo fallback**: useful for quick visual preview only (not metrically accurate).
- **Calibrated XYZ**: real 3D points derived from camera intrinsics + laser plane intersection.

---

## 7) Full hardwareless workflow (Windows/Linux, no Pi hardware)

This project now supports explicit opt-in mock mode via `SCANNER_MOCK_HW=1`.

In mock mode:

- camera frames are synthetic (deterministic RGB gradients)
- GPIO/stepper/laser calls are simulated
- production behavior remains unchanged when `SCANNER_MOCK_HW` is not set

### 7.1 Generate synthetic run + calibration fixtures

```bash
python tools/generate_hardwareless_fixtures.py --run-id synthetic_hwless_test --steps 96
```

This creates:

- `runs/synthetic_hwless_test/points/step_*.npz`
- `runs/synthetic_hwless_test/config.json`
- `runs/synthetic_hwless_test/plan.json`
- `calibration/calibration.json` (schema-compatible synthetic calibration)

### 7.2 Run local exporter verification harness

```bash
python tools/run_hardwareless_checks.py --run-id synthetic_hwless_test --steps 96
```

This runs generation + exporter and verifies the output JSON contains expected fields including triangulated arrays.

### 7.3 Start web server in hardwareless mode

Windows `cmd.exe`:

```bash
set SCANNER_MOCK_HW=1 && python -m uvicorn webapp:app --host 127.0.0.1 --port 8000
```

Linux/macOS:

```bash
SCANNER_MOCK_HW=1 python -m uvicorn webapp:app --host 127.0.0.1 --port 8000
```

### 7.4 API smoke checks while server is running

```bash
python tools/hardwareless_api_smoke.py --base http://127.0.0.1:8000 --run-id synthetic_hwless_test
```

Checks:

- `GET /api/system/mode` (expects `mock_hw=true`)
- `GET /api/scan/status`
- `GET /api/calibration/status`
- `GET /api/runs`
- `POST /api/runs/<run_id>/export`

### 7.5 Browser GUI validation targets

Validate in browser (manual or MCP automation):

- `/` main status/control page loads
- `/api/calibration/status` reachable and populated
- runs list visible on main page and includes synthetic run
- export trigger works from UI (`POST /api/runs/<run_id>/export`)
- `/viewer/` opens and is interactive


