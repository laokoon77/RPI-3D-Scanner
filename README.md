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

## 4) Scan workflow

Before scan, preflight checks in [`_scan_preflight_errors()`](webapp.py:569) require:

1. background captured
2. NORMAL profile saved
3. LASER profile saved
4. lasers explicitly enabled

### 4.1 Prepare camera/background from UI

In `http://<pi-ip>:8000/`:

1. set camera and focus (Auto / AF / Freeze / Manual)
2. save profiles:
   - **Save NORMAL profile**
   - **Save LASER profile**
3. remove object and click **Capture background**
4. click **Allow lasers (overlay blinking)**

### 4.2 Start scan

API example:

```bash
curl -X POST "http://<pi-ip>:8000/api/scan/start?step_deg=2.0&span_deg=360&speed=320&save_images=0"
```

Monitor status:

```bash
curl "http://<pi-ip>:8000/api/scan/status"
```

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

## 6) Pseudo fallback vs calibrated real XYZ

Exporter logic in [`export_run()`](tools/export_run_to_json.py:81):

- If calibration is valid, output includes triangulated `laser1_xyz` / `laser2_xyz` in millimeters.
- If calibration is missing/invalid, export still succeeds with 2D stripe points (`laser1`, `laser2`) and mapping defaults (`scale_y`, `scale_r`, `x_center`) for pseudo visualization.

Interpretation:

- **Pseudo fallback**: useful for quick visual preview only (not metrically accurate).
- **Calibrated XYZ**: real 3D points derived from camera intrinsics + laser plane intersection.

