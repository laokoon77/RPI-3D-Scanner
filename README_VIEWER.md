# WebGL Scan Preview Viewer

This repository now includes a standalone viewer in [`viewer/index.html`](viewer/index.html) and a run exporter script in [`tools/export_run_to_json.py`](tools/export_run_to_json.py).

## What this viewer shows

This is a **preview** of scanned geometry and is **not metrically accurate yet**.

It uses a placeholder pseudo-3D mapping:

- `angle_deg` -> rotation around Y axis
- image `y` -> world Y (height), scaled by `scale_y`
- image `x` -> radius from `x_center`, scaled by `scale_r`
- world coordinates:
  - `X = R * cos(theta)`
  - `Z = R * sin(theta)`

## 1) Export a run to JSON

From project root, run:

```bash
python tools/export_run_to_json.py runs/<timestamp>
```

This writes:

- `runs/<timestamp>/viewer_export.json`

Optional parameters:

```bash
python tools/export_run_to_json.py runs/<timestamp> --output runs/<timestamp>/my_export.json --scale-y 0.01 --scale-r 0.01 --x-center 640
```

## 2) Open the viewer

Open [`viewer/index.html`](viewer/index.html) directly in your browser.

Then load the exported JSON via:

- **Load export JSON** button, or
- drag & drop JSON file into the page

No backend changes are required.

## Controls

- Orbit controls: mouse rotate/pan/zoom
- Toggle visibility: Laser 1, Laser 2, Wireframe
- Point size slider
- Decimation:
  - row decimation (keep every Nth row-point)
  - step decimation (keep every Nth scan step)
- Inter-step wireframe matching:
  - nearest Y (default)
  - same sorted index
- Color mode:
  - by laser
  - by step angle
- Mapping fields:
  - `scale_y`, `scale_r`, `x_center`

After changing settings, click **Rebuild geometry** (or use controls that auto-rebuild).

