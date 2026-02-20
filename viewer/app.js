import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const viewerEl = document.getElementById("viewer");
const statusEl = document.getElementById("status");

const ui = {
  fileInput: document.getElementById("fileInput"),
  showLaser1: document.getElementById("showLaser1"),
  showLaser2: document.getElementById("showLaser2"),
  showWire: document.getElementById("showWire"),
  pointSize: document.getElementById("pointSize"),
  rowDecimation: document.getElementById("rowDecimation"),
  stepDecimation: document.getElementById("stepDecimation"),
  linkMode: document.getElementById("linkMode"),
  colorMode: document.getElementById("colorMode"),
  scaleY: document.getElementById("scaleY"),
  scaleR: document.getElementById("scaleR"),
  xCenter: document.getElementById("xCenter"),
  rebuildBtn: document.getElementById("rebuildBtn"),
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1117);

const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 5000);
camera.position.set(0.7, 0.8, 1.2);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewerEl.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0.35, 0);

scene.add(new THREE.HemisphereLight(0xffffff, 0x334466, 0.9));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.7);
dirLight.position.set(1.5, 2.0, 1.0);
scene.add(dirLight);

const grid = new THREE.GridHelper(2.0, 20, 0x2a3f74, 0x1c2537);
scene.add(grid);

const pivot = new THREE.Group();
scene.add(pivot);

let points1 = null;
let points2 = null;
let wire1 = null;
let wire2 = null;

let exported = null;

function setStatus(text) {
  statusEl.textContent = text;
}

window.addEventListener("error", (ev) => {
  setStatus(`Runtime error: ${ev.message}`);
});

window.addEventListener("unhandledrejection", (ev) => {
  const msg = ev?.reason?.message || String(ev?.reason || "unknown promise error");
  setStatus(`Unhandled error: ${msg}`);
});

function resize() {
  const w = Math.max(1, viewerEl.clientWidth);
  const h = Math.max(1, viewerEl.clientHeight);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}

window.addEventListener("resize", resize);
resize();

function disposeObject(obj) {
  if (!obj) {
    return;
  }
  pivot.remove(obj);
  if (obj.geometry) {
    obj.geometry.dispose();
  }
  if (obj.material) {
    obj.material.dispose();
  }
}

function clearGeometry() {
  disposeObject(points1);
  disposeObject(points2);
  disposeObject(wire1);
  disposeObject(wire2);
  points1 = null;
  points2 = null;
  wire1 = null;
  wire2 = null;
}

function centerAndFrameScene() {
  const box = new THREE.Box3().setFromObject(pivot);
  if (box.isEmpty()) {
    return false;
  }

  const center = box.getCenter(new THREE.Vector3());
  pivot.position.sub(center);

  const size = box.getSize(new THREE.Vector3()).length();
  const fitDist = Math.max(1.0, size * 1.1);

  controls.target.set(0, 0, 0);
  camera.position.set(fitDist * 0.6, fitDist * 0.45, fitDist * 0.85);
  camera.near = 0.01;
  camera.far = Math.max(5000, fitDist * 50);
  camera.updateProjectionMatrix();
  controls.update();
  return true;
}

function toPseudo3D(x, y, angleDeg, mapping) {
  const theta = (angleDeg * Math.PI) / 180.0;
  const radius = (x - mapping.xCenter) * mapping.scaleR;
  const worldX = radius * Math.cos(theta);
  const worldZ = radius * Math.sin(theta);
  const worldY = -y * mapping.scaleY;
  return [worldX, worldY, worldZ];
}

function colorByAngle(angleDeg) {
  const hue = ((angleDeg % 360) + 360) % 360;
  const c = new THREE.Color();
  c.setHSL(hue / 360.0, 0.8, 0.55);
  return c;
}

function parseNumeric(value, fallback, min = null) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return fallback;
  }
  if (min !== null && n < min) {
    return min;
  }
  return n;
}

function currentOptions() {
  return {
    showLaser1: ui.showLaser1.checked,
    showLaser2: ui.showLaser2.checked,
    showWire: ui.showWire.checked,
    pointSize: parseNumeric(ui.pointSize.value, 2, 1),
    rowDecimation: Math.floor(parseNumeric(ui.rowDecimation.value, 2, 1)),
    stepDecimation: Math.floor(parseNumeric(ui.stepDecimation.value, 1, 1)),
    linkMode: ui.linkMode.value,
    colorMode: ui.colorMode.value,
    mapping: {
      scaleY: parseNumeric(ui.scaleY.value, 0.01),
      scaleR: parseNumeric(ui.scaleR.value, 0.01),
      xCenter: parseNumeric(ui.xCenter.value, 640),
    },
  };
}

function collectRows(points, rowDecimation) {
  const sorted = [...points].sort((a, b) => a[1] - b[1]);
  const out = [];
  for (let i = 0; i < sorted.length; i += rowDecimation) {
    out.push(sorted[i]);
  }
  return out;
}

function buildLaserGeometry(steps, laserKey, options) {
  const pointPositions = [];
  const pointColors = [];
  const linePositions = [];

  const laserColor = laserKey === "laser1" ? new THREE.Color(0xff5f5f) : new THREE.Color(0x66c0ff);
  let prevRows = null;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (let stepIdx = 0; stepIdx < steps.length; stepIdx += options.stepDecimation) {
    const step = steps[stepIdx];
    const angle = Number(step.angle_deg) || 0;
    const src = Array.isArray(step[laserKey]) ? step[laserKey] : [];
    const rows = collectRows(src, options.rowDecimation);

    for (let i = 0; i < rows.length; i++) {
      const [x, y] = rows[i];
      const p = toPseudo3D(x, y, angle, options.mapping);
      pointPositions.push(p[0], p[1], p[2]);

      const c = options.colorMode === "step" ? colorByAngle(angle) : laserColor;
      pointColors.push(c.r, c.g, c.b);

      if (p[1] < minY) minY = p[1];
      if (p[1] > maxY) maxY = p[1];

      if (options.showWire && i > 0) {
        const [xPrev, yPrev] = rows[i - 1];
        const pPrev = toPseudo3D(xPrev, yPrev, angle, options.mapping);
        linePositions.push(pPrev[0], pPrev[1], pPrev[2], p[0], p[1], p[2]);
      }
    }

    if (options.showWire && prevRows && rows.length > 0) {
      if (options.linkMode === "same_index") {
        const n = Math.min(prevRows.length, rows.length);
        const prevAngle = Number(steps[Math.max(0, stepIdx - options.stepDecimation)].angle_deg) || 0;
        for (let i = 0; i < n; i++) {
          const pA = toPseudo3D(prevRows[i][0], prevRows[i][1], prevAngle, options.mapping);
          const pB = toPseudo3D(rows[i][0], rows[i][1], angle, options.mapping);
          linePositions.push(pA[0], pA[1], pA[2], pB[0], pB[1], pB[2]);
        }
      } else {
        const prevAngle = Number(steps[Math.max(0, stepIdx - options.stepDecimation)].angle_deg) || 0;
        for (const row of rows) {
          let best = null;
          let bestDist = Number.POSITIVE_INFINITY;
          for (const prow of prevRows) {
            const d = Math.abs(row[1] - prow[1]);
            if (d < bestDist) {
              best = prow;
              bestDist = d;
            }
          }
          if (!best) {
            continue;
          }
          const pA = toPseudo3D(best[0], best[1], prevAngle, options.mapping);
          const pB = toPseudo3D(row[0], row[1], angle, options.mapping);
          linePositions.push(pA[0], pA[1], pA[2], pB[0], pB[1], pB[2]);
        }
      }
    }

    prevRows = rows;
  }

  return {
    pointPositions,
    pointColors,
    linePositions,
    minY,
    maxY,
  };
}

function makePointsObject(pointPositions, pointColors, pointSize) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pointPositions, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(pointColors, 3));

  const m = new THREE.PointsMaterial({
    size: pointSize * 0.004,
    vertexColors: true,
    sizeAttenuation: true,
  });

  return new THREE.Points(g, m);
}

function makeWireObject(linePositions, colorHex) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));

  const m = new THREE.LineBasicMaterial({ color: colorHex, transparent: true, opacity: 0.8 });
  return new THREE.LineSegments(g, m);
}

async function rebuildGeometry() {
  if (!exported || !Array.isArray(exported.steps)) {
    setStatus("No data loaded.");
    return;
  }

  clearGeometry();
  setStatus("Building geometry...");

  const options = currentOptions();
  const steps = exported.steps;

  await new Promise((resolve) => setTimeout(resolve, 0));

  const g1 = buildLaserGeometry(steps, "laser1", options);
  const g2 = buildLaserGeometry(steps, "laser2", options);

  if (options.showLaser1 && g1.pointPositions.length > 0) {
    points1 = makePointsObject(g1.pointPositions, g1.pointColors, options.pointSize);
    pivot.add(points1);
  }
  if (options.showLaser2 && g2.pointPositions.length > 0) {
    points2 = makePointsObject(g2.pointPositions, g2.pointColors, options.pointSize);
    pivot.add(points2);
  }

  if (options.showWire && g1.linePositions.length > 0) {
    wire1 = makeWireObject(g1.linePositions, 0xff7777);
    pivot.add(wire1);
  }
  if (options.showWire && g2.linePositions.length > 0) {
    wire2 = makeWireObject(g2.linePositions, 0x77bbff);
    pivot.add(wire2);
  }

  const pointCount = (g1.pointPositions.length + g2.pointPositions.length) / 3;
  const segCount = (g1.linePositions.length + g2.linePositions.length) / 6;

  if (pointCount <= 0) {
    setStatus("Loaded JSON but no renderable points after decimation/settings.");
    return;
  }

  centerAndFrameScene();
  setStatus(`Loaded ${steps.length} steps | points: ${Math.round(pointCount)} | wire segments: ${Math.round(segCount)}`);
}

async function loadFromFile(file) {
  const txt = await file.text();
  const parsed = JSON.parse(txt);
  if (!parsed || !Array.isArray(parsed.steps)) {
    throw new Error("Invalid export JSON: expected { steps: [...] }");
  }

  exported = parsed;
  const map = parsed.mapping_defaults || {};

  if (Number.isFinite(Number(map.scale_y))) {
    ui.scaleY.value = String(map.scale_y);
  }
  if (Number.isFinite(Number(map.scale_r))) {
    ui.scaleR.value = String(map.scale_r);
  }
  if (Number.isFinite(Number(map.x_center))) {
    ui.xCenter.value = String(map.x_center);
  }

  await rebuildGeometry();
}

ui.fileInput.addEventListener("change", async (ev) => {
  const file = ev.target.files?.[0];
  if (!file) {
    return;
  }
  try {
    setStatus("Loading JSON...");
    await loadFromFile(file);
  } catch (err) {
    setStatus(`Load failed: ${err?.message || String(err)}`);
  }
});

document.addEventListener("dragover", (ev) => {
  ev.preventDefault();
});

document.addEventListener("drop", async (ev) => {
  ev.preventDefault();
  const file = ev.dataTransfer?.files?.[0];
  if (!file) {
    return;
  }
  try {
    setStatus("Loading dropped JSON...");
    await loadFromFile(file);
  } catch (err) {
    setStatus(`Load failed: ${err?.message || String(err)}`);
  }
});

ui.rebuildBtn.addEventListener("click", () => {
  rebuildGeometry();
});

[
  ui.showLaser1,
  ui.showLaser2,
  ui.showWire,
  ui.pointSize,
  ui.rowDecimation,
  ui.stepDecimation,
  ui.linkMode,
  ui.colorMode,
].forEach((el) => {
  el.addEventListener("change", () => {
    rebuildGeometry();
  });
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

animate();

