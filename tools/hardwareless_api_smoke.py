from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url: str) -> dict:
    req = urllib.request.Request(url, method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def _post_q(base: str, path: str, query: dict[str, object]) -> dict:
    qs = urllib.parse.urlencode(query)
    return _post(f"{base}{path}?{qs}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hardwareless API smoke tests")
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--run-id", default="synthetic_hwless_test")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    b = args.base.rstrip("/")

    checks: dict[str, dict] = {}
    checks["system_mode"] = _get(f"{b}/api/system/mode")
    checks["detector_telemetry_initial"] = _get(f"{b}/api/detector/telemetry")

    # OFF/ON-only scan path prep: lock profiles + enable lasers, no background capture.
    checks["lasers_enabled"] = _post_q(b, "/api/lasers/enabled", {"enabled": 1})
    checks["profile_save_normal"] = _post_q(b, "/api/camera/profile/save", {"name": "normal"})
    checks["profile_save_laser"] = _post_q(b, "/api/camera/profile/save", {"name": "laser"})
    checks["scan_start"] = _post_q(
        b,
        "/api/scan/start",
        {
            "step_deg": 120,
            "span_deg": 360,
            "speed": 320,
            "save_images": 0,
        },
    )

    # Give worker a moment to produce telemetry/status.
    time.sleep(0.8)
    checks["scan_status"] = _get(f"{b}/api/scan/status")
    checks["detector_telemetry_after_start"] = _get(f"{b}/api/detector/telemetry")
    checks["scan_stop"] = _post(f"{b}/api/scan/stop")
    checks["calibration_status"] = _get(f"{b}/api/calibration/status")
    checks["runs"] = _get(f"{b}/api/runs")
    checks["export"] = _post(f"{b}/api/runs/{args.run_id}/export")

    scan_start_ok = bool(checks["scan_start"].get("ok"))
    if not scan_start_ok:
        scan_err = str(checks["scan_start"].get("error", ""))
        # Regression guard: background capture must not be mandatory by default path.
        if "background" in scan_err.lower():
            print(json.dumps({"ok": False, "checks": checks, "reason": "scan_start blocked by background preflight"}, indent=2))
            return 1

    ok = (
        bool(checks["system_mode"].get("ok"))
        and bool(checks["detector_telemetry_initial"].get("ok"))
        and bool(checks["runs"].get("ok"))
        and bool(checks["export"].get("ok"))
        and scan_start_ok
        and bool(checks["detector_telemetry_after_start"].get("ok"))
    )
    print(json.dumps({"ok": ok, "checks": checks}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

