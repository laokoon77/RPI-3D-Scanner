from __future__ import annotations

import argparse
import json
import urllib.request


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url: str) -> dict:
    req = urllib.request.Request(url, method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


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
    checks["scan_status"] = _get(f"{b}/api/scan/status")
    checks["calibration_status"] = _get(f"{b}/api/calibration/status")
    checks["runs"] = _get(f"{b}/api/runs")
    checks["export"] = _post(f"{b}/api/runs/{args.run_id}/export")

    ok = bool(checks["system_mode"].get("ok")) and bool(checks["runs"].get("ok")) and bool(checks["export"].get("ok"))
    print(json.dumps({"ok": ok, "checks": checks}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

