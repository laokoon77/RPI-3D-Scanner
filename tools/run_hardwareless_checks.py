from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def _check_export_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps") if isinstance(payload, dict) else None
    if not isinstance(steps, list) or not steps:
        raise RuntimeError("export json has no steps")
    first = steps[0]
    if not isinstance(first, dict):
        raise RuntimeError("invalid first step")
    for key in ("laser1", "laser2", "laser1_xyz", "laser2_xyz"):
        if key not in first:
            raise RuntimeError(f"missing key in first step: {key}")
    return {
        "steps": len(steps),
        "first_laser1": len(first.get("laser1", [])),
        "first_laser2": len(first.get("laser2", [])),
        "first_laser1_xyz": len(first.get("laser1_xyz", [])),
        "first_laser2_xyz": len(first.get("laser2_xyz", [])),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run hardwareless fixture generation + export verification")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--run-id", default="synthetic_hwless_test")
    p.add_argument("--steps", type=int, default=90)
    p.add_argument("--export-name", default="viewer_export_hwless.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    py = str(args.python)
    run_dir = Path("runs") / args.run_id
    export_path = run_dir / args.export_name

    g = _run([
        py,
        "tools/generate_hardwareless_fixtures.py",
        "--run-id",
        args.run_id,
        "--steps",
        str(args.steps),
    ])
    if g.returncode != 0:
        print(g.stdout)
        print(g.stderr)
        return g.returncode

    e = _run([
        py,
        "tools/export_run_to_json.py",
        str(run_dir),
        "--output",
        str(export_path),
    ])
    if e.returncode != 0:
        print(e.stdout)
        print(e.stderr)
        return e.returncode

    summary = _check_export_json(export_path)
    print(json.dumps({"ok": True, "run_dir": run_dir.as_posix(), "export": export_path.as_posix(), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

