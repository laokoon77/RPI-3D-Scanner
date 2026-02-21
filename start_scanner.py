"""One-command launcher for the RPi 3D scanner web stack.

Run from project root:
    python start_scanner.py

Optional environment overrides:
    SCANNER_HOST (default: 0.0.0.0)
    SCANNER_PORT (default: 8000)
    SCANNER_MOCK_HW (optional passthrough to webapp)
    SCANNER_ENABLE_LEGACY_BACKGROUND_PATH (optional passthrough to webapp)
"""

from __future__ import annotations

import logging
import os

from webapp import run_server


def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _get_host() -> str:
    return os.getenv("SCANNER_HOST", "0.0.0.0").strip() or "0.0.0.0"


def _get_port() -> int:
    raw = os.getenv("SCANNER_PORT", "8000").strip()
    try:
        port = int(raw)
    except ValueError:
        logging.warning("Invalid SCANNER_PORT=%r; falling back to 8000", raw)
        return 8000

    if not (1 <= port <= 65535):
        logging.warning("Out-of-range SCANNER_PORT=%r; falling back to 8000", raw)
        return 8000
    return port


def main() -> None:
    logging.basicConfig(
        level=os.getenv("SCANNER_LOG_LEVEL", "INFO").upper(),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    host = _get_host()
    port = _get_port()
    mock_hw = _env_bool("SCANNER_MOCK_HW")
    legacy_bg = _env_bool("SCANNER_ENABLE_LEGACY_BACKGROUND_PATH")

    logging.info("Starting RPi 3D Scanner web server")
    logging.info("Host: %s", host)
    logging.info("Port: %d", port)
    logging.info(
        "SCANNER_MOCK_HW: %s",
        "unset" if mock_hw is None else ("enabled" if mock_hw else "disabled"),
    )
    logging.info(
        "SCANNER_ENABLE_LEGACY_BACKGROUND_PATH: %s",
        "unset" if legacy_bg is None else ("enabled" if legacy_bg else "disabled"),
    )

    run_server(host=host, port=port)


if __name__ == "__main__":
    main()
