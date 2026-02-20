from __future__ import annotations

import time
import threading
import os
from typing import Any

try:
    import lgpio  # type: ignore
except Exception:
    lgpio = None  # type: ignore

# ---- Stepper Motor HAT (B), channel M1 (BCM numbering) ----
M1_DIR = 13
M1_STEP = 19
M1_EN = 12  # Enable pin on HAT

_stepper_lock = threading.Lock()


def _is_mock_handle(h: Any) -> bool:
    return isinstance(h, dict) and bool(h.get("__mock_gpio__"))


def _mock_hw_enabled() -> bool:
    return str(os.getenv("SCANNER_MOCK_HW", "")).strip().lower() in {"1", "true", "yes", "on"}

def gpio_open(chip: int = 0):
    if _mock_hw_enabled():
        return {"__mock_gpio__": True, "chip": int(chip), "pins": {}, "tx": []}
    if lgpio is None:
        raise RuntimeError(
            "lgpio is unavailable in this environment. "
            "Set SCANNER_MOCK_HW=1 to run in hardwareless mode."
        )
    return lgpio.gpiochip_open(chip)

def gpio_close(h):
    if _is_mock_handle(h):
        return
    try:
        lgpio.gpiochip_close(h)
    except Exception:
        pass


# ---------------- Stepper ----------------

def stepper_init(h, dir_pin=M1_DIR, step_pin=M1_STEP, en_pin=M1_EN, en_active_low: bool = False):
    """
    IMPORTANT:
    Your observed behavior (holds at startup, goes limp on move) strongly suggests
    your enable polarity is ACTIVE-HIGH in practice, so default is en_active_low=False.

    If it's wrong on your board, flip it from UI via /api/stepper/polarity.
    """
    if _is_mock_handle(h):
        h["pins"][int(dir_pin)] = 0
        h["pins"][int(step_pin)] = 0
    else:
        lgpio.gpio_claim_output(h, dir_pin, 0)
        lgpio.gpio_claim_output(h, step_pin, 0)

    # Disabled by default:
    disabled_level = 1 if en_active_low else 0
    if _is_mock_handle(h):
        h["pins"][int(en_pin)] = int(disabled_level)
    else:
        lgpio.gpio_claim_output(h, en_pin, disabled_level)

    return {
        "dir": int(dir_pin),
        "step": int(step_pin),
        "en": int(en_pin),
        "en_active_low": bool(en_active_low),
        "position_steps": 0,
    }

def stepper_enable(h, st, enabled: bool):
    active_low = bool(st.get("en_active_low", False))
    if active_low:
        level = 0 if enabled else 1
    else:
        level = 1 if enabled else 0
    if _is_mock_handle(h):
        h["pins"][int(st["en"])] = int(level)
    else:
        lgpio.gpio_write(h, st["en"], level)

def stepper_set_dir(h, st, direction: int):
    level = 1 if direction >= 0 else 0
    if _is_mock_handle(h):
        h["pins"][int(st["dir"])] = int(level)
    else:
        lgpio.gpio_write(h, st["dir"], level)

def stepper_step(h, st, steps: int, speed_sps: float = 800.0, hold: bool = True):
    """
    Generates STEP pulses. Uses lgpio.tx_pulse if available (smoother timing).
    Keeps EN asserted until pulses are finished.
    """
    if steps == 0:
        return

    with _stepper_lock:
        direction = 1 if steps > 0 else -1
        stepper_set_dir(h, st, direction)

        count = abs(int(steps))
        speed_sps = max(1.0, float(speed_sps))

        period_us = int(round(1_000_000 / speed_sps))
        on_us = max(5, period_us // 2)
        off_us = max(5, period_us - on_us)

        # Enable driver
        stepper_enable(h, st, True)

        used_tx = (not _is_mock_handle(h)) and hasattr(lgpio, "tx_pulse")
        if used_tx:
            rc = lgpio.tx_pulse(h, st["step"], on_us, off_us, 0, count)
            if rc < 0:
                used_tx = False
            else:
                # Wait for expected duration (simple and reliable)
                time.sleep((count / speed_sps) + 0.02)

        if not used_tx:
            if _is_mock_handle(h):
                h["tx"].append({
                    "pin": int(st["step"]),
                    "count": int(count),
                    "speed_sps": float(speed_sps),
                    "on_us": int(on_us),
                    "off_us": int(off_us),
                })
            else:
                hi = on_us / 1_000_000.0
                lo = off_us / 1_000_000.0
                for _ in range(count):
                    lgpio.gpio_write(h, st["step"], 1)
                    time.sleep(hi)
                    lgpio.gpio_write(h, st["step"], 0)
                    time.sleep(lo)

        if _is_mock_handle(h):
            h["pins"][int(st["step"])] = 0
        else:
            lgpio.gpio_write(h, st["step"], 0)
        st["position_steps"] += int(steps)

        if not hold:
            stepper_enable(h, st, False)


# ---------------- Lasers / LED drivers ----------------

def laser_init(h, pin_bcm: int):
    if _is_mock_handle(h):
        h["pins"][int(pin_bcm)] = 0
    else:
        lgpio.gpio_claim_output(h, int(pin_bcm), 0)
    return {"pin": int(pin_bcm), "on": False}

def laser_set(h, laser, on: bool):
    level = 1 if on else 0
    if _is_mock_handle(h):
        h["pins"][int(laser["pin"])] = int(level)
    else:
        lgpio.gpio_write(h, laser["pin"], level)
    laser["on"] = bool(on)
