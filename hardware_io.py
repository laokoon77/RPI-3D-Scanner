from __future__ import annotations

import time
import threading
import lgpio

# ---- Stepper Motor HAT (B), channel M1 (BCM numbering) ----
M1_DIR = 13
M1_STEP = 19
M1_EN = 12  # Enable pin on HAT

_stepper_lock = threading.Lock()

def gpio_open(chip: int = 0):
    return lgpio.gpiochip_open(chip)

def gpio_close(h):
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
    lgpio.gpio_claim_output(h, dir_pin, 0)
    lgpio.gpio_claim_output(h, step_pin, 0)

    # Disabled by default:
    disabled_level = 1 if en_active_low else 0
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
    lgpio.gpio_write(h, st["en"], level)

def stepper_set_dir(h, st, direction: int):
    lgpio.gpio_write(h, st["dir"], 1 if direction >= 0 else 0)

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

        used_tx = hasattr(lgpio, "tx_pulse")
        if used_tx:
            rc = lgpio.tx_pulse(h, st["step"], on_us, off_us, 0, count)
            if rc < 0:
                used_tx = False
            else:
                # Wait for expected duration (simple and reliable)
                time.sleep((count / speed_sps) + 0.02)

        if not used_tx:
            hi = on_us / 1_000_000.0
            lo = off_us / 1_000_000.0
            for _ in range(count):
                lgpio.gpio_write(h, st["step"], 1)
                time.sleep(hi)
                lgpio.gpio_write(h, st["step"], 0)
                time.sleep(lo)

        lgpio.gpio_write(h, st["step"], 0)
        st["position_steps"] += int(steps)

        if not hold:
            stepper_enable(h, st, False)


# ---------------- Lasers / LED drivers ----------------

def laser_init(h, pin_bcm: int):
    lgpio.gpio_claim_output(h, int(pin_bcm), 0)
    return {"pin": int(pin_bcm), "on": False}

def laser_set(h, laser, on: bool):
    lgpio.gpio_write(h, laser["pin"], 1 if on else 0)
    laser["on"] = bool(on)
