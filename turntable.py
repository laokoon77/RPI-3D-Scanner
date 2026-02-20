# scanner/turntable.py
from __future__ import annotations
from dataclasses import dataclass

try:
    from .hardware_io import stepper_step
except ImportError:
    from hardware_io import stepper_step

@dataclass
class TurntableConfig:
    motor_steps_per_rev: int = 200
    microsteps: int = 32
    gear_driver: int = 10   # motor gear teeth
    gear_driven: int = 66   # turntable gear teeth

    @property
    def microsteps_per_motor_rev(self) -> int:
        return self.motor_steps_per_rev * self.microsteps

    @property
    def ratio(self) -> float:
        # motor revs per turntable rev
        return self.gear_driven / self.gear_driver

    @property
    def microsteps_per_turntable_rev(self) -> float:
        return self.microsteps_per_motor_rev * self.ratio

    @property
    def microsteps_per_degree(self) -> float:
        return self.microsteps_per_turntable_rev / 360.0


class Turntable:
    def __init__(self, gpio, stepper, cfg: TurntableConfig = TurntableConfig()):
        self.gpio = gpio
        self.stepper = stepper
        self.cfg = cfg
        self._residual = 0.0       # fractional microsteps carried forward
        self.angle_deg = 0.0       # best-effort angle tracking (no encoder)

    def degrees_to_steps(self, deg: float) -> int:
        exact = deg * self.cfg.microsteps_per_degree + self._residual
        steps = int(round(exact))
        self._residual = exact - steps
        return steps

    def move_deg(self, deg: float, speed_sps: float = 800.0, hold: bool = True) -> int:
        steps = self.degrees_to_steps(deg)
        if steps != 0:
            stepper_step(self.gpio, self.stepper, steps, speed_sps=speed_sps, hold=hold)
        self.angle_deg += deg
        return steps
