"""
core/mac_sync.py
================
Timing Advance (TA) and MAC Synchronization Engine for 200 km Operational Theater.
Includes Numerical Validation and Finite Boundary Checks.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MACSyncConfig:
    speed_of_light: float = 299_792_458.0
    frame_duration_sec: float = 0.010
    ta_quantization_step_sec: float = 0.1e-6
    clock_jitter_margin_sec: float = 2.0e-6
    safety_margin_sec: float = 3.0e-6

    def __post_init__(self):
        if not np.isfinite(self.speed_of_light) or self.speed_of_light <= 0:
            raise ValueError("speed_of_light must be strictly positive and finite")
        if not np.isfinite(self.frame_duration_sec) or self.frame_duration_sec <= 0:
            raise ValueError("frame_duration_sec must be strictly positive and finite")
        if not np.isfinite(self.ta_quantization_step_sec) or self.ta_quantization_step_sec <= 0:
            raise ValueError("ta_quantization_step_sec must be strictly positive and finite")
        if not np.isfinite(self.clock_jitter_margin_sec) or self.clock_jitter_margin_sec < 0:
            raise ValueError("clock_jitter_margin_sec must be non-negative and finite")
        if not np.isfinite(self.safety_margin_sec) or self.safety_margin_sec < 0:
            raise ValueError("safety_margin_sec must be non-negative and finite")


class MACSyncEngine:
    def __init__(self, cfg: MACSyncConfig = MACSyncConfig()):
        self.cfg = cfg

    def compute_one_way_delay(self, slant_distances_m: np.ndarray) -> np.ndarray:
        d = np.asarray(slant_distances_m, dtype=np.float64)
        if not np.all(np.isfinite(d)) or np.any(d < 0):
            raise ValueError("slant_distances_m must contain non-negative finite values")
        return d / self.cfg.speed_of_light

    def compute_timing_advance(self, slant_distances_m: np.ndarray) -> np.ndarray:
        d = np.asarray(slant_distances_m, dtype=np.float64)
        if not np.all(np.isfinite(d)) or np.any(d < 0):
            raise ValueError("slant_distances_m must contain non-negative finite values")
        exact_ta = 2.0 * d / self.cfg.speed_of_light
        return np.round(exact_ta / self.cfg.ta_quantization_step_sec) * self.cfg.ta_quantization_step_sec

    def compute_ta_drift_rate(self, range_rates_mps: np.ndarray) -> np.ndarray:
        rr = np.asarray(range_rates_mps, dtype=np.float64)
        if not np.all(np.isfinite(rr)):
            raise ValueError("range_rates_mps must be finite")
        return 2.0 * rr / self.cfg.speed_of_light

    def compute_guard_interval_requirement(self, with_timing_advance: bool, slant_distances_m: np.ndarray) -> float:
        if with_timing_advance:
            return (
                self.cfg.ta_quantization_step_sec
                + self.cfg.clock_jitter_margin_sec
                + self.cfg.safety_margin_sec
            )
        else:
            delays = self.compute_one_way_delay(slant_distances_m)
            if len(delays) == 0:
                raise ValueError("slant_distances_m cannot be empty")
            delta_tau = np.max(delays) - np.min(delays)
            return float(delta_tau + self.cfg.safety_margin_sec)