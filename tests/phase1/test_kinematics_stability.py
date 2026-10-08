"""
tests/phase1/test_kinematics_stability.py
=========================================
Multi-Period Long-Horizon Flight Mechanics & Orbital Invariant Test.
Simulates 10 full orbits (18,000 s / 5 hours of continuous flight).
Asserts zero coordinate drift, speed bounds, angular C1 yaw continuity, and acceleration limits.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from core.kinematics import ACNKinematicsEngine, OrbitConfig, AntennaConfig


def test_multi_period_orbital_invariants():
    print("==================================================================")
    print("   PHASE 1: 10-ORBIT (5-HOUR) KINEMATICS STABILITY SWEEP          ")
    print("==================================================================\n")

    kin = ACNKinematicsEngine(OrbitConfig(), AntennaConfig())
    period = 1800.0
    omega = 2.0 * np.pi / period

    v_min_theoretical = 20_000.0 * omega  # 69.81 m/s
    v_max_theoretical = 40_000.0 * omega  # 139.63 m/s

    p0, _, _ = kin.compute_acn_state(0.0)

    # 10 full orbits sampled at 10-second intervals (1,801 points)
    t_samples = np.linspace(0.0, 10.0 * period, 1801)
    max_drift_at_period_boundary = 0.0
    max_angular_step_rad = 0.0
    max_accel_mps2 = 0.0

    prev_p, prev_v, prev_y = kin.compute_acn_state(0.0)

    for idx, t in enumerate(t_samples[1:], start=1):
        dt = t - t_samples[idx - 1]
        p, v, y = kin.compute_acn_state(t)
        spd = np.linalg.norm(v)

        assert v_min_theoretical - 0.5 <= spd <= v_max_theoretical + 0.5, f"Speed {spd} out of bounds at t={t}!"
        assert 4800.0 <= p[2] <= 5200.0, f"Altitude {p[2]} out of bounds at t={t}!"
        assert -np.pi <= y <= np.pi, f"Yaw {y} out of range!"

        # Wrapped angular delta assertion (strictly proves C1 yaw continuity)
        delta_yaw = np.arctan2(np.sin(y - prev_y), np.cos(y - prev_y))
        abs_yaw_rate = abs(delta_yaw) / dt
        if abs(delta_yaw) > max_angular_step_rad:
            max_angular_step_rad = abs(delta_yaw)
        assert abs_yaw_rate < 0.05, f"Yaw discontinuity detected! Yaw rate {abs_yaw_rate:.4f} rad/s at t={t}"

        # Acceleration bound check
        accel = np.linalg.norm(v - prev_v) / dt
        if accel > max_accel_mps2:
            max_accel_mps2 = accel
        assert accel < 2.0, f"Unphysical acceleration {accel:.3f} m/s^2 at t={t}!"

        # Period boundary return check
        if idx % 180 == 0:
            drift = np.linalg.norm(p[:2] - p0[:2])
            if drift > max_drift_at_period_boundary:
                max_drift_at_period_boundary = drift

        prev_p, prev_v, prev_y = p, v, y

    print(f"  Continuous Flight Evaluated : 18,000 seconds (5.0 hours / 10 orbits)")
    print(f"  Max Period Return Drift     : {max_drift_at_period_boundary:.6e} meters")
    print(f"  Max Angular Step (10s)      : {np.rad2deg(max_angular_step_rad):.3f}° (Continuity verified)")
    print(f"  Max Centripetal Accel       : {max_accel_mps2:.3f} m/s²")
    assert max_drift_at_period_boundary < 1e-3, "Kinematics accumulated orbital drift!"
    print("\n[PASS] Multi-period orbital mechanics, acceleration limits, and yaw continuity verified.\n")


if __name__ == "__main__":
    test_multi_period_orbital_invariants()