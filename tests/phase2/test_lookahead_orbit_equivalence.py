"""
tests/phase2/test_lookahead_orbit_equivalence.py
================================================
Full-Orbit Equivalence Verification:
Compares Clearance-Culled Lookahead vs. Brute-Force Unculled Ray Tracer.
Samples the full 1800s orbit across all 16 nodes.
Asserts 100% agreement on current_los, tau_masking, and tau_recovery.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from envs.tactical_a2g_env import TacticalA2GEnv, EnvConfig


def test_full_orbit_lookahead_equivalence():
    print("==================================================================")
    print("   PHASE 2: ORBIT-WIDE LOOKAHEAD EQUIVALENCE TEST (1,800 SECONDS) ")
    print("==================================================================\n")

    env = TacticalA2GEnv(EnvConfig(seed=42))
    env.reset(seed=42)

    kin = env.kin_engine
    terrain = env.terrain_engine
    gns = env.gn_positions

    orbit_timestamps = np.arange(0.0, 1800.0, 20.0)
    mismatched_los = 0
    mismatched_masking = 0
    mismatched_recovery = 0

    for t in orbit_timestamps:
        acn_pos, _, _ = kin.compute_acn_state(t)
        los_mask, _, min_clearance_m = terrain.compute_los_batch(
            acn_pos, gns, num_samples=16, return_clearance=True
        )

        # Mode A: Clearance-Culled
        culled = terrain.compute_predictive_lookahead(
            t, kin, gns, horizon_sec=2.0, step_sec=0.5, current_clearance_margin_m=min_clearance_m
        )

        # Mode B: Brute-Force Unculled Reference
        reference = terrain.compute_predictive_lookahead(
            t, kin, gns, horizon_sec=2.0, step_sec=0.5, current_clearance_margin_m=None
        )

        if not np.array_equal(culled["current_los"], reference["current_los"]):
            mismatched_los += 1
        if not np.allclose(culled["tau_masking_sec"], reference["tau_masking_sec"], atol=1e-5):
            mismatched_masking += 1
        if not np.allclose(culled["tau_recovery_sec"], reference["tau_recovery_sec"], atol=1e-5):
            mismatched_recovery += 1

    print(f"  Orbital Positions Evaluated  : {len(orbit_timestamps)} steps across 360° orbit")
    print(f"  Total Link Checks Evaluated  : {len(orbit_timestamps) * 16} link-horizon vectors")
    print(f"  Current-LoS Mismatches       : {mismatched_los}")
    print(f"  Masking Countdown Mismatches : {mismatched_masking}")
    print(f"  Recovery Countdown Mismatches: {mismatched_recovery}")

    assert mismatched_los == 0, "Clearance culling corrupted current LoS classification!"
    assert mismatched_masking == 0, "Clearance culling diverged on masking countdown!"
    assert mismatched_recovery == 0, "Clearance culling diverged on recovery countdown!"
    print("\n[PASS] Full lookahead output (LoS, masking, and recovery) mathematically identical to reference.\n")


if __name__ == "__main__":
    test_full_orbit_lookahead_equivalence()