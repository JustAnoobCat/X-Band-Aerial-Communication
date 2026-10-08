"""
tests/phase1/test_terrain_convergence.py
========================================
Numerical Convergence Study for DEM Ray-Casting & Knife-Edge Diffraction.
Evaluates sample counts: [10, 16, 32, 64, 128, 256].
Proves that 16 samples with 11-point sub-sample peak refinement achieves <= 1.5 dB error vs 256 samples.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from core.kinematics import ACNKinematicsEngine, OrbitConfig, AntennaConfig
from core.terrain_engine import TerrainEngine, DEMConfig


def run_ray_sample_convergence_study():
    print("==================================================================")
    print("   PHASE 1: DEM RAY-CASTING NUMERICAL CONVERGENCE STUDY           ")
    print("==================================================================\n")

    terrain = TerrainEngine(DEMConfig(), seed=42)
    kin = ACNKinematicsEngine(OrbitConfig(), AntennaConfig())

    gn_targets = np.array([
        [      0.0,      0.0,  800.0],  # Plateau Center (Clear LoS)
        [ 30_000.0, 30_000.0,  800.0],  # Foothills
        [-40_000.0, 50_000.0, 1600.0],  # Radar North
        [-40_000.0, 50_000.0,  800.0],  # Valley Target behind Y=25km Ridge (NLoS)
    ], dtype=np.float64)

    sample_densities = [10, 16, 32, 64, 128, 256]
    test_timestamps = [0.0, 300.0, 680.0, 1200.0]

    print(f"{'Time (s)':<10} | {'Node':<6} | {'10 smp':<10} | {'16 smp':<10} | {'32 smp':<10} | {'64 smp':<10} | {'128 smp':<10} | {'256 (Ref)':<10}")
    print("-" * 88)

    max_error_16_vs_256 = 0.0
    los_disagreements = 0

    for t in test_timestamps:
        acn_pos, _, _ = kin.compute_acn_state(t)

        ref_los, ref_loss, _ = terrain.compute_los_batch(
            acn_pos, gn_targets, num_samples=256, return_clearance=True
        )

        for gn_idx in range(len(gn_targets)):
            losses = []
            for n_samples in sample_densities:
                los, loss, _ = terrain.compute_los_batch(
                    acn_pos, gn_targets, num_samples=n_samples, return_clearance=True
                )
                losses.append(loss[gn_idx])

                if n_samples >= 16 and los[gn_idx] != ref_los[gn_idx]:
                    los_disagreements += 1

                if n_samples == 16:
                    err = abs(loss[gn_idx] - ref_loss[gn_idx])
                    if err > max_error_16_vs_256:
                        max_error_16_vs_256 = err

            row_str = " | ".join(f"{l:6.2f} dB" for l in losses)
            print(f"{t:<10.1f} | GN {gn_idx:<3} | {row_str}")

    print("-" * 88)
    print(f"  Maximum Diffraction Error (16 vs 256 samples): {max_error_16_vs_256:.3f} dB (Tolerance <= 1.5 dB)")
    print(f"  LoS Classification Disagreements (16 vs 256) : {los_disagreements}")

    assert los_disagreements == 0, "16 samples produced a false LoS/NLoS classification vs 256 reference!"
    assert max_error_16_vs_256 <= 1.5, f"Diffraction error {max_error_16_vs_256:.2f} dB exceeded 1.5 dB tolerance!"
    print("\n[PASS] DEM ray-casting convergence verified: Sub-sample refined 16 samples achieves <= 1.5 dB error.\n")


if __name__ == "__main__":
    run_ray_sample_convergence_study()