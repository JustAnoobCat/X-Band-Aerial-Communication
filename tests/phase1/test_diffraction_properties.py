"""
tests/phase1/test_diffraction_properties.py
===========================================
Property-Based Mathematical Verification for ITU-R P.526 Knife-Edge Diffraction:
- Path Reciprocity / Symmetry (d1 <-> d2 invariance)
- Strict Monotonicity with Obstacle Height (h_obs > 0)
- Clear-Path Zero-Loss Invariant (h_obs <= 0 -> 0.0 dB)
- Frequency / Wavelength Scaling (v proportional to 1/sqrt(lambda))
- Asymptotic Saturation Cap (45.0 dB max)
- Adversarial Non-Positive and Non-Finite Input Rejection
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from core.terrain_engine import TerrainEngine, DEMConfig


def test_diffraction_mathematical_properties():
    print("==================================================================")
    print("   PHASE 1: ITU-R P.526 DIFFRACTION PROPERTY & SYMMETRY AUDIT     ")
    print("==================================================================\n")

    terrain = TerrainEngine(DEMConfig(), seed=42)
    wavelength = 0.0375  # 8.0 GHz (lambda = c / 8e9 = 0.0375 m)

    # ------------------------------------------------------------------
    # PROPERTY 1: Path Reciprocity / Symmetry: J(h, d1, d2) == J(h, d2, d1)
    # ------------------------------------------------------------------
    h_test = np.array([5.0, 20.0, 50.0, 100.0])
    d1_asym = np.array([15_000.0, 25_000.0, 80_000.0, 10_000.0])
    d2_asym = np.array([45_000.0, 75_000.0, 20_000.0, 90_000.0])

    loss_forward = terrain.compute_knife_edge_diffraction_db(h_test, d1_asym, d2_asym, wavelength)
    loss_reverse = terrain.compute_knife_edge_diffraction_db(h_test, d2_asym, d1_asym, wavelength)

    max_reciprocity_diff = np.max(np.abs(loss_forward - loss_reverse))
    print(f"  Property 1 (Path Reciprocity)   : Max asymmetry delta = {max_reciprocity_diff:.6e} dB")
    assert np.allclose(loss_forward, loss_reverse, atol=1e-7), (
        "Diffraction violated path reciprocity when swapping d1 and d2!"
    )

    # ------------------------------------------------------------------
    # PROPERTY 2: Strict Monotonicity with Obstacle Height (h > 0)
    # ------------------------------------------------------------------
    d1_fixed = np.full(50, 50_000.0)
    d2_fixed = np.full(50, 50_000.0)
    # Increasing obstacle heights from 1 m to 300 m
    heights_monotonic = np.linspace(1.0, 300.0, 50)

    losses_monotonic = terrain.compute_knife_edge_diffraction_db(heights_monotonic, d1_fixed, d2_fixed, wavelength)
    # Check that every consecutive loss strictly increases
    diffs = np.diff(losses_monotonic)
    min_growth = np.min(diffs)
    print(f"  Property 2 (Height Monotonicity): Strictly positive gradient (min delta = +{min_growth:.4f} dB)")
    assert np.all(diffs > 0.0), "Diffraction loss failed to strictly increase with obstacle height!"

    # ------------------------------------------------------------------
    # PROPERTY 3: Clear-Path Zero-Loss Invariant (h <= 0 -> 0.0 dB)
    # ------------------------------------------------------------------
    h_clear = np.array([-500.0, -100.0, -10.0, -0.001, 0.0])
    d1_c = np.full(len(h_clear), 30_000.0)
    d2_c = np.full(len(h_clear), 30_000.0)
    losses_clear = terrain.compute_knife_edge_diffraction_db(h_clear, d1_c, d2_c, wavelength)

    print(f"  Property 3 (Clear-Path Invariant): h <= 0 max loss = {np.max(losses_clear):.6f} dB")
    assert np.all(losses_clear == 0.0), "Diffraction produced non-zero attenuation for clear path!"

    # ------------------------------------------------------------------
    # PROPERTY 4: Frequency / Wavelength Scaling
    # Shorter wavelength (higher frequency) has tighter Fresnel zone -> higher loss
    # ------------------------------------------------------------------
    lambda_uhf = 0.3000   # 1.0 GHz UHF
    lambda_xband = 0.0375 # 8.0 GHz X-band
    h_scale = np.array([50.0])
    d1_s = np.array([40_000.0])
    d2_s = np.array([40_000.0])

    loss_uhf = terrain.compute_knife_edge_diffraction_db(h_scale, d1_s, d2_s, lambda_uhf)[0]
    loss_xband = terrain.compute_knife_edge_diffraction_db(h_scale, d1_s, d2_s, lambda_xband)[0]

    print(f"  Property 4 (Frequency Scaling)  : 1 GHz loss = {loss_uhf:5.2f} dB < 8 GHz loss = {loss_xband:5.2f} dB")
    assert loss_xband > loss_uhf, "X-band loss was not strictly higher than UHF loss for same obstacle!"

    # ------------------------------------------------------------------
    # PROPERTY 5: Asymptotic Saturation Cap (45.0 dB)
    # ------------------------------------------------------------------
    h_extreme = np.array([5000.0])  # Enormous 5 km mountain obstacle
    loss_capped = terrain.compute_knife_edge_diffraction_db(h_extreme, d1_s, d2_s, wavelength)[0]
    print(f"  Property 5 (Asymptotic Cap)     : Extreme 5000m obstacle loss = {loss_capped:.2f} dB (Cap = 45.0 dB)")
    assert loss_capped == 45.0, f"Diffraction exceeded 45.0 dB cap: got {loss_capped} dB!"

    # ------------------------------------------------------------------
    # PROPERTY 6: Adversarial Input Rejections
    # ------------------------------------------------------------------
    # Negative d1
    try:
        terrain.compute_knife_edge_diffraction_db(np.array([10.0]), np.array([-1000.0]), np.array([1000.0]), wavelength)
        assert False, "Failed to reject negative d1 distance!"
    except ValueError:
        pass

    # Zero d2
    try:
        terrain.compute_knife_edge_diffraction_db(np.array([10.0]), np.array([1000.0]), np.array([0.0]), wavelength)
        assert False, "Failed to reject zero d2 distance!"
    except ValueError:
        pass

    # Negative wavelength
    try:
        terrain.compute_knife_edge_diffraction_db(np.array([10.0]), np.array([1000.0]), np.array([1000.0]), -0.0375)
        assert False, "Failed to reject negative wavelength!"
    except ValueError:
        pass

    # NaN obstacle height
    try:
        terrain.compute_knife_edge_diffraction_db(np.array([np.nan]), np.array([1000.0]), np.array([1000.0]), wavelength)
        assert False, "Failed to reject NaN obstacle height!"
    except ValueError:
        pass

    print("  Property 6 (Adversarial Bounds) : All invalid inputs successfully rejected with ValueError.")
    print("\n[PASS] All ITU-R P.526 mathematical properties, reciprocity, and bounds verified.\n")


if __name__ == "__main__":
    test_diffraction_mathematical_properties()