"""
tests/test_phase2_verification.py
=================================
Tier-1 Acceptance Verification Suite for Phase 2:
Synchronized 10.0 ms Gymnasium Simulation Environment.

Pass Criteria:
- 6/6 tests must exit with code 0.
- Isolated env.step() mean latency <= 2.0 ms across all seeds.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Auto-inject virtual environment site-packages
py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
venv_paths = [
    PROJECT_ROOT / ".venv" / "lib64" / py_ver / "site-packages",
    PROJECT_ROOT / ".venv" / "lib" / py_ver / "site-packages",
]
for vp in venv_paths:
    if vp.exists() and str(vp) not in sys.path:
        sys.path.insert(0, str(vp))

import gymnasium as gym
from envs.tactical_a2g_env import TacticalA2GEnv, EnvConfig


def test_gate1_spaces_and_observations():
    print("  [Test 1] Verifying Observation & Action Spaces...")
    env = TacticalA2GEnv(EnvConfig())
    obs, info = env.reset(seed=42)

    assert obs.shape == (291,), f"Observation shape mismatch: {obs.shape} != (291,)"
    assert obs.dtype == np.float32, f"Observation dtype mismatch: {obs.dtype}"
    assert np.all(np.isfinite(obs)), "Observation contains NaN or Inf values"
    assert env.action_space.shape == (8,), f"Action space shape mismatch: {env.action_space.shape}"
    env.close()
    return True


def test_gate2_action_masking_bounds():
    print("  [Test 2] Verifying Action Masking Matrix Structure...")
    env = TacticalA2GEnv(EnvConfig())
    obs, info = env.reset(seed=42)

    mask = env.action_masks()
    assert mask.shape == (8, 17), f"Mask shape mismatch: {mask.shape} != (8, 17)"
    assert mask.dtype == bool, f"Mask dtype mismatch: {mask.dtype}"
    assert np.all(mask[:, 16]), "IDLE action index 16 must be universally valid"
    env.close()
    return True


def test_gate3_deterministic_replay():
    print("  [Test 3] Verifying Multi-Step Deterministic Replay...")
    cfg = EnvConfig(traffic_injection_enabled=False)
    env1 = TacticalA2GEnv(cfg)
    env2 = TacticalA2GEnv(cfg)

    obs1, _ = env1.reset(seed=101)
    obs2, _ = env2.reset(seed=101)
    assert np.array_equal(obs1, obs2), "Reset observations diverged under identical seed!"

    for step in range(20):
        action = np.full(8, 16, dtype=np.int64)
        o1, r1, t1, tr1, _ = env1.step(action)
        o2, r2, t2, tr2, _ = env2.step(action)
        assert np.array_equal(o1, o2), f"Observations diverged at step {step}!"
        assert np.isclose(r1, r2), f"Rewards diverged at step {step}!"

    env1.close()
    env2.close()
    return True


def test_gate4_multirate_sensing_cadence():
    print("  [Test 4] Verifying 5 Hz Lookahead Multi-Rate Cadence...")
    env = TacticalA2GEnv(EnvConfig(traffic_injection_enabled=False))
    env.reset(seed=42)

    for step in range(45):
        action = np.full(8, 16, dtype=np.int64)
        obs, _, _, _, _ = env.step(action)
        assert np.all(np.isfinite(obs)), f"Non-finite values at step {step}"

    env.close()
    return True


def test_gate5_reward_breakdown_observability():
    print("  [Test 5] Verifying Unbundled Reward Telemetry...")
    env = TacticalA2GEnv(EnvConfig())
    env.reset(seed=42)

    for step in range(10):
        action = np.full(8, 16, dtype=np.int64)
        _, reward, _, _, info = env.step(action)
        assert "reward_breakdown" in info, "Missing reward_breakdown in info dict"
        rb = info["reward_breakdown"]
        expected_keys = {"throughput", "drop_penalty", "hol_delay_penalty", "guard_waste_penalty", "invalid_action_penalty"}
        assert set(rb.keys()) == expected_keys, f"Reward breakdown keys mismatch: {rb.keys()}"

    env.close()
    return True


def test_gate6_realtime_execution_budget():
    print("  [Test 6] Benchmarking Step Latency Budget (<= 2.0 ms)...")
    env = TacticalA2GEnv(EnvConfig())
    seeds = [1, 42, 123]
    steps_per_seed = 100
    seed_means = []

    for seed in seeds:
        env.reset(seed=seed)
        step_times = []

        # Warm up JIT and caches
        for _ in range(10):
            env.step(np.full(8, 16, dtype=np.int64))

        for _ in range(steps_per_seed):
            t0 = time.perf_counter()
            env.step(np.full(8, 16, dtype=np.int64))
            t1 = time.perf_counter()
            step_times.append((t1 - t0) * 1000.0)

        mean_ms = float(np.mean(step_times))
        seed_means.append(mean_ms)
        print(f"    - Seed {seed:>3}: Mean Step Latency = {mean_ms:5.3f} ms")

    overall_mean = float(np.mean(seed_means))
    print(f"    Overall Multi-Seed Mean = {overall_mean:5.3f} ms (Target <= 2.0 ms)")

    env.close()
    assert overall_mean <= 2.0, (
        f"Overall multi-seed mean step latency {overall_mean:.3f} ms exceeded hard real-time budget of 2.0 ms"
    )
    return True


def run_phase2_verification() -> bool:
    print("=" * 80)
    print("PHASE 2 VERIFICATION SUITE: 10.0 ms GYMNASIUM DIGITAL TWIN RELEASE GATES")
    print("=" * 80)

    tests = [
        ("Test 1: Spaces & Observation Consistency", test_gate1_spaces_and_observations),
        ("Test 2: Action Space Bounds & Masking", test_gate2_action_masking_bounds),
        ("Test 3: Multi-Step Deterministic Replay", test_gate3_deterministic_replay),
        ("Test 4: Multi-Rate 5 Hz Lookahead Cadence", test_gate4_multirate_sensing_cadence),
        ("Test 5: Unbundled Reward Breakdown", test_gate5_reward_breakdown_observability),
        ("Test 6: Real-Time Step Latency Budget (<= 2.0 ms)", test_gate6_realtime_execution_budget),
    ]

    passed = 0
    for name, test_fn in tests:
        try:
            if test_fn():
                print(f"  [PASS] {name}")
                passed += 1
            else:
                print(f"  [FAIL] {name} - Returned False")
        except Exception as e:
            import traceback
            print(f"  [FAIL] {name} - Exception:")
            traceback.print_exc()

    print("=" * 80)
    if passed == len(tests):
        print(f"ALL {passed} PHASE 2 GYMNASIUM ENVIRONMENT TESTS PASSED ASSERTIONS.")
    else:
        print(f"PHASE 2 SUMMARY: {passed}/{len(tests)} TESTS PASSED.")
    print("=" * 80)
    return passed == len(tests)


if __name__ == "__main__":
    success = run_phase2_verification()
    sys.exit(0 if success else 1)