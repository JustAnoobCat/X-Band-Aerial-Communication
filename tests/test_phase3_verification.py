"""
File: tests/test_phase3_verification.py
Project: Adaptive Intelligent Resource Allocation & Scheduling for Tactical A2G Networks
Author: Aarush Mandoliya & Senior Defense Communications Advisor
Description:
    Tier-1 Phase 3 Acceptance Gate: Baseline Schedulers Verification Suite.
    Validates:
      Gate 1: Instantiation & Registry Integrity
      Gate 2: Action Dimensions, Typing & MultiDiscrete Bounds ([17]*8)
      Gate 3: Action-Mask Schema Enforcement & Invalid Partition Rejection
      Gate 4: Topology Locking, Telemetry Clamping & Zero-SNR Rate Telemetry Preservation
      Gate 5: M-LWDF Delay Prioritization
      Gate 6: M-LWDF-TA Exact Multipliers & End-to-End Action Selection Differentiation
      Gate 7: Round Robin Asymmetric Rotation & Anti-Starvation
      Gate 8: Proportional Fair EMA Equity & Dynamic Self-Throttling
      Gate 9: Closed-Loop Digital-Twin Integration & Faster-Than-Real-Time Execution (<= 5.0 ms)
      Gate 10: Empirical Soft-Real-Time Latency Benchmark (Nominal & Stress Workloads <= 250 us)
"""

import gc
import os
import sys
import time
from typing import Dict, Tuple, List, Optional
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithms.baselines import (
    BaseScheduler,
    RoundRobinScheduler,
    ProportionalFairScheduler,
    MLWDFScheduler,
    PredictiveMLWDFScheduler,
    make_scheduler,
    SCHEDULER_REGISTRY,
    NUM_NODES,
    NUM_RBGS,
    IDLE_ACTION_INDEX,
    CLASS_WEIGHTS,
    CLASS_DEADLINES_SEC,
    OBS_EXPECTED_DIM,
)


def create_synthetic_obs(
    queues: Optional[np.ndarray] = None,
    delays: Optional[np.ndarray] = None,
    snr_db: Optional[np.ndarray] = None,
    los: Optional[np.ndarray] = None,
    tau_masking: Optional[np.ndarray] = None,
    cac: Optional[np.ndarray] = None,
) -> np.ndarray:
    obs = np.zeros(OBS_EXPECTED_DIM, dtype=np.float64)

    _queues = queues if queues is not None else np.zeros((NUM_NODES, 4))
    _delays = delays if delays is not None else np.zeros((NUM_NODES, 4))
    _snr = snr_db if snr_db is not None else np.full(NUM_NODES, 20.0)
    _los = los if los is not None else np.ones(NUM_NODES)
    _tau = tau_masking if tau_masking is not None else np.full(NUM_NODES, 2.0)
    _cac = cac if cac is not None else np.ones(NUM_NODES)

    for i in range(NUM_NODES):
        offset = i * 18
        obs[offset : offset + 4] = _queues[i]
        obs[offset + 4 : offset + 8] = _delays[i]
        obs[offset + 8] = _snr[i]
        obs[offset + 9] = 100.0
        obs[offset + 10] = 1.0
        obs[offset + 11] = _los[i]
        obs[offset + 12] = _tau[i]
        obs[offset + 13] = _cac[i]
        obs[offset + 14] = 1.0

    obs[288] = 0.0
    obs[289] = 1.0
    obs[290] = 0.5
    return obs


def test_gate1_instantiation_registry() -> bool:
    print("  [Gate 1] Verifying Scheduler Registry and Instantiation...")
    expected_keys = {"round_robin", "proportional_fair", "mlwdf", "mlwdf_ta"}
    assert set(SCHEDULER_REGISTRY.keys()) == expected_keys
    for key in expected_keys:
        sched = make_scheduler(key)
        assert isinstance(sched, BaseScheduler)
        assert sched.num_nodes == NUM_NODES
        assert sched.num_rbgs == NUM_RBGS
        assert sched.idle_action == IDLE_ACTION_INDEX

    try:
        make_scheduler("non_existent_scheduler")
        assert False, "Failed to reject invalid scheduler name"
    except ValueError:
        pass
    return True


def test_gate2_action_dimensions_and_typing() -> bool:
    print("  [Gate 2] Verifying Action Dimensions and MultiDiscrete Bounds...")
    obs = create_synthetic_obs()
    for k in SCHEDULER_REGISTRY:
        sched = make_scheduler(k)
        sched.reset()
        action = sched.select_action(obs)
        assert isinstance(action, np.ndarray)
        assert action.shape == (NUM_RBGS,)
        assert issubclass(action.dtype.type, (np.integer, int))
        assert np.all(action >= 0) and np.all(action <= IDLE_ACTION_INDEX)
    return True


def test_gate3_hard_action_mask_compliance() -> bool:
    print("  [Gate 3] Verifying Action-Mask Compliance & Malformed Mask Rejection...")
    schedulers = [make_scheduler(k) for k in SCHEDULER_REGISTRY]
    obs = create_synthetic_obs()
    rng = np.random.RandomState(42)

    for _ in range(50):
        mask = rng.rand(NUM_RBGS, IDLE_ACTION_INDEX + 1) > 0.6
        mask[:, IDLE_ACTION_INDEX] = True
        mask[3, :IDLE_ACTION_INDEX] = False
        mask[7, :IDLE_ACTION_INDEX] = False

        for sched in schedulers:
            action = sched.select_action(obs, action_mask=mask)
            for rbg in range(NUM_RBGS):
                chosen_node = action[rbg]
                assert mask[rbg, chosen_node], f"Mask violation in {type(sched).__name__}"
            assert action[3] == IDLE_ACTION_INDEX
            assert action[7] == IDLE_ACTION_INDEX

    all_false_mask = np.ones((NUM_RBGS, IDLE_ACTION_INDEX + 1), dtype=bool)
    all_false_mask[2, :] = False
    for sched in schedulers:
        try:
            sched.select_action(obs, action_mask=all_false_mask)
            assert False, f"{type(sched).__name__} failed to reject all-false mask row"
        except ValueError:
            pass

    bad_shape_mask = np.ones((NUM_RBGS, 16), dtype=bool)
    for sched in schedulers:
        try:
            sched.select_action(obs, action_mask=bad_shape_mask)
            assert False, f"{type(sched).__name__} failed to reject (8, 16) mask shape"
        except ValueError:
            pass

    bad_dtype_mask = np.ones((NUM_RBGS, IDLE_ACTION_INDEX + 1), dtype=int)
    for sched in schedulers:
        try:
            sched.select_action(obs, action_mask=bad_dtype_mask)
            assert False, f"{type(sched).__name__} failed to reject non-boolean mask dtype"
        except ValueError:
            pass

    python_list_mask = [[True] * 17 for _ in range(8)]
    for sched in schedulers:
        try:
            sched.select_action(obs, action_mask=python_list_mask)  # type: ignore
            assert False, f"{type(sched).__name__} failed to reject Python list mask"
        except ValueError:
            pass

    return True


def test_gate4_cold_start_and_schema_validation() -> bool:
    print("  [Gate 4] Verifying Topology Locking & Telemetry Clamping...")
    schedulers = [make_scheduler(k) for k in SCHEDULER_REGISTRY]
    valid_obs = create_synthetic_obs()

    invalid_topologies = [
        {"num_nodes": 8, "idle_action": 8},
        {"num_rbgs": 4},
        {"idle_action": 15},
    ]
    for config in invalid_topologies:
        try:
            RoundRobinScheduler(**config)
            assert False, f"Failed to reject invalid topology: {config}"
        except ValueError:
            pass

    for sched in schedulers:
        try:
            sched.select_action(valid_obs[:-1])
            assert False, f"{type(sched).__name__} failed to reject short observation"
        except ValueError:
            pass

    for sched in schedulers:
        try:
            sched.select_action(np.append(valid_obs, 0.0))
            assert False, f"{type(sched).__name__} failed to reject oversized observation"
        except ValueError:
            pass

    nan_global_obs = valid_obs.copy()
    nan_global_obs[290] = np.nan
    for sched in schedulers:
        try:
            sched.select_action(nan_global_obs)
            assert False, f"{type(sched).__name__} failed to reject NaN in global features"
        except ValueError:
            pass

    for sched in schedulers:
        try:
            sched.select_action(create_synthetic_obs(queues=np.full((NUM_NODES, 4), -1.0)))
            assert False, f"{type(sched).__name__} accepted negative queue backlog"
        except ValueError:
            pass
        try:
            sched.select_action(create_synthetic_obs(delays=np.full((NUM_NODES, 4), -1.0)))
            assert False, f"{type(sched).__name__} accepted negative queue delay"
        except ValueError:
            pass
        try:
            sched.select_action(create_synthetic_obs(tau_masking=np.full(NUM_NODES, -1.0)))
            assert False, f"{type(sched).__name__} accepted negative tau_masking"
        except ValueError:
            pass

    bad_infos = [
        {"achievable_rates_per_rbg": np.ones(NUM_NODES - 1)},
        {"achievable_rates_per_rbg": np.full(NUM_NODES, np.nan)},
        {"achievable_rates_per_rbg": np.full(NUM_NODES, np.inf)},
        {"achievable_rates_per_rbg": np.full(NUM_NODES, -10.0)},
        {"achievable_rates_per_rbg": [1.0] * NUM_NODES},
    ]
    for sched in schedulers:
        for info in bad_infos:
            try:
                sched.select_action(valid_obs, info=info)
                assert False, f"{type(sched).__name__} accepted malformed telemetry: {info}"
            except ValueError:
                pass

    oversized_info = {"achievable_rates_per_rbg": np.full(NUM_NODES, 100.0e6)}
    snr_clean = np.full(NUM_NODES, 30.0)
    clamped_rates = schedulers[0].compute_achievable_rates(snr_clean, info=oversized_info)
    assert np.isclose(clamped_rates[0], 3.0 * 4.125e6), "MCP trunk rate was not clamped to 3.0 bps/Hz"
    assert np.isclose(clamped_rates[1], 1.5 * 4.125e6), "Soldier rate was not clamped to 1.5 bps/Hz"

    zero_snr_obs = np.zeros(NUM_NODES)
    authoritative_info = {"achievable_rates_per_rbg": np.full(NUM_NODES, 5.0e6)}
    for sched in schedulers:
        rates = sched.compute_achievable_rates(zero_snr_obs, info=authoritative_info)
        assert np.all(rates > 0.0), f"{type(sched).__name__} zeroed out authoritative rates when SNR was 0.0!"

    empty_obs = create_synthetic_obs()
    for sched in schedulers:
        sched.reset()
        for _ in range(100):
            action = sched.select_action(empty_obs)
            assert not np.any(np.isnan(sched.avg_throughput_bps))
            assert not np.any(np.isinf(sched.avg_throughput_bps))
            assert np.all(action >= 0) and np.all(action <= IDLE_ACTION_INDEX)

    return True


def test_gate5_mlwdf_qos_delay_prioritization() -> bool:
    print("  [Gate 5] Verifying M-LWDF Delay-Bounded Urgency Dynamics...")
    sched = MLWDFScheduler()
    sched.reset()

    queues = np.zeros((NUM_NODES, 4))
    delays = np.zeros((NUM_NODES, 4))
    snr_db = np.full(NUM_NODES, 10.0)

    queues[1, 3] = 1.0; delays[1, 3] = 0.01; snr_db[1] = 30.0
    queues[2, 0] = 0.5; delays[2, 0] = 0.90; snr_db[2] = 12.0

    obs = create_synthetic_obs(queues=queues, delays=delays, snr_db=snr_db)
    mask = np.zeros((NUM_RBGS, IDLE_ACTION_INDEX + 1), dtype=bool)
    mask[:, 1] = True
    mask[:, 2] = True
    mask[:, IDLE_ACTION_INDEX] = True

    action = sched.select_action(obs, action_mask=mask)
    assert np.all(action == 2), f"M-LWDF failed QoS priority gate! Got: {action}"
    return True


def test_gate6_predictive_mlwdf_ta_exact_multiplier() -> bool:
    print("  [Gate 6] Verifying M-LWDF-TA Exact Multipliers & Action Selection...")
    mlwdf = MLWDFScheduler()
    mlwdf_ta = PredictiveMLWDFScheduler(lookahead_gain=2.0, lookahead_sigma=0.5)
    mlwdf.reset()
    mlwdf_ta.reset()

    # Part A: Multiplier Invariant Math Check
    los_flag = np.ones(NUM_NODES)
    tau_masking = np.full(NUM_NODES, 2.0)
    tau_masking[4] = 2.00
    tau_masking[5] = 0.05

    boosts = mlwdf_ta.compute_terrain_boost(los_flag, tau_masking)
    expected_boost_4 = 1.0 + 2.0 * np.exp(-2.0 / 0.5)
    expected_boost_5 = 1.0 + 2.0 * np.exp(-0.05 / 0.5)

    assert np.isclose(boosts[4], expected_boost_4, atol=1e-4)
    assert np.isclose(boosts[5], expected_boost_5, atol=1e-4)

    # Part B: Controlled Action-Selection Behavioral Probe (Equal Channel + Equal Delay)
    queues = np.zeros((NUM_NODES, 4))
    delays = np.zeros((NUM_NODES, 4))
    snr_db = np.full(NUM_NODES, 20.0)
    tau_test = np.full(NUM_NODES, 2.0)

    queues[0, 0] = 1.0; delays[0, 0] = 0.50
    queues[1, 0] = 1.0; delays[1, 0] = 0.50
    tau_test[1] = 0.10

    obs = create_synthetic_obs(queues=queues, delays=delays, snr_db=snr_db, tau_masking=tau_test)
    mask = np.zeros((NUM_RBGS, IDLE_ACTION_INDEX + 1), dtype=bool)
    mask[:, 0] = True
    mask[:, 1] = True
    mask[:, IDLE_ACTION_INDEX] = True

    action_std = mlwdf.select_action(obs, action_mask=mask)
    assert np.all(action_std == 0), f"Standard M-LWDF tie-break mismatch: {action_std}"

    action_ta = mlwdf_ta.select_action(obs, action_mask=mask)
    assert np.all(action_ta == 1), f"M-LWDF-TA failed to prioritize imminent occlusion terminal: {action_ta}"

    return True


def test_gate7_round_robin_asymmetric_pointer_progression() -> bool:
    print("  [Gate 7] Verifying Round Robin Asymmetric Rotation & Anti-Starvation...")
    rr = RoundRobinScheduler()
    rr.reset()
    obs = create_synthetic_obs()

    mask = np.zeros((NUM_RBGS, IDLE_ACTION_INDEX + 1), dtype=bool)
    mask[:, [0, 1, 2, 3, 4]] = True
    mask[:, IDLE_ACTION_INDEX] = True

    expected_frames = [
        [0, 1, 2, 3, 4, 0, 1, 2],
        [3, 4, 0, 1, 2, 3, 4, 0],
        [1, 2, 3, 4, 0, 1, 2, 3],
        [4, 0, 1, 2, 3, 4, 0, 1],
        [2, 3, 4, 0, 1, 2, 3, 4],
    ]

    allocation_counts = {node: 0 for node in range(NUM_NODES)}

    for frame_idx, expected in enumerate(expected_frames):
        action = rr.select_action(obs, action_mask=mask)
        assert np.array_equal(action, expected), f"RR mismatch in frame {frame_idx + 1}!"
        for node in action:
            allocation_counts[node] += 1

    for node in range(5):
        assert allocation_counts[node] == 8, f"Anti-starvation failed for Node {node}"
    for node in range(5, NUM_NODES):
        assert allocation_counts[node] == 0, f"Inactive Node {node} received slots"

    return True


def test_gate8_proportional_fair_ema_equity() -> bool:
    print("  [Gate 8] Verifying Proportional Fair Long-Term Equity Self-Throttling...")
    pf = ProportionalFairScheduler(ema_window_frames=20)
    pf.reset()

    obs = create_synthetic_obs(snr_db=np.full(NUM_NODES, 20.0))
    pf.avg_throughput_bps[1] = 50.0e6
    pf.avg_throughput_bps[2] = 0.1e6

    mask = np.zeros((NUM_RBGS, IDLE_ACTION_INDEX + 1), dtype=bool)
    mask[:, 1] = True
    mask[:, 2] = True
    mask[:, IDLE_ACTION_INDEX] = True

    action = pf.select_action(obs, action_mask=mask)
    assert np.all(action == 2), f"PF failed equity gate: {action}"
    return True


def test_gate9_closed_loop_gym_step_integration() -> bool:
    print("  [Gate 9] Verifying Closed-Loop Digital-Twin Integration & Cadence (<= 5.0 ms)...")
    try:
        from envs.tactical_a2g_env import TacticalA2GEnv
    except ImportError as e:
        raise AssertionError(f"TacticalA2GEnv dependency unavailable for release gate: {e}") from e

    env = TacticalA2GEnv()
    eval_seeds = [42, 123, 456]
    steps_per_seed = 50
    schedulers = [
        make_scheduler("round_robin"),
        make_scheduler("proportional_fair"),
        make_scheduler("mlwdf"),
        make_scheduler("mlwdf_ta"),
    ]

    all_passed = True
    results = {}

    for sched in schedulers:
        seed_means = []

        for seed in eval_seeds:
            obs, info = env.reset(seed=seed)
            sched.reset()

            for _ in range(10):
                mask = env.action_masks() if hasattr(env, "action_masks") else info.get("action_mask", None)
                action = sched.select_action(obs, action_mask=mask, info=info)
                obs, _, _, _, info = env.step(action)

            latencies_ms = []

            for step in range(steps_per_seed):
                t_frame_start = time.perf_counter()
                mask = env.action_masks() if hasattr(env, "action_masks") else info.get("action_mask", None)
                action = sched.select_action(obs, action_mask=mask, info=info)
                obs, reward, terminated, truncated, info = env.step(action)
                latencies_ms.append((time.perf_counter() - t_frame_start) * 1000.0)

                assert not np.isnan(reward), f"NaN reward in {type(sched).__name__}"
                assert "reward_breakdown" in info, "Missing reward_breakdown in info"
                if terminated or truncated:
                    break

            lat_arr = np.asarray(latencies_ms, dtype=np.float64)
            assert lat_arr.size > 0, "No completed frames recorded in Gate 9!"
            seed_means.append(float(np.mean(lat_arr)))

        overall_mean_ms = float(np.mean(seed_means))
        worst_seed_mean_ms = float(np.max(seed_means))
        results[type(sched).__name__] = (overall_mean_ms, worst_seed_mean_ms)

        print(
            f"    - {type(sched).__name__:<25}: Multi-Seed Mean = {overall_mean_ms:5.2f} ms | "
            f"Worst Seed Mean = {worst_seed_mean_ms:5.2f} ms"
        )

        # Calibrated Digital Twin Budget: mean <= 4.5 ms, worst <= 5.0 ms (guarantees >= 2.0x faster than real time)
        if overall_mean_ms > 4.5 or worst_seed_mean_ms > 5.0:
            all_passed = False

    env.close()
    assert all_passed, f"Closed-loop frame execution exceeded digital-twin budget! {results}"
    return True


def test_gate10_realtime_execution_budget() -> bool:
    """
    Gate 10: Empirical Soft-Real-Time Latency Benchmark.
    Evaluates both Profile A (Nominal) and Profile B (Dynamic Tactical Stress) across 5 rounds x 1,000 steps.
    Empirical Decision Budget: Median Mean <= 250.0 us under dynamic stress.
    """
    import gc
    print("  [Gate 10] Profiling Empirical Soft-Real-Time Latencies (Nominal & Dynamic Stress)...")
    schedulers = [
        make_scheduler("round_robin"),
        make_scheduler("proportional_fair"),
        make_scheduler("mlwdf"),
        make_scheduler("mlwdf_ta"),
    ]

    obs_nominal = [create_synthetic_obs()]
    mask_nominal = [np.ones((NUM_RBGS, IDLE_ACTION_INDEX + 1), dtype=bool)]

    dynamic_obs = []
    dynamic_masks = []

    for frame_id in range(8):
        q = np.zeros((NUM_NODES, 4))
        w = np.zeros((NUM_NODES, 4))
        tau = np.full(NUM_NODES, 2.0)
        los = np.ones(NUM_NODES)
        snr = np.full(NUM_NODES, 20.0)

        if frame_id % 2 == 1:
            q[1, 0] = 0.8; w[1, 0] = 0.85
            q[4, 1] = 0.5; w[4, 1] = 0.60
            tau[4] = 0.05 + 0.1 * frame_id
        if frame_id in [3, 7]:
            snr[5] = 5.0
            los[5] = 0.0

        dynamic_obs.append(create_synthetic_obs(queues=q, delays=w, snr_db=snr, tau_masking=tau, los=los))

        m = np.ones((NUM_RBGS, IDLE_ACTION_INDEX + 1), dtype=bool)
        if frame_id % 2 == 1:
            m[3, :IDLE_ACTION_INDEX] = False
        if frame_id in [4, 5]:
            m[7, :IDLE_ACTION_INDEX] = False
        dynamic_masks.append(m)

    workloads = [
        ("Nominal", obs_nominal, mask_nominal),
        ("Dynamic Stress", dynamic_obs, dynamic_masks),
    ]

    num_rounds = 5
    steps_per_round = 1000
    all_passed = True
    results = {}

    for workload_name, obs_list, mask_list in workloads:
        print(f"    Evaluating Profile: {workload_name}...")
        num_patterns = len(obs_list)

        for sched in schedulers:
            sched.reset()
            round_means = []
            round_p95s = []
            round_p99s = []
            round_p99_9s = []
            round_maxs = []

            for w_idx in range(20):
                _ = sched.select_action(obs_list[w_idx % num_patterns], action_mask=mask_list[w_idx % num_patterns])

            for _ in range(num_rounds):
                latencies_us = np.zeros(steps_per_round, dtype=np.float64)
                gc.collect()
                gc.disable()
                try:
                    for i in range(steps_per_round):
                        t_obs = obs_list[i % num_patterns]
                        t_mask = mask_list[i % num_patterns]
                        t_start = time.perf_counter_ns()
                        _ = sched.select_action(t_obs, action_mask=t_mask)
                        t_end = time.perf_counter_ns()
                        latencies_us[i] = (t_end - t_start) / 1000.0
                finally:
                    gc.enable()

                round_means.append(float(np.mean(latencies_us)))
                round_p95s.append(float(np.percentile(latencies_us, 95)))
                round_p99s.append(float(np.percentile(latencies_us, 99)))
                round_p99_9s.append(float(np.percentile(latencies_us, 99.9)))
                round_maxs.append(float(np.max(latencies_us)))

            median_mean_us = float(np.median(round_means))
            median_p95_us = float(np.median(round_p95s))
            median_p99_us = float(np.median(round_p99s))
            median_p99_9_us = float(np.median(round_p99_9s))
            worst_max_us = float(np.max(round_maxs))

            results[f"{workload_name}_{type(sched).__name__}"] = (
                median_mean_us, median_p95_us, median_p99_us, median_p99_9_us, worst_max_us
            )

            print(
                f"      - {type(sched).__name__:<25}: Mean = {median_mean_us:5.2f} us | "
                f"P95 = {median_p95_us:5.2f} us | P99 = {median_p99_us:5.2f} us | "
                f"P99.9 = {median_p99_9_us:5.2f} us | Max = {worst_max_us:6.2f} us"
            )

            # Calibrated Decision Budget <= 250 us under dynamic stress
            if (
                median_mean_us > 250.0
                or median_p95_us > 500.0
                or median_p99_us > 700.0
                or median_p99_9_us > 1500.0
            ):
                all_passed = False

    assert all_passed, f"Scheduler latency breached empirical budget! {results}"
    return True


def run_phase3_verification() -> bool:
    print("=" * 80)
    print("PHASE 3 VERIFICATION SUITE: BASELINE SCHEDULERS & QOS ACCEPTANCE GATES")
    print("=" * 80)

    gates = [
        ("Gate 1: Instantiation & Registry Integrity", test_gate1_instantiation_registry),
        ("Gate 2: Action Dimensions, Typing & Bounds", test_gate2_action_dimensions_and_typing),
        ("Gate 3: Action-Mask Schema & Rejection", test_gate3_hard_action_mask_compliance),
        ("Gate 4: Topology Locking & Telemetry Clamping", test_gate4_cold_start_and_schema_validation),
        ("Gate 5: M-LWDF Delay Prioritization", test_gate5_mlwdf_qos_delay_prioritization),
        ("Gate 6: M-LWDF-TA Exact Multipliers & Action Selection", test_gate6_predictive_mlwdf_ta_exact_multiplier),
        ("Gate 7: Round Robin Asymmetric Rotation & Anti-Starvation", test_gate7_round_robin_asymmetric_pointer_progression),
        ("Gate 8: Proportional Fair EMA Equity", test_gate8_proportional_fair_ema_equity),
        ("Gate 9: Closed-Loop Digital-Twin Integration & Faster-Than-Real-Time Execution", test_gate9_closed_loop_gym_step_integration),
        ("Gate 10: Empirical Soft-Real-Time Latency Benchmark", test_gate10_realtime_execution_budget),
    ]

    passed = 0
    total = len(gates)

    for name, gate_fn in gates:
        try:
            if gate_fn():
                print(f"  [PASS] {name}")
                passed += 1
            else:
                print(f"  [FAIL] {name} - Returned False")
        except Exception as e:
            import traceback
            print(f"  [FAIL] {name} - Exception:")
            traceback.print_exc()

    print("=" * 80)
    print(f"PHASE 3 SUMMARY: {passed}/{total} GATES PASSED")
    print("=" * 80)
    return passed == total


if __name__ == "__main__":
    success = run_phase3_verification()
    sys.exit(0 if success else 1)