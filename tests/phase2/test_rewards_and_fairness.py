"""
tests/phase2/test_rewards_and_fairness.py
=========================================
Tier-2 Deep Numerical Suite: Reward Decomposition, Monotonicity & Comparative Fairness.

Gates:
  1. Invalid-action penalty linearity (-10.0 per demoted RBG).
  2. Guard-waste hinge -2.0 * max(0, N_unique - 4), swept over N_unique = 1..8.
  3. Throughput term = +1.0e-6 * delivered_bits; monotone in allocated RBGs; 0.0 when idle.
  4. Drop-penalty hierarchy C1 > C2 > C3 (-500 / -250 / -100) + linear aggregation.
  5. HoL delay term = -10 * w_c * (W / tau_c)^2: absolute values, convexity, class weighting.
  6. Jain fairness: Round Robin >= 2x Greedy - 1e-3 over 100 frames.
"""

import atexit
import itertools
import math
import sys
import traceback
from pathlib import Path

import numpy as np

# Ensure project root and venv site-packages are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
for _vp in (
    PROJECT_ROOT / ".venv" / "lib64" / _py_ver / "site-packages",
    PROJECT_ROOT / ".venv" / "lib" / _py_ver / "site-packages",
):
    if _vp.exists() and str(_vp) not in sys.path:
        sys.path.insert(0, str(_vp))

from core.queues import QOS_PROFILES, TrafficClass
from envs.tactical_a2g_env import EnvConfig, TacticalA2GEnv


# --------------------------------------------------------------------------------------
# Spec constants
# --------------------------------------------------------------------------------------
N_NODES = 16
N_RBG = 8
IDLE = N_NODES                      # action index that means "do not schedule"
GUARD_FREE_NODES = 4
STRESS_NODE = 3
STRESS_SHADOW_SEC = 1.20
T0 = 10.0

C1 = TrafficClass.C1_CRITICAL_C2
C2 = TrafficClass.C2_VOICE
C3 = TrafficClass.C3_VIDEO
C4 = TrafficClass.C4_BULK_LOGS
TTL_EXPIRING_CLASSES = (C1, C2, C3)

SPEC_WEIGHTS = {
    "weight_throughput": 1.0e-6,
    "weight_packet_drop": 50.0,
    "weight_hol_delay": 10.0,
    "weight_guard_waste": 2.0,
    "weight_invalid_action": 10.0,
}
SPEC_DROP_PENALTY = {C1: -500.0, C2: -250.0, C3: -100.0}

COMPONENTS = (
    "throughput",
    "drop_penalty",
    "hol_delay_penalty",
    "guard_waste_penalty",
    "invalid_action_penalty",
)

REL_TOL = 1e-6
HOL_REL_TOL = 1e-4
FAIRNESS_ABS_TOL = 1e-3

_PID = itertools.count(1_000)
_IDLE_ACTION = np.full(N_RBG, IDLE, dtype=np.int64)
_IDLE_ACTION.setflags(write=False)


# --------------------------------------------------------------------------------------
# Shared fixtures / helpers
# --------------------------------------------------------------------------------------
_ISO_ENV = None


def _verify_spec_config(cfg: EnvConfig) -> None:
    assert cfg.num_nodes == N_NODES and cfg.num_rbgs == N_RBG, (
        f"Suite assumes {N_NODES} nodes / {N_RBG} RBGs, got {cfg.num_nodes} / {cfg.num_rbgs}"
    )
    assert not cfg.traffic_injection_enabled, "Isolation env must have traffic injection disabled"
    for name, spec in SPEC_WEIGHTS.items():
        got = getattr(cfg, name)
        assert math.isclose(got, spec, rel_tol=1e-12), f"{name} drifted from spec: {got} != {spec}"


def _iso_env() -> TacticalA2GEnv:
    global _ISO_ENV
    if _ISO_ENV is None:
        env = TacticalA2GEnv(EnvConfig(traffic_injection_enabled=False))
        _verify_spec_config(env.cfg)
        _ISO_ENV = env
    return _ISO_ENV


def _close_iso_env() -> None:
    global _ISO_ENV
    if _ISO_ENV is not None:
        _ISO_ENV.close()
        _ISO_ENV = None


atexit.register(_close_iso_env)


def _reset_iso(env: TacticalA2GEnv, **options):
    return env.reset(seed=42, options=options or None)


def _enqueue(env, node, cls, nbytes, arrival=None):
    now = env.sim_time_sec if arrival is None else arrival
    env.node_queues[node].enqueue_packet(cls, nbytes, now, next(_PID))


def _inject_bits(env, node, bits, cls=C4, pkt_bytes=256, max_pkts=8192) -> float:
    n_pkts = min(max_pkts, max(1, math.ceil(bits / (8.0 * pkt_bytes))))
    now = env.sim_time_sec
    for _ in range(n_pkts):
        env.node_queues[node].enqueue_packet(cls, pkt_bytes, now, next(_PID))
    return n_pkts * pkt_bytes * 8.0


def _schedulable(env) -> np.ndarray:
    return np.flatnonzero(env.action_masks()[0, :N_NODES])


def _prime_active_backlog(env, nbytes=500):
    active = sorted(env.cac_engine.active_terminals)
    for node in active:
        _enqueue(env, node, C1, nbytes)
    return active, _schedulable(env)


def _require(cond: bool, msg: str) -> None:
    assert cond, f"fixture precondition failed: {msg}"


def _check_breakdown(rb: dict, rel_tol: float = REL_TOL, **expected) -> None:
    unknown = set(expected) - set(COMPONENTS)
    assert not unknown, f"unknown reward components: {unknown}"
    for key in COMPONENTS:
        want = float(expected.get(key, 0.0))
        got = rb[key]
        if want == 0.0:
            assert got == 0.0, f"zero-leakage violated: {key} = {got!r} (expected exactly 0.0)"
        else:
            assert math.isclose(got, want, rel_tol=rel_tol), f"{key}: expected {want!r}, got {got!r}"


def _step(env, action):
    _, reward, _, _, info = env.step(np.asarray(action, dtype=np.int64))
    rb = info["reward_breakdown"]
    assert set(rb) == set(COMPONENTS), f"unexpected breakdown keys: {sorted(rb)}"
    assert rb["throughput"] >= 0.0, f"throughput must be non-negative, got {rb['throughput']}"
    for key in COMPONENTS[1:]:
        assert rb[key] <= 0.0, f"{key} must be non-positive, got {rb[key]}"
    assert math.isclose(reward, sum(rb.values()), rel_tol=1e-12, abs_tol=1e-12), (
        f"reward {reward!r} != sum(reward_breakdown) {sum(rb.values())!r}"
    )
    return reward, rb, info


def calculate_jains_fairness(allocations: np.ndarray) -> float:
    x = np.asarray(allocations, dtype=np.float64)
    sum_sq_x = float(np.dot(x, x))
    if x.size == 0 or sum_sq_x <= 0.0:
        return 0.0
    return float(x.sum() ** 2 / (x.size * sum_sq_x))


# --------------------------------------------------------------------------------------
# Gate 1: invalid-action penalty
# --------------------------------------------------------------------------------------
def test_invalid_action_penalty_scaling():
    print("  [Deep Test 1] Verifying Invalid Action Penalty Linearity...")
    env = _iso_env()
    w_inv = env.cfg.weight_invalid_action

    # (a) Empty backlog: node 0 has Q = 0 at reset
    _reset_iso(env)
    mask = env.action_masks()
    assert mask.shape == (N_RBG, N_NODES + 1)
    assert not mask[:, 0].any(), "Node 0 has Q=0 but the mask marks it schedulable"
    assert mask[:, IDLE].all(), "IDLE must always be a valid action"

    for k in (0, 1, 2, 4, 8):
        action = np.full(N_RBG, IDLE, dtype=np.int64)
        action[:k] = 0
        reward, rb, info = _step(env, action)
        expected = -w_inv * k
        _check_breakdown(rb, invalid_action_penalty=expected)
        assert info["invalid_allocations"] == k
        assert info["active_scheduled_nodes"] == 0
        assert info["delivered_bits"] == 0.0
        assert math.isclose(reward, expected, abs_tol=1e-12)

    # (b) Un-admitted terminal that DOES have backlog -> still invalid.
    _reset_iso(env)
    unadmitted = [i for i in range(N_NODES) if i not in env.cac_engine.active_terminals]
    _require(len(unadmitted) > 0, "no unadmitted terminal exists")
    node = unadmitted[0]
    env.cac_engine.request_admission(env.node_queues[node])
    _enqueue(env, node, C1, 500)
    backlog_before = env.node_queues[node].get_total_backlog_bytes()
    assert backlog_before > 0
    assert not env.action_masks()[:, node].any(), f"unadmitted node {node} with backlog must be masked"

    k = N_RBG // 2
    action = _IDLE_ACTION.copy()
    action[:k] = node
    reward, rb, info = _step(env, action)
    _check_breakdown(rb, invalid_action_penalty=-w_inv * k)
    assert info["invalid_allocations"] == k and info["delivered_bits"] == 0.0
    assert env.node_queues[node].get_total_backlog_bytes() == backlog_before

    # (c) SNR outage: terrain_stress forces GN 3 into outage after shadow time
    _reset_iso(env, scenario="terrain_stress")
    _require(STRESS_NODE in env.cac_engine.active_terminals, f"GN {STRESS_NODE} is not CAC-admitted")
    for _ in range(int(round(STRESS_SHADOW_SEC / env.cfg.frame_duration_sec)) + 1):
        _step(env, _IDLE_ACTION)
    _require(bool(env.get_frame_channel_state()["is_outage"][STRESS_NODE]),
             f"GN {STRESS_NODE} is not in outage after shadow frame")
    _enqueue(env, STRESS_NODE, C1, 500)
    assert not env.action_masks()[:, STRESS_NODE].any(), "outage node with backlog must be masked"

    k = 2
    action = _IDLE_ACTION.copy()
    action[:k] = STRESS_NODE
    reward, rb, info = _step(env, action)
    _check_breakdown(rb, invalid_action_penalty=-w_inv * k)
    assert info["invalid_allocations"] == k and info["delivered_bits"] == 0.0
    assert math.isclose(reward, -w_inv * k, abs_tol=1e-12)


# --------------------------------------------------------------------------------------
# Gate 2: guard-interval waste hinge
# --------------------------------------------------------------------------------------
def test_guard_waste_penalty_threshold():
    print("  [Deep Test 2] Verifying Guard Interval Waste Convexity (Threshold = 4)...")
    env = _iso_env()
    w_guard = env.cfg.weight_guard_waste
    w_tput = env.cfg.weight_throughput

    cases = [
        ("A: 4 nodes, paired RBGs", [0, 0, 1, 1, 2, 2, 3, 3]),
        ("B: 6 nodes + 2 IDLE", [0, 1, 2, 3, 4, 5, -1, -1]),
    ]
    cases += [(f"sweep: {k} unique node(s)", [i % k for i in range(N_RBG)]) for k in range(1, N_RBG + 1)]

    for label, slots in cases:
        _reset_iso(env)
        # Unmask outage for the duration of this isolated guard test
        env._cached_frame_state["is_outage"].fill(False)
        active, valid = _prime_active_backlog(env)
        _require(valid.size >= N_RBG,
                 f"need {N_RBG} schedulable nodes, mask allows {valid.tolist()} "
                 f"(active={active})")
        action = np.array([valid[s] if s >= 0 else IDLE for s in slots], dtype=np.int64)
        n_unique = len({s for s in slots if s >= 0})

        reward, rb, info = _step(env, action)

        expected_guard = -w_guard * max(0, n_unique - GUARD_FREE_NODES)
        _check_breakdown(
            rb,
            throughput=w_tput * info["delivered_bits"],
            guard_waste_penalty=expected_guard,
        )
        assert info["active_scheduled_nodes"] == n_unique, f"{label}: scheduled {info['active_scheduled_nodes']}"
        assert info["invalid_allocations"] == 0, f"{label}: unexpected invalid allocations"
        assert math.isclose(reward, rb["throughput"] + expected_guard, rel_tol=1e-12, abs_tol=1e-12)


# --------------------------------------------------------------------------------------
# Gate 3: throughput reward
# --------------------------------------------------------------------------------------
def test_throughput_reward_monotonicity():
    print("  [Deep Test 3] Verifying Throughput Monotonicity & Zero-Leakage...")
    env = _iso_env()
    w_tput = env.cfg.weight_throughput
    dt = env.cfg.frame_duration_sec

    _reset_iso(env)
    reward, rb, info = _step(env, _IDLE_ACTION)
    _check_breakdown(rb)
    assert info["delivered_bits"] == 0.0 and info["throughput_mbps"] == 0.0 and reward == 0.0

    node = 0
    delivered, rewards = [], []
    for k in (1, 2, 4, 8):
        _reset_iso(env)
        cap_bits = float(env.get_frame_channel_state()["rbg_capacity_bits"][node])
        _require(cap_bits > 0.0, "Node 0 has zero RBG capacity (outage) at reset")
        target = 2.0 * N_RBG * cap_bits
        injected = _inject_bits(env, node, target)
        _require(injected >= target, "backlog injection hit packet cap")
        _require(node in _schedulable(env), "Node 0 is not schedulable after injection")

        action = _IDLE_ACTION.copy()
        action[:k] = node
        reward, rb, info = _step(env, action)

        bits = info["delivered_bits"]
        assert 0.0 < bits <= k * cap_bits * (1.0 + 1e-9)
        _check_breakdown(rb, throughput=w_tput * bits)
        assert reward > 0.0 and math.isclose(reward, rb["throughput"], rel_tol=1e-12)

        by_node = info["delivered_bits_by_node"]
        assert by_node[node] == bits and np.count_nonzero(by_node) == 1
        assert math.isclose(info["throughput_mbps"], bits / dt / 1e6, rel_tol=1e-9)

        delivered.append(bits)
        rewards.append(reward)

    assert all(b >= a for a, b in zip(delivered, delivered[1:]))
    assert all(b >= a for a, b in zip(rewards, rewards[1:]))
    assert delivered[-1] > delivered[0] and rewards[-1] > rewards[0]


# --------------------------------------------------------------------------------------
# Gate 4: packet-drop hierarchy
# --------------------------------------------------------------------------------------
def test_packet_drop_priority_ordering():
    print("  [Deep Test 4] Verifying QoS Packet Drop Penalty Hierarchy (C1 > C2 > C3)...")
    env = _iso_env()
    w_drop = env.cfg.weight_packet_drop
    frame = env.cfg.frame_duration_sec
    penalties = {}

    # Test classes with active TTL expiration in queue architecture (C1, C2, C3)
    for cls in TTL_EXPIRING_CLASSES:
        _reset_iso(env, start_time_sec=T0)
        ttl = QOS_PROFILES[cls].deadline_sec
        _enqueue(env, 1, cls, 256, arrival=env.sim_time_sec - ttl - frame)
        reward, rb, info = _step(env, _IDLE_ACTION)

        expected = -w_drop * QOS_PROFILES[cls].weight
        _check_breakdown(rb, drop_penalty=expected)
        assert math.isclose(rb["drop_penalty"], SPEC_DROP_PENALTY[cls], rel_tol=REL_TOL), (
            f"{cls.name}: {rb['drop_penalty']} != spec {SPEC_DROP_PENALTY[cls]}"
        )
        assert info["dropped_packets"] == 1
        assert info["drops_by_class"][cls] == 1 and sum(info["drops_by_class"].values()) == 1
        assert math.isclose(reward, rb["drop_penalty"], rel_tol=1e-12)
        penalties[cls] = rb["drop_penalty"]

    mags = [abs(penalties[c]) for c in TTL_EXPIRING_CLASSES]
    assert mags[0] > mags[1] > mags[2], f"drop hierarchy violated: {mags}"
    assert math.isclose(penalties[C1] / penalties[C3], 5.0, rel_tol=1e-9), "C1:C3 penalty ratio must be 5x"

    # Linear aggregation across classes in one frame (arrivals strictly monotonically increasing with j)
    mix = {C1: 2, C2: 1, C3: 2}
    _reset_iso(env, start_time_sec=T0)
    now = env.sim_time_sec
    for cls, count in mix.items():
        ttl = QOS_PROFILES[cls].deadline_sec
        for j in range(count):
            # Arrival time strictly monotonically increases with j while remaining fully expired (> TTL)
            _enqueue(env, 1, cls, 256, arrival=now - ttl - frame + 1e-3 * j)
    reward, rb, info = _step(env, _IDLE_ACTION)
    expected = -w_drop * sum(n * QOS_PROFILES[c].weight for c, n in mix.items())
    _check_breakdown(rb, drop_penalty=expected)
    assert info["dropped_packets"] == sum(mix.values())
    assert all(info["drops_by_class"][c] == mix.get(c, 0) for c in TTL_EXPIRING_CLASSES)


# --------------------------------------------------------------------------------------
# Gate 5: HoL delay quadratic convexity
# --------------------------------------------------------------------------------------
def test_hol_delay_penalty_scaling():
    print("  [Deep Test 5] Verifying Quadratic Head-of-Line Delay Convexity...")
    env = _iso_env()
    w_delay = env.cfg.weight_hol_delay
    w1 = QOS_PROFILES[C1].weight
    ttl = QOS_PROFILES[C1].deadline_sec

    _reset_iso(env, start_time_sec=T0)
    _, rb, _ = _step(env, _IDLE_ACTION)
    _check_breakdown(rb)

    fractions = (0.0, 0.2, 0.4, 0.6, 0.8)
    pen = {}
    for f in fractions:
        _reset_iso(env, start_time_sec=T0)
        _enqueue(env, 2, C1, 256, arrival=env.sim_time_sec - f * ttl)
        reward, rb, info = _step(env, _IDLE_ACTION)
        _check_breakdown(rb, rel_tol=HOL_REL_TOL, hol_delay_penalty=-w_delay * w1 * f * f)
        assert info["dropped_packets"] == 0
        pen[f] = abs(rb["hol_delay_penalty"])

    assert math.isclose(pen[0.2], 4.0, rel_tol=HOL_REL_TOL), f"Case A: {pen[0.2]}"
    assert math.isclose(pen[0.8], 64.0, rel_tol=HOL_REL_TOL), f"Case B: {pen[0.8]}"
    assert math.isclose(pen[0.8] / pen[0.2], 16.0, rel_tol=HOL_REL_TOL)

    vals = [pen[f] for f in fractions]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    second_diff = [vals[i + 1] - 2.0 * vals[i] + vals[i - 1] for i in range(1, len(vals) - 1)]
    assert all(d > 0.0 for d in second_diff)

    _reset_iso(env, start_time_sec=T0)
    now = env.sim_time_sec
    for cls in (C1, C2, C3, C4):
        _enqueue(env, 1, cls, 256, arrival=now - 0.5 * QOS_PROFILES[cls].deadline_sec)
    _, rb, info = _step(env, _IDLE_ACTION)
    expected = -w_delay * 0.25 * sum(QOS_PROFILES[c].weight for c in (C1, C2, C3, C4))
    _check_breakdown(rb, rel_tol=HOL_REL_TOL, hol_delay_penalty=expected)
    assert info["dropped_packets"] == 0


# --------------------------------------------------------------------------------------
# Gate 6: comparative fairness (Jain's index)
# --------------------------------------------------------------------------------------
class _RoundRobin:
    def __init__(self):
        self.ptr = 0

    def __call__(self, env) -> np.ndarray:
        action = np.full(N_RBG, IDLE, dtype=np.int64)
        valid = _schedulable(env)
        if valid.size:
            start = int(np.searchsorted(valid, self.ptr)) % valid.size
            picks = valid[(start + np.arange(N_RBG)) % valid.size]
            action[:] = picks
            self.ptr = (int(picks[-1]) + 1) % N_NODES
        return action


def _greedy_policy(env) -> np.ndarray:
    return np.zeros(N_RBG, dtype=np.int64)


def _run_episode(policy, seed: int, ep_len: int):
    env = TacticalA2GEnv(EnvConfig(seed=seed, traffic_injection_enabled=True))
    try:
        _, info0 = env.reset(seed=seed)
        admitted = set(info0["active_terminals"])
        bits = np.zeros(N_NODES, dtype=np.float64)
        invalid = 0
        for _ in range(ep_len):
            _, _, terminated, truncated, info = env.step(policy(env))
            bits += info["delivered_bits_by_node"]
            invalid += info["invalid_allocations"]
            if terminated or truncated:
                break
        return bits / 8.0, invalid, admitted
    finally:
        env.close()


def test_comparative_fairness_jains_index():
    print("  [Deep Test 6] Auditing Comparative Fairness (Round Robin vs Greedy Starvation)...")
    ep_len, seed = 100, 123

    bytes_greedy, _, _ = _run_episode(_greedy_policy, seed, ep_len)
    bytes_rr, invalid_rr, admitted = _run_episode(_RoundRobin(), seed, ep_len)

    jains_greedy = calculate_jains_fairness(bytes_greedy)
    jains_rr = calculate_jains_fairness(bytes_rr)

    print(f"    - Greedy Starvation Network Jain Index: {jains_greedy:6.4f}")
    print(f"    - Round Robin Fair Network Jain Index:  {jains_rr:6.4f}")
    print(f"    - Fairness Advantage Ratio:             {jains_rr / max(1e-6, jains_greedy):6.2f}x")

    assert bytes_greedy[0] > 0.0 and np.count_nonzero(bytes_greedy) == 1
    assert math.isclose(jains_greedy, 1.0 / N_NODES, rel_tol=1e-9)
    assert invalid_rr == 0
    served_rr = set(np.flatnonzero(bytes_rr).tolist())
    assert len(served_rr) >= 2
    assert served_rr <= admitted

    # Invariant: Round Robin >= 2x greedy, with the 1e-3 finite-window tolerance
    assert jains_rr >= 2.0 * jains_greedy - FAIRNESS_ABS_TOL, (
        f"Round Robin failed the 2x fairness advantage! RR: {jains_rr:.4f}, Greedy: {jains_greedy:.4f}"
    )


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------
def run_rewards_and_fairness_suite() -> bool:
    print("=" * 80)
    print("PHASE 2 DEEP NUMERICAL SUITE: REWARD MONOTONICITY & FAIRNESS AUDIT")
    print("=" * 80)

    tests = [
        ("Test 1: Invalid Action Penalty Linearity (-10.0 / slot)", test_invalid_action_penalty_scaling),
        ("Test 2: Guard Waste Threshold Convexity (N > 4)", test_guard_waste_penalty_threshold),
        ("Test 3: Throughput Reward Monotonicity (+1.0e-6 / bit)", test_throughput_reward_monotonicity),
        ("Test 4: QoS Packet Drop Penalty Hierarchy (C1..C3)", test_packet_drop_priority_ordering),
        ("Test 5: Quadratic HoL Delay Penalty Convexity", test_hol_delay_penalty_scaling),
        ("Test 6: Comparative Fairness Audit (Jain Index RR >= 2x Greedy)", test_comparative_fairness_jains_index),
    ]

    passed = 0
    try:
        for name, test_fn in tests:
            try:
                test_fn()
            except Exception:
                print(f"  [FAIL] {name} - Exception:")
                traceback.print_exc()
            else:
                print(f"  [PASS] {name}")
                passed += 1
    finally:
        _close_iso_env()

    print("=" * 80)
    print(f"REWARD & FAIRNESS SUITE SUMMARY: {passed}/{len(tests)} TESTS PASSED.")
    print("=" * 80)
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if run_rewards_and_fairness_suite() else 1)