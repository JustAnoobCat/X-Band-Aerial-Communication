"""
File: benchmarks/benchmark_baselines.py
Project: Adaptive Intelligent Resource Allocation & Scheduling for Tactical A2G Networks
Author: Aarush Mandoliya & Senior Defense Communications Advisor
Description:
    Production-grade Paired-Seed Monte Carlo Benchmarking Engine for Baseline Schedulers:
      1. Round Robin (RR)
      2. Proportional Fair (PF)
      3. Modified Largest Weighted Delay First (M-LWDF)
      4. Predictive Terrain-Aware M-LWDF (M-LWDF-TA)
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Auto-inject virtual environment site-packages
venv_paths = [
    PROJECT_ROOT / ".venv" / "lib64" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    PROJECT_ROOT / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
]
for vp in venv_paths:
    if vp.exists() and str(vp) not in sys.path:
        sys.path.insert(0, str(vp))

from envs.tactical_a2g_env import TacticalA2GEnv
from core.queues import TrafficClass
from algorithms.baselines import (
    BaseScheduler,
    make_scheduler,
    SCHEDULER_REGISTRY,
    NUM_NODES,
)


def calculate_jains_fairness(allocations: np.ndarray, active_only: bool = False) -> float:
    """Computes Jain's Fairness Index."""
    x = np.asarray(allocations, dtype=np.float64)
    if active_only:
        x = x[x > 0.0]
    if len(x) == 0:
        return 0.0
    sum_x = np.sum(x)
    sum_sq_x = np.sum(np.square(x))
    if sum_sq_x <= 0.0:
        return 0.0
    return float((sum_x ** 2) / (len(x) * sum_sq_x))


class BaselineBenchmarkHarness:
    def __init__(
        self,
        seeds: List[int],
        steps_per_episode: int = 1000,
        scheduler_names: Optional[List[str]] = None,
        output_dir: str = "data/benchmarks",
        scenario: str = "terrain_stress",
    ):
        self.seeds = [int(s) for s in seeds]
        self.steps_per_episode = int(steps_per_episode)
        self.scheduler_names = (
            scheduler_names if scheduler_names is not None else list(SCHEDULER_REGISTRY.keys())
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scenario = scenario.strip().lower()

    def run_single_episode(
        self,
        env: TacticalA2GEnv,
        scheduler: BaseScheduler,
        seed: int,
    ) -> Dict[str, Any]:
        options = {"scenario": self.scenario}
        obs, info = env.reset(seed=seed, options=options)
        scheduler.reset()

        frame_throughputs_mbps: List[float] = []
        sched_latencies_us: List[float] = []
        hol_delays_c1_ms: List[float] = []
        delivered_bytes_per_node = np.zeros(NUM_NODES, dtype=np.float64)

        cumulative_drops_by_class = {c: 0 for c in TrafficClass}
        cumulative_delivered_bits = 0.0

        evacuation_events = 0
        evacuation_allocations = 0
        evacuation_drains = 0
        completed_steps = 0

        # Snapshot initial queue packets for exact delta accounting
        initial_c1_drops = env.node_queues[3].queues[TrafficClass.C1_CRITICAL_C2].cumulative_dropped_packets

        for step in range(self.steps_per_episode):
            mask = env.action_masks() if hasattr(env, "action_masks") else info.get("action_mask", None)

            # 1. Pre-step threat detection on state s_t BEFORE action selection
            node_features = obs[: NUM_NODES * 18].reshape(NUM_NODES, 18)
            backlogs_norm = node_features[:, 0:4]
            los_flags = node_features[:, 11]
            raw_tau = node_features[:, 12]
            tau_sec = raw_tau * 2.0 if np.max(raw_tau) <= 1.0 and np.max(raw_tau) > 0.0 else raw_tau

            has_backlog = np.sum(backlogs_norm, axis=1) > 0.0
            imminent_nodes = set(np.flatnonzero(
                (los_flags > 0.5) & (tau_sec < 1.50) & (tau_sec > 0.0) & has_backlog
            ))

            # 2. Timed scheduler decision based on s_t
            t0 = time.perf_counter_ns()
            action = scheduler.select_action(obs, action_mask=mask, info=info)
            t1 = time.perf_counter_ns()
            sched_latencies_us.append((t1 - t0) / 1000.0)

            # 3. Step simulation environment
            obs, reward, terminated, truncated, info = env.step(action)
            completed_steps += 1

            tput_mbps = float(info.get("throughput_mbps", 0.0))
            frame_throughputs_mbps.append(tput_mbps)

            delivered_bits = float(info.get("delivered_bits", 0.0))
            cumulative_delivered_bits += delivered_bits

            node_bits = info.get("delivered_bits_by_node", None)
            if node_bits is not None:
                delivered_bytes_per_node += np.asarray(node_bits) / 8.0

            drops = info.get("drops_by_class", {})
            for c in TrafficClass:
                cumulative_drops_by_class[c] += drops.get(c, 0)

            # 4. Sample active C1 HoL delays
            current_time = float(getattr(env, "sim_time_sec", completed_steps * 0.010))
            for qbank in env.node_queues:
                c1_q = qbank.queues[TrafficClass.C1_CRITICAL_C2]
                if c1_q.total_bytes > 0:
                    hol_delays_c1_ms.append(c1_q.get_hol_delay(current_time) * 1000.0)

            # 5. Evaluate pre-occlusion service on threatened nodes identified at s_t
            if len(imminent_nodes) > 0:
                evacuation_events += len(imminent_nodes)
                allocated_nodes = set(int(n) for n in action if n < NUM_NODES)
                covered_nodes = imminent_nodes.intersection(allocated_nodes)
                evacuation_allocations += len(covered_nodes)

                if node_bits is not None:
                    drained_nodes = {n for n in covered_nodes if node_bits[n] > 0.0}
                    evacuation_drains += len(drained_nodes)

            if terminated or truncated:
                break

        if completed_steps != self.steps_per_episode:
            raise RuntimeError(
                f"Benchmark episode truncated unexpectedly: completed {completed_steps}/{self.steps_per_episode} steps!"
            )

        # Delta drop accounting
        c1_dropped = cumulative_drops_by_class[TrafficClass.C1_CRITICAL_C2]
        gn3_c1_drops = env.node_queues[3].queues[TrafficClass.C1_CRITICAL_C2].cumulative_dropped_packets - initial_c1_drops

        c1_delivered_approx = int((delivered_bytes_per_node.sum() * 0.15) / 128.0)
        c1_total = max(1, c1_dropped + c1_delivered_approx)
        c1_drop_ratio = float(c1_dropped / c1_total)

        total_dropped = sum(cumulative_drops_by_class.values())
        overall_total = max(1, total_dropped + int(delivered_bytes_per_node.sum() / 500.0))
        overall_drop_ratio = float(total_dropped / overall_total)

        jains_fairness_all = calculate_jains_fairness(delivered_bytes_per_node, active_only=False)
        jains_fairness_active = calculate_jains_fairness(delivered_bytes_per_node, active_only=True)

        evacuation_coverage_ratio = float(evacuation_allocations / max(1, evacuation_events))
        evacuation_coverage_ratio = float(np.clip(evacuation_coverage_ratio, 0.0, 1.0))

        evacuation_drain_ratio = float(evacuation_drains / max(1, evacuation_events))
        evacuation_drain_ratio = float(np.clip(evacuation_drain_ratio, 0.0, 1.0))

        throughput_mbps = float(np.mean(frame_throughputs_mbps)) if frame_throughputs_mbps else 0.0
        latencies_arr = np.asarray(sched_latencies_us, dtype=np.float64)

        return {
            "completed_steps": completed_steps,
            "throughput_mbps": throughput_mbps,
            "c1_dropped": c1_dropped,
            "c1_drop_ratio": c1_drop_ratio,
            "gn3_c1_drops": int(gn3_c1_drops),
            "overall_drop_ratio": overall_drop_ratio,
            "delivered_bytes_total": float(delivered_bytes_per_node.sum()),
            "jains_fairness_all": jains_fairness_all,
            "jains_fairness_active": jains_fairness_active,
            "evacuation_coverage_ratio": evacuation_coverage_ratio,
            "evacuation_drain_ratio": evacuation_drain_ratio,
            "evacuation_events": evacuation_events,
            "mean_sched_latency_us": float(np.mean(latencies_arr)),
            "p95_sched_latency_us": float(np.percentile(latencies_arr, 95)),
            "p99_sched_latency_us": float(np.percentile(latencies_arr, 99)),
            "p99_9_sched_latency_us": float(np.percentile(latencies_arr, 99.9)),
            "mean_c1_hol_ms": float(np.mean(hol_delays_c1_ms)) if hol_delays_c1_ms else 0.0,
            "p95_c1_hol_ms": float(np.percentile(hol_delays_c1_ms, 95)) if hol_delays_c1_ms else 0.0,
        }

    def execute_suite(self) -> Dict[str, Any]:
        print("=" * 105)
        print("TACTICAL A2G BASELINE SCHEDULERS: MONTE CARLO BENCHMARK HARNESS")
        print(f"Scenario: {self.scenario.upper()} | Seeds: {self.seeds} | Steps/Episode: {self.steps_per_episode}")
        print("=" * 105)

        env = TacticalA2GEnv()
        suite_results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in self.scheduler_names}

        for seed_idx, seed in enumerate(self.seeds):
            print(f"\n[ORBIT SEED {seed_idx + 1}/{len(self.seeds)}: SEED = {seed}]")

            seed_initial_digest: Optional[str] = None

            for name in self.scheduler_names:
                scheduler = make_scheduler(name)

                options = {"scenario": self.scenario}
                obs, _ = env.reset(seed=seed, options=options)

                mask = env.action_masks() if hasattr(env, "action_masks") else np.zeros((8, 17), bool)
                c_state = env._cached_frame_state if hasattr(env, "_cached_frame_state") else {}

                state_data = (
                    obs.tobytes()
                    + mask.tobytes()
                    + np.asarray(c_state.get("snr_db", np.zeros(NUM_NODES))).tobytes()
                    + np.asarray(c_state.get("los_mask", np.zeros(NUM_NODES))).tobytes()
                )
                obs_digest = hashlib.sha256(state_data).hexdigest()

                if seed_initial_digest is None:
                    seed_initial_digest = obs_digest
                else:
                    assert obs_digest == seed_initial_digest, (
                        f"Paired-seed initial physical state divergence detected on seed {seed} for {name}!"
                    )

                t_start = time.perf_counter()
                ep_metrics = self.run_single_episode(env, scheduler, seed)
                dt = time.perf_counter() - t_start

                suite_results[name].append(ep_metrics)
                print(
                    f"  - {name:<22}: Tput = {ep_metrics['throughput_mbps']:5.2f} Mbps | "
                    f"C1 Drop = {ep_metrics['c1_drop_ratio']*100:4.1f}% | "
                    f"GN3 Drops = {ep_metrics['gn3_c1_drops']} | "
                    f"Evac Cov = {ep_metrics['evacuation_coverage_ratio']*100:4.1f}% ({ep_metrics['evacuation_events']} events) | "
                    f"Latency = {ep_metrics['mean_sched_latency_us']:5.1f} us ({dt:4.1f}s)"
                )

        env.close()

        summary = self.aggregate_and_report(suite_results)
        self.save_results(suite_results, summary)
        return summary

    def aggregate_and_report(self, suite_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        summary = {}
        print("\n" + "=" * 110)
        print(f"EXECUTIVE DEFENSE RESEARCH BENCHMARK SUMMARY (SCENARIO: {self.scenario.upper()})")
        print("=" * 110)
        print(
            f"{'Scheduler':<22} | {'Tput (Mbps)':<14} | {'C1 Drop (%)':<12} | "
            f"{'GN3 Drops':<12} | {'Evac Cov (%)':<14} | {'Latency (us)':<12}"
        )
        print("-" * 110)

        for name in self.scheduler_names:
            runs = suite_results[name]
            tputs = [r["throughput_mbps"] for r in runs]
            c1_drops = [r["c1_drop_ratio"] * 100.0 for r in runs]
            gn3_drops = [r["gn3_c1_drops"] for r in runs]
            fairnesses_active = [r["jains_fairness_active"] for r in runs]
            evacuations = [r["evacuation_coverage_ratio"] * 100.0 for r in runs]
            latencies = [r["mean_sched_latency_us"] for r in runs]

            summary[name] = {
                "tput_mean": float(np.mean(tputs)),
                "tput_std": float(np.std(tputs)),
                "c1_drop_mean": float(np.mean(c1_drops)),
                "c1_drop_std": float(np.std(c1_drops)),
                "gn3_drops_mean": float(np.mean(gn3_drops)),
                "gn3_drops_std": float(np.std(gn3_drops)),
                "fairness_active_mean": float(np.mean(fairnesses_active)),
                "evacuation_coverage_mean": float(np.mean(evacuations)),
                "evacuation_coverage_std": float(np.std(evacuations)),
                "latency_mean": float(np.mean(latencies)),
                "latency_p99": float(np.mean([r["p99_sched_latency_us"] for r in runs])),
            }

            print(
                f"{name:<22} | "
                f"{summary[name]['tput_mean']:5.2f} +/- {summary[name]['tput_std']:4.2f} | "
                f"{summary[name]['c1_drop_mean']:4.1f} +/- {summary[name]['c1_drop_std']:3.1f}% | "
                f"{summary[name]['gn3_drops_mean']:5.1f} +/- {summary[name]['gn3_drops_std']:3.1f} | "
                f"{summary[name]['evacuation_coverage_mean']:5.1f}% +/- {summary[name]['evacuation_coverage_std']:4.1f}% | "
                f"{summary[name]['latency_mean']:5.1f} us"
            )

        print("=" * 110 + "\n")
        return summary

    def save_results(self, suite_results: Dict[str, Any], summary: Dict[str, Any]) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"benchmark_{self.scenario}_{timestamp}.json"
        npz_path = self.output_dir / f"benchmark_{self.scenario}_{timestamp}.npz"

        with open(json_path, "w") as f:
            json.dump({"scenario": self.scenario, "summary": summary, "suite_results": suite_results}, f, indent=2)

        np.savez_compressed(
            npz_path,
            scenario=self.scenario,
            schedulers=np.array(self.scheduler_names),
            seeds=np.array(self.seeds),
            tput_means=np.array([summary[s]["tput_mean"] for s in self.scheduler_names]),
            c1_drop_means=np.array([summary[s]["c1_drop_mean"] for s in self.scheduler_names]),
            gn3_drops_means=np.array([summary[s]["gn3_drops_mean"] for s in self.scheduler_names]),
            fairness_active_means=np.array([summary[s]["fairness_active_mean"] for s in self.scheduler_names]),
            evacuation_means=np.array([summary[s]["evacuation_coverage_mean"] for s in self.scheduler_names]),
            latency_means=np.array([summary[s]["latency_mean"] for s in self.scheduler_names]),
        )
        print(f"[ARTIFACTS] Telemetry saved to:\n  - {json_path}\n  - {npz_path}")


def main():
    parser = argparse.ArgumentParser(description="Tactical A2G Baseline Schedulers Benchmarking Harness")
    parser.add_argument("--fast", action="store_true", help="Run rapid smoke benchmark (2 seeds, 200 steps)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202, 303, 404],
                        help="List of random orbit seeds for paired Monte Carlo evaluation")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Simulation steps per episode (1000 steps = 10.0s)")
    parser.add_argument("--scenario", choices=["nominal", "terrain_stress"], default="terrain_stress",
                        help="Evaluation scenario identifier")
    parser.add_argument("--schedulers", type=str, nargs="+", default=list(SCHEDULER_REGISTRY.keys()),
                        help="Subset of schedulers to evaluate")
    parser.add_argument("--output-dir", type=str, default="data/benchmarks",
                        help="Directory to save benchmark metrics and NPZ data")
    args = parser.parse_args()

    if args.fast:
        seeds = [42, 101]
        steps = 200
        print(f"[FAST MODE] Executing rapid smoke benchmark ({args.scenario})...")
    else:
        seeds = args.seeds
        steps = args.steps

    harness = BaselineBenchmarkHarness(
        seeds=seeds,
        steps_per_episode=steps,
        scheduler_names=args.schedulers,
        output_dir=args.output_dir,
        scenario=args.scenario,
    )
    harness.execute_suite()


if __name__ == "__main__":
    main()