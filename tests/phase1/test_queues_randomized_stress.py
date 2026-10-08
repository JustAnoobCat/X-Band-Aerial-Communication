"""
tests/phase1/test_queues_randomized_stress.py
=============================================
Randomized Property-Based Stress Fuzzing for Tactical Queue Banks.
Executes 1,000 randomized operations across all 4 QoS classes.
Asserts mathematical mass conservation: Arrived == Queued + Delivered + Dropped.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from core.queues import NodeQueueBank, NodeTier, TrafficClass


def test_randomized_queue_mass_conservation():
    print("==================================================================")
    print("   PHASE 1: RANDOMIZED QUEUE MASS CONSERVATION STRESS (1000 OPS)  ")
    print("==================================================================\n")

    rng = np.random.default_rng(12345)
    node = NodeQueueBank(node_id=1, base_tier=NodeTier.TIER_3_COMBAT)
    current_time = 0.0
    packet_id = 0
    traffic_classes = list(TrafficClass)

    for op in range(1000):
        action_type = rng.choice(["enqueue", "drain", "advance_time"])

        if action_type == "enqueue":
            # Explicit index selection to preserve TrafficClass enum type
            c_idx = int(rng.integers(0, len(traffic_classes)))
            pkt_class = traffic_classes[c_idx]

            size = int(rng.integers(64, 30_000))
            packet_id += 1
            node.enqueue_packet(pkt_class, size, current_time, packet_id)

        elif action_type == "drain":
            granted_bits = float(rng.integers(1_000, 500_000))
            node.drain_capacity(granted_bits, current_time)

        elif action_type == "advance_time":
            dt = float(rng.uniform(0.001, 0.020))
            current_time += dt
            node.purge_all_expired(current_time)

        # STRICT CONSERVATION LAW CHECK ON EVERY OPERATION
        total_arrived = sum(q.cumulative_arrived_bytes for q in node.queues.values())
        total_queued = sum(q.total_bytes for q in node.queues.values())
        total_delivered = sum(q.cumulative_delivered_bytes for q in node.queues.values())
        total_dropped = sum(q.cumulative_dropped_bytes for q in node.queues.values())

        reconciled = total_queued + total_delivered + total_dropped
        assert total_arrived == reconciled, (
            f"LEAK DETECTED at op {op}: Arrived ({total_arrived}) != Reconciled ({reconciled})"
        )

    print(f"  Operations Processed    : 1,000 randomized actions")
    print(f"  Total Influx Traffic    : {total_arrived / 1e6:.2f} MB")
    print(f"  Total Data Delivered    : {total_delivered / 1e6:.2f} MB")
    print(f"  Total Expired / Dropped : {total_dropped / 1e6:.2f} MB")
    print(f"  Residual Queue Backlog  : {total_queued / 1e3:.2f} KB")
    print("\n[PASS] Queue mass conservation strictly preserved across 1,000 randomized operations.\n")


if __name__ == "__main__":
    test_randomized_queue_mass_conservation()