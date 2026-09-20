"""
core/queues.py
==============
Multi-Class Tactical Queue Bank with Mass Conservation Tracking,
Chronological Invariant Enforcement, and Preemptive Admission Control.
"""

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class TrafficClass(IntEnum):
    C1_CRITICAL_C2 = 0
    C2_VOICE = 1
    C3_VIDEO = 2
    C4_BULK_LOGS = 3


class NodeTier(IntEnum):
    TIER_1_STRATEGIC = 100
    TIER_2_TACTICAL = 50
    TIER_3_COMBAT = 20
    TIER_4_SENSOR = 5


@dataclass(frozen=True)
class ClassQoSProfile:
    name: str
    deadline_sec: float
    weight: float
    drop_on_expiry: bool

    def __post_init__(self):
        if not np.isfinite(self.deadline_sec) or self.deadline_sec <= 0:
            raise ValueError("deadline_sec must be strictly positive and finite")
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("weight must be strictly positive and finite")


QOS_PROFILES: dict[TrafficClass, ClassQoSProfile] = {
    TrafficClass.C1_CRITICAL_C2: ClassQoSProfile("C1_FlashC2", 0.020, 10.0, True),
    TrafficClass.C2_VOICE: ClassQoSProfile("C2_Voice", 0.050, 5.0, True),
    TrafficClass.C3_VIDEO: ClassQoSProfile("C3_Video", 0.150, 2.0, True),
    TrafficClass.C4_BULK_LOGS: ClassQoSProfile("C4_Bulk", 1.000, 0.5, False),
}


@dataclass
class TacticalPacket:
    packet_id: int
    traffic_class: TrafficClass
    total_bytes: int
    remaining_bytes: int
    arrival_time_sec: float
    deadline_sec: float

    def __post_init__(self):
        # Explicit boolean rejection (since bool is a subclass of int in Python)
        if isinstance(self.packet_id, bool) or not isinstance(self.packet_id, (int, np.integer)) or self.packet_id < 0:
            raise ValueError(f"packet_id must be a non-negative integer and not bool, got {self.packet_id}")
        if not isinstance(self.traffic_class, TrafficClass):
            raise ValueError(f"Invalid traffic_class: {self.traffic_class}")
        if self.total_bytes <= 0:
            raise ValueError("total_bytes must be strictly positive")
        if not (0 < self.remaining_bytes <= self.total_bytes):
            raise ValueError(f"remaining_bytes ({self.remaining_bytes}) must satisfy 0 < remaining_bytes <= total_bytes ({self.total_bytes})")
        if not np.isfinite(self.arrival_time_sec) or self.arrival_time_sec < 0:
            raise ValueError("arrival_time_sec must be finite and non-negative")
        if not np.isfinite(self.deadline_sec) or self.deadline_sec <= self.arrival_time_sec:
            raise ValueError("deadline_sec must be finite and strictly exceed arrival_time_sec")


class LogicalQueue:
    def __init__(self, traffic_class: TrafficClass):
        self.traffic_class = traffic_class
        self.profile = QOS_PROFILES[traffic_class]
        self.buffer: deque[TacticalPacket] = deque()
        self.total_bytes: int = 0
        self.cumulative_arrived_bytes: int = 0
        self.cumulative_dropped_bytes: int = 0
        self.cumulative_delivered_bytes: int = 0
        self.cumulative_dropped_packets: int = 0
        self.last_arrival_time_sec: float = -1.0

    def push(self, packet: TacticalPacket):
        if packet.arrival_time_sec < self.last_arrival_time_sec:
            raise ValueError(f"Non-monotonic arrival detected! Influx t={packet.arrival_time_sec} < last t={self.last_arrival_time_sec}")
        self.last_arrival_time_sec = packet.arrival_time_sec
        self.buffer.append(packet)
        self.total_bytes += packet.remaining_bytes
        self.cumulative_arrived_bytes += packet.remaining_bytes
        self._assert_conservation()

    def purge_expired_packets(self, current_time_sec: float) -> int:
        if not self.profile.drop_on_expiry:
            return 0
        purged_count = 0
        while self.buffer and (current_time_sec >= self.buffer[0].deadline_sec):
            expired = self.buffer.popleft()
            self.total_bytes -= expired.remaining_bytes
            self.cumulative_dropped_bytes += expired.remaining_bytes
            self.cumulative_dropped_packets += 1
            purged_count += 1

        self._assert_conservation()
        return purged_count

    def get_hol_delay(self, current_time_sec: float) -> float:
        if not self.buffer:
            return 0.0
        return max(0.0, current_time_sec - self.buffer[0].arrival_time_sec)

    def _assert_conservation(self):
        reconciled = self.total_bytes + self.cumulative_delivered_bytes + self.cumulative_dropped_bytes
        assert self.cumulative_arrived_bytes == reconciled, (
            f"Queue conservation breach: Arrived ({self.cumulative_arrived_bytes}) != Reconciled ({reconciled})"
        )


class NodeQueueBank:
    def __init__(self, node_id: int, base_tier: NodeTier):
        self.node_id = node_id
        self.base_tier = base_tier
        self.queues: dict[TrafficClass, LogicalQueue] = {
            c: LogicalQueue(c) for c in TrafficClass
        }
        self.window_arrived_bytes: int = 0
        self.window_served_bytes: int = 0
        self.arrival_rate_bps: float = 0.0
        self.service_rate_bps: float = 0.0
        self.growth_rate_bps: float = 0.0

    def enqueue_packet(self, pkt_class: TrafficClass, size_bytes: int, arrival_time_sec: float, pkt_id: int):
        if not isinstance(pkt_class, TrafficClass):
            raise ValueError(f"Invalid traffic class: {pkt_class}. Must be an instance of TrafficClass enum.")
        if size_bytes <= 0:
            raise ValueError(f"Packet size must be positive, got {size_bytes}")

        profile = QOS_PROFILES[pkt_class]
        deadline = arrival_time_sec + profile.deadline_sec
        packet = TacticalPacket(pkt_id, pkt_class, size_bytes, size_bytes, arrival_time_sec, deadline)
        self.queues[pkt_class].push(packet)
        self.window_arrived_bytes += size_bytes

    def update_growth_rate(self, dt_sec: float):
        if dt_sec <= 0 or not np.isfinite(dt_sec):
            raise ValueError(f"dt_sec must be strictly positive and finite, got {dt_sec}")
        self.arrival_rate_bps = (self.window_arrived_bytes * 8.0) / dt_sec
        self.service_rate_bps = (self.window_served_bytes * 8.0) / dt_sec
        self.growth_rate_bps = self.arrival_rate_bps - self.service_rate_bps
        self.window_arrived_bytes = 0
        self.window_served_bytes = 0

    def purge_all_expired(self, current_time_sec: float) -> dict[TrafficClass, int]:
        return {c: q.purge_expired_packets(current_time_sec) for c, q in self.queues.items()}

    def drain_capacity(self, allocated_bits: float, current_time_sec: float) -> float:
        if allocated_bits < 0 or not np.isfinite(allocated_bits):
            raise ValueError(f"allocated_bits must be non-negative finite number, got {allocated_bits}")

        remaining_budget_bytes = int(allocated_bits / 8.0)
        bytes_delivered = 0

        for c in [TrafficClass.C1_CRITICAL_C2, TrafficClass.C2_VOICE, TrafficClass.C3_VIDEO, TrafficClass.C4_BULK_LOGS]:
            q = self.queues[c]
            q.purge_expired_packets(current_time_sec)

            while q.buffer and remaining_budget_bytes > 0:
                pkt = q.buffer[0]
                if remaining_budget_bytes >= pkt.remaining_bytes:
                    bytes_delivered += pkt.remaining_bytes
                    remaining_budget_bytes -= pkt.remaining_bytes
                    q.total_bytes -= pkt.remaining_bytes
                    q.cumulative_delivered_bytes += pkt.remaining_bytes
                    q.buffer.popleft()
                else:
                    bytes_delivered += remaining_budget_bytes
                    pkt.remaining_bytes -= remaining_budget_bytes
                    q.total_bytes -= remaining_budget_bytes
                    q.cumulative_delivered_bytes += remaining_budget_bytes
                    remaining_budget_bytes = 0
                q._assert_conservation()

        self.window_served_bytes += bytes_delivered
        return float(bytes_delivered * 8.0)

    def get_total_backlog_bytes(self) -> int:
        return sum(q.total_bytes for q in self.queues.values())

    def get_priority_score(self) -> float:
        score = float(self.base_tier.value)
        if self.queues[TrafficClass.C1_CRITICAL_C2].total_bytes > 0:
            score += 15.0
        return score


class AdmissionControlEngine:
    def __init__(self, max_concurrent_connections: int = 8):
        if not isinstance(max_concurrent_connections, (int, np.integer)) or max_concurrent_connections <= 0:
            raise ValueError("max_concurrent_connections must be strictly positive integer")
        self.max_conns = max_concurrent_connections
        self.active_terminals: dict[int, NodeQueueBank] = {}
        self.standby_terminals: dict[int, NodeQueueBank] = {}

    def request_admission(self, terminal: NodeQueueBank) -> tuple[bool, int | None]:
        tid = terminal.node_id
        if tid in self.active_terminals:
            return True, None

        if len(self.active_terminals) < self.max_conns:
            self.active_terminals[tid] = terminal
            self.standby_terminals.pop(tid, None)
            return True, None

        incoming_score = terminal.get_priority_score()
        preemptible_victims = {
            k: v for k, v in self.active_terminals.items()
            if v.base_tier != NodeTier.TIER_1_STRATEGIC
        }

        if not preemptible_victims:
            self.standby_terminals[tid] = terminal
            return False, None

        victim_id = min(
            preemptible_victims,
            key=lambda k: (preemptible_victims[k].get_priority_score(), k)
        )
        victim = preemptible_victims[victim_id]

        if incoming_score > victim.get_priority_score():
            del self.active_terminals[victim_id]
            self.standby_terminals[victim_id] = victim
            self.active_terminals[tid] = terminal
            self.standby_terminals.pop(tid, None)
            return True, victim_id
        else:
            self.standby_terminals[tid] = terminal
            return False, None

    def release_connection(self, node_id: int):
        if node_id in self.active_terminals:
            del self.active_terminals[node_id]

        if self.standby_terminals and len(self.active_terminals) < self.max_conns:
            best_standby_id = max(
                self.standby_terminals,
                key=lambda k: (self.standby_terminals[k].get_priority_score(), -k)
            )
            promoted = self.standby_terminals.pop(best_standby_id)
            self.active_terminals[best_standby_id] = promoted