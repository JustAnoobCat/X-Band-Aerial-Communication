"""
tests/test_phase1_verification.py
=================================
Automated Defense Verification Test Suite for Phase 1.
Hard assertions across all physical invariants, mass conservation, and validation.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from core.channel_physics import ChannelPhysicsEngine, LinkType
from core.kinematics import ACNKinematicsEngine, OrbitConfig
from core.mac_sync import MACSyncEngine
from core.queues import (
    AdmissionControlEngine,
    NodeQueueBank,
    NodeTier,
    TacticalPacket,
    TrafficClass,
)
from core.terrain_engine import DEMConfig, TerrainEngine


def test_1_kinematics_and_speed_limits():
    kin = ACNKinematicsEngine()
    omega = (2.0 * np.pi) / 1800.0
    v_min_theoretical = 20_000.0 * omega
    v_max_theoretical = 40_000.0 * omega

    speeds = []
    for t in np.linspace(0, 1800, 100):
        pos, vel, _ = kin.compute_acn_state(t)
        spd = np.linalg.norm(vel)
        speeds.append(spd)
        assert 4800.0 <= pos[2] <= 5200.0, f"Altitude {pos[2]} out of bounds!"

    assert np.isclose(np.min(speeds), v_min_theoretical, rtol=1e-2), "v_min mismatch!"
    assert np.isclose(np.max(speeds), v_max_theoretical, rtol=1e-2), "v_max mismatch!"
    print("[PASS] Test 1: Kinematics and speed limits verified.")


def test_2_sector_handover_hysteresis():
    kin = ACNKinematicsEngine()
    pos = np.array([0.0, 0.0, 5000.0])
    vel = np.array([100.0, 0.0, 0.0])
    yaw = 0.0

    gn_pos = np.array([[30_000.0 * np.cos(np.deg2rad(46.0)), 30_000.0 * np.sin(np.deg2rad(46.0)), 800.0]])
    kin.serving_sectors = np.array([0], dtype=np.int64)
    geo = kin.compute_links_geometry(pos, vel, yaw, gn_pos)
    assert geo["best_sector"][0] == 0, "Handover occurred without meeting 3 dB hysteresis threshold!"

    gn_pos_strong = np.array([[30_000.0 * np.cos(np.deg2rad(65.0)), 30_000.0 * np.sin(np.deg2rad(65.0)), 800.0]])
    geo2 = kin.compute_links_geometry(pos, vel, yaw, gn_pos_strong)
    assert geo2["best_sector"][0] == 1, "Handover failed to trigger when gain exceeded 3 dB hysteresis!"
    print("[PASS] Test 2: Stateful 3 dB sector handover hysteresis verified.")


def test_3_terrain_bilinear_and_knife_edge_diffraction():
    terrain = TerrainEngine()

    p1 = terrain.get_elevation_bilinear(np.array([[1000.0, 1000.0]]))
    p2 = terrain.get_elevation_bilinear(np.array([[1000.1, 1000.1]]))
    assert np.abs(p1 - p2) < 0.1, "Bilinear interpolation discontinuous!"

    wavelength = 0.0375
    d1 = np.array([50_000.0])
    d2 = np.array([50_000.0])

    loss_clear = terrain.compute_knife_edge_diffraction_db(np.array([0.0]), d1, d2, wavelength)
    assert loss_clear[0] == 0.0, f"Clear LoS must have 0.0 dB diffraction loss, got {loss_clear[0]} dB!"

    loss_obs = terrain.compute_knife_edge_diffraction_db(np.array([100.0]), d1, d2, wavelength)
    assert loss_obs[0] > 15.0, f"Expected >15 dB diffraction loss for 100m obstacle, got {loss_obs[0]:.2f} dB!"

    # Multi-dimensional broadcasting check
    h_b = np.array([1.0, 2.0])
    d1_b = np.array([[1000.0], [2000.0]])
    d2_b = np.array([1000.0, 2000.0])
    loss_matrix = terrain.compute_knife_edge_diffraction_db(h_b, d1_b, d2_b, wavelength)
    assert loss_matrix.shape == (2, 2), f"Diffraction broadcast failed to output (2, 2) shape, got {loss_matrix.shape}"
    print(f"[PASS] Test 3: Bilinear DEM, ITU-R P.526 diffraction and broadcasting verified (100m obstacle = {loss_obs[0]:.2f} dB).")


def test_4_predictive_terrain_lookahead_transition():
    kin = ACNKinematicsEngine()
    terrain = TerrainEngine()
    gn_target = np.array([[-40_000.0, 50_000.0, 800.0]])

    t_timeline = np.arange(0.0, 1800.0, 5.0)
    los_record = np.zeros(len(t_timeline), dtype=bool)

    for idx, t in enumerate(t_timeline):
        pos, _, _ = kin.compute_acn_state(t)
        mask, _ = terrain.compute_los_batch(pos, gn_target, num_samples=25)
        los_record[idx] = mask[0]

    flip_indices = np.where(los_record[:-1] & ~los_record[1:])[0]
    assert len(flip_indices) > 0, "Target [-40km, 50km] does not produce an LoS -> NLoS transition!"

    idx_flip = flip_indices[0]
    t_flip = t_timeline[idx_flip]

    t_test = max(0.0, t_flip - 1.0)
    lookahead = terrain.compute_predictive_lookahead(t_test, kin, gn_target, horizon_sec=6.0, step_sec=0.2)

    assert lookahead["current_los"][0] == True, "Expected link to be LoS before transition!"
    tau_mask = lookahead["tau_masking_sec"][0]
    assert 0.0 < tau_mask <= 6.0, f"tau_masking ({tau_mask}s) out of valid horizon!"
    print(f"[PASS] Test 4: Predictive lookahead verified (LoS->NLoS at t={t_flip:.1f}s, tau_masking={tau_mask:.2f}s).")


def test_5_timing_advance_and_guard_reduction():
    sync = MACSyncEngine()
    dist_200km = np.array([200_000.0])
    exact_ta = 2.0 * 200_000.0 / 299_792_458.0

    ta = sync.compute_timing_advance(dist_200km)
    assert np.isclose(ta[0], exact_ta, atol=1e-7), "Timing advance calculation incorrect!"

    guard_with_ta = sync.compute_guard_interval_requirement(True, dist_200km)
    guard_no_ta = sync.compute_guard_interval_requirement(False, np.array([5_000.0, 200_000.0]))

    assert guard_with_ta <= 10.0e-6, f"Guard with TA exceeds 10 us! ({guard_with_ta*1e6:.2f} us)"
    assert guard_no_ta > 600.0e-6, "Guard without TA should exceed 600 us!"
    print(f"[PASS] Test 5: Timing Advance ({ta[0]*1e3:.3f} ms) and guard reduction ({guard_with_ta*1e6:.1f} us) verified.")


def test_6_itu_rain_fade_isolation_and_hard_outage():
    chan = ChannelPhysicsEngine()
    d = np.array([200_000.0])
    elev = np.array([np.deg2rad(1.5)])
    los = np.array([True])
    excess = np.array([0.0])
    gain = np.array([30.0])

    b_clear = chan.compute_link_budget(LinkType.DOWNLINK, d, elev, los, excess, gain, rain_rate_mmhr=0.0, apply_fading=False)
    b_rain = chan.compute_link_budget(LinkType.DOWNLINK, d, elev, los, excess, gain, rain_rate_mmhr=25.0, apply_fading=False)

    rain_diff = b_clear["snr_db"][0] - b_rain["snr_db"][0]
    expected_rain_loss = 0.0038 * (25.0 ** 1.36) * 25.0
    assert np.isclose(rain_diff, expected_rain_loss, atol=0.05), "Rain attenuation did not match ITU-R formula!"

    b_outage = chan.compute_link_budget(LinkType.DOWNLINK, d, elev, np.array([False]), np.array([35.0]), gain, apply_fading=False)
    assert b_outage["is_outage"][0] == True, "Link should be in outage!"
    assert b_outage["achievable_rate_bps"][0] == 0.0, "Hard outage must clamp achievable rate to 0 bps!"
    print(f"[PASS] Test 6: ITU-R P.838 rain fade ({rain_diff:.2f} dB) and hard outage clamping verified.")


def test_7_queue_segmentation_and_non_preemptible_tier1():
    node = NodeQueueBank(node_id=1, base_tier=NodeTier.TIER_3_COMBAT)
    node.enqueue_packet(TrafficClass.C3_VIDEO, 15_000, 0.0, 101)

    bits_served = node.drain_capacity(allocated_bits=80_000, current_time_sec=0.010)
    assert bits_served == 80_000.0, "Did not serve full capacity!"
    assert node.get_total_backlog_bytes() == 5_000, "Packet was not segmented! Remaining should be 5000 B."

    cac = AdmissionControlEngine(max_concurrent_connections=1)
    mcp = NodeQueueBank(node_id=0, base_tier=NodeTier.TIER_1_STRATEGIC)
    admitted_mcp, _ = cac.request_admission(mcp)
    assert admitted_mcp, "MCP failed to admit!"

    combat = NodeQueueBank(node_id=2, base_tier=NodeTier.TIER_2_TACTICAL)
    combat.enqueue_packet(TrafficClass.C1_CRITICAL_C2, 128, 0.0, 999)
    admitted_combat, victim = cac.request_admission(combat)

    assert not admitted_combat, "Tier 1 MCP was improperly preempted!"
    assert victim is None, "Victim was selected despite Tier 1 protection!"
    print("[PASS] Test 7: MAC packet segmentation and Tier 1 non-preemptibility verified.")


def test_8_standby_readmission():
    cac = AdmissionControlEngine(max_concurrent_connections=1)
    combat = NodeQueueBank(node_id=10, base_tier=NodeTier.TIER_3_COMBAT)
    sensor = NodeQueueBank(node_id=20, base_tier=NodeTier.TIER_4_SENSOR)

    cac.request_admission(combat)
    admitted_sensor, _ = cac.request_admission(sensor)
    assert not admitted_sensor, "Sensor should have been deferred to standby!"
    assert 20 in cac.standby_terminals, "Sensor missing from standby table!"

    cac.release_connection(10)
    assert 20 in cac.active_terminals, "Sensor was not promoted to active upon channel release!"
    assert 20 not in cac.standby_terminals, "Promoted node still in standby table!"
    assert len(cac.active_terminals) == 1, "Active terminals count incorrect!"
    print("[PASS] Test 8: Automatic standby re-admission verified.")


def test_9_mentor_reference_rate_calibration():
    chan = ChannelPhysicsEngine()
    d_short = np.array([10_000.0])
    elev_high = np.array([np.deg2rad(30.0)])
    los = np.array([True])
    excess = np.array([0.0])
    gain = np.array([30.0])

    r_dl = chan.compute_link_budget(LinkType.DOWNLINK, d_short, elev_high, los, excess, gain, apply_fading=False)
    r_ul = chan.compute_link_budget(LinkType.UPLINK, d_short, elev_high, los, excess, gain, apply_fading=False)
    r_mcp = chan.compute_link_budget(LinkType.MCP_TRUNK, d_short, elev_high, los, excess, gain, apply_fading=False)
    r_ctrl = chan.compute_link_budget(LinkType.CONTROL, d_short, elev_high, los, excess, gain, apply_fading=False)

    rate_dl_mbps = r_dl["achievable_rate_bps"][0] / 1e6
    rate_ul_mbps = r_ul["achievable_rate_bps"][0] / 1e6
    rate_mcp_mbps = r_mcp["achievable_rate_bps"][0] / 1e6
    rate_ctrl_mbps = r_ctrl["achievable_rate_bps"][0] / 1e6

    assert np.isclose(rate_dl_mbps, 49.5, atol=0.1), f"DL Rate {rate_dl_mbps} != 49.5 Mbps!"
    assert np.isclose(rate_ul_mbps, 1.5, atol=0.05), f"UL Rate {rate_ul_mbps} != 1.5 Mbps!"
    assert np.isclose(rate_mcp_mbps, 240.0, atol=0.5), f"MCP Rate {rate_mcp_mbps} != 240.0 Mbps!"
    assert np.isclose(rate_ctrl_mbps, 0.01, atol=0.001), f"Control Rate {rate_ctrl_mbps} != 0.01 Mbps!"
    print(f"[PASS] Test 9: Mentor link rate calibration verified (DL: {rate_dl_mbps:.1f} Mbps, UL: {rate_ul_mbps:.1f} Mbps, MCP: {rate_mcp_mbps:.1f} Mbps, Ctrl: {rate_ctrl_mbps*1e3:.0f} kbps).")


def test_10_queue_mass_conservation_and_validation():
    node = NodeQueueBank(node_id=5, base_tier=NodeTier.TIER_3_COMBAT)
    t = 0.0

    node.enqueue_packet(TrafficClass.C1_CRITICAL_C2, 200, t, 1)
    node.enqueue_packet(TrafficClass.C2_VOICE, 800, t, 2)
    node.enqueue_packet(TrafficClass.C3_VIDEO, 5000, t, 3)

    node.drain_capacity(4000, t)

    t = 0.060
    node.purge_all_expired(t)

    for _, q in node.queues.items():
        q._assert_conservation()

    # 1. Reject non-integral DEM grid ratio
    try:
        DEMConfig(theater_span_m=240001.0, grid_resolution_m=500.0)
        assert False, "Failed to reject non-integral DEM grid ratio!"
    except ValueError:
        pass

    # 2. Reject out-of-bounds DEM coordinates
    terrain = TerrainEngine()
    try:
        terrain.get_elevation_bilinear(np.array([[1e9, 0.0]]))
        assert False, "Failed to reject out-of-bounds terrain coordinates!"
    except ValueError:
        pass

    # 3. Reject NaN rain rate
    chan = ChannelPhysicsEngine()
    try:
        chan.compute_rain_attenuation_db(np.array([1000.0]), float("nan"))
        assert False, "Failed to reject NaN rain rate!"
    except ValueError:
        pass

    # 4. Reject negative rain distance
    try:
        chan.compute_rain_attenuation_db(np.array([-100.0]), 10.0)
        assert False, "Failed to reject negative rain distance!"
    except ValueError:
        pass

    # 5. Reject negative excess terrain loss
    try:
        chan.compute_link_budget(
            LinkType.DOWNLINK,
            np.array([1000.0]),
            np.array([0.5]),
            np.array([True]),
            np.array([-10.0]),
            np.array([30.0]),
            apply_fading=False
        )
        assert False, "Failed to reject negative excess terrain loss!"
    except ValueError:
        pass

    # 6. Reject link budget shape mismatch
    try:
        chan.compute_link_budget(
            LinkType.DOWNLINK,
            np.array([1000.0]),
            np.array([0.5]),
            np.array([True, False]),
            np.array([0.0]),
            np.array([30.0]),
            apply_fading=False
        )
        assert False, "Failed to reject link budget shape mismatch!"
    except ValueError:
        pass

    # 7. Reject impossible packet remaining_bytes > total_bytes
    try:
        TacticalPacket(
            packet_id=1,
            traffic_class=TrafficClass.C1_CRITICAL_C2,
            total_bytes=10,
            remaining_bytes=11,
            arrival_time_sec=0.0,
            deadline_sec=0.020
        )
        assert False, "Failed to reject remaining_bytes > total_bytes!"
    except ValueError:
        pass

    # 8. Reject boolean packet_id
    try:
        TacticalPacket(
            packet_id=True,
            traffic_class=TrafficClass.C1_CRITICAL_C2,
            total_bytes=10,
            remaining_bytes=10,
            arrival_time_sec=0.0,
            deadline_sec=0.020
        )
        assert False, "Failed to reject boolean packet_id!"
    except ValueError:
        pass

    # 9. Reject invalid non-enum traffic class
    try:
        q_test = NodeQueueBank(1, NodeTier.TIER_3_COMBAT)
        q_test.enqueue_packet(99, 10, 0.0, 1)  # type: ignore
        assert False, "Failed to reject invalid non-enum traffic class!"
    except ValueError:
        pass

    # 10. Reject non-monotonic arrival timestamps
    try:
        q_order = NodeQueueBank(1, NodeTier.TIER_3_COMBAT)
        q_order.enqueue_packet(TrafficClass.C1_CRITICAL_C2, 10, 1.0, 1)
        q_order.enqueue_packet(TrafficClass.C1_CRITICAL_C2, 10, 0.0, 2)
        assert False, "Failed to reject out-of-order arrival timestamp!"
    except ValueError:
        pass

    # 11. Reject coincident zero-distance geometry
    try:
        kin = ACNKinematicsEngine()
        kin.compute_links_geometry(np.zeros(3), np.zeros(3), 0.0, np.zeros((1, 3)))
        assert False, "Failed to reject coincident zero-distance geometry!"
    except ValueError:
        pass

    # 12. Reject NaN ACN yaw
    try:
        kin = ACNKinematicsEngine()
        kin.compute_links_geometry(np.array([0.0, 0.0, 5000.0]), np.array([100.0, 0.0, 0.0]), float("nan"), np.array([[100.0, 0.0, 800.0]]))
        assert False, "Failed to reject NaN ACN yaw!"
    except ValueError:
        pass

    # 13. Reject negative diffraction distances
    try:
        terrain.compute_knife_edge_diffraction_db(np.array([1.0]), np.array([-1.0]), np.array([10.0]), 0.0375)
        assert False, "Failed to reject negative diffraction distance!"
    except ValueError:
        pass

    # 14. Reject non-finite OrbitConfig center
    try:
        OrbitConfig(center_enu_m=(np.nan, 0.0, 5000.0))
        assert False, "Failed to reject NaN orbit center!"
    except ValueError:
        pass

    print("[PASS] Test 10: Queue mass conservation and adversarial input rejection verified.")


def main():
    print("==================================================================")
    print("      RUNNING FORMAL PHASE 1 DEFENSE VERIFICATION SUITE           ")
    print("==================================================================\n")
    test_1_kinematics_and_speed_limits()
    test_2_sector_handover_hysteresis()
    test_3_terrain_bilinear_and_knife_edge_diffraction()
    test_4_predictive_terrain_lookahead_transition()
    test_5_timing_advance_and_guard_reduction()
    test_6_itu_rain_fade_isolation_and_hard_outage()
    test_7_queue_segmentation_and_non_preemptible_tier1()
    test_8_standby_readmission()
    test_9_mentor_reference_rate_calibration()
    test_10_queue_mass_conservation_and_validation()
    print("\n==================================================================")
    print("   ALL 10 PHASE 1 PHYSICAL & NETWORKING TESTS PASSED ASSERTIONS.  ")
    print("==================================================================\n")


if __name__ == "__main__":
    main()