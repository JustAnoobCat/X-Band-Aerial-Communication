"""
core/kinematics.py
==================
Tactical ACN Kinematics & 4-Sector Directional Array with 3 dB Hysteresis.
Includes Configuration Validation, Yaw Verification, and Coincident Geometry Rejection.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OrbitConfig:
    semi_major_m: float = 40_000.0
    semi_minor_m: float = 20_000.0
    orbit_angle_rad: float = np.deg2rad(30.0)
    center_enu_m: tuple = (0.0, 0.0, 5000.0)
    altitude_var_m: float = 200.0
    period_s: float = 1800.0

    def __post_init__(self):
        if not np.isfinite(self.semi_major_m) or self.semi_major_m <= 0:
            raise ValueError("semi_major_m must be positive and finite")
        if not np.isfinite(self.semi_minor_m) or self.semi_minor_m <= 0:
            raise ValueError("semi_minor_m must be positive and finite")
        if self.semi_major_m < self.semi_minor_m:
            raise ValueError("semi_major_m cannot be smaller than semi_minor_m")
        if not np.isfinite(self.period_s) or self.period_s <= 0:
            raise ValueError("period_s must be positive and finite")
        if not np.isfinite(self.orbit_angle_rad):
            raise ValueError("orbit_angle_rad must be finite")
        if not np.isfinite(self.altitude_var_m) or self.altitude_var_m < 0:
            raise ValueError("altitude_var_m must be non-negative and finite")
        if len(self.center_enu_m) != 3 or not all(np.isfinite(self.center_enu_m)):
            raise ValueError("center_enu_m must be a 3-element finite tuple (x, y, z)")


@dataclass(frozen=True)
class AntennaConfig:
    num_sectors: int = 4
    boresight_offsets_rad: tuple = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
    peak_gain_dbi: float = 30.0
    beamwidth_3db_rad: float = np.deg2rad(70.0)
    max_attenuation_db: float = 30.0
    handover_hysteresis_db: float = 3.0

    def __post_init__(self):
        if not isinstance(self.num_sectors, (int, np.integer)) or self.num_sectors <= 0:
            raise ValueError("num_sectors must be positive integer")
        if self.num_sectors != len(self.boresight_offsets_rad):
            raise ValueError("num_sectors must match length of boresight_offsets_rad")
        if not all(np.isfinite(self.boresight_offsets_rad)):
            raise ValueError("boresight_offsets_rad must be finite")
        if not np.isfinite(self.peak_gain_dbi):
            raise ValueError("peak_gain_dbi must be finite")
        if not np.isfinite(self.beamwidth_3db_rad) or self.beamwidth_3db_rad <= 0:
            raise ValueError("beamwidth_3db_rad must be positive and finite")
        if not np.isfinite(self.max_attenuation_db) or self.max_attenuation_db < 0:
            raise ValueError("max_attenuation_db must be non-negative and finite")
        if not np.isfinite(self.handover_hysteresis_db) or self.handover_hysteresis_db < 0:
            raise ValueError("handover_hysteresis_db must be non-negative and finite")


class ACNKinematicsEngine:
    def __init__(self, orbit_cfg: OrbitConfig = OrbitConfig(), ant_cfg: AntennaConfig = AntennaConfig()):
        self.orbit = orbit_cfg
        self.ant = ant_cfg
        self.omega = (2.0 * np.pi) / self.orbit.period_s
        self.serving_sectors: np.ndarray | None = None

    def reset_handover_state(self, num_nodes: int):
        self.serving_sectors = np.zeros(num_nodes, dtype=np.int64)

    def compute_acn_state(self, t_sec: float) -> tuple[np.ndarray, np.ndarray, float]:
        if not np.isfinite(t_sec):
            raise ValueError(f"t_sec must be finite, got {t_sec}")

        wt = self.omega * t_sec
        cos_rot = np.cos(self.orbit.orbit_angle_rad)
        sin_rot = np.sin(self.orbit.orbit_angle_rad)

        x_orb = self.orbit.semi_major_m * np.cos(wt)
        y_orb = self.orbit.semi_minor_m * np.sin(wt)
        vx_orb = -self.orbit.semi_major_m * self.omega * np.sin(wt)
        vy_orb = self.orbit.semi_minor_m * self.omega * np.cos(wt)

        x_enu = x_orb * cos_rot - y_orb * sin_rot + self.orbit.center_enu_m[0]
        y_enu = x_orb * sin_rot + y_orb * cos_rot + self.orbit.center_enu_m[1]
        z_enu = self.orbit.center_enu_m[2] + self.orbit.altitude_var_m * np.sin(2 * wt)

        vx_enu = vx_orb * cos_rot - vy_orb * sin_rot
        vy_enu = vx_orb * sin_rot + vy_orb * cos_rot
        vz_enu = 2 * self.omega * self.orbit.altitude_var_m * np.cos(2 * wt)

        pos = np.array([x_enu, y_enu, z_enu], dtype=np.float64)
        vel = np.array([vx_enu, vy_enu, vz_enu], dtype=np.float64)
        yaw_rad = np.arctan2(vy_enu, vx_enu)

        return pos, vel, yaw_rad

    def compute_links_geometry(
        self, acn_pos: np.ndarray, acn_vel: np.ndarray, acn_yaw: float, gn_positions: np.ndarray
    ) -> dict:
        if not np.isfinite(acn_yaw):
            raise ValueError(f"acn_yaw must be a finite float, got {acn_yaw}")

        acn_p = np.asarray(acn_pos, dtype=np.float64)
        acn_v = np.asarray(acn_vel, dtype=np.float64)
        gn_p = np.asarray(gn_positions, dtype=np.float64)

        if not np.all(np.isfinite(acn_p)) or not np.all(np.isfinite(acn_v)):
            raise ValueError("ACN position and velocity must be finite")
        if gn_p.ndim != 2 or gn_p.shape[1] != 3 or not np.all(np.isfinite(gn_p)):
            raise ValueError("gn_positions must be shape (N, 3) finite array")

        num_nodes = gn_p.shape[0]
        if self.serving_sectors is None or len(self.serving_sectors) != num_nodes:
            self.reset_handover_state(num_nodes)

        delta_p = gn_p - acn_p.reshape(1, 3)
        raw_distances = np.linalg.norm(delta_p, axis=1)

        # Reject coincident geometry
        if np.any(raw_distances < 1.0):
            raise ValueError(f"Coincident geometry detected! Minimum physical slant range is 1.0 m, got {np.min(raw_distances):.3f} m")

        slant_distances = raw_distances
        elevations_rad = np.arcsin(np.clip(-delta_p[:, 2] / slant_distances, -1.0, 1.0))
        azimuths_rad = np.arctan2(delta_p[:, 1], delta_p[:, 0])

        range_rates = np.sum(delta_p * (-acn_v.reshape(1, 3)), axis=1) / slant_distances

        sector_gains_dbi = np.zeros((num_nodes, self.ant.num_sectors), dtype=np.float64)

        for m in range(self.ant.num_sectors):
            boresight = (acn_yaw + self.ant.boresight_offsets_rad[m]) % (2 * np.pi)
            delta_phi = (azimuths_rad - boresight + np.pi) % (2 * np.pi) - np.pi
            attenuation = np.minimum(
                12.0 * (delta_phi / self.ant.beamwidth_3db_rad) ** 2,
                self.ant.max_attenuation_db
            )
            sector_gains_dbi[:, m] = self.ant.peak_gain_dbi - attenuation

        for i in range(num_nodes):
            current_sec = self.serving_sectors[i]
            current_gain = sector_gains_dbi[i, current_sec]
            best_cand_sec = int(np.argmax(sector_gains_dbi[i, :]))
            cand_gain = sector_gains_dbi[i, best_cand_sec]

            if cand_gain >= (current_gain + self.ant.handover_hysteresis_db):
                self.serving_sectors[i] = best_cand_sec

        return {
            "slant_distances_m": slant_distances,
            "elevations_rad": elevations_rad,
            "azimuths_rad": azimuths_rad,
            "range_rates_mps": range_rates,
            "sector_gains_dbi": sector_gains_dbi,
            "best_sector": self.serving_sectors.copy(),
        }