"""
core/channel_physics.py
=======================
X-Band (8.0 GHz) Tactical Propagation & Physical Link Budget Engine.
Accurate SNR Taxonomy, Configuration Validation, and Array Shape Invariants.
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np


class LinkType(Enum):
    DOWNLINK = "DL"
    UPLINK = "UL"
    MCP_TRUNK = "MCP"
    CONTROL = "CTRL"


@dataclass(frozen=True)
class LinkProfile:
    bandwidth_hz: float
    tx_power_dbm: float
    tx_antenna_gain_dbi: float
    rx_antenna_gain_dbi: float
    snr_threshold_db: float
    spectral_eff_cap_bps_hz: float

    def __post_init__(self):
        if not np.isfinite(self.bandwidth_hz) or self.bandwidth_hz <= 0:
            raise ValueError(f"bandwidth_hz must be positive and finite, got {self.bandwidth_hz}")
        if not np.isfinite(self.spectral_eff_cap_bps_hz) or self.spectral_eff_cap_bps_hz <= 0:
            raise ValueError("spectral_eff_cap_bps_hz must be positive and finite")


LINK_SPECS: dict[LinkType, LinkProfile] = {
    LinkType.DOWNLINK: LinkProfile(33.0e6, 51.13, 30.0, 6.0, 8.46, 1.50),
    LinkType.UPLINK: LinkProfile(1.0e6, 29.44, 6.0, 30.0, 8.46, 1.50),
    LinkType.MCP_TRUNK: LinkProfile(80.0e6, 50.78, 30.0, 30.0, 18.27, 3.00),
    LinkType.CONTROL: LinkProfile(10.0e3, 32.18, 12.0, 6.0, 6.70, 1.00),
}


class ChannelPhysicsEngine:
    def __init__(self, carrier_freq_hz: float = 8.0e9, noise_figure_db: float = 4.0, seed: int = 42):
        if not np.isfinite(carrier_freq_hz) or carrier_freq_hz <= 0:
            raise ValueError(f"carrier_freq_hz must be positive and finite, got {carrier_freq_hz}")
        self.fc = carrier_freq_hz
        self.c = 299_792_458.0
        self.wavelength = self.c / self.fc
        self.thermal_noise_density_dbm_hz = -174.0
        self.noise_figure_db = noise_figure_db
        self.rain_k = 0.0038
        self.rain_alpha = 1.36
        self.snr_gap_beta = 0.8
        self.rng = np.random.default_rng(seed)

    def compute_fspl_db(self, slant_distances_m: np.ndarray) -> np.ndarray:
        d = np.asarray(slant_distances_m, dtype=np.float64)
        if not np.all(np.isfinite(d)) or np.any(d <= 0):
            raise ValueError("slant_distances_m must contain strictly positive finite values")
        return 20.0 * np.log10(d) + 20.0 * np.log10(self.fc) + 20.0 * np.log10(4.0 * np.pi / self.c)

    def compute_rain_attenuation_db(self, slant_distances_m: np.ndarray, rain_rate_mmhr: float) -> np.ndarray:
        d = np.asarray(slant_distances_m, dtype=np.float64)
        if not np.all(np.isfinite(d)) or np.any(d < 0):
            raise ValueError("slant_distances_m must contain non-negative finite values")
        if not np.isfinite(rain_rate_mmhr) or rain_rate_mmhr < 0:
            raise ValueError(f"rain_rate_mmhr must be non-negative finite float, got {rain_rate_mmhr}")
        if rain_rate_mmhr == 0.0:
            return np.zeros_like(d, dtype=np.float64)
        gamma_r = self.rain_k * (rain_rate_mmhr ** self.rain_alpha)
        effective_path_km = np.minimum(d / 1000.0, 25.0)
        return gamma_r * effective_path_km

    def generate_fast_fading_gain_linear(
        self, elevations_rad: np.ndarray, los_mask: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        elev = np.asarray(elevations_rad, dtype=np.float64)
        los = np.asarray(los_mask, dtype=bool)

        if elev.shape != los.shape or elev.ndim != 1:
            raise ValueError(f"Shape mismatch: elevations_rad {elev.shape} vs los_mask {los.shape}")
        if not np.all(np.isfinite(elev)):
            raise ValueError("elevations_rad must be finite")

        gen = rng if rng is not None else self.rng
        num_nodes = len(elev)
        elevations_deg = np.rad2deg(np.maximum(elev, 0.0))
        k_factor_linear = np.where(los, np.clip(10.0 ** (elevations_deg / 10.0), 1.0, 20.0), 0.0)

        fading_power = np.zeros(num_nodes, dtype=np.float64)
        for i in range(num_nodes):
            k = k_factor_linear[i]
            if k > 0.0:
                mean_spec = np.sqrt(k / (k + 1.0))
                sigma = np.sqrt(1.0 / (2.0 * (k + 1.0)))
                re = gen.normal(mean_spec, sigma)
                im = gen.normal(0.0, sigma)
            else:
                sigma = 1.0 / np.sqrt(2.0)
                re = gen.normal(0.0, sigma)
                im = gen.normal(0.0, sigma)
            fading_power[i] = re**2 + im**2
        return fading_power

    def compute_link_budget(
        self,
        link_type: LinkType,
        slant_distances_m: np.ndarray,
        elevations_rad: np.ndarray,
        los_mask: np.ndarray,
        excess_terrain_loss_db: np.ndarray,
        sector_gains_dbi: np.ndarray,
        rain_rate_mmhr: float = 0.0,
        rng: np.random.Generator | None = None,
        apply_fading: bool = True,
    ) -> dict:
        d = np.asarray(slant_distances_m, dtype=np.float64)
        elev = np.asarray(elevations_rad, dtype=np.float64)
        los = np.asarray(los_mask, dtype=bool)
        excess = np.asarray(excess_terrain_loss_db, dtype=np.float64)
        gains = np.asarray(sector_gains_dbi, dtype=np.float64)

        if not (d.shape == elev.shape == los.shape == excess.shape == gains.shape):
            raise ValueError(f"Input array shape mismatch: d={d.shape}, elev={elev.shape}, los={los.shape}, excess={excess.shape}, gains={gains.shape}")
        if d.ndim != 1 or len(d) == 0:
            raise ValueError("Per-link input arrays must be non-empty 1D arrays")
        if not np.all(np.isfinite(d)) or np.any(d <= 0):
            raise ValueError("slant_distances_m must contain strictly positive finite values")
        if not np.all(np.isfinite(elev)) or not np.all(np.isfinite(gains)):
            raise ValueError("elevations_rad and sector_gains_dbi must be finite numbers")
        if not np.all(np.isfinite(excess)) or np.any(excess < 0):
            raise ValueError("excess_terrain_loss_db must contain non-negative finite values")

        spec = LINK_SPECS[link_type]
        fspl = self.compute_fspl_db(d)
        rain_loss = self.compute_rain_attenuation_db(d, rain_rate_mmhr)

        if link_type in (LinkType.DOWNLINK, LinkType.MCP_TRUNK):
            tx_gain = gains
            rx_gain = spec.rx_antenna_gain_dbi
        elif link_type == LinkType.UPLINK:
            tx_gain = spec.tx_antenna_gain_dbi
            rx_gain = gains
        else:
            tx_gain = spec.tx_antenna_gain_dbi
            rx_gain = spec.rx_antenna_gain_dbi

        if apply_fading:
            fading_linear = self.generate_fast_fading_gain_linear(elev, los, rng)
            fading_db = 10.0 * np.log10(np.maximum(fading_linear, 1e-6))
        else:
            fading_db = np.zeros_like(d)

        rx_power_dbm = (
            spec.tx_power_dbm + tx_gain + rx_gain - fspl - rain_loss - excess + fading_db
        )
        noise_floor_dbm = self.thermal_noise_density_dbm_hz + 10.0 * np.log10(spec.bandwidth_hz) + self.noise_figure_db

        snr_db = rx_power_dbm - noise_floor_dbm
        is_outage = snr_db < spec.snr_threshold_db

        snr_linear = 10.0 ** (snr_db / 10.0)
        effective_snr = np.maximum(0.0, self.snr_gap_beta * snr_linear)

        spectral_eff = np.minimum(np.log2(1.0 + effective_snr), spec.spectral_eff_cap_bps_hz)
        achievable_rate_bps = spec.bandwidth_hz * spectral_eff

        achievable_rate_bps[is_outage] = 0.0
        spectral_eff[is_outage] = 0.0

        return {
            "rx_power_dbm": rx_power_dbm,
            "noise_floor_dbm": noise_floor_dbm,
            "snr_db": snr_db,
            "achievable_rate_bps": achievable_rate_bps,
            "spectral_efficiency": spectral_eff,
            "is_outage": is_outage,
            "rain_loss_db": rain_loss,
            "fspl_db": fspl,
        }