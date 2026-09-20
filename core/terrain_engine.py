"""
core/terrain_engine.py
======================
2.5D Digital Elevation Model (DEM) with Exact Grid Spacing,
Boundary Bounding, and ITU-R P.526 Knife-Edge Diffraction with True Broadcasting.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DEMConfig:
    theater_span_m: float = 240_000.0
    grid_resolution_m: float = 500.0
    base_elevation_m: float = 800.0
    mountain_peak_m: float = 3400.0

    def __post_init__(self):
        if not np.isfinite(self.theater_span_m) or self.theater_span_m <= 0:
            raise ValueError(f"theater_span_m must be positive and finite, got {self.theater_span_m}")
        if not np.isfinite(self.grid_resolution_m) or self.grid_resolution_m <= 0:
            raise ValueError(f"grid_resolution_m must be positive and finite, got {self.grid_resolution_m}")
        if not np.isfinite(self.base_elevation_m) or self.base_elevation_m < 0:
            raise ValueError(f"base_elevation_m cannot be negative, got {self.base_elevation_m}")
        if not np.isfinite(self.mountain_peak_m) or self.mountain_peak_m <= self.base_elevation_m:
            raise ValueError("mountain_peak_m must exceed base_elevation_m and be finite")

        ratio = self.theater_span_m / self.grid_resolution_m
        if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-7):
            raise ValueError(
                f"theater_span_m ({self.theater_span_m}) must be an integer multiple of grid_resolution_m ({self.grid_resolution_m})"
            )


class TerrainEngine:
    def __init__(self, cfg: DEMConfig = DEMConfig(), seed: int = 42):
        self.cfg = cfg
        self.num_cells = int(round(self.cfg.theater_span_m / self.cfg.grid_resolution_m))
        self.grid_size = self.num_cells + 1  # Exactly 481 samples
        self.half_span = self.cfg.theater_span_m / 2.0
        self.seed = seed
        self.dem_matrix = self._generate_synthetic_tactical_terrain(seed)

    def _generate_synthetic_tactical_terrain(self, seed: int) -> np.ndarray:
        x = np.linspace(-self.half_span, self.half_span, self.grid_size, endpoint=True)
        y = np.linspace(-self.half_span, self.half_span, self.grid_size, endpoint=True)
        xx, yy = np.meshgrid(x, y)

        ridge = np.exp(-((yy - 25_000.0) ** 2) / (2 * (8_000.0 ** 2))) * (self.cfg.mountain_peak_m - 1200.0)
        ridge *= np.exp(-((xx) ** 2) / (2 * (60_000.0 ** 2)))

        peak = np.exp(-((xx - 40_000.0) ** 2 + (yy - 50_000.0) ** 2) / (2 * (15_000.0 ** 2))) * self.cfg.mountain_peak_m

        rng = np.random.default_rng(seed)
        roughness = rng.uniform(-10.0, 10.0, size=(self.grid_size, self.grid_size))

        dem = self.cfg.base_elevation_m + ridge + peak + roughness
        return dem.astype(np.float32)

    def get_elevation_bilinear(self, xy_coords: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy_coords, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"xy_coords must have shape (N, 2), got {xy.shape}")
        if not np.all(np.isfinite(xy)):
            raise ValueError("xy_coords contains NaN or Infinite coordinates!")

        if np.any(xy < -self.half_span) or np.any(xy > self.half_span):
            raise ValueError(f"xy_coords contains positions outside theater bounds [{-self.half_span}, {self.half_span}]")

        u = (xy[:, 0] + self.half_span) / self.cfg.grid_resolution_m
        v = (xy[:, 1] + self.half_span) / self.cfg.grid_resolution_m

        u = np.clip(u, 0.0, float(self.num_cells))
        v = np.clip(v, 0.0, float(self.num_cells))

        x0 = np.floor(u).astype(int)
        y0 = np.floor(v).astype(int)
        x1 = np.minimum(x0 + 1, self.grid_size - 1)
        y1 = np.minimum(y0 + 1, self.grid_size - 1)

        dx = u - x0
        dy = v - y0

        q00 = self.dem_matrix[y0, x0]
        q10 = self.dem_matrix[y0, x1]
        q01 = self.dem_matrix[y1, x0]
        q11 = self.dem_matrix[y1, x1]

        elev = (
            q00 * (1.0 - dx) * (1.0 - dy)
            + q10 * dx * (1.0 - dy)
            + q01 * (1.0 - dx) * dy
            + q11 * dx * dy
        )
        return elev

    def compute_knife_edge_diffraction_db(
        self, h_obs: np.ndarray, d1: np.ndarray, d2: np.ndarray, wavelength_m: float
    ) -> np.ndarray:
        if not np.isfinite(wavelength_m) or wavelength_m <= 0:
            raise ValueError(f"wavelength_m must be positive and finite, got {wavelength_m}")

        h_arr = np.asarray(h_obs, dtype=np.float64)
        d1_arr = np.asarray(d1, dtype=np.float64)
        d2_arr = np.asarray(d2, dtype=np.float64)

        if not np.all(np.isfinite(h_arr)):
            raise ValueError("h_obs must contain finite numbers")
        if not np.all(np.isfinite(d1_arr)) or np.any(d1_arr <= 0):
            raise ValueError("d1 distances must be strictly positive and finite")
        if not np.all(np.isfinite(d2_arr)) or np.any(d2_arr <= 0):
            raise ValueError("d2 distances must be strictly positive and finite")

        # Explicit Multi-Dimensional Broadcasting
        try:
            h_b, d1_b, d2_b = np.broadcast_arrays(h_arr, d1_arr, d2_arr)
        except ValueError as err:
            raise ValueError(
                f"Incompatible shapes for diffraction calculation: h_obs {h_arr.shape}, d1 {d1_arr.shape}, d2 {d2_arr.shape}"
            ) from err

        loss_db = np.zeros(h_b.shape, dtype=np.float64)
        obstructed = h_b > 0.0
        if not np.any(obstructed):
            return loss_db

        h_sub = h_b[obstructed]
        d1_sub = d1_b[obstructed]
        d2_sub = d2_b[obstructed]

        denom = np.maximum(wavelength_m * d1_sub * d2_sub / (d1_sub + d2_sub), 1e-9)
        v = h_sub * np.sqrt(2.0 / denom)

        j_v = 6.9 + 20.0 * np.log10(np.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)
        loss_db[obstructed] = np.clip(j_v, 0.0, 45.0)
        return loss_db

    def compute_los_batch(
        self, acn_pos: np.ndarray, gn_positions: np.ndarray, wavelength_m: float = 0.0375, num_samples: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(num_samples, (int, np.integer)) or num_samples <= 0:
            raise ValueError(f"num_samples must be a positive integer, got {num_samples}")

        acn_p = np.asarray(acn_pos, dtype=np.float64)
        gn_p = np.asarray(gn_positions, dtype=np.float64)

        if acn_p.shape != (3,) or not np.all(np.isfinite(acn_p)):
            raise ValueError(f"acn_pos must be a finite 3D vector, got {acn_p}")
        if gn_p.ndim != 2 or gn_p.shape[1] != 3 or not np.all(np.isfinite(gn_p)):
            raise ValueError(f"gn_positions must be shape (N, 3) finite array, got {gn_p.shape}")

        num_gn = gn_p.shape[0]
        alphas = np.linspace(0.01, 0.99, num_samples, dtype=np.float32).reshape(1, num_samples, 1)

        p_acn = acn_p.reshape(1, 1, 3)
        p_gns = gn_p.reshape(num_gn, 1, 3)
        ray_points = (1.0 - alphas) * p_acn + alphas * p_gns

        flat_xy = ray_points[:, :, :2].reshape(-1, 2)
        terrain_heights = self.get_elevation_bilinear(flat_xy).reshape(num_gn, num_samples)

        ray_z = ray_points[:, :, 2]
        clearance = ray_z - terrain_heights
        los_mask = np.all(clearance > 0.0, axis=1)

        h_obs = np.maximum(0.0, -np.min(clearance, axis=1))
        worst_idx = np.argmin(clearance, axis=1)
        alpha_worst = alphas.flatten()[worst_idx]

        total_dist = np.maximum(np.linalg.norm(gn_p - acn_p.reshape(1, 3), axis=1), 1e-3)
        d1 = total_dist * (1.0 - alpha_worst)
        d2 = total_dist * alpha_worst

        diffraction_loss_db = self.compute_knife_edge_diffraction_db(h_obs, d1, d2, wavelength_m)
        diffraction_loss_db[los_mask] = 0.0

        return los_mask, diffraction_loss_db

    def compute_predictive_lookahead(
        self,
        t_now_sec: float,
        kinematics_engine,
        gn_positions: np.ndarray,
        horizon_sec: float = 2.0,
        step_sec: float = 0.1,
    ) -> dict:
        if not np.isfinite(t_now_sec):
            raise ValueError(f"t_now_sec must be finite, got {t_now_sec}")
        if horizon_sec <= 0 or step_sec <= 0 or not np.isfinite(horizon_sec) or not np.isfinite(step_sec):
            raise ValueError("horizon_sec and step_sec must be strictly positive and finite")

        gn_p = np.asarray(gn_positions, dtype=np.float64)
        if gn_p.ndim != 2 or gn_p.shape[1] != 3 or not np.all(np.isfinite(gn_p)):
            raise ValueError(f"gn_positions must be shape (N, 3) finite array, got {gn_p.shape}")

        num_gn = gn_p.shape[0]
        eval_times = np.arange(t_now_sec, t_now_sec + horizon_sec + 1e-6, step_sec)
        num_steps = len(eval_times)

        los_timeline = np.zeros((num_gn, num_steps), dtype=bool)
        for s_idx, t_eval in enumerate(eval_times):
            fut_pos, _, _ = kinematics_engine.compute_acn_state(t_eval)
            los_mask, _ = self.compute_los_batch(fut_pos, gn_positions, num_samples=30)
            los_timeline[:, s_idx] = los_mask

        tau_masking = np.full(num_gn, horizon_sec, dtype=np.float64)
        tau_recovery = np.full(num_gn, horizon_sec, dtype=np.float64)

        for i in range(num_gn):
            if los_timeline[i, 0]:
                transitions = np.where(~los_timeline[i, :])[0]
                if len(transitions) > 0:
                    tau_masking[i] = transitions[0] * step_sec
            else:
                recoveries = np.where(los_timeline[i, :])[0]
                if len(recoveries) > 0:
                    tau_recovery[i] = recoveries[0] * step_sec

        return {
            "current_los": los_timeline[:, 0],
            "tau_masking_sec": tau_masking,
            "tau_recovery_sec": tau_recovery,
        }