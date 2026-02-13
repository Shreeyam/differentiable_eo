"""
Differentiable Constellation Optimization using dSGP4

Optimizes satellite constellation geometry (inclination, RAAN, mean anomaly)
at a fixed altitude to maximize ground coverage and minimize revisit gaps,
using gradient-based optimization through a differentiable orbit propagator.

Key ideas:
  1. SGP4 propagation is differentiable via dsgp4 (PyTorch autograd)
  2. Coverage is smoothed with a soft sigmoid → differentiable
  3. Revisit gap is tracked with a leaky integrator → differentiable
  4. Worst-case revisit uses LogSumExp soft-max → differentiable
"""

import torch
import dsgp4
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Config:
    # Constellation
    n_planes: int = 4
    n_sats_per_plane: int = 3
    target_alt_km: float = 550.0

    # Initial (deliberately suboptimal) configuration
    initial_inc_deg: float = 30.0  # too low for global coverage
    initial_raan_offsets_deg: list = None  # irregular spacing

    # Propagation
    prop_duration_hours: float = 24.0
    n_time_steps: int = 240

    # Ground grid
    n_lat: int = 18 * 2
    n_lon: int = 36 * 2

    # Coverage model
    min_elevation_deg: float = 10.0
    softness_deg: float = 2.0

    # Revisit model
    revisit_logsumexp_temp: float = 10.0  # temperature for soft-max (minutes)
    revisit_weight: float = 0.005  # weight of revisit loss vs coverage loss

    # Optimization
    n_iterations: int = 2000
    lr: float = 2e-3

    # Constants
    mu: float = 398600.4418
    R_earth: float = 6378.137
    earth_rot_rad_per_min: float = 7.2921159e-5 * 60

    def __post_init__(self):
        if self.initial_raan_offsets_deg is None:
            self.initial_raan_offsets_deg = [0, 40, 100, 200]  # irregular


cfg = Config()

# ============================================================
# Coordinate transforms (all differentiable via torch)
# ============================================================

def gmst_at_epoch(epoch_year, epoch_days):
    y = epoch_year
    jd = 367 * y - int(7 * (y + int(10 / 12)) / 4) + int(275 / 9) + epoch_days + 1721013.5
    T = (jd - 2451545.0) / 36525.0
    gmst_deg = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T**2
    return math.radians(gmst_deg % 360)


def teme_to_ecef(positions, gmst_rad):
    cos_g = math.cos(gmst_rad)
    sin_g = math.sin(gmst_rad)
    x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
    return torch.stack([cos_g * x + sin_g * y, -sin_g * x + cos_g * y, z], dim=-1)


def compute_elevation(sat_ecef, ground_ecef):
    """Elevation angle (deg) from ground to satellite. [N_sat, 3] x [N_ground, 3] -> [N_sat, N_ground]"""
    rel = sat_ecef.unsqueeze(1) - ground_ecef.unsqueeze(0)
    ground_unit = ground_ecef / torch.norm(ground_ecef, dim=-1, keepdim=True)
    rel_dist = torch.norm(rel, dim=-1)
    sin_el = (rel * ground_unit.unsqueeze(0)).sum(dim=-1) / (rel_dist + 1e-10)
    return torch.rad2deg(torch.asin(torch.clamp(sin_el, -1.0, 1.0)))


def soft_coverage(elevation_deg, min_el, temperature):
    return torch.sigmoid((elevation_deg - min_el) / temperature)


def hard_coverage(elevation_deg, min_el):
    return (elevation_deg >= min_el).float()


# ============================================================
# TLE helpers
# ============================================================

def mean_motion_from_alt(alt_km):
    a = cfg.R_earth + alt_km
    return math.sqrt(cfg.mu / a**3)  # goes into TLE dict (rad/s units)


def alt_from_no_kozai(no_kozai_rad_per_min):
    n_rad_s = no_kozai_rad_per_min / 60.0
    if n_rad_s <= 0:
        return 0.0
    return (cfg.mu / n_rad_s**2) ** (1.0 / 3.0) - cfg.R_earth


def make_tle(inc_rad, raan_rad, ma_rad, alt_km, sat_id=99999):
    return dsgp4.tle.TLE(dict(
        satellite_catalog_number=sat_id, classification='U',
        international_designator='24001A',
        epoch_year=2024, epoch_days=1.0, ephemeris_type=0,
        element_number=999, revolution_number_at_epoch=1,
        mean_motion=mean_motion_from_alt(alt_km),
        mean_motion_first_derivative=0.0, mean_motion_second_derivative=0.0,
        eccentricity=0.001, inclination=inc_rad, argument_of_perigee=0.0,
        raan=raan_rad, mean_anomaly=ma_rad, b_star=1e-5,
    ))


def update_tle_from_elements(tle, elements):
    """Write optimized values back to TLE instance attributes (not _data dict)."""
    tle._bstar = elements[0].detach()
    tle._ndot = elements[1].detach()
    tle._nddot = elements[2].detach()
    tle._ecco = elements[3].detach()
    tle._argpo = elements[4].detach()
    tle._inclo = elements[5].detach()
    tle._mo = elements[6].detach()
    tle._no_kozai = elements[7].detach()
    tle._nodeo = elements[8].detach()


# Gradient mask: only optimize inc(5), ma(6), raan(8)
GRAD_MASK = torch.tensor([0, 0, 0, 0, 0, 1, 1, 0, 1], dtype=torch.float32)


# ============================================================
# Forward pass: coverage + revisit loss
# ============================================================

def compute_loss(tles, tsinces, gmst_array, ground_ecef):
    """
    Propagate constellation, compute differentiable coverage and revisit losses.

    Returns: total_loss, coverage_fraction, mean_max_revisit_gap_minutes
    """
    n_sats = len(tles)
    n_time = len(tsinces)
    n_ground = ground_ecef.shape[0]
    dt = (tsinces[-1] - tsinces[0]).item() / (n_time - 1)  # minutes between steps

    # Propagate all satellites
    all_positions = []
    for tle in tles:
        try:
            state = dsgp4.propagate(tle, tsinces)
            all_positions.append(state[:, 0, :])
        except Exception:
            all_positions.append(torch.zeros(n_time, 3))
    all_positions = torch.stack(all_positions)  # [N_sat, T, 3]

    # --- Per-timestep coverage and revisit tracking ---
    total_coverage = torch.tensor(0.0)
    gap = torch.zeros(n_ground)  # running gap per ground point (minutes)
    all_gaps = []  # store gaps for soft-max revisit computation

    for t_idx in range(n_time):
        sat_teme = all_positions[:, t_idx, :]
        sat_ecef = teme_to_ecef(sat_teme, gmst_array[t_idx])
        elevation = compute_elevation(sat_ecef, ground_ecef)
        cov = soft_coverage(elevation, cfg.min_elevation_deg, cfg.softness_deg)

        # P(any satellite covers ground point) = 1 - prod(1 - cov_i)
        any_covered = 1.0 - torch.prod(1.0 - cov, dim=0)  # [N_ground]
        total_coverage = total_coverage + any_covered.sum()

        # Leaky integrator for revisit gap:
        #   gap[t] = (gap[t-1] + dt) * (1 - coverage[t])
        # Resets to 0 when covered, grows by dt when not covered
        gap = (gap + dt) * (1.0 - any_covered)
        all_gaps.append(gap.clone())

    # Coverage fraction
    coverage_frac = total_coverage / (n_time * n_ground)

    # Worst-case revisit gap per ground point via LogSumExp soft-max
    # max_gap[g] ≈ τ * log(Σ_t exp(gap[g,t] / τ))
    gap_stack = torch.stack(all_gaps, dim=0)  # [T, N_ground]
    tau = cfg.revisit_logsumexp_temp
    soft_max_gap = tau * torch.logsumexp(gap_stack / tau, dim=0)  # [N_ground]
    mean_max_revisit = soft_max_gap.mean()

    # Combined loss: maximize coverage, minimize revisit gaps
    loss = mean_max_revisit

    return loss, coverage_frac, mean_max_revisit


def compute_hard_metrics(tles, tsinces, gmst_array, ground_ecef):
    """
    Compute the exact discrete (hard) coverage and revisit metrics.

    Uses binary visibility indicator, exact Boolean OR, exact gap tracking
    (reset to 0 on coverage, grow by dt otherwise), and true max over time.
    """
    n_sats = len(tles)
    n_time = len(tsinces)
    n_ground = ground_ecef.shape[0]
    dt = (tsinces[-1] - tsinces[0]).item() / (n_time - 1)

    # Propagate all satellites
    all_positions = []
    for tle in tles:
        try:
            state = dsgp4.propagate(tle, tsinces)
            all_positions.append(state[:, 0, :])
        except Exception:
            all_positions.append(torch.zeros(n_time, 3))
    all_positions = torch.stack(all_positions)

    total_coverage = 0.0
    gap = torch.zeros(n_ground)
    max_gap = torch.zeros(n_ground)

    for t_idx in range(n_time):
        sat_teme = all_positions[:, t_idx, :]
        sat_ecef = teme_to_ecef(sat_teme, gmst_array[t_idx])
        elevation = compute_elevation(sat_ecef, ground_ecef)

        # Hard binary coverage
        cov = hard_coverage(elevation, cfg.min_elevation_deg)
        any_covered = (cov.sum(dim=0) > 0).float()
        total_coverage += any_covered.sum().item()

        # Exact gap tracking: reset to 0 when covered, grow by dt otherwise
        gap = (gap + dt) * (1.0 - any_covered)
        max_gap = torch.maximum(max_gap, gap)

    hard_cov_frac = total_coverage / (n_time * n_ground)
    hard_mean_max_revisit = max_gap.mean().item()

    return hard_cov_frac, hard_mean_max_revisit


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 65)
    print("  Differentiable Constellation Optimization via dSGP4")
    print("  Fixed altitude, optimizing geometry: inc, RAAN, mean anomaly")
    print("=" * 65)

    n_sats = cfg.n_planes * cfg.n_sats_per_plane

    # Ground grid
    lat_deg = torch.linspace(-70, 70, cfg.n_lat)
    lon_deg = torch.linspace(-180, 180, cfg.n_lon + 1)[:-1]
    lat_mesh, lon_mesh = torch.meshgrid(lat_deg, lon_deg, indexing='ij')
    lat_rad = torch.deg2rad(lat_mesh.flatten())
    lon_rad = torch.deg2rad(lon_mesh.flatten())
    ground_ecef = torch.stack([
        cfg.R_earth * torch.cos(lat_rad) * torch.cos(lon_rad),
        cfg.R_earth * torch.cos(lat_rad) * torch.sin(lon_rad),
        cfg.R_earth * torch.sin(lat_rad),
    ], dim=-1)
    n_ground = ground_ecef.shape[0]

    # Time array
    prop_min = cfg.prop_duration_hours * 60
    tsinces = torch.linspace(0, prop_min, cfg.n_time_steps)
    gmst_0 = gmst_at_epoch(2024, 1.0)
    gmst_array = [gmst_0 + cfg.earth_rot_rad_per_min * t.item() for t in tsinces]

    print(f"\nConstellation: {n_sats} sats ({cfg.n_planes}P x {cfg.n_sats_per_plane}S)")
    print(f"Grid: {cfg.n_lat}x{cfg.n_lon}={n_ground} pts | {cfg.n_time_steps} time steps over {cfg.prop_duration_hours}h")
    print(f"Altitude: {cfg.target_alt_km} km (FIXED)")
    print(f"Initial: inc={cfg.initial_inc_deg}°, RAAN offsets={cfg.initial_raan_offsets_deg}°")

    # --- Create initial (suboptimal) constellation ---
    tles = []
    for p in range(cfg.n_planes):
        raan = math.radians(cfg.initial_raan_offsets_deg[p])
        for s in range(cfg.n_sats_per_plane):
            ma = 2 * math.pi * s / cfg.n_sats_per_plane
            tles.append(make_tle(
                inc_rad=math.radians(cfg.initial_inc_deg),
                raan_rad=raan, ma_rad=ma,
                alt_km=cfg.target_alt_km,
                sat_id=40000 + p * cfg.n_sats_per_plane + s,
            ))

    # Save initial TLEs for comparison plots later
    initial_tles = []
    for p in range(cfg.n_planes):
        raan = math.radians(cfg.initial_raan_offsets_deg[p])
        for s in range(cfg.n_sats_per_plane):
            ma = 2 * math.pi * s / cfg.n_sats_per_plane
            initial_tles.append(make_tle(
                inc_rad=math.radians(cfg.initial_inc_deg),
                raan_rad=raan, ma_rad=ma,
                alt_km=cfg.target_alt_km,
                sat_id=40000 + p * cfg.n_sats_per_plane + s,
            ))

    # --- Create persistent parameter tensors for torch.optim.Adam ---
    # Initialize each from the TLE's current elements
    for tle in tles:
        dsgp4.initialize_tle(tle, with_grad=False)

    params = []
    for tle in tles:
        p = torch.tensor([
            tle._bstar, tle._ndot, tle._nddot, tle._ecco, tle._argpo,
            tle._inclo, tle._mo, tle._no_kozai, tle._nodeo,
        ], dtype=torch.float64, requires_grad=True)
        params.append(p)

    optimizer = torch.optim.AdamW(params, lr=cfg.lr)

    # --- Compute initial metrics ---
    with torch.no_grad():
        _, init_cov, init_revisit = compute_loss(tles, tsinces, gmst_array, ground_ecef)
    print(f"\nInitial coverage: {init_cov.item()*100:.2f}%")
    print(f"Initial mean max revisit gap: {init_revisit.item():.1f} min")

    # --- Compute initial hard metrics ---
    for tle in tles:
        dsgp4.initialize_tle(tle, with_grad=False)
    with torch.no_grad():
        init_hard_cov, init_hard_rev = compute_hard_metrics(tles, tsinces, gmst_array, ground_ecef)
    print(f"Initial hard coverage: {init_hard_cov*100:.2f}%")
    print(f"Initial hard mean max revisit gap: {init_hard_rev:.1f} min")

    # --- Optimization ---
    loss_history, cov_history, revisit_history = [], [], []
    hard_cov_history, hard_revisit_history, hard_eval_iters = [], [], []

    print(f"\n{'It':>4} {'Cov%':>7} {'Revisit':>8} {'HardCov%':>9} {'HardRev':>8} {'AvgInc':>7} {'AvgRAANsep':>11} {'Loss':>8}")
    print("-" * 76)

    for iteration in range(cfg.n_iterations):
        optimizer.zero_grad()

        # 1. Write current param values into TLEs, then initialize for autograd
        for i, tle in enumerate(tles):
            update_tle_from_elements(tle, params[i])
        tle_elements_list = []
        for tle in tles:
            tle_elements_list.append(dsgp4.initialize_tle(tle, with_grad=True))

        # 2. Forward
        loss, cov_frac, mean_revisit = compute_loss(tles, tsinces, gmst_array, ground_ecef)

        # 3. Backward (through dsgp4's ephemeral tensors)
        loss.backward()

        # 4. Copy gradients from dsgp4 tensors → our persistent params
        for i in range(n_sats):
            g = tle_elements_list[i].grad.clone()
            g = torch.clamp(g, -5.0, 5.0) * GRAD_MASK  # zero non-geometry grads
            params[i].grad = g

        # 5. Adam step on persistent params
        optimizer.step()

        # 6. Clamp to physical ranges
        with torch.no_grad():
            for p in params:
                p[3].clamp_(1e-6, 0.05)       # eccentricity
                p[5].clamp_(0.05, 3.09)        # inclination ~3°-177°

        # Logging
        cov_pct = cov_frac.item() * 100
        rev_min = mean_revisit.item()
        loss_history.append(loss.item())
        cov_history.append(cov_pct)
        revisit_history.append(rev_min)

        if iteration % 20 == 0 or iteration == cfg.n_iterations - 1:
            # Compute hard metrics (reuses already-initialized TLEs)
            with torch.no_grad():
                h_cov, h_rev = compute_hard_metrics(tles, tsinces, gmst_array, ground_ecef)
            hard_cov_history.append(h_cov * 100)
            hard_revisit_history.append(h_rev)
            hard_eval_iters.append(iteration)

            avg_inc = np.mean([math.degrees(tle_elements_list[i][5].item()) for i in range(n_sats)])
            raans = sorted([math.degrees(tle_elements_list[i][8].item()) % 360 for i in range(0, n_sats, cfg.n_sats_per_plane)])
            raan_seps = [raans[i+1] - raans[i] for i in range(len(raans)-1)]
            avg_raan_sep = np.mean(raan_seps) if raan_seps else 0
            print(f"{iteration:4d} {cov_pct:6.2f}% {rev_min:7.1f}m {h_cov*100:8.2f}% {h_rev:7.1f}m {avg_inc:6.1f}° {avg_raan_sep:10.1f}°  {loss.item():7.4f}")

    # --- Final summary ---
    print(f"\n{'=' * 76}")
    print(f"Relaxed coverage:  {cov_history[0]:.2f}% -> {cov_history[-1]:.2f}%  ({cov_history[-1]-cov_history[0]:+.2f}%)")
    print(f"Relaxed revisit:   {revisit_history[0]:.1f} min -> {revisit_history[-1]:.1f} min  ({revisit_history[-1]-revisit_history[0]:+.1f} min)")
    print(f"Hard coverage:     {hard_cov_history[0]:.2f}% -> {hard_cov_history[-1]:.2f}%  ({hard_cov_history[-1]-hard_cov_history[0]:+.2f}%)")
    print(f"Hard revisit:      {hard_revisit_history[0]:.1f} min -> {hard_revisit_history[-1]:.1f} min  ({hard_revisit_history[-1]-hard_revisit_history[0]:+.1f} min)")

    print(f"\nFinal constellation (altitude fixed at {cfg.target_alt_km} km):")
    for p in range(cfg.n_planes):
        idx = p * cfg.n_sats_per_plane
        elem = tle_elements_list[idx]
        print(f"  Plane {p}: inc={math.degrees(elem[5].item()):.2f}°, RAAN={math.degrees(elem[8].item())%360:.2f}°")

    # Write final params to TLEs for plotting
    for i, tle in enumerate(tles):
        update_tle_from_elements(tle, params[i])

    # ==================================================================
    # PLOTS
    # ==================================================================

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # --- Row 1: Optimization curves ---
    axes[0, 0].plot(cov_history, 'b-', lw=2)
    axes[0, 0].set_ylabel('Coverage %')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_title('Coverage Fraction')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(revisit_history, 'r-', lw=2)
    axes[0, 1].set_ylabel('Mean Max Revisit Gap (min)')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_title('Revisit Gap')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(loss_history, 'k-', lw=2)
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_title('Total Loss')
    axes[0, 2].grid(True, alpha=0.3)

    # --- Row 2: Coverage heatmaps (initial vs optimized) + ground tracks ---
    dense_tsinces = torch.linspace(0, prop_min, 500)

    def compute_coverage_map(tle_set):
        for t in tle_set:
            dsgp4.initialize_tle(t, with_grad=False)
        cmap = torch.zeros(cfg.n_lat, cfg.n_lon)
        with torch.no_grad():
            all_pos = []
            for t in tle_set:
                try:
                    state = dsgp4.propagate(t, tsinces)
                    all_pos.append(state[:, 0, :])
                except Exception:
                    all_pos.append(torch.zeros(len(tsinces), 3))
            all_pos = torch.stack(all_pos)
            for t_idx in range(cfg.n_time_steps):
                sat_ecef = teme_to_ecef(all_pos[:, t_idx, :], gmst_array[t_idx])
                el = compute_elevation(sat_ecef, ground_ecef)
                cov = (el > cfg.min_elevation_deg).float()
                any_cov = 1.0 - torch.prod(1.0 - cov, dim=0)
                cmap += any_cov.reshape(cfg.n_lat, cfg.n_lon)
        return (cmap / cfg.n_time_steps * 100).numpy()

    cmap_initial = compute_coverage_map(initial_tles)
    cmap_optimized = compute_coverage_map(tles)
    shared_vmax = max(cmap_initial.max(), cmap_optimized.max(), 1)

    for ax_idx, (label, cmap_data) in enumerate([
        ("Initial", cmap_initial), ("Optimized", cmap_optimized)
    ]):
        ax = axes[1, ax_idx]
        im = ax.imshow(cmap_data, extent=[-180, 180, -70, 70], origin='lower',
                       aspect='auto', cmap='turbo', vmin=0, vmax=shared_vmax)
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        avg = cmap_data.mean()
        ax.set_title(f'{label} Coverage (avg={avg:.1f}%, max={cmap_data.max():.1f}%)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Ground tracks
    ax = axes[1, 2]
    colors = plt.cm.tab10(np.linspace(0, 1, cfg.n_planes))

    # Draw initial (gray) and optimized (colored) ground tracks
    for tle_set, alpha_val, cmap_name, lbl_prefix in [
        (initial_tles, 0.15, None, 'Init'),
        (tles, 0.6, None, 'Opt'),
    ]:
        for t in tle_set:
            dsgp4.initialize_tle(t, with_grad=False)
        for i, tle in enumerate(tle_set):
            try:
                state = dsgp4.propagate(tle, dense_tsinces)
                pos = state[:, 0, :].detach().numpy()
                r = np.linalg.norm(pos, axis=-1)
                lat = np.degrees(np.arcsin(np.clip(pos[:, 2] / r, -1, 1)))
                lon = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))
                lon_c = (lon - np.degrees(cfg.earth_rot_rad_per_min * dense_tsinces.numpy())) % 360
                lon_c[lon_c > 180] -= 360
                plane_idx = i // cfg.n_sats_per_plane
                c = 'gray' if lbl_prefix == 'Init' else colors[plane_idx]
                lbl = None
                if i % cfg.n_sats_per_plane == 0:
                    lbl = f'{lbl_prefix} P{plane_idx}'
                ax.scatter(lon_c, lat, s=0.2, c=[c], alpha=alpha_val, label=lbl)
            except Exception:
                pass

    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title('Ground Tracks: Initial (gray) vs Optimized (color)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.legend(fontsize=6, markerscale=15, loc='lower left', ncol=2)
    ax.grid(True, alpha=0.2)

    # --- Row 3: Relaxed vs Hard comparison ---
    # Coverage comparison
    ax = axes[2, 0]
    ax.plot(cov_history, 'b-', lw=2, alpha=0.4, label='Relaxed (every iter)')
    ax.plot(hard_eval_iters, hard_cov_history, 'b-o', lw=2, markersize=3, label='Hard (discrete)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Coverage %')
    ax.set_title('Relaxed vs Hard: Coverage')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Revisit comparison
    ax = axes[2, 1]
    ax.plot(revisit_history, 'r-', lw=2, alpha=0.4, label='Relaxed (every iter)')
    ax.plot(hard_eval_iters, hard_revisit_history, 'r-o', lw=2, markersize=3, label='Hard (discrete)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Max Revisit Gap (min)')
    ax.set_title('Relaxed vs Hard: Revisit')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Scatter: relaxed vs hard (correlation)
    ax = axes[2, 2]
    relaxed_cov_at_hard = [cov_history[i] for i in hard_eval_iters]
    relaxed_rev_at_hard = [revisit_history[i] for i in hard_eval_iters]
    ax.scatter(relaxed_cov_at_hard, hard_cov_history, c='blue', s=20, alpha=0.7, label='Coverage %')
    ax2 = ax.twinx()
    ax2.scatter(relaxed_rev_at_hard, hard_revisit_history, c='red', s=20, alpha=0.7, label='Revisit (min)')
    # Reference line y=x
    lims = [min(min(relaxed_cov_at_hard), min(hard_cov_history)),
            max(max(relaxed_cov_at_hard), max(hard_cov_history))]
    ax.plot(lims, lims, 'b--', alpha=0.3, lw=1)
    lims_r = [min(min(relaxed_rev_at_hard), min(hard_revisit_history)),
              max(max(relaxed_rev_at_hard), max(hard_revisit_history))]
    ax2.plot(lims_r, lims_r, 'r--', alpha=0.3, lw=1)
    ax.set_xlabel('Relaxed metric')
    ax.set_ylabel('Hard coverage %', color='blue')
    ax2.set_ylabel('Hard revisit (min)', color='red')
    ax.set_title('Relaxed vs Hard: Correlation')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/shreeyam/Projects/relaxed_eo/constellation_optimization.png', dpi=150)
    print(f"\nPlot saved to constellation_optimization.png")
    plt.close()

    # --- Loss landscape visualization ---
    plot_loss_landscape(params, tles, tsinces, gmst_array, ground_ecef)


def plot_loss_landscape(params, tles, tsinces, gmst_array, ground_ecef, grid_size=50, scale=1.0):
    """
    Visualize the loss landscape around the optimum using two random directions,
    à la "Visualizing the Loss Landscape of Neural Nets" (Li et al. 2018).

    Projects the high-dimensional parameter space onto a 2D plane through the
    optimum, defined by two random direction vectors, then evaluates the loss
    on a grid and plots contours.
    """
    print(f"\nComputing loss landscape ({grid_size}x{grid_size} = {grid_size**2} evaluations)...")
    n_sats = len(params)

    # Flatten optimized params into a single vector (only free params: inc, ma, raan)
    # Full param vector for all satellites
    theta_star = torch.stack([p.detach().clone() for p in params])  # [N_sat, 9]

    # Generate two random directions with same shape, normalized
    torch.manual_seed(0)
    d1_raw = torch.randn_like(theta_star)
    d2_raw = torch.randn_like(theta_star)

    # Apply the same gradient mask — only perturb free parameters (inc, ma, raan)
    d1_raw *= GRAD_MASK.unsqueeze(0)
    d2_raw *= GRAD_MASK.unsqueeze(0)

    # Filter-wise normalization (normalize per-satellite, matching the scale of θ*)
    for i in range(n_sats):
        free_theta = theta_star[i] * GRAD_MASK
        norm_theta = free_theta.norm()
        if norm_theta > 0:
            d1_raw[i] = d1_raw[i] / d1_raw[i].norm() * norm_theta
            d2_raw[i] = d2_raw[i] / d2_raw[i].norm() * norm_theta

    # Grid of (alpha, beta) values
    alphas = torch.linspace(-scale, scale, grid_size)
    betas = torch.linspace(-scale, scale, grid_size)

    loss_grid = np.zeros((grid_size, grid_size))
    cov_grid = np.zeros((grid_size, grid_size))
    revisit_grid = np.zeros((grid_size, grid_size))

    total = grid_size * grid_size
    for ai, alpha in enumerate(alphas):
        for bi, beta in enumerate(betas):
            # Perturbed parameters: θ* + α·d1 + β·d2
            perturbed = theta_star + alpha * d1_raw + beta * d2_raw

            # Write into TLEs
            for i, tle in enumerate(tles):
                update_tle_from_elements(tle, perturbed[i])

            # Initialize and evaluate
            for tle in tles:
                dsgp4.initialize_tle(tle, with_grad=False)

            with torch.no_grad():
                try:
                    l, c, r = compute_loss(tles, tsinces, gmst_array, ground_ecef)
                    loss_grid[bi, ai] = l.item()
                    cov_grid[bi, ai] = c.item() * 100
                    revisit_grid[bi, ai] = r.item()
                except Exception:
                    loss_grid[bi, ai] = float('nan')
                    cov_grid[bi, ai] = float('nan')
                    revisit_grid[bi, ai] = float('nan')

            done = ai * grid_size + bi + 1
            if done % max(1, total // 10) == 0 or done == total:
                print(f"  {done}/{total}")

    # Restore optimized params
    for i, tle in enumerate(tles):
        update_tle_from_elements(tle, theta_star[i])

    # Plot
    A, B = np.meshgrid(alphas.numpy(), betas.numpy())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, data, title, cmap in [
        (axes[0], loss_grid, 'Total Loss', 'turbo'),
        (axes[1], cov_grid, 'Coverage %', 'turbo_r'),
        (axes[2], revisit_grid, 'Mean Max Revisit (min)', 'turbo'),
    ]:
        levels = 30
        cs = ax.contourf(A, B, data, levels=levels, cmap=cmap)
        ax.contour(A, B, data, levels=levels, colors='k', linewidths=0.3, alpha=0.4)
        ax.plot(0, 0, 'r*', markersize=15, markeredgecolor='k', markeredgewidth=0.5)
        ax.set_xlabel(r'$\alpha$ (direction 1)')
        ax.set_ylabel(r'$\beta$ (direction 2)')
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(cs, ax=ax, shrink=0.8)

    fig.suptitle('Loss Landscape around Optimum (random 2D slice)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('/Users/shreeyam/Projects/relaxed_eo/loss_landscape.png', dpi=150, bbox_inches='tight')
    print("Loss landscape saved to loss_landscape.png")
    plt.close()


if __name__ == '__main__':
    main()
