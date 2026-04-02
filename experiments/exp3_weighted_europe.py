"""
Experiment 3: Weighted Target Optimization (Europe)

Sample ground points from a GeoJSON of Europe with uniform weights.
Free all orbital parameters (inc, RAAN, MA) to see if the optimizer
discovers a non-Walker geometry tailored to European coverage.

Compares against a Walker baseline optimized for uniform global coverage.
"""

import sys
import os
import json
import math
import numpy as np
import torch
import dsgp4

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from differentiable_eo import (
    Config, ConstellationOptimizer,
    make_constellation, make_gmst_array,
    compute_loss, compute_hard_metrics, extract_elements,
    IDX_INCLO, IDX_MO, IDX_NODEO, IDX_ECCO, IDX_ARGPO, IDX_NO_KOZAI,
)
from differentiable_eo.constraints import (
    FixedConstraint, UnboundedConstraint, BoxConstraint,
    PeriapsisApoapsisConstraint, default_parameter_specs,
)
from differentiable_eo.tle_utils import no_kozai_from_alt, alt_from_no_kozai
from differentiable_eo.coordinates import make_ground_grid_with_weights
from shapely.geometry import shape, Point


def sample_europe_targets(geojson_path, n_points=500, seed=42):
    """Sample random points inside European country boundaries.

    Returns:
        ground_ecef: [N, 3] tensor of ECEF positions
        weights: [N] tensor of uniform weights
        lats: list of latitudes (for plotting)
        lons: list of longitudes (for plotting)
    """
    with open(geojson_path) as f:
        data = json.load(f)

    # Merge all European country polygons
    polygons = []
    for feat in data['features']:
        try:
            geom = shape(feat['geometry'])
            if geom.is_valid:
                polygons.append(geom)
        except Exception:
            continue

    from shapely.ops import unary_union
    europe = unary_union(polygons)
    minx, miny, maxx, maxy = europe.bounds

    rng = np.random.RandomState(seed)
    lats, lons = [], []

    while len(lats) < n_points:
        # Sample random points in bounding box
        batch_lon = rng.uniform(minx, maxx, size=n_points * 2)
        batch_lat = rng.uniform(miny, maxy, size=n_points * 2)
        for lon, lat in zip(batch_lon, batch_lat):
            if europe.contains(Point(lon, lat)):
                lats.append(lat)
                lons.append(lon)
                if len(lats) >= n_points:
                    break

    # Convert to ECEF
    R_EARTH = 6378.137
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    x = R_EARTH * np.cos(lats_rad) * np.cos(lons_rad)
    y = R_EARTH * np.cos(lats_rad) * np.sin(lons_rad)
    z = R_EARTH * np.sin(lats_rad)

    ground_ecef = torch.tensor(np.stack([x, y, z], axis=-1), dtype=torch.float64)
    weights = torch.tensor(np.cos(np.radians(lats)), dtype=torch.float64)

    return ground_ecef, weights, lats, lons


def free_all_geometry_specs():
    """Free inc, RAAN, MA, eccentricity, arg of perigee, and altitude.

    Uses coupled periapsis/apoapsis parameterization to guarantee
    perigee stays within bounds even at high eccentricity.
    """
    specs = default_parameter_specs()
    specs[IDX_INCLO] = BoxConstraint(math.radians(30), math.radians(90))
    specs[IDX_ARGPO] = UnboundedConstraint()
    # Coupled constraint: perigee ∈ [400, 600] km, excess (apogee - perigee) ∈ [0, 1500] km
    coupled = PeriapsisApoapsisConstraint(
        perigee_bounds_km=(400.0, 800.0),
        excess_bounds_km=(0.0, 11000.0),  # capped to keep period < 225 min (dSGP4 limit)
    )
    specs[IDX_NO_KOZAI] = coupled  # register on one index
    specs[IDX_ECCO] = coupled      # same object on both — ReparameterizedElements detects this
    return specs


def evaluate_constellation(tles, tsinces, gmst_tensor, ground_ecef, ground_weights,
                           ground_unit, min_el, revisit_reduce='mean',
                           softness=5.0, revisit_softness=2.0, revisit_tau=10.0,
                           revisit_weight=0.1):
    """Evaluate hard and soft metrics."""
    for t in tles:
        dsgp4.initialize_tle(t, with_grad=False)
    with torch.no_grad():
        h_cov, h_rev = compute_hard_metrics(
            tles, tsinces, gmst_tensor, ground_ecef,
            min_el=min_el, ground_weights=ground_weights,
            revisit_reduce=revisit_reduce, ground_unit=ground_unit,
        )
        _, s_cov, s_rev = compute_loss(
            tles, tsinces, gmst_tensor, ground_ecef,
            min_el=min_el, softness=softness,
            revisit_tau=revisit_tau, revisit_weight=revisit_weight,
            ground_weights=ground_weights, revisit_reduce=revisit_reduce,
            ground_unit=ground_unit, revisit_softness=revisit_softness,
        )
    return {
        'hard_cov': h_cov * 100 if isinstance(h_cov, float) else h_cov.item() * 100,
        'hard_rev': h_rev if isinstance(h_rev, float) else h_rev.item(),
        'soft_cov': s_cov.item() * 100,
        'soft_rev': s_rev.item(),
    }


def main():
    # ---- Configuration ----
    N_PLANES = 2
    N_SATS_PER_PLANE = 2
    N_SATS = N_PLANES * N_SATS_PER_PLANE
    ALT_KM = 550.0
    N_ITERATIONS = 3000
    N_TIME_STEPS = 240
    MIN_EL = 10.0
    PROP_HOURS = 24.0
    REVISIT_REDUCE = 'mean'
    N_TARGET_POINTS = 500

    # Relaxation params
    SOFTNESS = 5.0
    REVISIT_SOFTNESS = 2.0
    REVISIT_TAU = 10.0
    REVISIT_WEIGHT = 0.5
    LR = 1e-2

    INIT_INC = 60.0
    INIT_RAANS = [i * 360.0 / N_PLANES for i in range(N_PLANES)]
    WALKER_F = 1
    REVISIT_WEIGHT = 1.0  # heavier revisit weight with fewer sats

    # ---- Sample Europe targets ----
    geojson_path = os.path.join(os.path.dirname(__file__), 'europe.geojson')
    print(f"Sampling {N_TARGET_POINTS} points from Europe...")
    ground_ecef, ground_weights, target_lats, target_lons = sample_europe_targets(
        geojson_path, n_points=N_TARGET_POINTS)
    ground_unit = ground_ecef / torch.norm(ground_ecef, dim=-1, keepdim=True)

    print(f"  Lat range: {min(target_lats):.1f} to {max(target_lats):.1f}")
    print(f"  Lon range: {min(target_lons):.1f} to {max(target_lons):.1f}")

    # ---- Time grid ----
    prop_min = PROP_HOURS * 60
    tsinces = torch.linspace(0, prop_min, N_TIME_STEPS)
    gmst_array = make_gmst_array(tsinces)
    gmst_tensor = torch.tensor(gmst_array, dtype=torch.float64)

    # ---- Walker baseline (evaluated on Europe targets) ----
    walker_tles = make_constellation(
        n_planes=N_PLANES, n_sats_per_plane=N_SATS_PER_PLANE,
        inc_deg=INIT_INC, raan_offsets_deg=INIT_RAANS,
        alt_km=ALT_KM, phase_offset_f=WALKER_F,
    )
    walker_metrics = evaluate_constellation(
        walker_tles, tsinces, gmst_tensor, ground_ecef, ground_weights,
        ground_unit, MIN_EL, REVISIT_REDUCE,
        SOFTNESS, REVISIT_SOFTNESS, REVISIT_TAU, REVISIT_WEIGHT)
    print(f"\nWalker {N_SATS}/{N_PLANES}/{WALKER_F} on Europe targets:")
    print(f"  Hard: cov={walker_metrics['hard_cov']:.2f}%, rev={walker_metrics['hard_rev']:.1f} min")

    # ---- Random initial MAs ----
    rng = np.random.RandomState(123)
    INIT_MAS = rng.uniform(0, 360, size=N_SATS).tolist()

    # ---- Config: free inc + RAAN + MA, per-plane RAAN sharing ----
    config = Config(
        n_planes=N_PLANES,
        n_sats_per_plane=N_SATS_PER_PLANE,
        target_alt_km=ALT_KM,
        initial_inc_deg=INIT_INC,
        initial_raan_offsets_deg=INIT_RAANS,
        initial_ma_offsets_deg=INIT_MAS,
        prop_duration_hours=PROP_HOURS,
        n_time_steps=N_TIME_STEPS,
        n_lat=1, n_lon=1,  # unused, we override ground grid
        lat_bounds_deg=(-70, 70),
        min_elevation_deg=MIN_EL,
        n_iterations=N_ITERATIONS,
        lr=LR,
        softness_deg=SOFTNESS,
        revisit_softness_deg=REVISIT_SOFTNESS,
        revisit_logsumexp_temp=REVISIT_TAU,
        revisit_weight=REVISIT_WEIGHT,
        revisit_reduce=REVISIT_REDUCE,
        randomize_gmst=True,
        parameter_specs=free_all_geometry_specs(),
        per_plane_params=[IDX_NODEO, IDX_INCLO, IDX_ECCO, IDX_ARGPO, IDX_NO_KOZAI],
    )

    # ---- Override ground grid with Europe targets ----
    torch.manual_seed(13)  # seed GMST randomization for reproducibility
    opt = ConstellationOptimizer(config)
    opt.ground_ecef = ground_ecef
    opt.ground_weights = ground_weights
    opt.ground_unit = ground_unit
    opt.n_ground = ground_ecef.shape[0]

    # ---- Record snapshots ----
    snapshots = []
    SNAPSHOT_EVERY = 10

    init_elems = opt.get_current_elements()
    init_positions = []
    for e in init_elems:
        inc = math.degrees(e[IDX_INCLO].item())
        raan = math.degrees(e[IDX_NODEO].item()) % 360
        ma = math.degrees(e[IDX_MO].item()) % 360
        init_positions.append((inc, raan, ma))
    snapshots.append((-1, init_positions))

    def record_snapshot(iteration, step_result, optimizer_obj):
        if iteration % SNAPSHOT_EVERY == 0 or iteration == N_ITERATIONS - 1:
            elems = optimizer_obj.get_current_elements()
            positions = []
            for e in elems:
                inc = math.degrees(e[IDX_INCLO].item())
                raan = math.degrees(e[IDX_NODEO].item()) % 360
                ma = math.degrees(e[IDX_MO].item()) % 360
                positions.append((inc, raan, ma))
            snapshots.append((iteration, positions))

    result = opt.run(callback=record_snapshot)

    # ---- Evaluate final ----
    final_metrics = evaluate_constellation(
        result.final_tles, tsinces, gmst_tensor, ground_ecef, ground_weights,
        ground_unit, MIN_EL, REVISIT_REDUCE,
        SOFTNESS, REVISIT_SOFTNESS, REVISIT_TAU, REVISIT_WEIGHT)

    init_metrics = evaluate_constellation(
        result.initial_tles, tsinces, gmst_tensor, ground_ecef, ground_weights,
        ground_unit, MIN_EL, REVISIT_REDUCE,
        SOFTNESS, REVISIT_SOFTNESS, REVISIT_TAU, REVISIT_WEIGHT)

    # ---- Print results ----
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: WEIGHTED TARGET OPTIMIZATION (EUROPE)")
    print("=" * 70)
    # Extract final orbital elements per plane
    final_elems = extract_elements(result.final_tles[0]) if hasattr(result.final_tles[0], '_data') else None
    print("  Per-plane final geometry:")
    for p in range(N_PLANES):
        e = opt.get_current_elements()[p * N_SATS_PER_PLANE]
        inc = math.degrees(e[IDX_INCLO].item())
        raan = math.degrees(e[IDX_NODEO].item()) % 360
        ecc = e[IDX_ECCO].item()
        argp = math.degrees(e[IDX_ARGPO].item()) % 360
        alt = alt_from_no_kozai(e[IDX_NO_KOZAI].item())
        print(f"    Plane {p}: inc={inc:.1f}°, RAAN={raan:.1f}°, ecc={ecc:.4f}, argp={argp:.1f}°, alt={alt:.0f} km")
    print()
    print(f"  {'Metric':<30} {'Initial':>10} {'Optimized':>10} {'Walker':>10}")
    print(f"  {'-'*62}")
    print(f"  {'Hard coverage [%]':<30} {init_metrics['hard_cov']:>9.2f}% {final_metrics['hard_cov']:>9.2f}% {walker_metrics['hard_cov']:>9.2f}%")
    print(f"  {'Hard revisit [min]':<30} {init_metrics['hard_rev']:>10.1f} {final_metrics['hard_rev']:>10.1f} {walker_metrics['hard_rev']:>10.1f}")
    print(f"  {'Soft coverage [%]':<30} {init_metrics['soft_cov']:>9.2f}% {final_metrics['soft_cov']:>9.2f}% {walker_metrics['soft_cov']:>9.2f}%")
    print(f"  {'Soft revisit [min]':<30} {init_metrics['soft_rev']:>10.1f} {final_metrics['soft_rev']:>10.1f} {walker_metrics['soft_rev']:>10.1f}")
    print("=" * 70)

    # ---- Paper figure: target map + convergence ----
    print("\nGenerating paper figures...")
    COV_COLOR = '#3F51B5'
    REV_COLOR = '#E91E63'
    paper_dir = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')

    # (a) Europe target map
    fig_map, ax_map = plt.subplots(figsize=(4, 3.5))
    ax_map.scatter(target_lons, target_lats, s=2, c=COV_COLOR, alpha=0.5)
    ax_map.set_xlabel('Longitude [deg]')
    ax_map.set_ylabel('Latitude [deg]')
    ax_map.set_aspect('equal')
    ax_map.grid(True, alpha=0.2)
    fig_map.tight_layout()
    fig_map.savefig(os.path.join(paper_dir, 'exp3_europe_targets.pdf'), bbox_inches='tight')
    print(f"  Saved exp3_europe_targets.pdf")
    plt.close(fig_map)

    # (b) Convergence
    fig_conv, ax_conv = plt.subplots(figsize=(4, 3.5))
    ax_conv_rev = ax_conv.twinx()

    ax_conv.plot(result.cov_history, '-', color=COV_COLOR, lw=1.2, alpha=0.3, label='Soft coverage')
    ax_conv.plot(result.hard_eval_iters, result.hard_cov_history, 'o-',
                 color=COV_COLOR, lw=1.5, markersize=2, label='Hard coverage')
    ax_conv.axhline(walker_metrics['hard_cov'], color=COV_COLOR, ls='--', alpha=0.5,
                    label=f'Walker ({walker_metrics["hard_cov"]:.1f}%)')
    ax_conv.set_xlabel('Iteration')
    ax_conv.set_ylabel('Coverage [%]', color=COV_COLOR)
    ax_conv.tick_params(axis='y', labelcolor=COV_COLOR)

    ax_conv_rev.plot(result.revisit_history, '-', color=REV_COLOR, lw=1.2, alpha=0.3, label='Soft revisit')
    ax_conv_rev.plot(result.hard_eval_iters, result.hard_revisit_history, 's-',
                     color=REV_COLOR, lw=1.5, markersize=2, label='Hard revisit')
    ax_conv_rev.axhline(walker_metrics['hard_rev'], color=REV_COLOR, ls='--', alpha=0.5,
                        label=f'Walker ({walker_metrics["hard_rev"]:.0f} min)')
    ax_conv_rev.set_ylabel('Mean max revisit [min]', color=REV_COLOR)
    ax_conv_rev.tick_params(axis='y', labelcolor=REV_COLOR)

    lines1, labels1 = ax_conv.get_legend_handles_labels()
    lines2, labels2 = ax_conv_rev.get_legend_handles_labels()
    ax_conv.legend(lines1 + lines2, labels1 + labels2, fontsize=6, ncol=2, loc='upper left')
    ax_conv.set_xlim(0, N_ITERATIONS)
    ax_conv.set_ylim(0, 120)
    ax_conv_rev.set_ylim(0, 300)
    ax_conv.grid(True, alpha=0.2)

    fig_conv.tight_layout()
    fig_conv.savefig(os.path.join(paper_dir, 'exp3_convergence.pdf'), bbox_inches='tight')
    print(f"  Saved exp3_convergence.pdf")
    plt.close(fig_conv)

    # (c) Ground tracks on cartopy map (optimized vs Walker)
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    dense_tsinces = torch.linspace(0, prop_min, 1000)
    dense_gmst = make_gmst_array(dense_tsinces)

    def compute_ground_tracks(tles, tsinces_dense, gmst_dense):
        """Compute lat/lon ground tracks for all satellites."""
        from differentiable_eo.constants import EARTH_ROT_RAD_PER_MIN
        tracks = []
        for t in tles:
            dsgp4.initialize_tle(t, with_grad=False)
        for tle in tles:
            try:
                state = dsgp4.propagate(tle, tsinces_dense)
                pos = state[:, 0, :].detach().numpy()
                r = np.linalg.norm(pos, axis=-1)
                lat = np.degrees(np.arcsin(np.clip(pos[:, 2] / r, -1, 1)))
                lon = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))
                # Rotate to ECEF (ground-fixed)
                lon_ecef = (lon - np.degrees(EARTH_ROT_RAD_PER_MIN * tsinces_dense.numpy())) % 360
                lon_ecef[lon_ecef > 180] -= 360
                tracks.append((lat, lon_ecef))
            except Exception:
                tracks.append((np.array([]), np.array([])))
        return tracks

    colors_planes = plt.cm.tab10(np.linspace(0, 1, N_PLANES))

    # Compute visibility footprint density: for each satellite position,
    # mark all ground bins within the min elevation footprint
    lon_bins = np.linspace(-180, 180, 361)
    lat_bins = np.linspace(-90, 90, 181)
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    GRID_LON, GRID_LAT = np.meshgrid(lon_centers, lat_centers)  # [n_lat, n_lon]

    # Precompute grid ECEF positions
    R_E = 6378.137

    # Precompute ECEF for all 4 corners of each grid cell
    def latlon_to_ecef(lat_rad, lon_rad):
        return np.stack([
            R_E * np.cos(lat_rad) * np.cos(lon_rad),
            R_E * np.cos(lat_rad) * np.sin(lon_rad),
            R_E * np.sin(lat_rad),
        ], axis=-1)

    # 4 corners per cell: (lat_lo, lon_lo), (lat_lo, lon_hi), (lat_hi, lon_lo), (lat_hi, lon_hi)
    n_lat_cells = len(lat_bins) - 1
    n_lon_cells = len(lon_bins) - 1
    corner_ecefs = []  # list of 4 arrays, each [n_cells, 3]
    corner_units = []
    for dlat, dlon in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        lat_idx = np.repeat(np.arange(n_lat_cells) + dlat, n_lon_cells)
        lon_idx = np.tile(np.arange(n_lon_cells) + dlon, n_lat_cells)
        clat = np.radians(lat_bins[lat_idx])
        clon = np.radians(lon_bins[lon_idx])
        ecef = latlon_to_ecef(clat, clon)
        corner_ecefs.append(ecef)
        corner_units.append(ecef / np.linalg.norm(ecef, axis=-1, keepdims=True))

    # Also keep center for backwards compat
    grid_lat_rad = np.radians(GRID_LAT.ravel())
    grid_lon_rad = np.radians(GRID_LON.ravel())
    grid_ecef = latlon_to_ecef(grid_lat_rad, grid_lon_rad)
    grid_unit = grid_ecef / np.linalg.norm(grid_ecef, axis=-1, keepdims=True)

    def compute_visibility_density(tles, tsinces_dense, gmst_dense, min_el_deg=MIN_EL):
        """Count how many timesteps each ground bin is visible from any satellite."""
        from differentiable_eo.constants import EARTH_ROT_RAD_PER_MIN
        density = np.zeros(len(grid_lat_rad))
        min_el_rad = np.radians(min_el_deg)

        for t in tles:
            dsgp4.initialize_tle(t, with_grad=False)

        n_time = len(tsinces_dense)
        # Subsample for speed
        step = max(1, n_time // 200)
        for t_idx in range(0, n_time, step):
            for tle in tles:
                try:
                    state = dsgp4.propagate(tle, tsinces_dense[t_idx:t_idx+1])
                    # Single timestep returns [2, 3], multiple returns [T, 2, 3]
                    if state.dim() == 2:
                        pos_teme = state[0, :].detach().numpy()
                    else:
                        pos_teme = state[0, 0, :].detach().numpy()
                except Exception:
                    continue
                # Rotate to ECEF
                gmst = gmst_dense[t_idx]
                cg, sg = np.cos(gmst), np.sin(gmst)
                pos_ecef = np.array([
                    cg * pos_teme[0] + sg * pos_teme[1],
                    -sg * pos_teme[0] + cg * pos_teme[1],
                    pos_teme[2],
                ])
                # Compute elevation at all 4 corners — bin counts only if all 4 visible
                all_visible = np.ones(len(corner_ecefs[0]), dtype=bool)
                for c_ecef, c_unit in zip(corner_ecefs, corner_units):
                    diff = pos_ecef - c_ecef
                    dist = np.linalg.norm(diff, axis=-1)
                    elev = np.arcsin(np.clip(
                        np.sum(diff * c_unit, axis=-1) / dist, -1, 1))
                    all_visible &= (elev >= min_el_rad)
                density += all_visible.astype(float)

        return density.reshape(GRID_LAT.shape)

    all_densities = []
    for tag, tles, title in [('optimized', result.final_tles, 'Optimized for Europe'),
                              ('walker', walker_tles, f'Walker {N_SATS}/{N_PLANES}/{WALKER_F}')]:
        print(f"  Computing visibility density for {tag}...")
        density = compute_visibility_density(tles, dense_tsinces, dense_gmst)
        all_densities.append(density)

    vmax = max(d.max() for d in all_densities)

    for idx, (tag, tles, title) in enumerate([
            ('optimized', result.final_tles, 'Optimized for Europe'),
            ('walker', walker_tles, f'Walker {N_SATS}/{N_PLANES}/{WALKER_F}')]):

        fig_gt = plt.figure(figsize=(10, 4.5))
        ax_gt = fig_gt.add_subplot(111, projection=ccrs.PlateCarree())
        ax_gt.set_global()
        ax_gt.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='#cccccc', linewidth=0.3)
        ax_gt.add_feature(cfeature.OCEAN, facecolor='white')
        ax_gt.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='#cccccc')
        ax_gt.coastlines(linewidth=0.3, color='#999999')

        # Visibility density heatmap
        im = ax_gt.imshow(all_densities[idx], cmap='turbo',
                          vmin=0, vmax=vmax, alpha=0.7,
                          extent=[-180, 180, -90, 90], origin='lower',
                          interpolation='nearest',
                          transform=ccrs.PlateCarree(), zorder=2)

        # Europe target points
        ax_gt.scatter(target_lons, target_lats, s=3, c='white', edgecolors='k',
                     linewidths=0.2, alpha=0.3, transform=ccrs.PlateCarree(), zorder=4)

        # ax_gt.set_title(title)
        plt.colorbar(im, ax=ax_gt, shrink=0.8, pad=0.02, label='Visibility count')
        ax_gt.set_xticks(range(-180, 181, 60), crs=ccrs.PlateCarree())
        ax_gt.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
        ax_gt.set_xticklabels([f'{abs(x)}$^\\circ$W' if x < 0 else f'{x}$^\\circ$E' if x > 0 else '0$^\\circ$'
                               for x in range(-180, 181, 60)])
        ax_gt.set_yticklabels([f'{abs(y)}$^\\circ$S' if y < 0 else f'{y}$^\\circ$N' if y > 0 else '0$^\\circ$'
                               for y in range(-90, 91, 30)])
        ax_gt.grid(linestyle='--', color='gray', alpha=0.5, linewidth=0.3)

        fig_gt.tight_layout()
        gt_path = os.path.join(paper_dir, f'exp3_groundtrack_{tag}.pdf')
        fig_gt.savefig(gt_path, bbox_inches='tight')
        print(f"  Saved {os.path.basename(gt_path)}")
        plt.close(fig_gt)


    # ---- Globe figures (blue ball, before/after) ----
    print("\nGenerating globe figures...")
    from differentiable_eo.globe import eci_xyz, camera_direction, is_occluded
    from differentiable_eo.constants import R_EARTH as R_E
    from mpl_toolkits.mplot3d import Axes3D

    def render_blue_globe(raans_deg, mas_deg, inc_deg, alt_km, n_planes, n_sats_pp,
                          save_path, eccs=None, argps_deg=None, elev=25, azim=45):
        r_orbit = R_E + alt_km
        inc_rad = math.radians(inc_deg)
        lim = r_orbit * 0.72
        # For eccentric orbits, make sure apogee fits in frame
        if eccs:
            max_r = r_orbit * (1 + max(eccs))
            lim = max(lim, max_r * 0.72)
        cam = camera_direction(elev, azim)
        colors = plt.cm.tab10(np.linspace(0, 1, n_planes))
        ring_th = np.linspace(0, 2 * math.pi, 200)
        if eccs is None:
            eccs = [0.0] * n_planes
        if argps_deg is None:
            argps_deg = [0.0] * n_planes

        fig = plt.figure(figsize=(4, 4), facecolor='none')
        ax = fig.add_subplot(111, projection='3d', facecolor='none')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Blue sphere
        u_s = np.linspace(0, 2 * np.pi, 40)
        v_s = np.linspace(0, np.pi, 20)
        ax.plot_surface(
            R_E * np.outer(np.cos(u_s), np.sin(v_s)),
            R_E * np.outer(np.sin(u_s), np.sin(v_s)),
            R_E * np.outer(np.ones_like(u_s), np.cos(v_s)),
            alpha=0.15, color='steelblue', linewidth=0)

        # Orbital rings
        for p in range(n_planes):
            raan_rad = math.radians(raans_deg[p])
            argp_rad = math.radians(argps_deg[p])
            pts = np.array([eci_xyz(inc_rad, raan_rad, th, r_orbit,
                                    ecc=eccs[p], argp_rad=argp_rad)
                            for th in ring_th])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-',
                    color=colors[p], alpha=0.4, lw=1.0)

        # Satellites
        n_sats = n_planes * n_sats_pp
        for i in range(n_sats):
            p = i // n_sats_pp
            pos = np.array(eci_xyz(inc_rad, math.radians(raans_deg[p]),
                                   math.radians(mas_deg[i]), r_orbit,
                                   ecc=eccs[p],
                                   argp_rad=math.radians(argps_deg[p])))
            if is_occluded(pos, cam, R_E):
                ax.scatter(*pos, s=60, facecolors='none',
                          edgecolors=colors[p], linewidths=1.2,
                          depthshade=False, alpha=0.5)
            else:
                ax.scatter(*pos, s=60, color=colors[p],
                          edgecolors='k', linewidths=0.5, zorder=5,
                          depthshade=False)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')
        ax.view_init(elev=elev, azim=azim)
        fig.savefig(save_path, bbox_inches='tight', transparent=True)
        plt.close()
        print(f"  Saved {os.path.basename(save_path)}")

    # Initial constellation (Walker-like circular)
    init_pos = snapshots[0][1]
    init_raans = [init_pos[p * N_SATS_PER_PLANE][1] for p in range(N_PLANES)]
    init_mas = [init_pos[i][2] for i in range(N_SATS)]
    init_inc = init_pos[0][0]
    render_blue_globe(init_raans, init_mas, init_inc, ALT_KM,
                      N_PLANES, N_SATS_PER_PLANE,
                      os.path.join(paper_dir, 'exp3_globe_initial.pdf'))

    # Final optimized constellation — extract per-plane ecc and argp
    final_pos = snapshots[-1][1]
    final_raans = [final_pos[p * N_SATS_PER_PLANE][1] for p in range(N_PLANES)]
    final_mas = [final_pos[i][2] for i in range(N_SATS)]
    final_inc = final_pos[0][0]
    final_elems = opt.get_current_elements()
    final_alt = alt_from_no_kozai(final_elems[0][IDX_NO_KOZAI].item())
    final_eccs = [final_elems[p * N_SATS_PER_PLANE][IDX_ECCO].item() for p in range(N_PLANES)]
    final_argps = [math.degrees(final_elems[p * N_SATS_PER_PLANE][IDX_ARGPO].item()) % 360
                   for p in range(N_PLANES)]
    render_blue_globe(final_raans, final_mas, final_inc, final_alt,
                      N_PLANES, N_SATS_PER_PLANE,
                      os.path.join(paper_dir, 'exp3_globe_optimized.pdf'),
                      eccs=final_eccs, argps_deg=final_argps)


if __name__ == '__main__':
    main()
