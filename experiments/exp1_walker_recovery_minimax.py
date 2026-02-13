"""
Experiment 1b: Walker Recovery with Minimax Revisit

Same setup as exp1 but uses minimax revisit objective (LogSumExp over
ground points) instead of mean. Walker delta minimizes the worst-case
revisit gap, so minimax should recover Walker-like geometry.
"""

import sys
import os
import math
import numpy as np
import torch
import dsgp4

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from differentiable_eo import (
    Config, ConstellationOptimizer,
    make_constellation, make_ground_grid_with_weights, make_gmst_array,
    compute_hard_metrics, extract_elements,
)
from differentiable_eo.constraints import alt_from_no_kozai
from differentiable_eo.visualization import compute_coverage_map, compute_revisit_map
from differentiable_eo.constants import EARTH_ROT_RAD_PER_MIN


def evaluate_constellation(tles, tsinces, gmst_array, ground_ecef, ground_weights, min_el,
                           revisit_reduce='max'):
    """Evaluate a constellation and return hard metrics."""
    for t in tles:
        dsgp4.initialize_tle(t, with_grad=False)
    with torch.no_grad():
        h_cov, h_rev = compute_hard_metrics(
            tles, tsinces, gmst_array, ground_ecef,
            min_el=min_el, ground_weights=ground_weights,
            revisit_reduce=revisit_reduce,
        )
    return h_cov * 100, h_rev


def extract_constellation_stats(tles, n_planes, n_per_plane):
    """Extract mean inclination and RAAN spacings from a set of TLEs."""
    for t in tles:
        dsgp4.initialize_tle(t, with_grad=False)
    elements = [extract_elements(t) for t in tles]

    incs = [math.degrees(e[5].item()) for e in elements]
    raans = [math.degrees(e[8].item()) % 360 for e in elements]

    # Plane RAANs (first sat of each plane)
    plane_raans = [raans[p * n_per_plane] for p in range(n_planes)]
    plane_raans_sorted = sorted(plane_raans)

    # RAAN spacings (circular)
    spacings = []
    for i in range(len(plane_raans_sorted)):
        diff = plane_raans_sorted[(i + 1) % len(plane_raans_sorted)] - plane_raans_sorted[i]
        if diff < 0:
            diff += 360
        spacings.append(diff)

    return {
        'mean_inc': np.mean(incs),
        'plane_raans': plane_raans_sorted,
        'raan_spacings': spacings,
        'raan_spacing_std': np.std(spacings),
    }


def main():
    # ---- Configuration ----
    N_PLANES = 4
    N_SATS_PER_PLANE = 3
    ALT_KM = 550.0
    N_ITERATIONS = 1000
    N_TIME_STEPS = 200
    N_LAT, N_LON = 18, 36
    LAT_BOUNDS = (-60.0, 60.0)
    MIN_EL = 10.0
    PROP_HOURS = 24.0

    # ---- Ground grid with cos(lat) weights ----
    ground_ecef, ground_weights = make_ground_grid_with_weights(N_LAT, N_LON, LAT_BOUNDS)
    prop_min = PROP_HOURS * 60
    tsinces = torch.linspace(0, prop_min, N_TIME_STEPS)
    gmst_array = make_gmst_array(tsinces)

    # ---- Walker 12/4/1 reference ----
    walker_tles = make_constellation(
        n_planes=N_PLANES, n_sats_per_plane=N_SATS_PER_PLANE,
        inc_deg=60.0, raan_offsets_deg=[0.0, 90.0, 180.0, 270.0],
        alt_km=ALT_KM, phase_offset_f=1,
    )
    walker_cov, walker_rev = evaluate_constellation(
        walker_tles, tsinces, gmst_array, ground_ecef, ground_weights, MIN_EL)
    walker_stats = extract_constellation_stats(walker_tles, N_PLANES, N_SATS_PER_PLANE)

    print(f"Walker 12/4/1 reference: cov={walker_cov:.2f}%, revisit={walker_rev:.1f} min")

    # ---- Suboptimal initial guess ----
    config = Config(
        n_planes=N_PLANES,
        n_sats_per_plane=N_SATS_PER_PLANE,
        target_alt_km=ALT_KM,
        initial_inc_deg=30.0,
        initial_raan_offsets_deg=[0.0, 30.0, 100.0, 200.0],
        prop_duration_hours=PROP_HOURS,
        n_time_steps=N_TIME_STEPS,
        n_lat=N_LAT, n_lon=N_LON,
        lat_bounds_deg=LAT_BOUNDS,
        min_elevation_deg=MIN_EL,
        n_iterations=N_ITERATIONS,
        lr=5e-3,
        revisit_weight=0.005,
        revisit_reduce='max',  # minimax over ground points
        revisit_spatial_tau=5.0,  # sharper spatial max than temporal (10 min)
    )

    # ---- Run optimization ----
    optimizer = ConstellationOptimizer(config)
    result = optimizer.run()

    # ---- Collect stats ----
    init_stats = extract_constellation_stats(result.initial_tles, N_PLANES, N_SATS_PER_PLANE)
    final_stats = extract_constellation_stats(result.final_tles, N_PLANES, N_SATS_PER_PLANE)

    init_cov, init_rev = evaluate_constellation(
        result.initial_tles, tsinces, gmst_array, ground_ecef, ground_weights, MIN_EL)
    final_cov, final_rev = evaluate_constellation(
        result.final_tles, tsinces, gmst_array, ground_ecef, ground_weights, MIN_EL)

    # ---- Print comparison table ----
    print("\n" + "=" * 72)
    print("  COMPARISON TABLE (minimax revisit)")
    print("=" * 72)
    print(f"{'Metric':<25} {'Initial':>14} {'Optimized':>14} {'Walker 12/4/1':>14}")
    print("-" * 72)
    print(f"{'Coverage %':<25} {init_cov:>13.2f}% {final_cov:>13.2f}% {walker_cov:>13.2f}%")
    print(f"{'Max revisit (min)':<25} {init_rev:>14.1f} {final_rev:>14.1f} {walker_rev:>14.1f}")
    print(f"{'Mean inc (deg)':<25} {init_stats['mean_inc']:>14.1f} {final_stats['mean_inc']:>14.1f} {walker_stats['mean_inc']:>14.1f}")
    print(f"{'RAAN spacing std (deg)':<25} {init_stats['raan_spacing_std']:>14.1f} {final_stats['raan_spacing_std']:>14.1f} {walker_stats['raan_spacing_std']:>14.1f}")
    print(f"{'RAAN spacings (deg)':<25} {[f'{s:.0f}' for s in init_stats['raan_spacings']]} {[f'{s:.0f}' for s in final_stats['raan_spacings']]} {[f'{s:.0f}' for s in walker_stats['raan_spacings']]}")
    print("=" * 72)

    # ---- Generate figure ----
    fig = plt.figure(figsize=(18, 20))

    # Row 1: Convergence curves with Walker reference dashed lines
    ax1 = fig.add_subplot(4, 3, 1)
    ax1.plot(result.cov_history, 'b-', lw=2, alpha=0.4, label='Relaxed')
    ax1.plot(result.hard_eval_iters, result.hard_cov_history, 'b-o', lw=2, ms=3, label='Hard')
    ax1.axhline(walker_cov, color='b', ls='--', alpha=0.7, label=f'Walker ({walker_cov:.1f}%)')
    ax1.set_ylabel('Coverage %')
    ax1.set_xlabel('Iteration')
    ax1.set_title('Coverage')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(4, 3, 2)
    ax2.plot(result.revisit_history, 'r-', lw=2, alpha=0.4, label='Relaxed')
    ax2.plot(result.hard_eval_iters, result.hard_revisit_history, 'r-o', lw=2, ms=3, label='Hard')
    ax2.axhline(walker_rev, color='r', ls='--', alpha=0.7, label=f'Walker ({walker_rev:.0f} min)')
    ax2.set_ylabel('Worst-Case Revisit (min)')
    ax2.set_xlabel('Iteration')
    ax2.set_title('Minimax Revisit')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(4, 3, 3)
    ax3.plot(result.loss_history, 'k-', lw=2)
    ax3.set_ylabel('Loss')
    ax3.set_xlabel('Iteration')
    ax3.set_title('Total Loss')
    ax3.grid(True, alpha=0.3)

    lat_lo, lat_hi = LAT_BOUNDS

    # Row 2: Coverage maps (initial, optimized, Walker)
    cmap_initial = compute_coverage_map(
        result.initial_tles, tsinces, gmst_array, ground_ecef,
        N_LAT, N_LON, MIN_EL)
    cmap_optimized = compute_coverage_map(
        result.final_tles, tsinces, gmst_array, ground_ecef,
        N_LAT, N_LON, MIN_EL)
    cmap_walker = compute_coverage_map(
        walker_tles, tsinces, gmst_array, ground_ecef,
        N_LAT, N_LON, MIN_EL)
    cov_vmax = max(cmap_initial.max(), cmap_optimized.max(), cmap_walker.max(), 1)

    for ax_idx, (label, cmap_data) in enumerate([
        ("Initial", cmap_initial), ("Optimized", cmap_optimized), ("Walker 12/4/1", cmap_walker)
    ]):
        ax = fig.add_subplot(4, 3, 4 + ax_idx)
        im = ax.imshow(cmap_data, extent=[-180, 180, lat_lo, lat_hi], origin='lower',
                       aspect='auto', cmap='turbo', vmin=0, vmax=cov_vmax)
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        avg = cmap_data.mean()
        ax.set_title(f'{label} Coverage (avg={avg:.1f}%)')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Coverage %')

    # Row 3: Revisit maps (initial, optimized, Walker)
    rev_initial = compute_revisit_map(
        result.initial_tles, tsinces, gmst_array, ground_ecef,
        N_LAT, N_LON, MIN_EL)
    rev_optimized = compute_revisit_map(
        result.final_tles, tsinces, gmst_array, ground_ecef,
        N_LAT, N_LON, MIN_EL)
    rev_walker = compute_revisit_map(
        walker_tles, tsinces, gmst_array, ground_ecef,
        N_LAT, N_LON, MIN_EL)
    rev_vmax = max(rev_initial.max(), rev_optimized.max(), rev_walker.max(), 1)

    for ax_idx, (label, rev_data) in enumerate([
        ("Initial", rev_initial), ("Optimized", rev_optimized), ("Walker 12/4/1", rev_walker)
    ]):
        ax = fig.add_subplot(4, 3, 7 + ax_idx)
        im = ax.imshow(rev_data, extent=[-180, 180, lat_lo, lat_hi], origin='lower',
                       aspect='auto', cmap='turbo_r', vmin=0, vmax=rev_vmax)
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        worst = rev_data.max()
        ax.set_title(f'{label} Revisit (max={worst:.0f} min)')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Max revisit gap (min)')

    # Row 4: Ground tracks (initial, optimized, Walker)
    dense_tsinces = torch.linspace(0, prop_min, 500)
    colors_planes = plt.cm.tab10(np.linspace(0, 1, N_PLANES))

    for ax_idx, (label, tle_set) in enumerate([
        ("Initial", result.initial_tles),
        ("Optimized", result.final_tles),
        ("Walker 12/4/1", walker_tles),
    ]):
        ax = fig.add_subplot(4, 3, 10 + ax_idx)
        for t in tle_set:
            dsgp4.initialize_tle(t, with_grad=False)
        for i, tle in enumerate(tle_set):
            try:
                state = dsgp4.propagate(tle, dense_tsinces)
                pos = state[:, 0, :].detach().numpy()
                r = np.linalg.norm(pos, axis=-1)
                lat = np.degrees(np.arcsin(np.clip(pos[:, 2] / r, -1, 1)))
                lon = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))
                lon_c = (lon - np.degrees(EARTH_ROT_RAD_PER_MIN * dense_tsinces.numpy())) % 360
                lon_c[lon_c > 180] -= 360
                plane_idx = i // N_SATS_PER_PLANE
                c = colors_planes[plane_idx]
                lbl = f'P{plane_idx}' if i % N_SATS_PER_PLANE == 0 else None
                ax.scatter(lon_c, lat, s=0.3, c=[c], alpha=0.5, label=lbl)
            except Exception:
                pass
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        ax.set_title(f'{label} Ground Tracks')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.legend(fontsize=6, markerscale=15, loc='lower left')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'exp1_walker_recovery_minimax.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nFigure saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
