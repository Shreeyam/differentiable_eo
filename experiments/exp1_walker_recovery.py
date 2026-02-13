"""
Experiment 1: Walker Recovery (Toy Problem)

Fix inclination at 60 deg and altitude at 550 km. Allow the optimizer to
move RAAN and mean anomaly. Starting from badly clustered RAANs, the
optimizer should recover evenly-spaced Walker-like geometry.

This demonstrates that gradient-based optimization through the differentiable
relaxations converges to a known-optimal configuration.
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
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from differentiable_eo import (
    Config, ConstellationOptimizer,
    make_constellation, make_ground_grid_with_weights, make_gmst_array,
    compute_hard_metrics, extract_elements,
    IDX_INCLO, IDX_MO, IDX_NODEO,
)
from differentiable_eo.constraints import (
    FixedConstraint, UnboundedConstraint, BoxConstraint, alt_from_no_kozai,
    default_parameter_specs,
)
from differentiable_eo.coordinates import teme_to_ecef, compute_elevation
from differentiable_eo.visualization import compute_coverage_map
from differentiable_eo.constants import EARTH_ROT_RAD_PER_MIN


def ma_and_raan_specs():
    """Parameter specs that free mean anomaly and RAAN (inc + alt fixed)."""
    specs = default_parameter_specs()
    specs[IDX_INCLO] = FixedConstraint()  # Lock inclination at 60 deg
    return specs


def evaluate_constellation(tles, tsinces, gmst_array, ground_ecef, ground_weights, min_el):
    """Evaluate a constellation and return hard metrics."""
    for t in tles:
        dsgp4.initialize_tle(t, with_grad=False)
    with torch.no_grad():
        h_cov, h_rev = compute_hard_metrics(
            tles, tsinces, gmst_array, ground_ecef,
            min_el=min_el, ground_weights=ground_weights,
        )
    return h_cov * 100, h_rev


def extract_constellation_stats(tles, n_planes, n_sats_per_plane):
    """Extract inclination, RAAN, and MA from a set of TLEs."""
    for t in tles:
        dsgp4.initialize_tle(t, with_grad=False)
    elements = [extract_elements(t) for t in tles]

    incs = [math.degrees(e[5].item()) for e in elements]
    raans = [math.degrees(e[8].item()) % 360 for e in elements]
    mas = [math.degrees(e[6].item()) % 360 for e in elements]

    # Per-plane MA spacing
    plane_mas = []
    for p in range(n_planes):
        start = p * n_sats_per_plane
        plane_ma = sorted([mas[start + s] for s in range(n_sats_per_plane)])
        plane_mas.append(plane_ma)

    return {
        'mean_inc': np.mean(incs),
        'incs': incs,
        'raans': raans,
        'mas': mas,
        'plane_mas': plane_mas,
    }


def main():
    # ---- Configuration ----
    N_PLANES = 6
    N_SATS_PER_PLANE = 4
    ALT_KM = 550.0
    N_ITERATIONS = 600
    N_TIME_STEPS = 200
    N_LAT, N_LON = 36, 72
    LAT_BOUNDS = (-70.0, 70.0)
    MIN_EL = 10.0
    PROP_HOURS = 24.0

    WALKER_RAANS = [i * 360.0 / N_PLANES for i in range(N_PLANES)]
    WALKER_INC = 60.0
    WALKER_F = 1

    # ---- Ground grid with cos(lat) weights ----
    ground_ecef, ground_weights = make_ground_grid_with_weights(N_LAT, N_LON, LAT_BOUNDS)
    prop_min = PROP_HOURS * 60
    tsinces = torch.linspace(0, prop_min, N_TIME_STEPS)
    gmst_array = make_gmst_array(tsinces)

    # ---- Walker 6/3/1 reference ----
    walker_tles = make_constellation(
        n_planes=N_PLANES, n_sats_per_plane=N_SATS_PER_PLANE,
        inc_deg=WALKER_INC, raan_offsets_deg=WALKER_RAANS,
        alt_km=ALT_KM, phase_offset_f=WALKER_F,
    )
    walker_cov, walker_rev = evaluate_constellation(
        walker_tles, tsinces, gmst_array, ground_ecef, ground_weights, MIN_EL)
    walker_stats = extract_constellation_stats(walker_tles, N_PLANES, N_SATS_PER_PLANE)

    print(f"Walker 6/3/1 reference: cov={walker_cov:.2f}%, revisit={walker_rev:.1f} min")
    print(f"  Inc={walker_stats['mean_inc']:.1f} deg, MAs={[f'{m:.0f}' for m in walker_stats['mas']]}")

    # ---- Suboptimal initial: correct inc but messed up RAANs ----
    BAD_RAANS = [0.0, 10.0, 30.0, 35.0, 170.0, 180.0]  # Clustered, far from uniform

    config = Config(
        n_planes=N_PLANES,
        n_sats_per_plane=N_SATS_PER_PLANE,
        target_alt_km=ALT_KM,
        initial_inc_deg=WALKER_INC,  # Inc fixed at 60 deg
        initial_raan_offsets_deg=BAD_RAANS,  # Badly spaced RAANs
        prop_duration_hours=PROP_HOURS,
        n_time_steps=N_TIME_STEPS,
        n_lat=N_LAT, n_lon=N_LON,
        lat_bounds_deg=LAT_BOUNDS,
        min_elevation_deg=MIN_EL,
        n_iterations=N_ITERATIONS,
        lr=5e-3,
        revisit_weight=0.005,
        randomize_gmst=False,
        parameter_specs=ma_and_raan_specs(),  # MA + RAAN free, inc fixed
    )

    # ---- Run optimization, recording snapshots for animation ----
    opt = ConstellationOptimizer(config)

    # Collect (RAAN, MA) snapshots every few iterations
    snapshots = []  # list of (iteration, [(raan_deg, ma_deg), ...])
    SNAPSHOT_EVERY = 5

    def record_snapshot(iteration, step_result, optimizer_obj):
        if iteration % SNAPSHOT_EVERY == 0 or iteration == N_ITERATIONS - 1:
            elems = optimizer_obj.get_current_elements()
            positions = []
            for e in elems:
                raan = math.degrees(e[8].item()) % 360
                ma = math.degrees(e[6].item()) % 360
                positions.append((raan, ma))
            snapshots.append((iteration, positions))

    result = opt.run(callback=record_snapshot)

    # ---- Collect stats ----
    init_stats = extract_constellation_stats(result.initial_tles, N_PLANES, N_SATS_PER_PLANE)
    final_stats = extract_constellation_stats(result.final_tles, N_PLANES, N_SATS_PER_PLANE)

    init_cov, init_rev = evaluate_constellation(
        result.initial_tles, tsinces, gmst_array, ground_ecef, ground_weights, MIN_EL)
    final_cov, final_rev = evaluate_constellation(
        result.final_tles, tsinces, gmst_array, ground_ecef, ground_weights, MIN_EL)

    # ---- Print comparison table ----
    print("\n" + "=" * 72)
    print("  WALKER RECOVERY: COMPARISON TABLE")
    print("  (inc fixed at 60 deg; RAAN + MA optimized)")
    print("=" * 72)
    print(f"{'Metric':<25} {'Initial':>12} {'Optimized':>12} {'Walker 6/3/1':>14}")
    print("-" * 72)
    print(f"{'Coverage %':<25} {init_cov:>11.2f}% {final_cov:>11.2f}% {walker_cov:>13.2f}%")
    print(f"{'Max revisit (min)':<25} {init_rev:>12.1f} {final_rev:>12.1f} {walker_rev:>14.1f}")
    print(f"{'Mean inc (deg)':<25} {init_stats['mean_inc']:>12.1f} {final_stats['mean_inc']:>12.1f} {walker_stats['mean_inc']:>14.1f}")

    print(f"\nPer-satellite mean anomalies (deg):")
    print(f"  Initial:  {[f'{m:.0f}' for m in init_stats['mas']]}")
    print(f"  Optimized: {[f'{m:.0f}' for m in final_stats['mas']]}")
    print(f"  Walker:   {[f'{m:.0f}' for m in walker_stats['mas']]}")
    print("=" * 72)

    # ---- Generate figure ----
    fig = plt.figure(figsize=(18, 15))

    # Row 1: Convergence curves (relaxed + hard) with Walker reference
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(result.cov_history, 'b-', lw=2, alpha=0.4, label='Relaxed')
    ax1.plot(result.hard_eval_iters, result.hard_cov_history, 'b-o', lw=2, markersize=3, label='Hard')
    ax1.axhline(walker_cov, color='b', ls='--', alpha=0.7, label=f'Walker ({walker_cov:.1f}%)')
    ax1.set_ylabel('Coverage %')
    ax1.set_xlabel('Iteration')
    ax1.set_title('Coverage Convergence')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(result.revisit_history, 'r-', lw=2, alpha=0.4, label='Relaxed')
    ax2.plot(result.hard_eval_iters, result.hard_revisit_history, 'r-o', lw=2, markersize=3, label='Hard')
    ax2.axhline(walker_rev, color='r', ls='--', alpha=0.7, label=f'Walker ({walker_rev:.0f} min)')
    ax2.set_ylabel('Mean Max Revisit (min)')
    ax2.set_xlabel('Iteration')
    ax2.set_title('Revisit Convergence')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(result.loss_history, 'k-', lw=2)
    ax3.set_ylabel('Loss')
    ax3.set_xlabel('Iteration')
    ax3.set_title('Total Loss')
    ax3.grid(True, alpha=0.3)

    # Row 2: Coverage maps (initial, optimized, Walker)
    lat_lo, lat_hi = LAT_BOUNDS

    cmap_initial = compute_coverage_map(
        result.initial_tles, tsinces, gmst_array, ground_ecef,
        N_LAT, N_LON, MIN_EL)
    cmap_optimized = compute_coverage_map(
        result.final_tles, tsinces, gmst_array, ground_ecef,
        N_LAT, N_LON, MIN_EL)
    cmap_walker = compute_coverage_map(
        walker_tles, tsinces, gmst_array, ground_ecef,
        N_LAT, N_LON, MIN_EL)

    shared_vmax = max(cmap_initial.max(), cmap_optimized.max(), cmap_walker.max(), 1)

    for ax_idx, (label, cmap_data) in enumerate([
        ("Initial", cmap_initial), ("Optimized", cmap_optimized), ("Walker 6/3/1", cmap_walker)
    ]):
        ax = fig.add_subplot(3, 3, 4 + ax_idx)
        im = ax.imshow(cmap_data, extent=[-180, 180, lat_lo, lat_hi], origin='lower',
                       aspect='auto', cmap='turbo', vmin=0, vmax=shared_vmax)
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        avg = cmap_data.mean()
        ax.set_title(f'{label} (avg={avg:.1f}%)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 3: Ground tracks (initial, optimized, Walker)
    dense_tsinces = torch.linspace(0, prop_min, 500)
    colors_planes = plt.cm.tab10(np.linspace(0, 1, N_PLANES))

    for ax_idx, (label, tle_set) in enumerate([
        ("Initial", result.initial_tles),
        ("Optimized", result.final_tles),
        ("Walker 6/3/1", walker_tles),
    ]):
        ax = fig.add_subplot(3, 3, 7 + ax_idx)
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
    save_path = os.path.join(os.path.dirname(__file__), 'exp1_walker_recovery.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nFigure saved to {save_path}")
    plt.close()

    # ---- Animation: 3D Earth + satellites with movement arrows ----
    print("\nGenerating 3D animation...")
    from mpl_toolkits.mplot3d import Axes3D
    from differentiable_eo.constants import R_EARTH

    n_sats = N_PLANES * N_SATS_PER_PLANE
    r_orbit = R_EARTH + ALT_KM  # km

    def orbital_elements_to_xyz(inc_rad, raan_rad, ma_rad, r):
        """Convert (inc, RAAN, MA) to ECI (x, y, z) for a circular orbit."""
        # Position in orbital plane
        x_orb = r * math.cos(ma_rad)
        y_orb = r * math.sin(ma_rad)
        # Rotate by inclination (around x)
        ci, si = math.cos(inc_rad), math.sin(inc_rad)
        x_inc = x_orb
        y_inc = y_orb * ci
        z_inc = y_orb * si
        # Rotate by RAAN (around z)
        cr, sr = math.cos(raan_rad), math.sin(raan_rad)
        x = x_inc * cr - y_inc * sr
        y = x_inc * sr + y_inc * cr
        z = z_inc
        return x, y, z

    def snapshot_to_xyz(positions_elem):
        """Convert list of (inc, raan, ma) element tuples to xyz arrays."""
        xs, ys, zs = [], [], []
        for inc, raan, ma in positions_elem:
            x, y, z = orbital_elements_to_xyz(inc, raan, ma, r_orbit)
            xs.append(x)
            ys.append(y)
            zs.append(z)
        return np.array(xs), np.array(ys), np.array(zs)

    # Re-collect snapshots with inc/raan/ma in radians for 3D conversion
    snapshots_3d = []
    for iteration, positions in snapshots:
        elems_rad = []
        for raan_deg, ma_deg in positions:
            elems_rad.append((math.radians(WALKER_INC), math.radians(raan_deg), math.radians(ma_deg)))
        snapshots_3d.append((iteration, elems_rad))

    # Draw Earth sphere
    u_sphere = np.linspace(0, 2 * np.pi, 40)
    v_sphere = np.linspace(0, np.pi, 20)
    xe = R_EARTH * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    ye = R_EARTH * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    ze = R_EARTH * np.outer(np.ones_like(u_sphere), np.cos(v_sphere))

    # Precompute orbital rings for each plane (thin lines showing full orbit)
    ring_theta = np.linspace(0, 2 * np.pi, 100)

    all_losses = result.loss_history

    fig_anim = plt.figure(figsize=(14, 7))
    ax3d = fig_anim.add_subplot(121, projection='3d')
    ax_loss = fig_anim.add_subplot(122)

    # Loss axis setup
    ax_loss.set_xlim(0, N_ITERATIONS)
    ax_loss.set_ylim(min(all_losses) - 0.05, max(all_losses) + 0.05)
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Optimization Loss')
    ax_loss.grid(True, alpha=0.3)

    def draw_frame(frame_idx):
        ax3d.cla()

        iteration, elems_rad = snapshots_3d[frame_idx]

        # Earth
        ax3d.plot_surface(xe, ye, ze, alpha=0.15, color='steelblue', linewidth=0)

        # Orbital rings (one per plane, using current RAAN)
        for p in range(N_PLANES):
            idx = p * N_SATS_PER_PLANE
            inc, raan, _ = elems_rad[idx]
            ring_x, ring_y, ring_z = [], [], []
            for theta in ring_theta:
                x, y, z = orbital_elements_to_xyz(inc, raan, theta, r_orbit)
                ring_x.append(x)
                ring_y.append(y)
                ring_z.append(z)
            ax3d.plot(ring_x, ring_y, ring_z, '-',
                     color=colors_planes[p], alpha=0.25, lw=0.8)

        # Satellite positions
        xs, ys, zs = snapshot_to_xyz(elems_rad)
        for i in range(n_sats):
            plane_idx = i // N_SATS_PER_PLANE
            ax3d.scatter([xs[i]], [ys[i]], [zs[i]], s=50,
                        color=colors_planes[plane_idx],
                        edgecolors='k', linewidths=0.5, zorder=5, depthshade=False)

        # Movement arrows from previous frame
        if frame_idx > 0:
            _, prev_elems = snapshots_3d[frame_idx - 1]
            prev_xs, prev_ys, prev_zs = snapshot_to_xyz(prev_elems)
            arrow_scale = 8.0
            for i in range(n_sats):
                dx = (xs[i] - prev_xs[i]) * arrow_scale
                dy = (ys[i] - prev_ys[i]) * arrow_scale
                dz = (zs[i] - prev_zs[i]) * arrow_scale
                length = math.sqrt(dx**2 + dy**2 + dz**2)
                if length > 5:  # Only draw visible arrows
                    plane_idx = i // N_SATS_PER_PLANE
                    ax3d.quiver(xs[i], ys[i], zs[i], dx, dy, dz,
                               color=colors_planes[plane_idx], alpha=0.7,
                               arrow_length_ratio=0.2, linewidth=1.5)

        # Formatting
        lim = r_orbit * 1
        ax3d.set_xlim(-lim, lim)
        ax3d.set_ylim(-lim, lim)
        ax3d.set_zlim(-lim, lim)
        ax3d.set_box_aspect([1, 1, 1])
        ax3d.set_title(f'Iteration {iteration}', fontsize=12)
        ax3d.axis('off')
        ax3d.view_init(elev=25, azim=45)

        # Loss curve (progressive)
        ax_loss.cla()
        ax_loss.set_xlim(0, N_ITERATIONS)
        ax_loss.set_ylim(min(all_losses) - 0.05, max(all_losses) + 0.05)
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Optimization Loss')
        ax_loss.grid(True, alpha=0.3)
        ax_loss.plot(range(iteration + 1), all_losses[:iteration + 1], 'k-', lw=2)
        ax_loss.plot(iteration, all_losses[iteration], 'ro', markersize=6)

    # Generate frames
    anim = FuncAnimation(fig_anim, draw_frame,
                         frames=len(snapshots_3d), interval=100, blit=False)

    anim_path = os.path.join(os.path.dirname(__file__), 'exp1_animation.gif')
    anim.save(anim_path, writer=PillowWriter(fps=20))
    print(f"Animation saved to {anim_path}")
    plt.close(fig_anim)


if __name__ == '__main__':
    main()
