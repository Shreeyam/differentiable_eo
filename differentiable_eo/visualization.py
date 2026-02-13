"""Visualization utilities for constellation optimization results."""

import numpy as np
import torch
import dsgp4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .coordinates import teme_to_ecef, compute_elevation, make_ground_grid, make_gmst_array
from .constants import EARTH_ROT_RAD_PER_MIN
from .objective import compute_loss
from .tle_utils import update_tle_from_elements


def compute_coverage_map(tles, tsinces, gmst_array, ground_ecef, n_lat, n_lon, min_el):
    """Compute per-grid-cell coverage fraction over the propagation window."""
    for t in tles:
        dsgp4.initialize_tle(t, with_grad=False)
    n_time = len(tsinces)
    cmap = torch.zeros(n_lat, n_lon)
    with torch.no_grad():
        all_pos = []
        for t in tles:
            try:
                state = dsgp4.propagate(t, tsinces)
                all_pos.append(state[:, 0, :])
            except Exception:
                all_pos.append(torch.zeros(n_time, 3))
        all_pos = torch.stack(all_pos)
        for t_idx in range(n_time):
            sat_ecef = teme_to_ecef(all_pos[:, t_idx, :], gmst_array[t_idx])
            el = compute_elevation(sat_ecef, ground_ecef)
            cov = (el > min_el).float()
            any_cov = 1.0 - torch.prod(1.0 - cov, dim=0)
            cmap += any_cov.reshape(n_lat, n_lon)
    return (cmap / n_time * 100).numpy()


def compute_revisit_map(tles, tsinces, gmst_array, ground_ecef, n_lat, n_lon, min_el):
    """Compute per-grid-cell max revisit gap (minutes) over the propagation window."""
    for t in tles:
        dsgp4.initialize_tle(t, with_grad=False)
    n_time = len(tsinces)
    dt = (tsinces[-1] - tsinces[0]).item() / (n_time - 1)
    with torch.no_grad():
        all_pos = []
        for t in tles:
            try:
                state = dsgp4.propagate(t, tsinces)
                all_pos.append(state[:, 0, :])
            except Exception:
                all_pos.append(torch.zeros(n_time, 3))
        all_pos = torch.stack(all_pos)
        n_ground = ground_ecef.shape[0]
        gap = torch.zeros(n_ground)
        max_gap = torch.zeros(n_ground)
        for t_idx in range(n_time):
            sat_ecef = teme_to_ecef(all_pos[:, t_idx, :], gmst_array[t_idx])
            el = compute_elevation(sat_ecef, ground_ecef)
            any_cov = ((el > min_el).float().sum(dim=0) > 0).float()
            gap = (gap + dt) * (1.0 - any_cov)
            max_gap = torch.maximum(max_gap, gap)
    return max_gap.reshape(n_lat, n_lon).numpy()


def plot_optimization_results(result, config, save_path=None):
    """
    Generate the 3x3 optimization results plot.

    Args:
        result: OptimizationResult from ConstellationOptimizer.run()
        config: Config used for the run
        save_path: path to save the figure (default: constellation_optimization.png)
    """
    if save_path is None:
        save_path = 'constellation_optimization.png'

    cfg = config
    ground_ecef = make_ground_grid(cfg.n_lat, cfg.n_lon, cfg.lat_bounds_deg)
    prop_min = cfg.prop_duration_hours * 60
    tsinces = torch.linspace(0, prop_min, cfg.n_time_steps)
    gmst_array = make_gmst_array(tsinces)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # --- Row 1: Optimization curves ---
    axes[0, 0].plot(result.cov_history, 'b-', lw=2)
    axes[0, 0].set_ylabel('Coverage %')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_title('Coverage Fraction')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(result.revisit_history, 'r-', lw=2)
    axes[0, 1].set_ylabel('Mean Max Revisit Gap (min)')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_title('Revisit Gap')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(result.loss_history, 'k-', lw=2)
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_title('Total Loss')
    axes[0, 2].grid(True, alpha=0.3)

    # --- Row 2: Coverage heatmaps + ground tracks ---
    cmap_initial = compute_coverage_map(
        result.initial_tles, tsinces, gmst_array, ground_ecef,
        cfg.n_lat, cfg.n_lon, cfg.min_elevation_deg)
    cmap_optimized = compute_coverage_map(
        result.final_tles, tsinces, gmst_array, ground_ecef,
        cfg.n_lat, cfg.n_lon, cfg.min_elevation_deg)
    shared_vmax = max(cmap_initial.max(), cmap_optimized.max(), 1)

    lat_lo, lat_hi = cfg.lat_bounds_deg
    for ax_idx, (label, cmap_data) in enumerate([
        ("Initial", cmap_initial), ("Optimized", cmap_optimized)
    ]):
        ax = axes[1, ax_idx]
        im = ax.imshow(cmap_data, extent=[-180, 180, lat_lo, lat_hi], origin='lower',
                       aspect='auto', cmap='turbo', vmin=0, vmax=shared_vmax)
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        avg = cmap_data.mean()
        ax.set_title(f'{label} Coverage (avg={avg:.1f}%, max={cmap_data.max():.1f}%)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Ground tracks
    ax = axes[1, 2]
    dense_tsinces = torch.linspace(0, prop_min, 500)
    colors = plt.cm.tab10(np.linspace(0, 1, cfg.n_planes))

    for tle_set, alpha_val, lbl_prefix in [
        (result.initial_tles, 0.15, 'Init'),
        (result.final_tles, 0.6, 'Opt'),
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
                lon_c = (lon - np.degrees(EARTH_ROT_RAD_PER_MIN * dense_tsinces.numpy())) % 360
                lon_c[lon_c > 180] -= 360
                plane_idx = i // cfg.n_sats_per_plane
                c = 'gray' if lbl_prefix == 'Init' else colors[plane_idx]
                lbl = None
                if i % cfg.n_sats_per_plane == 0:
                    lbl = f'{lbl_prefix} P{plane_idx}'
                ax.scatter(lon_c, lat, s=0.2, c=[c], alpha=alpha_val, label=lbl)
            except Exception:
                pass

    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    ax.set_title('Ground Tracks: Initial (gray) vs Optimized (color)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.legend(fontsize=6, markerscale=15, loc='lower left', ncol=2)
    ax.grid(True, alpha=0.2)

    # --- Row 3: Relaxed vs Hard comparison ---
    ax = axes[2, 0]
    ax.plot(result.cov_history, 'b-', lw=2, alpha=0.4, label='Relaxed (every iter)')
    ax.plot(result.hard_eval_iters, result.hard_cov_history, 'b-o', lw=2, markersize=3, label='Hard (discrete)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Coverage %')
    ax.set_title('Relaxed vs Hard: Coverage')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(result.revisit_history, 'r-', lw=2, alpha=0.4, label='Relaxed (every iter)')
    ax.plot(result.hard_eval_iters, result.hard_revisit_history, 'r-o', lw=2, markersize=3, label='Hard (discrete)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Max Revisit Gap (min)')
    ax.set_title('Relaxed vs Hard: Revisit')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    relaxed_cov_at_hard = [result.cov_history[i] for i in result.hard_eval_iters]
    relaxed_rev_at_hard = [result.revisit_history[i] for i in result.hard_eval_iters]
    ax.scatter(relaxed_cov_at_hard, result.hard_cov_history, c='blue', s=20, alpha=0.7, label='Coverage %')
    ax2 = ax.twinx()
    ax2.scatter(relaxed_rev_at_hard, result.hard_revisit_history, c='red', s=20, alpha=0.7, label='Revisit (min)')
    lims = [min(min(relaxed_cov_at_hard), min(result.hard_cov_history)),
            max(max(relaxed_cov_at_hard), max(result.hard_cov_history))]
    ax.plot(lims, lims, 'b--', alpha=0.3, lw=1)
    lims_r = [min(min(relaxed_rev_at_hard), min(result.hard_revisit_history)),
              max(max(relaxed_rev_at_hard), max(result.hard_revisit_history))]
    ax2.plot(lims_r, lims_r, 'r--', alpha=0.3, lw=1)
    ax.set_xlabel('Relaxed metric')
    ax.set_ylabel('Hard coverage %', color='blue')
    ax2.set_ylabel('Hard revisit (min)', color='red')
    ax.set_title('Relaxed vs Hard: Correlation')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_loss_landscape(optimizer, grid_size=50, scale=1.0, save_path=None):
    """
    Visualize loss landscape around the optimum using two random directions in z-space.

    Args:
        optimizer: ConstellationOptimizer instance (after optimization)
        grid_size: resolution of the landscape grid
        scale: range of perturbation in z-space
        save_path: path to save the figure
    """
    if save_path is None:
        save_path = 'loss_landscape.png'

    cfg = optimizer.config
    print(f"\nComputing loss landscape ({grid_size}x{grid_size} = {grid_size**2} evaluations)...")

    # Collect current z values
    z_star = [re.optimizer_param.detach().clone() for re in optimizer.reparam_elements]
    z_flat = torch.cat(z_star)

    # Random directions in z-space
    torch.manual_seed(0)
    d1 = torch.randn_like(z_flat)
    d2 = torch.randn_like(z_flat)
    d1 = d1 / d1.norm() * z_flat.norm()
    d2 = d2 / d2.norm() * z_flat.norm()

    alphas = torch.linspace(-scale, scale, grid_size)
    betas = torch.linspace(-scale, scale, grid_size)

    loss_grid = np.zeros((grid_size, grid_size))
    cov_grid = np.zeros((grid_size, grid_size))
    revisit_grid = np.zeros((grid_size, grid_size))

    # Split directions back per-satellite
    sizes = [re.n_free for re in optimizer.reparam_elements]
    total = grid_size * grid_size

    for ai, alpha in enumerate(alphas):
        for bi, beta in enumerate(betas):
            perturbed = z_flat + alpha * d1 + beta * d2

            # Write perturbed z into reparam elements
            offset = 0
            for re in optimizer.reparam_elements:
                if re.n_free > 0:
                    re._z = torch.nn.Parameter(perturbed[offset:offset + re.n_free].clone())
                    offset += re.n_free

            # Map to elements -> TLEs -> propagate
            for i, tle in enumerate(optimizer.tles):
                elements = optimizer.reparam_elements[i].to_elements()
                update_tle_from_elements(tle, elements)
            for tle in optimizer.tles:
                dsgp4.initialize_tle(tle, with_grad=False)

            with torch.no_grad():
                try:
                    l, c, r = compute_loss(
                        optimizer.tles, optimizer.tsinces, optimizer.gmst_array,
                        optimizer.ground_ecef, cfg.min_elevation_deg, cfg.softness_deg,
                        cfg.revisit_logsumexp_temp, cfg.revisit_weight,
                    )
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

    # Restore original z values
    offset = 0
    for re, z_orig in zip(optimizer.reparam_elements, z_star):
        if re.n_free > 0:
            re._z = torch.nn.Parameter(z_orig)
            offset += re.n_free

    # Restore TLEs
    for i, tle in enumerate(optimizer.tles):
        elements = optimizer.reparam_elements[i].to_elements()
        update_tle_from_elements(tle, elements)

    # Plot
    A, B = np.meshgrid(alphas.numpy(), betas.numpy())
    fig, axes_arr = plt.subplots(1, 3, figsize=(18, 5))

    for ax, data, title, cmap in [
        (axes_arr[0], loss_grid, 'Total Loss', 'turbo'),
        (axes_arr[1], cov_grid, 'Coverage %', 'turbo_r'),
        (axes_arr[2], revisit_grid, 'Mean Max Revisit (min)', 'turbo'),
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

    fig.suptitle('Loss Landscape around Optimum (random 2D slice in z-space)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Loss landscape saved to {save_path}")
    plt.close()
