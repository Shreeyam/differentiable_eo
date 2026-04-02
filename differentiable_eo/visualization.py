"""Visualization utilities for constellation optimization results."""

import numpy as np
import torch
import dsgp4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .coordinates import teme_to_ecef, compute_elevation, make_ground_grid, make_gmst_array
from .constants import EARTH_ROT_RAD_PER_MIN
from .objective import compute_loss, compute_hard_metrics
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
                        optimizer.tles, optimizer.tsinces, optimizer.gmst_tensor,
                        optimizer.ground_ecef, cfg.min_elevation_deg, cfg.softness_deg,
                        cfg.revisit_logsumexp_temp, cfg.revisit_weight,
                        ground_weights=optimizer.ground_weights,
                        revisit_reduce=cfg.revisit_reduce,
                        revisit_spatial_tau=cfg.revisit_spatial_tau,
                        ground_unit=optimizer.ground_unit,
                        revisit_softness=cfg.revisit_softness_deg,
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
        ax.plot(0, 0, '*', markerfacecolor='orange', markersize=15, markeredgecolor='k', markeredgewidth=0.5)
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


def plot_loss_landscape_pca(optimizer, z_center, d1, d2, trajectories,
                            grid_size=50, padding=0.3, save_path=None):
    """
    Visualize loss landscape on a PCA-defined 2D slice with trajectory overlays.

    Args:
        optimizer: ConstellationOptimizer instance (for TLE structure, grid, etc.)
        z_center: [D] tensor, center of the landscape (typically best final z)
        d1, d2: [D] unit vectors defining the 2D slice (PCA directions)
        trajectories: list of (label, z_snapshots, color) where z_snapshots is list of [D] tensors
        grid_size: resolution of the landscape grid
        padding: fractional padding beyond trajectory extent
        save_path: path to save the figure
    """
    if save_path is None:
        save_path = 'loss_landscape_pca.png'

    cfg = optimizer.config

    # Project all trajectory points to find axis ranges
    all_alphas, all_betas = [], []
    for _label, z_snapshots, _color in trajectories:
        for z in z_snapshots:
            delta = z - z_center
            all_alphas.append(torch.dot(delta, d1).item())
            all_betas.append(torch.dot(delta, d2).item())

    alpha_min, alpha_max = min(all_alphas), max(all_alphas)
    beta_min, beta_max = min(all_betas), max(all_betas)

    # Add padding and ensure minimum range
    alpha_span = max(alpha_max - alpha_min, 0.1)
    beta_span = max(beta_max - beta_min, 0.1)
    alpha_pad = alpha_span * padding
    beta_pad = beta_span * padding
    alpha_range = (alpha_min - alpha_pad, alpha_max + alpha_pad)
    beta_range = (beta_min - beta_pad, beta_max + beta_pad)

    alphas = torch.linspace(alpha_range[0], alpha_range[1], grid_size)
    betas = torch.linspace(beta_range[0], beta_range[1], grid_size)

    loss_grid = np.zeros((grid_size, grid_size))
    cov_grid = np.zeros((grid_size, grid_size))
    revisit_grid = np.zeros((grid_size, grid_size))
    hard_loss_grid = np.zeros((grid_size, grid_size))
    hard_cov_grid = np.zeros((grid_size, grid_size))
    hard_revisit_grid = np.zeros((grid_size, grid_size))

    total = grid_size * grid_size
    print(f"\nComputing PCA loss landscape ({grid_size}x{grid_size} = {total} evaluations)...")

    # Save original z values
    z_orig_list = [re.optimizer_param.detach().clone() for re in optimizer.reparam_elements]

    for ai, alpha in enumerate(alphas):
        for bi, beta in enumerate(betas):
            perturbed = z_center + alpha * d1 + beta * d2

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
                        optimizer.tles, optimizer.tsinces, optimizer.gmst_tensor,
                        optimizer.ground_ecef, cfg.min_elevation_deg, cfg.softness_deg,
                        cfg.revisit_logsumexp_temp, cfg.revisit_weight,
                        ground_weights=optimizer.ground_weights,
                        revisit_reduce=cfg.revisit_reduce,
                        revisit_spatial_tau=cfg.revisit_spatial_tau,
                        ground_unit=optimizer.ground_unit,
                        revisit_softness=cfg.revisit_softness_deg,
                    )
                    loss_grid[bi, ai] = l.item()
                    cov_grid[bi, ai] = c.item() * 100
                    revisit_grid[bi, ai] = r.item()

                    h_cov, h_rev = compute_hard_metrics(
                        optimizer.tles, optimizer.tsinces, optimizer.gmst_tensor,
                        optimizer.ground_ecef, cfg.min_elevation_deg,
                        ground_weights=optimizer.ground_weights,
                        revisit_reduce=cfg.revisit_reduce,
                        ground_unit=optimizer.ground_unit,
                    )
                    hard_cov_grid[bi, ai] = h_cov * 100
                    hard_revisit_grid[bi, ai] = h_rev
                    hard_loss_grid[bi, ai] = -h_cov + cfg.revisit_weight * h_rev
                except Exception:
                    loss_grid[bi, ai] = float('nan')
                    cov_grid[bi, ai] = float('nan')
                    revisit_grid[bi, ai] = float('nan')
                    hard_loss_grid[bi, ai] = float('nan')
                    hard_cov_grid[bi, ai] = float('nan')
                    hard_revisit_grid[bi, ai] = float('nan')

            done = ai * grid_size + bi + 1
            if done % max(1, total // 10) == 0 or done == total:
                print(f"  {done}/{total}")

    # Restore original z values
    for re, z_orig in zip(optimizer.reparam_elements, z_orig_list):
        if re.n_free > 0:
            re._z = torch.nn.Parameter(z_orig)

    # Restore TLEs
    for i, tle in enumerate(optimizer.tles):
        elements = optimizer.reparam_elements[i].to_elements()
        update_tle_from_elements(tle, elements)

    # Plot: 2 rows (relaxed top, hard bottom) x 3 columns
    A, B = np.meshgrid(alphas.numpy(), betas.numpy())
    fig, axes_arr = plt.subplots(2, 3, figsize=(20, 12))

    row_configs = [
        ('Relaxed', [
            (loss_grid, 'Total Loss (relaxed)', 'turbo'),
            (cov_grid, 'Coverage % (relaxed)', 'turbo_r'),
            (revisit_grid, 'Mean Max Revisit (relaxed, min)', 'turbo'),
        ]),
        ('Hard', [
            (hard_loss_grid, 'Total Loss (hard)', 'turbo'),
            (hard_cov_grid, 'Coverage % (hard)', 'turbo_r'),
            (hard_revisit_grid, 'Mean Max Revisit (hard, min)', 'turbo'),
        ]),
    ]

    for row_idx, (row_label, panels) in enumerate(row_configs):
        for col_idx, (data, title, cmap) in enumerate(panels):
            ax = axes_arr[row_idx, col_idx]
            levels = 30
            cs = ax.contourf(A, B, data, levels=levels, cmap=cmap)
            ax.contour(A, B, data, levels=levels, colors='k', linewidths=0.3, alpha=0.4)

            # Overlay trajectories
            for label, z_snapshots, color in trajectories:
                proj_a = [torch.dot(z - z_center, d1).item() for z in z_snapshots]
                proj_b = [torch.dot(z - z_center, d2).item() for z in z_snapshots]
                show_label = (row_idx == 0 and col_idx == 0)
                ax.plot(proj_a, proj_b, '-', color=color, lw=1.5, alpha=0.8,
                        label=label if show_label else None)
                # Start marker (hollow circle)
                ax.plot(proj_a[0], proj_b[0], 'o', color=color,
                        markerfacecolor='none', markersize=8, markeredgewidth=1.5)
                # End marker (filled)
                ax.plot(proj_a[-1], proj_b[-1], 'o', color=color,
                        markersize=6, markeredgecolor='k', markeredgewidth=0.5)
                # Direction arrow at ~40% of trajectory
                arrow_idx = len(proj_a) * 0.5
                if arrow_idx > 0 and arrow_idx < len(proj_a) - 1:
                    ax.annotate('', xy=(proj_a[arrow_idx + 1], proj_b[arrow_idx + 1]),
                                xytext=(proj_a[arrow_idx], proj_b[arrow_idx]),
                                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

            # Optimum marker
            ax.plot(0, 0, 'r*', markersize=15, markeredgecolor='k', markeredgewidth=0.5,
                    label='Optimum' if (row_idx == 0 and col_idx == 0) else None)

            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(title)
            ax.set_aspect('equal')
            plt.colorbar(cs, ax=ax, shrink=0.8)

    axes_arr[0, 0].legend(fontsize=7, loc='upper left')

    fig.suptitle('Loss Landscape (PCA 2D slice) with Optimization Trajectories', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"PCA loss landscape saved to {save_path}")
    plt.close()


def plot_loss_landscapes_per_trajectory(optimizer, trajectory_configs, grid_size=50,
                                         padding=0.3, global_view=None,
                                         save_path=None, fig_width=10):
    """
    Visualize loss landscape on per-trajectory PCA slices.

    Produces a 2-row figure: top = relaxed loss, bottom = hard loss.
    Each per-trajectory column uses its own PCA directions. An optional
    global_view column uses caller-specified directions (e.g. random) to
    show the broader landscape.

    Args:
        optimizer: ConstellationOptimizer instance (for TLE structure, grid, etc.)
        trajectory_configs: list of (label, z_center, d1, d2, z_snapshots, color)
        global_view: optional (label, z_center, d1, d2, scale) for a zoomed-out column
        grid_size: resolution of the landscape grid
        padding: fractional padding beyond trajectory extent
        save_path: path to save the figure
        fig_width: figure width in inches (default 6.5 for paper column width)
    """
    if save_path is None:
        save_path = 'loss_landscape_per_trajectory.pdf'

    cfg = optimizer.config

    # Build unified column configs: global view (if any) then per-trajectory
    columns = []

    if global_view is not None:
        g_label, g_center, g_d1, g_d2, g_scale = global_view
        columns.append({
            'label': g_label,
            'z_center': g_center, 'd1': g_d1, 'd2': g_d2,
            'alpha_range': (-g_scale, g_scale),
            'beta_range': (-g_scale, g_scale),
            'overlays': [],
        })

    for label, z_center, d1, d2, z_snapshots, color in trajectory_configs:
        proj_a = [torch.dot(z - z_center, d1).item() for z in z_snapshots]
        proj_b = [torch.dot(z - z_center, d2).item() for z in z_snapshots]
        a_min, a_max = min(proj_a), max(proj_a)
        b_min, b_max = min(proj_b), max(proj_b)
        a_span = max(a_max - a_min, 0.1)
        b_span = max(b_max - b_min, 0.1)
        columns.append({
            'label': label,
            'z_center': z_center, 'd1': d1, 'd2': d2,
            'alpha_range': (a_min - a_span * padding, a_max + a_span * padding),
            'beta_range': (b_min - b_span * padding, b_max + b_span * padding),
            'overlays': [(label, proj_a, proj_b, 'white')],
        })

    n_cols = len(columns)
    subplot_size = fig_width / (n_cols + 0.15)
    fig_height = 2 * subplot_size + 0.4
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(2, n_cols + 1, width_ratios=[1] * n_cols + [0.03],
                          wspace=0.35, hspace=0.4)
    axes_arr = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(2)])
    cbar_axes = [fig.add_subplot(gs[r, n_cols]) for r in range(2)]

    # Save original z values
    z_orig_list = [re.optimizer_param.detach().clone() for re in optimizer.reparam_elements]
    last_cs = [None, None]  # track last contourf mappable per row

    for col_idx, col_cfg in enumerate(columns):
        z_center = col_cfg['z_center']
        d1, d2 = col_cfg['d1'], col_cfg['d2']

        alphas = torch.linspace(col_cfg['alpha_range'][0], col_cfg['alpha_range'][1], grid_size)
        betas = torch.linspace(col_cfg['beta_range'][0], col_cfg['beta_range'][1], grid_size)

        loss_grid = np.zeros((grid_size, grid_size))
        hard_loss_grid = np.zeros((grid_size, grid_size))

        total = grid_size * grid_size
        print(f"alpha: {col_cfg['alpha_range']}, beta: {col_cfg['beta_range']}")
        print(f"\nComputing landscape for '{col_cfg['label']}' "
              f"({grid_size}x{grid_size} = {total} evaluations)...")
        

        for ai, alpha in enumerate(alphas):
            for bi, beta in enumerate(betas):
                perturbed = z_center + alpha * d1 + beta * d2

                offset = 0
                for re in optimizer.reparam_elements:
                    if re.n_free > 0:
                        re._z = torch.nn.Parameter(perturbed[offset:offset + re.n_free].clone())
                        offset += re.n_free

                for i, tle in enumerate(optimizer.tles):
                    elements = optimizer.reparam_elements[i].to_elements()
                    update_tle_from_elements(tle, elements)
                for tle in optimizer.tles:
                    dsgp4.initialize_tle(tle, with_grad=False)

                with torch.no_grad():
                    try:
                        l, c, r = compute_loss(
                            optimizer.tles, optimizer.tsinces, optimizer.gmst_tensor,
                            optimizer.ground_ecef, cfg.min_elevation_deg, cfg.softness_deg,
                            cfg.revisit_logsumexp_temp, cfg.revisit_weight,
                            ground_weights=optimizer.ground_weights,
                            revisit_reduce=cfg.revisit_reduce,
                            revisit_spatial_tau=cfg.revisit_spatial_tau,
                            ground_unit=optimizer.ground_unit,
                            revisit_softness=cfg.revisit_softness_deg,
                        )
                        loss_grid[bi, ai] = l.item()

                        for tle in optimizer.tles:
                            dsgp4.initialize_tle(tle, with_grad=False)

                        h_cov, h_rev = compute_hard_metrics(
                            optimizer.tles, optimizer.tsinces, optimizer.gmst_tensor,
                            optimizer.ground_ecef, cfg.min_elevation_deg,
                            ground_weights=optimizer.ground_weights,
                            revisit_reduce=cfg.revisit_reduce,
                            ground_unit=optimizer.ground_unit,
                        )
                        hard_loss_grid[bi, ai] = -h_cov + cfg.revisit_weight * h_rev
                    except Exception:
                        loss_grid[bi, ai] = float('nan')
                        hard_loss_grid[bi, ai] = float('nan')

                done = ai * grid_size + bi + 1
                if done % max(1, total // 10) == 0 or done == total:
                    print(f"  {done}/{total}")

        # Plot this column
        A, B = np.meshgrid(alphas.numpy(), betas.numpy())

        for row_idx, (data, row_label) in enumerate([
            (loss_grid, 'Relaxed'),
            (hard_loss_grid, 'Hard'),
        ]):
            ax = axes_arr[row_idx, col_idx]
            levels = 30
            cs = ax.contourf(A, B, data, levels=levels, cmap='turbo')
            ax.contour(A, B, data, levels=levels, colors='k', linewidths=0.3, alpha=0.4)
            last_cs[row_idx] = cs

            # Overlay trajectories (exp1 style: white line, circle start, orange star end)
            for ovl_label, proj_a, proj_b, ovl_color in col_cfg['overlays']:
                ax.plot(proj_a, proj_b, '-', color='w', lw=1.5, alpha=0.8)
                ax.plot(proj_a[0], proj_b[0], 'wo', markersize=5,
                        markeredgecolor='k', markeredgewidth=0.5)
                ax.plot(proj_a[-1], proj_b[-1], '*', markersize=8,
                        markerfacecolor='orange', markeredgecolor='k', markeredgewidth=0.5)

            ax.set_xlabel("PC1")
            if col_idx == 0:
                ax.set_ylabel("PC2")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            if row_idx == 0:
                ax.set_title(f'{col_cfg["label"]}\n({row_label} loss)')
            else:
                ax.set_title(f'({row_label} loss)')

    # Restore original z values
    for re, z_orig in zip(optimizer.reparam_elements, z_orig_list):
        if re.n_free > 0:
            re._z = torch.nn.Parameter(z_orig)

    for i, tle in enumerate(optimizer.tles):
        elements = optimizer.reparam_elements[i].to_elements()
        update_tle_from_elements(tle, elements)

    # Colorbars in dedicated axes
    for row_idx in range(2):
        if last_cs[row_idx] is not None:
            fig.colorbar(last_cs[row_idx], cax=cbar_axes[row_idx])

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-trajectory loss landscape saved to {save_path}")
    plt.close()
