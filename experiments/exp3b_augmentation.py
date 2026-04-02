"""
Experiment 3b: Constellation Augmentation

Start with a fixed 6-satellite Walker 3/2/1 constellation at 550 km.
Add 2 new satellites with all orbital elements free.
Optimize the new satellites to maximize coverage + minimize revisit
over European targets, while the Walker fleet remains fixed.

This demonstrates a problem class that parametric families cannot address:
optimizing new satellites to complement an existing, potentially suboptimal fleet.
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
    Config, make_constellation, make_gmst_array,
    compute_loss, compute_hard_metrics, extract_elements,
    IDX_INCLO, IDX_MO, IDX_NODEO, IDX_ECCO, IDX_ARGPO, IDX_NO_KOZAI,
)
from differentiable_eo.constraints import (
    FixedConstraint, UnboundedConstraint, BoxConstraint,
    PeriapsisApoapsisConstraint, ReparameterizedElements,
)
from differentiable_eo.tle_utils import (
    no_kozai_from_alt, alt_from_no_kozai, make_tle, extract_elements,
    update_tle_from_elements,
)

# Reuse Europe sampling from exp3
from exp3_weighted_europe import sample_europe_targets


def augmentation_specs():
    """All geometry free for augmentation satellites."""
    coupled = PeriapsisApoapsisConstraint(
        perigee_bounds_km=(400.0, 600.0),
        excess_bounds_km=(0.0, 1500.0),
    )
    return {
        IDX_INCLO: BoxConstraint(math.radians(30), math.radians(90)),
        IDX_NODEO: UnboundedConstraint(),
        IDX_MO: UnboundedConstraint(),
        IDX_ARGPO: UnboundedConstraint(),
        IDX_NO_KOZAI: coupled,
        IDX_ECCO: coupled,
        # Fixed metadata
        0: FixedConstraint(),  # bstar
        1: FixedConstraint(),  # ndot
        2: FixedConstraint(),  # nddot
    }


def main():
    # ---- Configuration ----
    N_ITERATIONS = 2000
    N_TIME_STEPS = 240
    MIN_EL = 10.0
    PROP_HOURS = 24.0
    N_TARGET_POINTS = 500
    LR = 1e-2

    # Relaxation params
    SOFTNESS = 5.0
    REVISIT_SOFTNESS = 2.0
    REVISIT_TAU = 10.0
    REVISIT_WEIGHT = 1.0

    # Existing fleet: small Walker constellation
    WALKER_N_PLANES = 3
    WALKER_N_PER_PLANE = 2
    WALKER_INC = 60.0
    WALKER_ALT = 550.0
    WALKER_RAANS = [i * 360.0 / WALKER_N_PLANES for i in range(WALKER_N_PLANES)]
    WALKER_F = 1

    # New satellites to optimize
    N_NEW = 2
    NEW_INIT_INC = 60.0
    NEW_INIT_ALT = 550.0

    # ---- Sample Europe targets ----
    geojson_path = os.path.join(os.path.dirname(__file__), 'europe.geojson')
    print(f"Sampling {N_TARGET_POINTS} points from Europe...")
    ground_ecef, ground_weights, target_lats, target_lons = sample_europe_targets(
        geojson_path, n_points=N_TARGET_POINTS)
    ground_unit = ground_ecef / torch.norm(ground_ecef, dim=-1, keepdim=True)

    # ---- Time grid ----
    prop_min = PROP_HOURS * 60
    tsinces = torch.linspace(0, prop_min, N_TIME_STEPS)
    gmst_array = make_gmst_array(tsinces)
    gmst_tensor = torch.tensor(gmst_array, dtype=torch.float64)

    # ---- Create fixed Walker fleet ----
    walker_tles = make_constellation(
        WALKER_N_PLANES, WALKER_N_PER_PLANE,
        WALKER_INC, WALKER_RAANS, WALKER_ALT,
        phase_offset_f=WALKER_F,
    )
    n_walker = len(walker_tles)
    print(f"Fixed Walker fleet: {n_walker} satellites")

    # ---- Evaluate Walker-only baseline ----
    for t in walker_tles:
        dsgp4.initialize_tle(t, with_grad=False)
    with torch.no_grad():
        walker_only_cov, walker_only_rev = compute_hard_metrics(
            walker_tles, tsinces, gmst_tensor, ground_ecef,
            min_el=MIN_EL, ground_weights=ground_weights,
            revisit_reduce='mean', ground_unit=ground_unit)
    print(f"Walker-only: cov={walker_only_cov*100:.2f}%, rev={walker_only_rev:.1f} min")

    # ---- Create new satellites (to be optimized) ----
    rng = np.random.RandomState(42)
    new_tles = []
    new_reparam = []
    specs = augmentation_specs()

    for i in range(N_NEW):
        raan = rng.uniform(0, 360)
        ma = rng.uniform(0, 360)
        tle = make_tle(math.radians(NEW_INIT_INC), math.radians(raan),
                       math.radians(ma), NEW_INIT_ALT)
        dsgp4.initialize_tle(tle, with_grad=False)
        elems = extract_elements(tle)
        reparam = ReparameterizedElements(elems, specs)
        new_tles.append(tle)
        new_reparam.append(reparam)

    # Optimizer over new satellites only
    z_params = [r.optimizer_param for r in new_reparam if r.n_free > 0]
    optimizer = torch.optim.AdamW(z_params, lr=LR)

    # ---- Optimization loop ----
    torch.manual_seed(7)
    loss_history = []
    hard_cov_history = []
    hard_rev_history = []
    hard_eval_iters = []

    print(f"\nOptimizing {N_NEW} new satellites for {N_ITERATIONS} iterations...")
    for iteration in range(N_ITERATIONS):
        optimizer.zero_grad()

        # Update new TLEs from reparameterized elements
        tle_elem_list = []
        for i, tle in enumerate(new_tles):
            elements = new_reparam[i].to_elements()
            update_tle_from_elements(tle, elements)
            tle_elem_list.append(dsgp4.initialize_tle(tle, with_grad=True))

        # Initialize Walker TLEs (no grad)
        for t in walker_tles:
            dsgp4.initialize_tle(t, with_grad=False)

        # GMST randomization
        gmst_offset = torch.rand(1).item() * 2 * math.pi
        gmst_cur = gmst_tensor + gmst_offset

        # Forward pass with ALL satellites (Walker + new)
        all_tles = walker_tles + new_tles
        loss, cov, rev = compute_loss(
            all_tles, tsinces, gmst_cur, ground_ecef,
            min_el=MIN_EL, softness=SOFTNESS,
            revisit_tau=REVISIT_TAU, revisit_weight=REVISIT_WEIGHT,
            ground_weights=ground_weights, revisit_reduce='mean',
            ground_unit=ground_unit, revisit_softness=REVISIT_SOFTNESS,
        )

        loss.backward()

        # Chain rule through reparameterization (new sats only)
        for i in range(N_NEW):
            ephemeral_grad = tle_elem_list[i].grad
            if ephemeral_grad is not None:
                new_reparam[i].compute_z_grad(ephemeral_grad)

        optimizer.step()
        loss_history.append(loss.item())

        # Periodic hard eval
        if iteration % 20 == 0 or iteration == N_ITERATIONS - 1:
            for t in walker_tles:
                dsgp4.initialize_tle(t, with_grad=False)
            for i, tle in enumerate(new_tles):
                elements = new_reparam[i].to_elements()
                update_tle_from_elements(tle, elements)
                dsgp4.initialize_tle(tle, with_grad=False)

            with torch.no_grad():
                h_cov, h_rev = compute_hard_metrics(
                    all_tles, tsinces, gmst_tensor, ground_ecef,
                    min_el=MIN_EL, ground_weights=ground_weights,
                    revisit_reduce='mean', ground_unit=ground_unit)
            hard_cov_history.append(h_cov * 100 if isinstance(h_cov, float) else h_cov.item() * 100)
            hard_rev_history.append(h_rev if isinstance(h_rev, float) else h_rev.item())
            hard_eval_iters.append(iteration)

            if iteration % 100 == 0 or iteration == N_ITERATIONS - 1:
                print(f"  {iteration:4d}  cov={hard_cov_history[-1]:.2f}%  rev={hard_rev_history[-1]:.1f} min  loss={loss.item():.4f}")

    # ---- Final results ----
    final_cov = hard_cov_history[-1]
    final_rev = hard_rev_history[-1]

    print("\n" + "=" * 70)
    print("  EXPERIMENT 3b: CONSTELLATION AUGMENTATION")
    print("=" * 70)
    print(f"  Walker-only ({n_walker} sats):  cov={walker_only_cov*100:.2f}%  rev={walker_only_rev:.1f} min")
    print(f"  Augmented (+{N_NEW} sats):     cov={final_cov:.2f}%  rev={final_rev:.1f} min")
    print()
    print("  New satellite geometry:")
    for i in range(N_NEW):
        e = new_reparam[i].to_elements()
        inc = math.degrees(e[IDX_INCLO].item())
        raan = math.degrees(e[IDX_NODEO].item()) % 360
        ecc = e[IDX_ECCO].item()
        argp = math.degrees(e[IDX_ARGPO].item()) % 360
        alt = alt_from_no_kozai(e[IDX_NO_KOZAI].item())
        a = 6378.137 + alt
        rp = a * (1 - ecc) - 6378.137
        ra = a * (1 + ecc) - 6378.137
        print(f"    Sat {i}: inc={inc:.1f}°  RAAN={raan:.1f}°  ecc={ecc:.4f}  argp={argp:.1f}°  alt={alt:.0f} km  (rp={rp:.0f}/ra={ra:.0f} km)")
    print("=" * 70)

    # ---- Figures ----
    print("\nGenerating figures...")
    COV_COLOR = '#3F51B5'
    REV_COLOR = '#E91E63'
    paper_dir = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')

    # Convergence
    fig_conv, ax_conv = plt.subplots(figsize=(4, 3.5))
    ax_conv_rev = ax_conv.twinx()

    ax_conv.plot(hard_eval_iters, hard_cov_history, 'o-',
                 color=COV_COLOR, lw=1.5, markersize=2, label='Coverage (augmented)')
    ax_conv.axhline(walker_only_cov * 100, color=COV_COLOR, ls='--', alpha=0.5,
                    label=f'Walker-only ({walker_only_cov*100:.1f}\\%)')
    ax_conv.set_xlabel('Iteration')
    ax_conv.set_ylabel('Coverage [\\%]', color=COV_COLOR)
    ax_conv.tick_params(axis='y', labelcolor=COV_COLOR)

    ax_conv_rev.plot(hard_eval_iters, hard_rev_history, 's-',
                     color=REV_COLOR, lw=1.5, markersize=2, label='Revisit (augmented)')
    ax_conv_rev.axhline(walker_only_rev, color=REV_COLOR, ls='--', alpha=0.5,
                        label=f'Walker-only ({walker_only_rev:.0f} min)')
    ax_conv_rev.set_ylabel('Mean max revisit [min]', color=REV_COLOR)
    ax_conv_rev.tick_params(axis='y', labelcolor=REV_COLOR)

    lines1, labels1 = ax_conv.get_legend_handles_labels()
    lines2, labels2 = ax_conv_rev.get_legend_handles_labels()
    ax_conv.legend(lines1 + lines2, labels1 + labels2, fontsize=6, ncol=2, loc='upper left')
    ax_conv.set_xlim(0, N_ITERATIONS)
    ax_conv.grid(True, alpha=0.2)
    fig_conv.tight_layout()
    fig_conv.savefig(os.path.join(paper_dir, 'exp3b_convergence.pdf'), bbox_inches='tight')
    print(f"  Saved exp3b_convergence.pdf")
    plt.close(fig_conv)

    # Globe: Walker (gray) + new sats (colored)
    from differentiable_eo.globe import eci_xyz, camera_direction, is_occluded
    from differentiable_eo.constants import R_EARTH as R_E
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(5, 5), facecolor='none')
    ax = fig.add_subplot(111, projection='3d', facecolor='none')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Determine frame size from largest orbit
    max_r = R_E + WALKER_ALT
    for i in range(N_NEW):
        e = new_reparam[i].to_elements()
        alt = alt_from_no_kozai(e[IDX_NO_KOZAI].item())
        ecc = e[IDX_ECCO].item()
        max_r = max(max_r, (R_E + alt) * (1 + ecc))
    lim = max_r * 0.72

    cam = camera_direction(25, 45)
    ring_th = np.linspace(0, 2 * math.pi, 200)

    # Blue sphere
    u_s = np.linspace(0, 2 * np.pi, 40)
    v_s = np.linspace(0, np.pi, 20)
    ax.plot_surface(
        R_E * np.outer(np.cos(u_s), np.sin(v_s)),
        R_E * np.outer(np.sin(u_s), np.sin(v_s)),
        R_E * np.outer(np.ones_like(u_s), np.cos(v_s)),
        alpha=0.15, color='steelblue', linewidth=0)

    # Walker fleet (gray)
    walker_stats = []
    for t in walker_tles:
        dsgp4.initialize_tle(t, with_grad=False)
        e = extract_elements(t)
        walker_stats.append(e)

    for si, e in enumerate(walker_stats):
        inc_r = e[IDX_INCLO].item()
        raan_r = e[IDX_NODEO].item()
        r = R_E + WALKER_ALT
        plane_idx = si // WALKER_N_PER_PLANE

        # Ring (only draw once per plane)
        if si % WALKER_N_PER_PLANE == 0:
            pts = np.array([eci_xyz(inc_r, raan_r, th, r) for th in ring_th])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-',
                    color='gray', alpha=0.3, lw=0.8)

        # Satellite
        ma_r = e[IDX_MO].item()
        pos = np.array(eci_xyz(inc_r, raan_r, ma_r, r))
        if is_occluded(pos, cam, R_E):
            ax.scatter(*pos, s=40, facecolors='none', edgecolors='gray',
                      linewidths=1.0, depthshade=False, alpha=0.4)
        else:
            ax.scatter(*pos, s=40, color='gray', edgecolors='k',
                      linewidths=0.4, depthshade=False, zorder=4)

    # New satellites (colored)
    new_colors = ['#E91E63', '#3F51B5']  # pink, indigo
    for i in range(N_NEW):
        e = new_reparam[i].to_elements()
        inc_r = e[IDX_INCLO].item()
        raan_r = e[IDX_NODEO].item()
        ecc = e[IDX_ECCO].item()
        argp_r = e[IDX_ARGPO].item()
        ma_r = e[IDX_MO].item()
        alt = alt_from_no_kozai(e[IDX_NO_KOZAI].item())
        a = R_E + alt

        # Eccentric ring
        pts = np.array([eci_xyz(inc_r, raan_r, th, a, ecc=ecc, argp_rad=argp_r)
                        for th in ring_th])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-',
                color=new_colors[i], alpha=0.6, lw=1.2)

        # Satellite position
        pos = np.array(eci_xyz(inc_r, raan_r, ma_r, a, ecc=ecc, argp_rad=argp_r))
        if is_occluded(pos, cam, R_E):
            ax.scatter(*pos, s=80, facecolors='none', edgecolors=new_colors[i],
                      linewidths=1.5, depthshade=False, alpha=0.6)
        else:
            ax.scatter(*pos, s=80, color=new_colors[i], edgecolors='k',
                      linewidths=0.5, depthshade=False, zorder=5)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')
    ax.view_init(elev=25, azim=45)

    globe_path = os.path.join(paper_dir, 'exp3b_augmented_globe.pdf')
    fig.savefig(globe_path, bbox_inches='tight', transparent=True)
    print(f"  Saved exp3b_augmented_globe.pdf")
    plt.close(fig)

    # Visibility density comparison
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    dense_tsinces = torch.linspace(0, prop_min, 1000)
    dense_gmst = make_gmst_array(dense_tsinces)

    lon_bins = np.linspace(-180, 180, 181)
    lat_bins = np.linspace(-90, 90, 91)
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    GRID_LON, GRID_LAT = np.meshgrid(lon_centers, lat_centers)

    grid_lat_rad = np.radians(GRID_LAT.ravel())
    grid_lon_rad = np.radians(GRID_LON.ravel())
    grid_ecef = np.stack([
        R_E * np.cos(grid_lat_rad) * np.cos(grid_lon_rad),
        R_E * np.cos(grid_lat_rad) * np.sin(grid_lon_rad),
        R_E * np.sin(grid_lat_rad),
    ], axis=-1)
    grid_unit_np = grid_ecef / np.linalg.norm(grid_ecef, axis=-1, keepdims=True)

    def compute_vis_density(tles):
        density = np.zeros(len(grid_lat_rad))
        min_el_rad = np.radians(MIN_EL)
        for t in tles:
            dsgp4.initialize_tle(t, with_grad=False)
        n_time = len(dense_tsinces)
        step = max(1, n_time // 200)
        for t_idx in range(0, n_time, step):
            for tle in tles:
                try:
                    state = dsgp4.propagate(tle, dense_tsinces[t_idx:t_idx+1])
                    if state.dim() == 2:
                        pos_teme = state[0, :].detach().numpy()
                    else:
                        pos_teme = state[0, 0, :].detach().numpy()
                except Exception:
                    continue
                gmst = dense_gmst[t_idx]
                cg, sg = np.cos(gmst), np.sin(gmst)
                pos_ecef = np.array([
                    cg * pos_teme[0] + sg * pos_teme[1],
                    -sg * pos_teme[0] + cg * pos_teme[1],
                    pos_teme[2],
                ])
                diff = pos_ecef - grid_ecef
                dist = np.linalg.norm(diff, axis=-1)
                elev = np.arcsin(np.clip(np.sum(diff * grid_unit_np, axis=-1) / dist, -1, 1))
                density += (elev >= min_el_rad).astype(float)
        return density.reshape(GRID_LAT.shape)

    print("  Computing visibility density (Walker-only)...")
    density_walker = compute_vis_density(walker_tles)
    print("  Computing visibility density (augmented)...")
    # Update new TLEs to final state
    for i, tle in enumerate(new_tles):
        elements = new_reparam[i].to_elements()
        update_tle_from_elements(tle, elements)
    density_augmented = compute_vis_density(all_tles)

    vmax = max(density_walker.max(), density_augmented.max())

    for tag, density, title in [
        ('walker_only', density_walker, f'Walker-only ({n_walker} sats)'),
        ('augmented', density_augmented, f'Augmented (+{N_NEW} new)'),
    ]:
        fig_gt = plt.figure(figsize=(6, 3.5))
        ax_gt = fig_gt.add_subplot(111, projection=ccrs.PlateCarree())
        ax_gt.set_global()
        ax_gt.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='#cccccc', linewidth=0.3)
        ax_gt.add_feature(cfeature.OCEAN, facecolor='white')
        ax_gt.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='#cccccc')
        ax_gt.coastlines(linewidth=0.3, color='#999999')

        im = ax_gt.imshow(density, cmap='turbo', vmin=0, vmax=vmax, alpha=0.7,
                          extent=[-180, 180, -90, 90], origin='lower',
                          interpolation='bilinear',
                          transform=ccrs.PlateCarree(), zorder=2)

        ax_gt.scatter(target_lons, target_lats, s=3, c='white', edgecolors='k',
                     linewidths=0.2, alpha=0.3, transform=ccrs.PlateCarree(), zorder=4)

        ax_gt.set_title(title)
        plt.colorbar(im, ax=ax_gt, shrink=0.7, pad=0.02, label='Visibility count')
        ax_gt.gridlines(draw_labels=True, linewidth=0.2, alpha=0.3,
                       xlocs=range(-180, 181, 60), ylocs=range(-90, 91, 30))
        fig_gt.tight_layout()
        fig_gt.savefig(os.path.join(paper_dir, f'exp3b_density_{tag}.pdf'), bbox_inches='tight')
        print(f"  Saved exp3b_density_{tag}.pdf")
        plt.close(fig_gt)


if __name__ == '__main__':
    main()
