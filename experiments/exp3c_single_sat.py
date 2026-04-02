"""
Experiment 3c: Single Satellite Targeting London

1 satellite, all orbital elements free, optimized to minimize revisit
over a single target (London, 51.5°N 0°W). With one satellite and one
target, the optimizer must discover a resonant orbit to revisit the
target periodically.
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
import scienceplots
plt.style.use(['science', 'no-latex'])

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from differentiable_eo import (
    make_gmst_array, compute_loss, compute_hard_metrics,
    extract_elements, IDX_INCLO, IDX_MO, IDX_NODEO,
    IDX_ECCO, IDX_ARGPO, IDX_NO_KOZAI,
)
from differentiable_eo.constraints import (
    FixedConstraint, UnboundedConstraint, BoxConstraint,
    PeriapsisApoapsisConstraint, ReparameterizedElements,
)
from differentiable_eo.tle_utils import (
    no_kozai_from_alt, alt_from_no_kozai, make_tle, extract_elements,
    update_tle_from_elements,
)


def all_free_specs():
    coupled = PeriapsisApoapsisConstraint(
        perigee_bounds_km=(300.0, 600.0),
        excess_bounds_km=(0.0, 35000.0),  # allow up to GEO-like apogee
    )
    return {
        IDX_INCLO: BoxConstraint(math.radians(20), math.radians(90)),
        IDX_NODEO: UnboundedConstraint(),
        IDX_MO: UnboundedConstraint(),
        IDX_ARGPO: UnboundedConstraint(),
        IDX_NO_KOZAI: coupled,
        IDX_ECCO: coupled,
        0: FixedConstraint(),  # bstar
        1: FixedConstraint(),  # ndot
        2: FixedConstraint(),  # nddot
    }


def main():
    N_ITERATIONS = 3000
    N_TIME_STEPS = 360  # finer time resolution for single target
    MIN_EL = 10.0
    PROP_HOURS = 48.0  # 2 days to see periodicity
    SOFTNESS = 5.0
    REVISIT_SOFTNESS = 2.0
    REVISIT_TAU = 10.0
    REVISIT_WEIGHT = 5.0  # heavy revisit pressure
    LR = 1e-2

    # Target: Europe
    R_EARTH = 6378.137
    from exp3_weighted_europe import sample_europe_targets
    geojson_path = os.path.join(os.path.dirname(__file__), 'europe.geojson')
    ground_ecef, ground_weights, target_lats, target_lons = sample_europe_targets(
        geojson_path, n_points=300)
    ground_unit = ground_ecef / torch.norm(ground_ecef, dim=-1, keepdim=True)
    LONDON_LAT, LONDON_LON = 51.5, -0.1  # for marking on plot

    # Time grid
    prop_min = PROP_HOURS * 60
    tsinces = torch.linspace(0, prop_min, N_TIME_STEPS)
    gmst_array = make_gmst_array(tsinces)
    gmst_tensor = torch.tensor(gmst_array, dtype=torch.float64)

    # Create single satellite
    rng = np.random.RandomState(42)
    init_raan = rng.uniform(0, 360)
    init_ma = rng.uniform(0, 360)
    tle = make_tle(math.radians(55.0), math.radians(init_raan),
                   math.radians(init_ma), 550.0)
    dsgp4.initialize_tle(tle, with_grad=False)
    elems = extract_elements(tle)
    specs = all_free_specs()
    reparam = ReparameterizedElements(elems, specs)

    optimizer = torch.optim.AdamW([reparam.optimizer_param], lr=LR)

    # Run optimization
    torch.manual_seed(42)
    loss_history = []
    hard_rev_history = []
    hard_eval_iters = []

    print(f"Optimizing 1 satellite for London revisit, {N_ITERATIONS} iterations...")
    for iteration in range(N_ITERATIONS):
        optimizer.zero_grad()

        elements = reparam.to_elements()
        update_tle_from_elements(tle, elements)
        tle_elems = dsgp4.initialize_tle(tle, with_grad=True)

        gmst_offset = torch.rand(1).item() * 2 * math.pi
        gmst_cur = gmst_tensor + gmst_offset

        loss, cov, rev = compute_loss(
            [tle], tsinces, gmst_cur, ground_ecef,
            min_el=MIN_EL, softness=SOFTNESS,
            revisit_tau=REVISIT_TAU, revisit_weight=REVISIT_WEIGHT,
            ground_weights=ground_weights, revisit_reduce='mean',
            ground_unit=ground_unit, revisit_softness=REVISIT_SOFTNESS,
        )

        loss.backward()

        if tle_elems.grad is not None:
            reparam.compute_z_grad(tle_elems.grad)

        optimizer.step()
        loss_history.append(loss.item())

        if iteration % 20 == 0 or iteration == N_ITERATIONS - 1:
            elements = reparam.to_elements()
            update_tle_from_elements(tle, elements)
            dsgp4.initialize_tle(tle, with_grad=False)
            with torch.no_grad():
                h_cov, h_rev = compute_hard_metrics(
                    [tle], tsinces, gmst_tensor, ground_ecef,
                    min_el=MIN_EL, ground_weights=ground_weights,
                    revisit_reduce='mean', ground_unit=ground_unit)
            hard_rev_history.append(h_rev if isinstance(h_rev, float) else h_rev.item())
            hard_eval_iters.append(iteration)

            if iteration % 200 == 0 or iteration == N_ITERATIONS - 1:
                e = reparam.to_elements()
                inc = math.degrees(e[IDX_INCLO].item())
                ecc = e[IDX_ECCO].item()
                alt = alt_from_no_kozai(e[IDX_NO_KOZAI].item())
                a = R_EARTH + alt
                T_min = 2 * math.pi * math.sqrt(a**3 / 398600.4418) / 60
                revs = 23.9345 * 60 / T_min
                print(f"  {iteration:4d}  rev={hard_rev_history[-1]:.1f} min  "
                      f"inc={inc:.1f}°  ecc={ecc:.4f}  alt={alt:.0f} km  "
                      f"T={T_min:.1f} min  revs/day={revs:.2f}")

    # Final results
    e = reparam.to_elements()
    inc = math.degrees(e[IDX_INCLO].item())
    raan = math.degrees(e[IDX_NODEO].item()) % 360
    ma = math.degrees(e[IDX_MO].item()) % 360
    ecc = e[IDX_ECCO].item()
    argp = math.degrees(e[IDX_ARGPO].item()) % 360
    alt = alt_from_no_kozai(e[IDX_NO_KOZAI].item())
    a = R_EARTH + alt
    T_min = 2 * math.pi * math.sqrt(a**3 / 398600.4418) / 60
    T_hr = T_min / 60
    revs_per_day = 23.9345 / T_hr
    rp = a * (1 - ecc) - R_EARTH
    ra = a * (1 + ecc) - R_EARTH

    print("\n" + "=" * 70)
    print("  EXPERIMENT 3c: SINGLE SATELLITE - LONDON REVISIT")
    print("=" * 70)
    print(f"  inc={inc:.1f}°  RAAN={raan:.1f}°  ecc={ecc:.4f}  argp={argp:.1f}°")
    print(f"  Mean alt={alt:.0f} km  Perigee={rp:.0f} km  Apogee={ra:.0f} km")
    print(f"  Period={T_min:.1f} min = {T_hr:.3f} hr")
    print(f"  Revs/sidereal day={revs_per_day:.3f}  (nearest integer: {round(revs_per_day)})")
    print(f"  Deviation from {round(revs_per_day)}:1 = {abs(revs_per_day - round(revs_per_day))/revs_per_day*100:.3f}%")
    print(f"  Final revisit: {hard_rev_history[-1]:.1f} min")
    print("=" * 70)

    # ---- Figures ----
    paper_dir = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')

    # Ground track density
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from differentiable_eo.constants import EARTH_ROT_RAD_PER_MIN

    dense_tsinces = torch.linspace(0, prop_min, 2000)
    elements = reparam.to_elements()
    update_tle_from_elements(tle, elements)
    dsgp4.initialize_tle(tle, with_grad=False)
    state = dsgp4.propagate(tle, dense_tsinces)
    pos = state[:, 0, :].detach().numpy()
    r = np.linalg.norm(pos, axis=-1)
    lat = np.degrees(np.arcsin(np.clip(pos[:, 2] / r, -1, 1)))
    lon = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))
    lon_ecef = (lon - np.degrees(EARTH_ROT_RAD_PER_MIN * dense_tsinces.numpy())) % 360
    lon_ecef[lon_ecef > 180] -= 360

    fig_gt = plt.figure(figsize=(6, 3.5))
    ax_gt = fig_gt.add_subplot(111, projection=ccrs.PlateCarree())
    ax_gt.set_global()
    ax_gt.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='#cccccc', linewidth=0.3)
    ax_gt.add_feature(cfeature.OCEAN, facecolor='white')
    ax_gt.coastlines(linewidth=0.3, color='#999999')

    ax_gt.scatter(lon_ecef, lat, s=0.5, c=np.linspace(0, 1, len(lat)), cmap='turbo',
                 alpha=0.6, transform=ccrs.PlateCarree(), zorder=2)

    # Mark London
    ax_gt.scatter([LONDON_LON], [LONDON_LAT], s=100, marker='*', c='red',
                 edgecolors='k', linewidths=0.5, transform=ccrs.PlateCarree(), zorder=5)

    ax_gt.set_title(f'Single satellite ground track ({PROP_HOURS:.0f}h)')
    ax_gt.gridlines(draw_labels=True, linewidth=0.2, alpha=0.3,
                   xlocs=range(-180, 181, 60), ylocs=range(-90, 91, 30))
    fig_gt.tight_layout()
    fig_gt.savefig(os.path.join(paper_dir, 'exp3c_groundtrack.pdf'), bbox_inches='tight')
    print(f"  Saved exp3c_groundtrack.pdf")
    plt.close(fig_gt)

    # Globe figure
    from differentiable_eo.globe import eci_xyz, camera_direction, is_occluded
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(4, 4), facecolor='none')
    ax = fig.add_subplot(111, projection='3d', facecolor='none')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    max_r = a * (1 + ecc)
    lim = max_r * 0.72
    cam = camera_direction(25, 45)
    ring_th = np.linspace(0, 2 * math.pi, 300)

    u_s = np.linspace(0, 2 * np.pi, 40)
    v_s = np.linspace(0, np.pi, 20)
    ax.plot_surface(
        R_EARTH * np.outer(np.cos(u_s), np.sin(v_s)),
        R_EARTH * np.outer(np.sin(u_s), np.sin(v_s)),
        R_EARTH * np.outer(np.ones_like(u_s), np.cos(v_s)),
        alpha=0.15, color='steelblue', linewidth=0)

    inc_r = e[IDX_INCLO].item()
    raan_r = e[IDX_NODEO].item()
    argp_r = e[IDX_ARGPO].item()
    ma_r = e[IDX_MO].item()

    pts = np.array([eci_xyz(inc_r, raan_r, th, a, ecc=ecc, argp_rad=argp_r)
                    for th in ring_th])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-', color='#E91E63', alpha=0.6, lw=1.2)

    sat_pos = np.array(eci_xyz(inc_r, raan_r, ma_r, a, ecc=ecc, argp_rad=argp_r))
    ax.scatter(*sat_pos, s=80, color='#E91E63', edgecolors='k', linewidths=0.5,
              depthshade=False, zorder=5)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')
    ax.view_init(elev=25, azim=45)
    fig.savefig(os.path.join(paper_dir, 'exp3c_globe.pdf'), bbox_inches='tight', transparent=True)
    print(f"  Saved exp3c_globe.pdf")
    plt.close(fig)

    # Convergence
    fig_conv, ax_conv = plt.subplots(figsize=(4, 3.5))
    ax_conv.plot(hard_eval_iters, hard_rev_history, 's-', color='#E91E63',
                 lw=1.5, markersize=2)
    ax_conv.set_xlabel('Iteration')
    ax_conv.set_ylabel('Mean max revisit [min]')
    ax_conv.set_xlim(0, N_ITERATIONS)
    ax_conv.grid(True, alpha=0.2)
    fig_conv.tight_layout()
    fig_conv.savefig(os.path.join(paper_dir, 'exp3c_convergence.pdf'), bbox_inches='tight')
    print(f"  Saved exp3c_convergence.pdf")
    plt.close(fig_conv)


if __name__ == '__main__':
    main()
