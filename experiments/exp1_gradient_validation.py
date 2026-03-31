"""
Experiment 1: Gradient Validation (Toy Problem)

2-satellite, 1-plane MA-only optimization. Both satellites at 550 km, inc=60 deg,
RAAN=0 deg fixed. Only M_1 and M_2 are free. The optimal solution
is |M_1 - M_2| = 180 deg.

Produces a 2x3 grid of contour plots (relaxed & hard) over the (M_1,
M_2) domain, with the optimization trajectory overlaid.
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
plt.style.use(['science'])

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from differentiable_eo import (
    Config, ConstellationOptimizer,
    make_constellation, make_ground_grid_with_weights, make_gmst_array,
    compute_loss, compute_hard_metrics, extract_elements,
    IDX_INCLO, IDX_MO, IDX_NODEO,
)
from differentiable_eo.constraints import (
    FixedConstraint, UnboundedConstraint, default_parameter_specs,
)


def ma_only_specs():
    """Parameter specs that free only MA (inc, RAAN, alt all fixed)."""
    specs = default_parameter_specs()
    specs[IDX_INCLO] = FixedConstraint()
    specs[IDX_NODEO] = FixedConstraint()
    return specs


def evaluate_grid_point(ma1_deg, ma2_deg, alt_km, inc_deg,
                        tsinces, gmst_tensor, ground_ecef, ground_weights, ground_unit,
                        min_el, softness, revisit_softness, revisit_tau, revisit_weight):
    """Evaluate both relaxed and hard metrics at a single (M_1, M_2) point."""
    tles = make_constellation(
        n_planes=1, n_sats_per_plane=2,
        inc_deg=inc_deg,
        alt_km=alt_km,
        raan_offsets_deg=[0.0],
        ma_offsets_deg=[ma1_deg, ma2_deg],
    )
    for t in tles:
        dsgp4.initialize_tle(t, with_grad=False)

    with torch.no_grad():
        loss, cov, rev = compute_loss(
            tles, tsinces, gmst_tensor, ground_ecef,
            min_el=min_el, softness=softness,
            revisit_tau=revisit_tau, revisit_weight=revisit_weight,
            ground_weights=ground_weights,
            revisit_reduce='mean',
            ground_unit=ground_unit,
            revisit_softness=revisit_softness,
        )
        h_cov, h_rev = compute_hard_metrics(
            tles, tsinces, gmst_tensor, ground_ecef,
            min_el=min_el, ground_weights=ground_weights,
            revisit_reduce='mean', ground_unit=ground_unit,
        )

    return {
        'loss': loss.item(),
        'cov': cov.item() * 100,
        'rev': rev.item(),
        'h_cov': h_cov * 100,
        'h_rev': h_rev,
    }


def main():
    # ---- Configuration ----
    N_GRID = 72           # grid resolution per axis
    ALT_KM = 550.0
    INC_DEG = 60.0
    N_TIME_STEPS = 360
    N_LAT, N_LON = 72, 144
    LAT_BOUNDS = (-70.0, 70.0)
    MIN_EL = 10.0
    PROP_HOURS = 24.0

    # Relaxation hyperparams (matching exp3)
    SOFTNESS = 5.0
    REVISIT_SOFTNESS = 2.0
    REVISIT_TAU = 10.0
    REVISIT_WEIGHT = 2.0

    # Optimization
    N_ITERATIONS = 800
    LR = 1e-2
    INIT_MA1 = 179.0
    INIT_MA2 = 181.0

    # ---- Shared setup ----
    ground_ecef, ground_weights = make_ground_grid_with_weights(N_LAT, N_LON, LAT_BOUNDS)
    ground_unit = ground_ecef / torch.norm(ground_ecef, dim=-1, keepdim=True)
    prop_min = PROP_HOURS * 60
    tsinces = torch.linspace(0, prop_min, N_TIME_STEPS)
    gmst_array = make_gmst_array(tsinces)
    gmst_tensor = torch.tensor(gmst_array, dtype=torch.float64)

    # ---- Phase 1: Dense grid evaluation ----
    print(f"Evaluating {N_GRID}x{N_GRID} grid...")
    ma_vals = np.linspace(0, 360, N_GRID, endpoint=False)

    grid_loss = np.zeros((N_GRID, N_GRID))
    grid_cov = np.zeros((N_GRID, N_GRID))
    grid_rev = np.zeros((N_GRID, N_GRID))
    grid_h_cov = np.zeros((N_GRID, N_GRID))
    grid_h_rev = np.zeros((N_GRID, N_GRID))

    total = N_GRID * N_GRID
    done = 0
    for i, m1 in enumerate(ma_vals):
        for j, m2 in enumerate(ma_vals):
            result = evaluate_grid_point(
                m1, m2, ALT_KM, INC_DEG,
                tsinces, gmst_tensor, ground_ecef, ground_weights, ground_unit,
                MIN_EL, SOFTNESS, REVISIT_SOFTNESS, REVISIT_TAU, REVISIT_WEIGHT,
            )
            grid_loss[j, i] = result['loss']
            grid_cov[j, i] = result['cov']
            grid_rev[j, i] = result['rev']
            grid_h_cov[j, i] = result['h_cov']
            grid_h_rev[j, i] = result['h_rev']
            done += 1
            if done % max(1, total // 10) == 0:
                print(f"  {done}/{total} ({100*done/total:.0f}%)")

    # Compute hard loss from hard metrics (same formula as relaxed)
    grid_h_loss = -(grid_h_cov / 100) + REVISIT_WEIGHT * grid_h_rev

    # ---- Phase 2: Optimization trajectory ----
    print(f"\nRunning optimization from MA=({INIT_MA1}, {INIT_MA2}) deg...")

    config = Config(
        n_planes=1,
        n_sats_per_plane=2,
        target_alt_km=ALT_KM,
        initial_inc_deg=INC_DEG,
        initial_raan_offsets_deg=[0.0],
        initial_ma_offsets_deg=[INIT_MA1, INIT_MA2],
        prop_duration_hours=PROP_HOURS,
        n_time_steps=N_TIME_STEPS,
        n_lat=N_LAT, n_lon=N_LON,
        lat_bounds_deg=LAT_BOUNDS,
        min_elevation_deg=MIN_EL,
        n_iterations=N_ITERATIONS,
        lr=LR,
        softness_deg=SOFTNESS,
        revisit_softness_deg=REVISIT_SOFTNESS,
        revisit_logsumexp_temp=REVISIT_TAU,
        revisit_weight=REVISIT_WEIGHT,
        revisit_reduce='mean',
        randomize_gmst=False,  # fixed GMST for reproducible trajectory
        parameter_specs=ma_only_specs(),
    )

    opt = ConstellationOptimizer(config)

    # Record MA trajectory
    trajectory = []  # list of (ma1_deg, ma2_deg)

    # Initial position
    init_elems = opt.get_current_elements()
    trajectory.append((
        math.degrees(init_elems[0][IDX_MO].item()) % 360,
        math.degrees(init_elems[1][IDX_MO].item()) % 360,
    ))

    def record_mas(iteration, step_result, optimizer_obj):
        elems = optimizer_obj.get_current_elements()
        m1 = math.degrees(elems[0][IDX_MO].item()) % 360
        m2 = math.degrees(elems[1][IDX_MO].item()) % 360
        trajectory.append((m1, m2))

    result = opt.run(callback=record_mas)

    traj_m1 = [t[0] for t in trajectory]
    traj_m2 = [t[1] for t in trajectory]

    print(f"\nTrajectory: ({traj_m1[0]:.1f}, {traj_m2[0]:.1f}) -> "
          f"({traj_m1[-1]:.1f}, {traj_m2[-1]:.1f}) deg")
    print(f"Final separation: {abs(traj_m1[-1] - traj_m2[-1]):.1f} deg")

    # ---- Phase 3: Plot 2x3 grid ----
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.8))

    extent = [0, 360, 0, 360]

    panels = [
        # (row, col, data, title, cmap)
        (0, 0, grid_loss, 'Relaxed loss', 'turbo'),
        (0, 1, grid_cov, 'Relaxed coverage [\\%]', 'turbo'),
        (0, 2, grid_rev, 'Relaxed revisit [min]', 'turbo'),
        (1, 0, grid_h_loss, 'Hard loss', 'turbo'),
        (1, 1, grid_h_cov, 'Hard coverage [\\%]', 'turbo'),
        (1, 2, grid_h_rev, 'Hard revisit [min]', 'turbo'),
    ]

    for row, col, data, title, cmap in panels:
        ax = axes[row, col]
        im = ax.imshow(data, extent=extent, origin='lower', aspect='equal', cmap=cmap)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

        # Anti-diagonal reference line (|M_1 - M_2| = 180)
        ax.plot([0, 180], [180, 360], 'w--', alpha=0.5, lw=1)
        ax.plot([180, 360], [0, 180], 'w--', alpha=0.5, lw=1)

        # Optimization trajectory
        ax.plot(traj_m1, traj_m2, 'w-', lw=1.5, alpha=0.8)
        ax.plot(traj_m1[0], traj_m2[0], 'wo', markersize=5, markeredgecolor='k', markeredgewidth=0.5)
        ax.plot(traj_m1[-1], traj_m2[-1], '*', markersize=8, markerfacecolor='orange', markeredgecolor='k', markeredgewidth=0.5)

        ax.set_title(title)
        ax.set_xlabel('$\\mathcal{M}_1$ [deg]')
        ax.set_ylabel('$\\mathcal{M}_2$ [deg]')

    plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.5)

    save_path = os.path.join(os.path.dirname(__file__), 'exp1_gradient_validation.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
