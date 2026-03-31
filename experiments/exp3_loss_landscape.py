"""
Experiment 3: Loss Landscape with Optimization Trajectories

Runs multiple optimizations from different initial RAAN configurations,
records z-space trajectories, then uses per-trajectory PCA to find a
meaningful 2D slice of the loss landscape for each run. An additional
global view column uses random directions to show the broader landscape
structure (local minima).

Compares:
  - Near-uniform RAANs (should converge quickly)
  - Exp2 default RAANs (converges but slowly)
  - Clustered RAANs (likely gets stuck)
  - Two-cluster RAANs (may find local minimum)

Usage:
  python exp3_loss_landscape.py            # full run (optimization + plot)
  python exp3_loss_landscape.py --plot-only # regenerate plot from saved data
"""

import sys
import os
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from differentiable_eo import (
    Config, ConstellationOptimizer,
    IDX_INCLO, IDX_NODEO,
)
from differentiable_eo.constraints import FixedConstraint, default_parameter_specs
from differentiable_eo.visualization import plot_loss_landscapes_per_trajectory

# ---- Experiment constants ----
N_PLANES = 6
N_SATS_PER_PLANE = 4
ALT_KM = 550.0
N_ITERATIONS = 800
N_TIME_STEPS = 240
N_LAT, N_LON = 36, 72
LAT_BOUNDS = (-70.0, 70.0)
MIN_EL = 10.0
PROP_HOURS = 24.0
REVISIT_REDUCE = 'mean'
SNAPSHOT_EVERY = 5

CONFIGS = [
    ("Near-uniform", [0, 60, 120, 180, 240, 300]),
    ("Exp2 default", [0, 30, 120, 200, 210, 300]),
    ("Clustered",    [0, 10, 20, 30, 40, 50]),
    ("Two-cluster",  [0, 5, 180, 185, 270, 275]),
]
COLORS = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0']


def ma_and_raan_specs():
    """Parameter specs that free mean anomaly and RAAN (inc + alt fixed)."""
    specs = default_parameter_specs()
    specs[IDX_INCLO] = FixedConstraint()
    return specs


def make_config(raans):
    """Create config with given RAAN offsets (deterministic MAs from seed 42)."""
    rng = np.random.RandomState(42)
    n_total = N_PLANES * N_SATS_PER_PLANE
    common_mas = rng.uniform(0, 360, size=n_total).tolist()
    return Config(
        n_planes=N_PLANES,
        n_sats_per_plane=N_SATS_PER_PLANE,
        target_alt_km=ALT_KM,
        initial_inc_deg=60.0,
        initial_raan_offsets_deg=raans,
        initial_ma_offsets_deg=common_mas,
        prop_duration_hours=PROP_HOURS,
        n_time_steps=N_TIME_STEPS,
        n_lat=N_LAT, n_lon=N_LON,
        lat_bounds_deg=LAT_BOUNDS,
        min_elevation_deg=MIN_EL,
        n_iterations=N_ITERATIONS,
        lr=1e-2,
        softness_deg=5.0,
        revisit_softness_deg=2.0,
        revisit_logsumexp_temp=10,
        revisit_weight=1.0,
        revisit_reduce=REVISIT_REDUCE,
        randomize_gmst=True,
        parameter_specs=ma_and_raan_specs(),
        per_plane_params=[IDX_NODEO],
    )


def main():
    exp_dir = os.path.dirname(__file__)
    save_data_path = os.path.join(exp_dir, 'exp3_data.pt')
    save_path = os.path.join(exp_dir, 'exp3_loss_landscape.pdf')

    # ---- Plot-only mode: load saved data and regenerate figure ----
    if '--plot-only' in sys.argv:
        if not os.path.exists(save_data_path):
            print(f"Error: {save_data_path} not found. Run without --plot-only first.")
            sys.exit(1)
        print(f"Loading saved data from {save_data_path}")
        data = torch.load(save_data_path, weights_only=False)
        opt = ConstellationOptimizer(make_config(CONFIGS[0][1]))
        # Re-do scaleing for global view based on loaded data
        global_view = data['global_view']   
        new_global_view = (global_view[0], global_view[1], global_view[2], global_view[3], global_view[4] * 0.3)

        plot_loss_landscapes_per_trajectory(
            opt, data['per_traj_configs'],
            global_view=new_global_view,
            grid_size=50, save_path=save_path,
        )
        return

    # ---- Run each config and record z-space trajectories ----
    run_results = []  # (label, OptimizationResult, optimizer)
    z_trajectories = []  # (label, [z_t0, z_t1, ...], color)

    for (label, raans), color in zip(CONFIGS, COLORS):
        print(f"\n{'=' * 65}")
        print(f"  Config: {label} -- RAANs = {raans}")
        print(f"{'=' * 65}")

        config = make_config(raans)
        opt = ConstellationOptimizer(config)

        # Record z-space snapshots
        z_snaps = []

        # Record initial z
        z_flat = torch.cat([re.optimizer_param.detach().clone()
                            for re in opt.reparam_elements])
        z_snaps.append(z_flat)

        def record_z(iteration, step_result, optimizer_obj, z_list=z_snaps):
            if iteration % SNAPSHOT_EVERY == 0 or iteration == N_ITERATIONS - 1:
                z = torch.cat([re.optimizer_param.detach().clone()
                               for re in optimizer_obj.reparam_elements])
                z_list.append(z)

        result = opt.run(callback=record_z)

        run_results.append((label, result, opt))
        z_trajectories.append((label, z_snaps, color))

        print(f"  {label}: final loss={result.loss_history[-1]:.4f}, "
              f"cov={result.cov_history[-1]:.2f}%, "
              f"revisit={result.revisit_history[-1]:.1f} min, "
              f"z snapshots={len(z_snaps)}")

    # ---- Find best final z (lowest final loss) ----
    best_idx = min(range(len(run_results)),
                   key=lambda i: run_results[i][1].loss_history[-1])
    best_label, best_result, best_opt = run_results[best_idx]
    print(f"\nBest config: {best_label} "
          f"(final loss={best_result.loss_history[-1]:.4f})")

    # ---- Per-trajectory PCA ----
    per_traj_configs = []
    for (label, z_snaps, color), (_, result, opt) in zip(z_trajectories, run_results):
        Z = torch.stack(z_snaps)  # [N_snapshots, D]
        z_center = Z[-1]  # center at final z
        Z_centered = Z - z_center.unsqueeze(0)

        U, S, Vt = torch.linalg.svd(Z_centered, full_matrices=False)
        d1 = Vt[0]
        d2 = Vt[1]

        total_var = (S ** 2).sum()
        var_explained = (S[:2] ** 2) / total_var * 100
        print(f"  {label}: PC1={var_explained[0]:.1f}%, PC2={var_explained[1]:.1f}% "
              f"(total={var_explained.sum():.1f}%)")

        per_traj_configs.append((label, z_center, d1, d2, z_snaps, color))

    # ---- Global view: random directions centered at best result ----
    z_global_center = torch.cat([re.optimizer_param.detach().clone()
                                  for re in best_opt.reparam_elements])
    max_dist = 0
    for _label, z_snaps, _color in z_trajectories:
        for z in z_snaps:
            dist = (z - z_global_center).norm().item()
            max_dist = max(max_dist, dist)
    global_scale = max_dist * 1.0

    torch.manual_seed(0)
    d1_rand = torch.randn_like(z_global_center)
    d1_rand = d1_rand / d1_rand.norm()
    d2_rand = torch.randn_like(z_global_center)
    d2_rand = d2_rand - torch.dot(d2_rand, d1_rand) * d1_rand  # Gram-Schmidt
    d2_rand = d2_rand / d2_rand.norm()

    global_view = ('Global view', z_global_center, d1_rand, d2_rand, global_scale)
    print(f"\nGlobal view scale: {global_scale:.2f}")

    # ---- Save data for --plot-only ----
    torch.save({
        'per_traj_configs': per_traj_configs,
        'global_view': global_view,
    }, save_data_path)
    print(f"Saved trajectory data to {save_data_path}")

    # ---- Plot ----
    plot_loss_landscapes_per_trajectory(
        best_opt, per_traj_configs,
        global_view=global_view,
        grid_size=50, save_path=save_path,
    )


if __name__ == '__main__':
    main()
