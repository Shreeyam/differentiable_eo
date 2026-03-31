"""
Experiment 0: Hyperparameter Tuning for Relaxation Tightness

The relaxed loss function has hyperparameters that control how tightly the
smooth surrogates approximate the true discrete metrics. Key insight: coverage
and revisit can use *different* softness values. Coverage tolerates loose
softness (good gradients), but the revisit leaky integrator is sensitive —
partial coverage from sub-threshold passes "partially resets" the gap timer.

This experiment:
  1. Runs 4 optimizations from different RAAN initializations (same as exp3)
  2. Collects the final solutions and their hard metrics
  3. Grid-searches over relaxation hyperparameters (including separate
     revisit_softness)
  4. Finds settings where the relaxed loss correctly ranks solutions
     (good hard metrics -> low relaxed loss)
"""

import sys
import os
import numpy as np
import torch
import dsgp4

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from differentiable_eo import (
    Config, ConstellationOptimizer,
    IDX_INCLO, IDX_NODEO,
)
from differentiable_eo.constraints import FixedConstraint, default_parameter_specs
from differentiable_eo.objective import compute_loss
from differentiable_eo.tle_utils import update_tle_from_elements


def ma_and_raan_specs():
    """Parameter specs that free mean anomaly and RAAN (inc + alt fixed)."""
    specs = default_parameter_specs()
    specs[IDX_INCLO] = FixedConstraint()
    return specs


def evaluate_at_z(optimizer, z, softness, revisit_softness, revisit_tau, revisit_weight):
    """Evaluate relaxed loss at a given z-vector with specific hyperparams."""
    # Set z in reparam elements
    offset = 0
    for re in optimizer.reparam_elements:
        if re.n_free > 0:
            re._z = torch.nn.Parameter(z[offset:offset + re.n_free].clone())
            offset += re.n_free

    # Map to elements -> TLEs -> initialize
    for i, tle in enumerate(optimizer.tles):
        elements = optimizer.reparam_elements[i].to_elements()
        update_tle_from_elements(tle, elements)
    for tle in optimizer.tles:
        dsgp4.initialize_tle(tle, with_grad=False)

    # Compute loss with custom hyperparams
    cfg = optimizer.config
    with torch.no_grad():
        l, c, r = compute_loss(
            optimizer.tles, optimizer.tsinces, optimizer.gmst_tensor,
            optimizer.ground_ecef, cfg.min_elevation_deg, softness,
            revisit_tau, revisit_weight,
            ground_weights=optimizer.ground_weights,
            revisit_reduce=cfg.revisit_reduce,
            revisit_spatial_tau=cfg.revisit_spatial_tau,
            ground_unit=optimizer.ground_unit,
            revisit_softness=revisit_softness,
        )
    return l.item(), c.item() * 100, r.item()


def main():
    # ---- Configuration (matches exp2/exp3) ----
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

    # Common random MAs (same seed for all configs)
    rng = np.random.RandomState(42)
    n_total = N_PLANES * N_SATS_PER_PLANE
    COMMON_MAS = rng.uniform(0, 360, size=n_total).tolist()

    # ---- Initial RAAN configs ----
    CONFIGS = [
        ("Near-uniform", [0, 60, 120, 180, 240, 300]),
        ("Exp2 default", [0, 30, 120, 200, 210, 300]),
        ("Clustered",    [0, 10, 20, 30, 40, 50]),
        ("Two-cluster",  [0, 5, 180, 185, 270, 275]),
    ]

    # ---- Phase 1: Run optimizations to get final solutions ----
    print("=" * 65)
    print("  Phase 1: Running optimizations to collect final solutions")
    print("=" * 65)

    run_results = []  # (label, result, optimizer)

    for label, raans in CONFIGS:
        print(f"\n--- {label}: RAANs = {raans} ---")

        config = Config(
            n_planes=N_PLANES,
            n_sats_per_plane=N_SATS_PER_PLANE,
            target_alt_km=ALT_KM,
            initial_inc_deg=60.0,
            initial_raan_offsets_deg=raans,
            initial_ma_offsets_deg=COMMON_MAS,
            prop_duration_hours=PROP_HOURS,
            n_time_steps=N_TIME_STEPS,
            n_lat=N_LAT, n_lon=N_LON,
            lat_bounds_deg=LAT_BOUNDS,
            min_elevation_deg=MIN_EL,
            n_iterations=N_ITERATIONS,
            lr=1e-2,
            revisit_weight=0.1,
            revisit_reduce=REVISIT_REDUCE,
            randomize_gmst=True,
            parameter_specs=ma_and_raan_specs(),
            per_plane_params=[IDX_NODEO],
        )

        opt = ConstellationOptimizer(config)
        result = opt.run()
        run_results.append((label, result, opt))

    # ---- Collect final z-vectors and hard metrics ----
    final_zs = []
    for label, result, opt in run_results:
        z = torch.cat([re.optimizer_param.detach().clone()
                       for re in opt.reparam_elements])
        final_zs.append(z)

    print("\n" + "=" * 65)
    print("  Final hard metrics from each config")
    print("=" * 65)
    print(f"{'Config':<16} {'Hard Cov%':>10} {'Hard Rev (min)':>15} {'Relaxed Loss':>14}")
    print("-" * 58)
    for (label, result, _), z in zip(run_results, final_zs):
        print(f"{label:<16} {result.hard_cov_history[-1]:>9.2f}% "
              f"{result.hard_revisit_history[-1]:>14.1f} "
              f"{result.loss_history[-1]:>14.4f}")

    # ---- Phase 2: Grid search over relaxation hyperparams ----
    print("\n" + "=" * 65)
    print("  Phase 2: Hyperparameter grid search (with separate revisit softness)")
    print("=" * 65)

    # Coverage softness: can be loose for smooth gradients
    softness_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    # Revisit softness: needs to be tighter to prevent gaming
    rev_softness_values = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    lse_temp_values = [3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
    rev_weight_values = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

    total_combos = (len(softness_values) * len(rev_softness_values)
                    * len(lse_temp_values) * len(rev_weight_values))
    print(f"Searching {total_combos} combos "
          f"({len(softness_values)} cov_soft x {len(rev_softness_values)} rev_soft "
          f"x {len(lse_temp_values)} lse_temp x {len(rev_weight_values)} rev_wt)...")

    # Use first optimizer for evaluation (all have same structure)
    eval_opt = run_results[0][2]
    z_orig = [re.optimizer_param.detach().clone() for re in eval_opt.reparam_elements]

    valid_combos = []
    n_done = 0

    for softness in softness_values:
        for rev_soft in rev_softness_values:
            for lse_temp in lse_temp_values:
                for rev_weight in rev_weight_values:
                    losses = []
                    coverages = []
                    revisits = []
                    for z in final_zs:
                        l, c, r = evaluate_at_z(
                            eval_opt, z, softness, rev_soft, lse_temp, rev_weight)
                        losses.append(l)
                        coverages.append(c)
                        revisits.append(r)

                    # Check: good configs (0,1) < bad configs (2,3)
                    good_max = max(losses[0], losses[1])
                    bad_min = min(losses[2], losses[3])

                    if good_max < bad_min:
                        margin = bad_min - good_max
                        valid_combos.append({
                            'softness': softness,
                            'rev_soft': rev_soft,
                            'lse_temp': lse_temp,
                            'rev_weight': rev_weight,
                            'losses': losses,
                            'coverages': coverages,
                            'revisits': revisits,
                            'margin': margin,
                        })

                    n_done += 1
                    if n_done % max(1, total_combos // 10) == 0:
                        print(f"  {n_done}/{total_combos} evaluated, "
                              f"{len(valid_combos)} valid so far")

    # Restore original z
    for re, z_o in zip(eval_opt.reparam_elements, z_orig):
        if re.n_free > 0:
            re._z = torch.nn.Parameter(z_o)
    for i, tle in enumerate(eval_opt.tles):
        elements = eval_opt.reparam_elements[i].to_elements()
        update_tle_from_elements(tle, elements)

    # ---- Results ----
    print(f"\n{'=' * 65}")
    print(f"  Results: {len(valid_combos)}/{total_combos} combos correctly rank solutions")
    print(f"{'=' * 65}")

    if not valid_combos:
        print("\nNo valid combos found!")
        return

    labels = [r[0] for r in run_results]

    # Save all valid combos to file
    results_path = os.path.join(os.path.dirname(__file__), 'exp0_valid_combos.txt')
    with open(results_path, 'w') as f:
        f.write(f"# Hyperparameter tuning: {len(valid_combos)}/{total_combos} valid combos\n")
        f.write(f"# Valid = loss(Near-uniform) < loss(Clustered) AND loss(Exp1) < loss(Two-cluster)\n")
        f.write(f"#\n")
        header = (f"{'cov_soft':>8} {'rev_soft':>8} {'lse_t':>7} {'rev_wt':>7} {'margin':>8}  "
                  + "  ".join(f"{l[:8]:>10}" for l in labels))
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for combo in sorted(valid_combos, key=lambda x: (-x['softness'], -x['rev_soft'], -x['margin'])):
            loss_str = "  ".join(f"{l:>10.4f}" for l in combo['losses'])
            f.write(f"{combo['softness']:>8.1f} {combo['rev_soft']:>8.2f} "
                    f"{combo['lse_temp']:>7.1f} {combo['rev_weight']:>7.2f} "
                    f"{combo['margin']:>8.4f}  {loss_str}\n")
    print(f"\nAll valid combos saved to {results_path}")

    # Print top 30 by margin
    valid_combos.sort(key=lambda x: -x['margin'])
    print(f"\n{'cov_soft':>8} {'rev_soft':>8} {'lse_t':>7} {'rev_wt':>7} {'margin':>8}  "
          + "  ".join(f"{l[:8]:>10}" for l in labels))
    print("-" * (42 + 12 * len(labels)))

    for combo in valid_combos[:30]:
        loss_str = "  ".join(f"{l:>10.4f}" for l in combo['losses'])
        print(f"{combo['softness']:>8.1f} {combo['rev_soft']:>8.2f} "
              f"{combo['lse_temp']:>7.1f} {combo['rev_weight']:>7.2f} "
              f"{combo['margin']:>8.4f}  {loss_str}")

    # Recommended combo: maximum coverage softness (smoothest gradients),
    # then maximum revisit softness, then largest margin as tiebreaker
    best = max(valid_combos, key=lambda x: (x['softness'], x['rev_soft'], x['margin']))
    print(f"\nRecommended combo (max softness with correct ranking, margin={best['margin']:.4f}):")
    print(f"  softness_deg          = {best['softness']}")
    print(f"  revisit_softness_deg  = {best['rev_soft']}")
    print(f"  revisit_logsumexp_temp = {best['lse_temp']}")
    print(f"  revisit_weight        = {best['rev_weight']}")
    print(f"\n  Relaxed losses:  {dict(zip(labels, [f'{l:.4f}' for l in best['losses']]))}")
    print(f"  Relaxed cov %:   {dict(zip(labels, [f'{c:.2f}' for c in best['coverages']]))}")
    print(f"  Relaxed revisit: {dict(zip(labels, [f'{r:.1f}' for r in best['revisits']]))}")

    # Show which coverage softness values have valid combos
    print("\n  Valid combos by coverage softness:")
    for s in softness_values:
        count = sum(1 for v in valid_combos if v['softness'] == s)
        if count > 0:
            best_rev_soft = max(v['rev_soft'] for v in valid_combos if v['softness'] == s)
            print(f"    softness_deg={s}: {count} valid combos "
                  f"(max rev_soft={best_rev_soft})")


if __name__ == '__main__':
    main()
