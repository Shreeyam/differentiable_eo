"""
Experiment 2 Baselines: GA, SA, and nevergrad NGOpt for Walker Recovery

Implements tuned Genetic Algorithm and Simulated Annealing baselines
(Paek et al., 2019) and nevergrad's NGOpt auto-selector, adapted to the
exp2 Walker recovery problem.
Same setup: 6 planes x 4 sats/plane, 550 km, 60 deg inc, optimize RAAN + MA.

Decision variables: 6 RAANs (per-plane) + 24 MAs (per-satellite) = 30 continuous.
Fitness = -(coverage_frac) + revisit_weight * revisit_minutes  (minimize).

Tuning:
  SA: probe-calibrated T0 (target 80% init acceptance), geometric cooling to
      T_final (1% final acceptance), adaptive step size targeting 0.44
      acceptance rate (Vanderbilt-Louie).
  GA: tournament selection (size 3), uniform crossover (cyclic-angle safe),
      Gaussian mutation with decaying sigma (30->5 deg), elitism.
  DE: nevergrad's TwoPointsDE (differential evolution), which handles
      multi-modal landscapes better than the CMA that NGOpt's meta-selector
      picked. Warm-started with the bad init (same as SA/GA).
"""

import sys
import os
import math
import time
import numpy as np
import torch
import dsgp4

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex', 'shreeyam.mplstyle'])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from differentiable_eo import (
    make_constellation, make_ground_grid_with_weights, make_gmst_array,
    compute_hard_metrics,
)

# Force line-buffered stdout so progress shows even when piped
sys.stdout.reconfigure(line_buffering=True)

# ---- Problem configuration (matches exp2) ----
N_PLANES = 6
N_SATS_PER_PLANE = 4
N_SATS = N_PLANES * N_SATS_PER_PLANE
ALT_KM = 550.0
INC_DEG = 60.0
N_TIME_STEPS = 240
N_LAT, N_LON = 36, 72
LAT_BOUNDS = (-70.0, 70.0)
MIN_EL = 10.0
PROP_HOURS = 24.0
REVISIT_REDUCE = 'mean'
REVISIT_WEIGHT = 0.1
WALKER_F = 1

# Bad initial configuration (same seed as exp2)
BAD_RAANS = [0.0, 30.0, 120.0, 200.0, 210.0, 300.0]
_rng_init = np.random.RandomState(42)
BAD_MAS = _rng_init.uniform(0, 360, size=N_SATS).tolist()

WALKER_RAANS = [i * 360.0 / N_PLANES for i in range(N_PLANES)]


# ---- Evaluation ----

def evaluate(raans_deg, mas_deg, tsinces, gmst_array, ground_ecef, ground_weights):
    """Evaluate a constellation and return (coverage_pct, revisit_min, fitness)."""
    tles = make_constellation(
        n_planes=N_PLANES, n_sats_per_plane=N_SATS_PER_PLANE,
        inc_deg=INC_DEG, raan_offsets_deg=list(raans_deg),
        alt_km=ALT_KM, phase_offset_f=0,
        ma_offsets_deg=list(mas_deg),
    )
    for t in tles:
        dsgp4.initialize_tle(t, with_grad=False)
    with torch.no_grad():
        h_cov, h_rev = compute_hard_metrics(
            tles, tsinces, gmst_array, ground_ecef,
            min_el=MIN_EL, ground_weights=ground_weights,
            revisit_reduce=REVISIT_REDUCE,
        )
    cov_pct = h_cov * 100
    fitness = -(cov_pct / 100) + REVISIT_WEIGHT * h_rev
    return cov_pct, h_rev, fitness


def decode_chromosome(chromo):
    """Split 30-vector into (raans[6], mas[24])."""
    return chromo[:N_PLANES], chromo[N_PLANES:]


# ---- Simulated Annealing (tuned) ----

def simulated_annealing(tsinces, gmst_array, ground_ecef, ground_weights,
                        n_iterations=2000, seed=0, snapshot_every=20):
    """
    SA with probe-calibrated temperature schedule and adaptive step size.

    Calibration:
      1. Probe 50 random perturbations from init, measure median positive dE.
      2. T0 = median_dE / -ln(0.8)   -> ~80% acceptance of typical worse move
      3. T_final = median_dE / -ln(0.01) -> ~1% acceptance at end
      4. Geometric cooling: alpha = (T_final / T0)^(1/n_iterations)

    Adaptive step size (Vanderbilt-Louie): track acceptance rate over last 100
    iters, multiply step by 1.05 if > 0.6, by 0.95 if < 0.3; target ~0.44.

    Returns (best_raans, best_mas, best_cov, best_rev, best_fitness, history, snapshots).
    """
    rng = np.random.RandomState(seed)

    raans = np.array(BAD_RAANS, dtype=np.float64)
    mas = np.array(BAD_MAS, dtype=np.float64)
    cov, rev, fitness = evaluate(raans, mas, tsinces, gmst_array, ground_ecef, ground_weights)

    best_raans, best_mas = raans.copy(), mas.copy()
    best_fitness, best_cov, best_rev = fitness, cov, rev

    # ---- Probe phase: calibrate T0, T_final ----
    n_probe = 50
    probe_step = 30.0
    pos_dEs = []
    for _ in range(n_probe):
        probe_r = (raans + rng.normal(0, probe_step, N_PLANES)) % 360.0
        probe_m = (mas + rng.normal(0, probe_step, N_SATS)) % 360.0
        _, _, probe_fit = evaluate(probe_r, probe_m, tsinces, gmst_array,
                                   ground_ecef, ground_weights)
        dE = probe_fit - fitness
        if dE > 0:
            pos_dEs.append(dE)

    median_dE = float(np.median(pos_dEs)) if pos_dEs else 1.0
    T0 = median_dE / -math.log(0.8)       # ~80% initial acceptance of typical worse move
    T_final = median_dE / -math.log(0.01) # ~1% final acceptance
    T = T0
    alpha = (T_final / T0) ** (1.0 / n_iterations)

    # Adaptive step (in degrees)
    step = 20.0
    step_min, step_max = 1.0, 60.0
    target_accept = 0.44
    window = 100
    accept_window = []

    history = {'fitness': [fitness], 'best_fitness': [best_fitness],
               'cov': [cov], 'rev': [rev],
               'best_cov': [best_cov], 'best_rev': [best_rev],
               'step': [step], 'temperature': [T]}
    snapshots = [(0, raans.copy(), mas.copy(), fitness, cov, rev)]

    n_accepted = 0
    print(f"  SA [{seed}] probe: median_dE={median_dE:.3f}, "
          f"T0={T0:.3f}, T_final={T_final:.4f}, alpha={alpha:.5f}")

    for i in range(n_iterations):
        new_raans = raans.copy()
        new_mas = mas.copy()

        # Perturb random subset, scale = current step
        n_raan_perturb = rng.randint(1, min(4, N_PLANES + 1))
        n_ma_perturb = rng.randint(1, min(7, N_SATS + 1))
        raan_idx = rng.choice(N_PLANES, size=n_raan_perturb, replace=False)
        ma_idx = rng.choice(N_SATS, size=n_ma_perturb, replace=False)

        new_raans[raan_idx] += rng.normal(0, step, n_raan_perturb)
        new_mas[ma_idx] += rng.normal(0, step, n_ma_perturb)
        new_raans %= 360.0
        new_mas %= 360.0

        new_cov, new_rev, new_fitness = evaluate(
            new_raans, new_mas, tsinces, gmst_array, ground_ecef, ground_weights)

        dE = new_fitness - fitness
        accepted = (dE < 0) or (rng.random() < math.exp(-dE / max(T, 1e-12)))
        if accepted:
            raans, mas, fitness = new_raans, new_mas, new_fitness
            cov, rev = new_cov, new_rev
            n_accepted += 1
            if fitness < best_fitness:
                best_raans, best_mas = raans.copy(), mas.copy()
                best_fitness, best_cov, best_rev = fitness, cov, rev

        # Track recent acceptance, adapt step every `window` iters
        accept_window.append(1 if accepted else 0)
        if len(accept_window) > window:
            accept_window.pop(0)
        if (i + 1) % window == 0 and len(accept_window) == window:
            recent_rate = sum(accept_window) / window
            if recent_rate > 0.6:
                step = min(step * 1.15, step_max)
            elif recent_rate < 0.3:
                step = max(step * 0.85, step_min)

        T *= alpha

        history['fitness'].append(fitness)
        history['best_fitness'].append(best_fitness)
        history['cov'].append(cov)
        history['rev'].append(rev)
        history['best_cov'].append(best_cov)
        history['best_rev'].append(best_rev)
        history['step'].append(step)
        history['temperature'].append(T)

        if (i + 1) % snapshot_every == 0:
            snapshots.append((i + 1, raans.copy(), mas.copy(), fitness, cov, rev))

        if (i + 1) % 200 == 0:
            recent = sum(accept_window) / max(1, len(accept_window))
            print(f"  SA [{seed}] iter {i+1}/{n_iterations}: "
                  f"best_fit={best_fitness:.4f}, T={T:.4f}, step={step:.1f}d, "
                  f"cov={best_cov:.2f}%, rev={best_rev:.1f}m, "
                  f"recent_accept={recent:.2f}")

    return best_raans, best_mas, best_cov, best_rev, best_fitness, history, snapshots


# ---- Genetic Algorithm (tuned) ----

def genetic_algorithm(tsinces, gmst_array, ground_ecef, ground_weights,
                      pop_size=50, n_generations=40, seed=0):
    """
    Real-valued GA with tournament selection (size 3), uniform crossover,
    Gaussian mutation with decaying sigma. Tournament avoids roulette-wheel
    fragility; uniform crossover is safe for cyclic angle variables.

    Budget: pop_size * n_generations + pop_size (initial eval).
    Returns (best_raans, best_mas, best_cov, best_rev, best_fitness, history, snapshots).
    """
    rng = np.random.RandomState(seed)
    n_genes = N_PLANES + N_SATS  # 30

    # Initialize: random population, inject bad init config as individual 0
    population = rng.uniform(0, 360, size=(pop_size, n_genes))
    population[0, :N_PLANES] = BAD_RAANS
    population[0, N_PLANES:] = BAD_MAS

    # Hyperparams (tuned defaults)
    crossover_rate = 0.9
    mutation_rate = 1.0 / n_genes  # ~0.033, Schaffer's rule
    sigma_start = 30.0
    sigma_end = 5.0
    tournament_size = 3
    elitism = 2

    def eval_pop(pop):
        fit = np.zeros(len(pop))
        cov = np.zeros(len(pop))
        rev = np.zeros(len(pop))
        for j, ch in enumerate(pop):
            r, m = decode_chromosome(ch)
            cov[j], rev[j], fit[j] = evaluate(
                r, m, tsinces, gmst_array, ground_ecef, ground_weights)
        return fit, cov, rev

    def tournament_select(fits):
        """Pick winner of a random tournament (lowest fitness wins)."""
        contenders = rng.choice(pop_size, size=tournament_size, replace=False)
        return contenders[np.argmin(fits[contenders])]

    fitnesses, covs, revs = eval_pop(population)

    best_idx = np.argmin(fitnesses)
    best_chromo = population[best_idx].copy()
    best_fitness, best_cov, best_rev = fitnesses[best_idx], covs[best_idx], revs[best_idx]

    history = {'best_fitness': [best_fitness], 'mean_fitness': [np.mean(fitnesses)],
               'best_cov': [best_cov], 'best_rev': [best_rev]}
    snapshots = [(0, best_chromo[:N_PLANES].copy(), best_chromo[N_PLANES:].copy(),
                  best_fitness, best_cov, best_rev)]

    for gen in range(n_generations):
        # Anneal mutation sigma linearly
        frac = gen / max(1, n_generations - 1)
        sigma = sigma_start * (1 - frac) + sigma_end * frac

        new_pop = np.zeros_like(population)

        # Elitism
        elite_idx = np.argsort(fitnesses)[:elitism]
        for e, ei in enumerate(elite_idx):
            new_pop[e] = population[ei]

        # Generate children via tournament + crossover + mutation
        for k in range(elitism, pop_size, 2):
            p1 = population[tournament_select(fitnesses)]
            p2 = population[tournament_select(fitnesses)]

            if rng.random() < crossover_rate:
                mask = rng.random(n_genes) < 0.5
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()

            for g in range(n_genes):
                if rng.random() < mutation_rate:
                    c1[g] += rng.normal(0, sigma)
                if rng.random() < mutation_rate:
                    c2[g] += rng.normal(0, sigma)

            new_pop[k] = c1 % 360.0
            if k + 1 < pop_size:
                new_pop[k + 1] = c2 % 360.0

        population = new_pop
        fitnesses, covs, revs = eval_pop(population)

        gen_best = np.argmin(fitnesses)
        if fitnesses[gen_best] < best_fitness:
            best_chromo = population[gen_best].copy()
            best_fitness = fitnesses[gen_best]
            best_cov, best_rev = covs[gen_best], revs[gen_best]

        history['best_fitness'].append(best_fitness)
        history['mean_fitness'].append(np.mean(fitnesses))
        history['best_cov'].append(best_cov)
        history['best_rev'].append(best_rev)
        snapshots.append((gen + 1,
                          best_chromo[:N_PLANES].copy(),
                          best_chromo[N_PLANES:].copy(),
                          best_fitness, best_cov, best_rev))

        if (gen + 1) % 10 == 0:
            print(f"  GA [{seed}] gen {gen+1}/{n_generations}: "
                  f"best_fit={best_fitness:.4f}, mean={np.mean(fitnesses):.4f}, "
                  f"sigma={sigma:.1f}d, "
                  f"cov={best_cov:.2f}%, rev={best_rev:.1f}m")

    best_raans, best_mas = decode_chromosome(best_chromo)
    return best_raans, best_mas, best_cov, best_rev, best_fitness, history, snapshots


# ---- Nevergrad TwoPointsDE ----

def nevergrad_de(tsinces, gmst_array, ground_ecef, ground_weights,
                 budget=2050, seed=0):
    """
    Nevergrad's TwoPointsDE (differential evolution). Warm-starts with the
    bad-init config (same as SA/GA). Records best-so-far cov/rev after every
    evaluation for a smooth convergence curve.
    """
    import nevergrad as ng

    n_genes = N_PLANES + N_SATS  # 30

    # Box-bounded parametrization in [0, 360); nevergrad handles box internally.
    param = ng.p.Array(shape=(n_genes,)).set_bounds(0.0, 360.0)
    init_vec = np.concatenate([np.array(BAD_RAANS), np.array(BAD_MAS)])
    param.value = init_vec

    optimizer = ng.optimizers.TwoPointsDE(parametrization=param, budget=budget)
    optimizer.parametrization.random_state.seed(seed)

    best_fit = float('inf')
    best_cov = best_rev = 0.0
    best_chromo = init_vec.copy()

    history = {'best_fitness': [], 'best_cov': [], 'best_rev': []}
    snapshots = []
    snapshot_every = max(1, budget // 100)

    for i in range(budget):
        cand = optimizer.ask()
        x = np.asarray(cand.value)
        raans = x[:N_PLANES]
        mas = x[N_PLANES:]
        cov, rev, fit = evaluate(raans, mas, tsinces, gmst_array,
                                 ground_ecef, ground_weights)
        optimizer.tell(cand, float(fit))

        if fit < best_fit:
            best_fit, best_cov, best_rev = fit, cov, rev
            best_chromo = x.copy()

        history['best_fitness'].append(best_fit)
        history['best_cov'].append(best_cov)
        history['best_rev'].append(best_rev)

        if (i + 1) % snapshot_every == 0 or i == budget - 1:
            snapshots.append((i + 1,
                              best_chromo[:N_PLANES].copy(),
                              best_chromo[N_PLANES:].copy(),
                              best_fit, best_cov, best_rev))

        if (i + 1) % 200 == 0:
            print(f"  DE [{seed}] eval {i+1}/{budget}: "
                  f"best_fit={best_fit:.4f}, cov={best_cov:.2f}%, rev={best_rev:.1f}m")

    best_raans, best_mas = best_chromo[:N_PLANES], best_chromo[N_PLANES:]
    return best_raans, best_mas, best_cov, best_rev, best_fit, history, snapshots


# ---- Main ----

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', default='all',
                        help='Comma-separated subset of {sa,ga,ng,all}. '
                             'Methods not listed are loaded from existing '
                             'exp2_baselines_data.pt if present.')
    args = parser.parse_args()
    requested = set(args.methods.lower().split(','))
    if 'all' in requested:
        requested = {'sa', 'ga', 'ng'}
    print(f"Running methods: {sorted(requested)}")

    N_SEEDS = 5
    SA_ITERS = 4000
    GA_POP = 50
    GA_GENS = 80  # 50 * 80 = 4000 evals, plus 50 initial

    # ---- Setup ----
    ground_ecef, ground_weights = make_ground_grid_with_weights(N_LAT, N_LON, LAT_BOUNDS)
    prop_min = PROP_HOURS * 60
    tsinces = torch.linspace(0, prop_min, N_TIME_STEPS)
    gmst_array = make_gmst_array(tsinces)

    walker_mas = [(360.0 * s / N_SATS_PER_PLANE + WALKER_F * (360.0 / N_SATS) * p) % 360
                  for p in range(N_PLANES) for s in range(N_SATS_PER_PLANE)]
    walker_cov, walker_rev, walker_fit = evaluate(
        WALKER_RAANS, walker_mas,
        tsinces, gmst_array, ground_ecef, ground_weights)
    print(f"Walker 24/6/1 reference: cov={walker_cov:.2f}%, rev={walker_rev:.1f} min, "
          f"fitness={walker_fit:.4f}")

    init_cov, init_rev, init_fit = evaluate(
        BAD_RAANS, BAD_MAS, tsinces, gmst_array, ground_ecef, ground_weights)
    print(f"Bad initial config:      cov={init_cov:.2f}%, rev={init_rev:.1f} min, "
          f"fitness={init_fit:.4f}")

    # Load existing pickle if we're only running a subset
    data_path = os.path.join(os.path.dirname(__file__), 'exp2_baselines_data.pt')
    existing = {}
    if requested != {'sa', 'ga', 'ng'} and os.path.exists(data_path):
        existing = torch.load(data_path, weights_only=False)
        print(f"Loaded existing data from {data_path}")

    NG_BUDGET = SA_ITERS + 50  # match SA total eval budget (probe+main)

    # ---- Run or load SA ----
    if 'sa' in requested:
        print(f"\n{'='*60}")
        print(f"  SIMULATED ANNEALING (tuned, {N_SEEDS} seeds x {SA_ITERS} iters)")
        print(f"{'='*60}")
        sa_results, sa_histories, sa_snapshots = [], [], []
        for s in range(N_SEEDS):
            t0 = time.time()
            raans, mas, cov, rev, fit, hist, snaps = simulated_annealing(
                tsinces, gmst_array, ground_ecef, ground_weights,
                n_iterations=SA_ITERS, seed=s)
            elapsed = time.time() - t0
            sa_results.append((raans, mas, cov, rev, fit))
            sa_histories.append(hist)
            sa_snapshots.append(snaps)
            print(f"  SA seed {s}: cov={cov:.2f}%, rev={rev:.1f} min, "
                  f"fitness={fit:.4f}, time={elapsed:.1f}s")
    else:
        sa_results = existing.get('sa_results', [])
        sa_histories = existing.get('sa_histories', [])
        sa_snapshots = existing.get('sa_snapshots', [])
        print(f"SA: loaded {len(sa_results)} seeds from existing data")

    # ---- Run or load GA ----
    if 'ga' in requested:
        print(f"\n{'='*60}")
        print(f"  GENETIC ALGORITHM (tuned, {N_SEEDS} seeds x {GA_POP}pop x {GA_GENS}gen)")
        print(f"{'='*60}")
        ga_results, ga_histories, ga_snapshots = [], [], []
        for s in range(N_SEEDS):
            t0 = time.time()
            raans, mas, cov, rev, fit, hist, snaps = genetic_algorithm(
                tsinces, gmst_array, ground_ecef, ground_weights,
                pop_size=GA_POP, n_generations=GA_GENS, seed=s)
            elapsed = time.time() - t0
            ga_results.append((raans, mas, cov, rev, fit))
            ga_histories.append(hist)
            ga_snapshots.append(snaps)
            print(f"  GA seed {s}: cov={cov:.2f}%, rev={rev:.1f} min, "
                  f"fitness={fit:.4f}, time={elapsed:.1f}s")
    else:
        ga_results = existing.get('ga_results', [])
        ga_histories = existing.get('ga_histories', [])
        ga_snapshots = existing.get('ga_snapshots', [])
        print(f"GA: loaded {len(ga_results)} seeds from existing data")

    # ---- Run or load NGOpt ----
    if 'ng' in requested:
        print(f"\n{'='*60}")
        print(f"  NEVERGRAD TwoPointsDE ({N_SEEDS} seeds x {NG_BUDGET} evals)")
        print(f"{'='*60}")
        ng_results, ng_histories, ng_snapshots = [], [], []
        for s in range(N_SEEDS):
            t0 = time.time()
            raans, mas, cov, rev, fit, hist, snaps = nevergrad_de(
                tsinces, gmst_array, ground_ecef, ground_weights,
                budget=NG_BUDGET, seed=s)
            elapsed = time.time() - t0
            ng_results.append((raans, mas, cov, rev, fit))
            ng_histories.append(hist)
            ng_snapshots.append(snaps)
            print(f"  NG seed {s}: cov={cov:.2f}%, rev={rev:.1f} min, "
                  f"fitness={fit:.4f}, time={elapsed:.1f}s")
    else:
        ng_results = existing.get('ng_results', [])
        ng_histories = existing.get('ng_histories', [])
        ng_snapshots = existing.get('ng_snapshots', [])
        print(f"NG: loaded {len(ng_results)} seeds from existing data")

    # Best-idx helpers (guard against empty lists from skipped methods)
    def best_idx(results):
        return int(np.argmin([r[4] for r in results])) if results else 0
    sa_best_idx = best_idx(sa_results)
    ga_best_idx = best_idx(ga_results)
    ng_best_idx = best_idx(ng_results)

    # ---- Save merged data ----
    torch.save({
        'sa_results': sa_results, 'sa_histories': sa_histories, 'sa_snapshots': sa_snapshots,
        'ga_results': ga_results, 'ga_histories': ga_histories, 'ga_snapshots': ga_snapshots,
        'ng_results': ng_results, 'ng_histories': ng_histories, 'ng_snapshots': ng_snapshots,
        'walker': (walker_cov, walker_rev, walker_fit, WALKER_RAANS, walker_mas),
        'initial': (init_cov, init_rev, init_fit, BAD_RAANS, BAD_MAS),
        'config': {
            'n_seeds': N_SEEDS, 'sa_iters': SA_ITERS,
            'ga_pop': GA_POP, 'ga_gens': GA_GENS,
            'ng_budget': NG_BUDGET,
            'revisit_weight': REVISIT_WEIGHT,
        },
    }, data_path)
    print(f"\nRaw data saved to {data_path}")

    sa_covs = [r[2] for r in sa_results]; sa_revs = [r[3] for r in sa_results]; sa_fits = [r[4] for r in sa_results]
    ga_covs = [r[2] for r in ga_results]; ga_revs = [r[3] for r in ga_results]; ga_fits = [r[4] for r in ga_results]
    ng_covs = [r[2] for r in ng_results]; ng_revs = [r[3] for r in ng_results]; ng_fits = [r[4] for r in ng_results]

    # ---- Comparison table ----
    print(f"\n{'='*72}")
    print(f"  BASELINE COMPARISON (exp2 Walker recovery, TUNED)")
    print(f"  fitness = -(cov/100) + {REVISIT_WEIGHT} * revisit_min")
    print(f"{'='*72}")
    print(f"{'Method':<20} {'Coverage %':>12} {'Revisit (min)':>14} {'Fitness':>10}")
    print(f"{'-'*72}")
    print(f"{'Initial':<20} {init_cov:>11.2f}% {init_rev:>14.1f} {init_fit:>10.4f}")
    print(f"{'Walker 24/6/1':<20} {walker_cov:>11.2f}% {walker_rev:>14.1f} {walker_fit:>10.4f}")
    print(f"{'-'*72}")
    print(f"{'SA (best)':<20} {sa_covs[sa_best_idx]:>11.2f}% "
          f"{sa_revs[sa_best_idx]:>14.1f} {sa_fits[sa_best_idx]:>10.4f}")
    print(f"{'SA (mean +/- std)':<20} "
          f"{np.mean(sa_covs):>7.2f}+/-{np.std(sa_covs):.2f}% "
          f"{np.mean(sa_revs):>8.1f}+/-{np.std(sa_revs):.1f} "
          f"{np.mean(sa_fits):>10.4f}")
    print(f"{'GA (best)':<20} {ga_covs[ga_best_idx]:>11.2f}% "
          f"{ga_revs[ga_best_idx]:>14.1f} {ga_fits[ga_best_idx]:>10.4f}")
    print(f"{'GA (mean +/- std)':<20} "
          f"{np.mean(ga_covs):>7.2f}+/-{np.std(ga_covs):.2f}% "
          f"{np.mean(ga_revs):>8.1f}+/-{np.std(ga_revs):.1f} "
          f"{np.mean(ga_fits):>10.4f}")
    if ng_results:
        print(f"{'NG (best)':<20} {ng_covs[ng_best_idx]:>11.2f}% "
              f"{ng_revs[ng_best_idx]:>14.1f} {ng_fits[ng_best_idx]:>10.4f}")
        print(f"{'NG (mean +/- std)':<20} "
              f"{np.mean(ng_covs):>7.2f}+/-{np.std(ng_covs):.2f}% "
              f"{np.mean(ng_revs):>8.1f}+/-{np.std(ng_revs):.1f} "
              f"{np.mean(ng_fits):>10.4f}")
    print(f"{'='*72}")
    return  # downstream SA/GA-only plots below are stale; use plots/fig_exp2_method_comparison.py

    # ---- Plots ----
    SA_COLOR = colors[2]
    GA_COLOR = colors[3]
    paper_dir = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')

    # --- 1. Convergence: fitness vs evaluations ---
    fig_conv, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    ax = axes[0]
    for s, hist in enumerate(sa_histories):
        lw = 1.5 if s == sa_best_idx else 0.6
        al = 0.9 if s == sa_best_idx else 0.3
        ax.plot(hist['best_fitness'], color=SA_COLOR, lw=lw, alpha=al,
                label='SA' if s == 0 else None)
    ax.axhline(walker_fit, color='k', ls='--', alpha=0.5, label='Walker')
    ax.set_xlabel('Evaluation')
    ax.set_ylabel('Best fitness')
    ax.set_title('Simulated Annealing')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    for s, hist in enumerate(ga_histories):
        evals = np.arange(len(hist['best_fitness'])) * GA_POP
        lw = 1.5 if s == ga_best_idx else 0.6
        al = 0.9 if s == ga_best_idx else 0.3
        ax.plot(evals, hist['best_fitness'], color=GA_COLOR, lw=lw, alpha=al,
                label='GA' if s == 0 else None)
        ax.plot(evals, hist['mean_fitness'], ':', color=GA_COLOR,
                lw=0.5, alpha=al * 0.5)
    ax.axhline(walker_fit, color='k', ls='--', alpha=0.5, label='Walker')
    ax.set_xlabel('Evaluation')
    ax.set_ylabel('Best fitness')
    ax.set_title('Genetic Algorithm')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    fig_conv.tight_layout()
    conv_path = os.path.join(os.path.dirname(__file__), 'exp2_baselines_convergence.png')
    fig_conv.savefig(conv_path, dpi=150)
    print(f"\nConvergence plot saved to {conv_path}")
    plt.close(fig_conv)

    # --- 2. RAAN vs MA scatter (best SA, best GA, Walker) ---
    fig_rm, axes_rm = plt.subplots(1, 3, figsize=(12, 4))
    colors_planes = [colors[i % len(colors)] for i in range(N_PLANES)]

    for ax_idx, (label, raans, mas) in enumerate([
        ("SA (best)", sa_results[sa_best_idx][0], sa_results[sa_best_idx][1]),
        ("GA (best)", ga_results[ga_best_idx][0], ga_results[ga_best_idx][1]),
        ("Walker 24/6/1", WALKER_RAANS, walker_mas),
    ]):
        ax = axes_rm[ax_idx]
        for s_idx in range(N_SATS):
            p = s_idx // N_SATS_PER_PLANE
            raan_val = raans[p] if len(raans) == N_PLANES else raans[s_idx]
            ma_val = mas[s_idx]
            ax.scatter(raan_val % 360, ma_val % 360, s=40,
                       color=colors_planes[p], edgecolors='k',
                       linewidths=0.4, zorder=5)
        ax.set_xlim(-5, 365)
        ax.set_ylim(-5, 365)
        ax.set_xlabel('RAAN [deg]')
        ax.set_ylabel('Mean anomaly [deg]')
        ax.set_title(label)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')

    fig_rm.tight_layout()
    rm_path = os.path.join(os.path.dirname(__file__), 'exp2_baselines_raan_ma.png')
    fig_rm.savefig(rm_path, dpi=150, bbox_inches='tight')
    print(f"RAAN-MA scatter saved to {rm_path}")
    plt.close(fig_rm)

    # --- 3. Paper figure: combined convergence (coverage + revisit) ---
    fig_paper, (ax_cov, ax_rev) = plt.subplots(1, 2, figsize=(8, 3.5))

    for s, hist in enumerate(sa_histories):
        al = 0.8 if s == sa_best_idx else 0.2
        lw = 1.5 if s == sa_best_idx else 0.5
        ax_cov.plot(hist['best_cov'], color=SA_COLOR, lw=lw, alpha=al,
                    label='SA' if s == 0 else None)
    for s, hist in enumerate(ga_histories):
        evals = np.arange(len(hist['best_cov'])) * GA_POP
        al = 0.8 if s == ga_best_idx else 0.2
        lw = 1.5 if s == ga_best_idx else 0.5
        ax_cov.plot(evals, hist['best_cov'], color=GA_COLOR, lw=lw, alpha=al,
                    label='GA' if s == 0 else None)
    ax_cov.axhline(walker_cov, color='k', ls='--', alpha=0.5,
                   label=f'Walker ({walker_cov:.1f}%)')
    ax_cov.set_xlabel('Evaluations')
    ax_cov.set_ylabel('Coverage [%]')
    ax_cov.legend(fontsize=7)
    ax_cov.grid(True, alpha=0.2)

    for s, hist in enumerate(sa_histories):
        al = 0.8 if s == sa_best_idx else 0.2
        lw = 1.5 if s == sa_best_idx else 0.5
        ax_rev.plot(hist['best_rev'], color=SA_COLOR, lw=lw, alpha=al,
                    label='SA' if s == 0 else None)
    for s, hist in enumerate(ga_histories):
        evals = np.arange(len(hist['best_rev'])) * GA_POP
        al = 0.8 if s == ga_best_idx else 0.2
        lw = 1.5 if s == ga_best_idx else 0.5
        ax_rev.plot(evals, hist['best_rev'], color=GA_COLOR, lw=lw, alpha=al,
                    label='GA' if s == 0 else None)
    ax_rev.axhline(walker_rev, color='k', ls='--', alpha=0.5,
                   label=f'Walker ({walker_rev:.0f} min)')
    ax_rev.set_xlabel('Evaluations')
    ax_rev.set_ylabel('Mean max revisit [min]')
    ax_rev.legend(fontsize=7)
    ax_rev.grid(True, alpha=0.2)

    fig_paper.tight_layout()
    fig_paper.savefig(os.path.join(paper_dir, 'exp2_baselines.pdf'), bbox_inches='tight')
    print(f"Paper figure saved to {paper_dir}/exp2_baselines.pdf")
    plt.close(fig_paper)

    # ---- Paper values ----
    print(f"\n{'='*60}")
    print(f"  VALUES FOR PAPER TABLE")
    print(f"{'='*60}")
    print(f"  SA coverage (best):   {sa_covs[sa_best_idx]:.2f}%")
    print(f"  SA coverage (mean):   {np.mean(sa_covs):.2f} +/- {np.std(sa_covs):.2f}%")
    print(f"  SA revisit  (best):   {sa_revs[sa_best_idx]:.1f} min")
    print(f"  SA revisit  (mean):   {np.mean(sa_revs):.1f} +/- {np.std(sa_revs):.1f} min")
    print(f"  GA coverage (best):   {ga_covs[ga_best_idx]:.2f}%")
    print(f"  GA coverage (mean):   {np.mean(ga_covs):.2f} +/- {np.std(ga_covs):.2f}%")
    print(f"  GA revisit  (best):   {ga_revs[ga_best_idx]:.1f} min")
    print(f"  GA revisit  (mean):   {np.mean(ga_revs):.1f} +/- {np.std(ga_revs):.1f} min")
    print(f"  Walker coverage:      {walker_cov:.2f}%")
    print(f"  Walker revisit:       {walker_rev:.1f} min")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
