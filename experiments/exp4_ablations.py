"""
Experiment 4: Design-choice ablations on the Walker recovery testbed.

Runs five ablation studies against the Exp 2 setup (24 sats, 6x4, 60 deg inc,
RAAN + MA free, irregular RAAN init, seeded random MAs):

  A. Loss composition: coverage-only, revisit-only, combined.
  B. Split vs shared sigmoid temperature (tau_cov vs tau_rev).
  C. Revisit weight lambda sweep.
  D. LogSumExp temperature beta sweep.
  E. Initialization sensitivity across seeds.

For each run we record the final hard coverage %, hard mean-max revisit (min),
and a coarse history. Results are written to paper/data/ablations.json and a
LaTeX table fragment to paper/tables/ablations.tex.

All runs use N_ITERATIONS=1000 to match Exp 2's budget, so ablation
numbers are directly comparable to the Walker recovery result.
"""

import sys
import os
import json
import math
import time
import subprocess
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


# ---- Shared setup ----------------------------------------------------------

DEFAULT_RAANS = [0.0, 30.0, 120.0, 200.0, 210.0, 300.0]
N_ITERATIONS = 1000
WALKER_REF = {'hard_cov_pct': 40.33, 'hard_revisit_min': 48.0}


def ma_and_raan_specs():
    specs = default_parameter_specs()
    specs[IDX_INCLO] = FixedConstraint()
    return specs


def make_base_config(overrides=None, seed=42):
    rng = np.random.RandomState(seed)
    mas = rng.uniform(0, 360, size=24).tolist()
    cfg_kwargs = dict(
        n_planes=6,
        n_sats_per_plane=4,
        target_alt_km=550.0,
        initial_inc_deg=60.0,
        initial_raan_offsets_deg=list(DEFAULT_RAANS),
        initial_ma_offsets_deg=mas,
        prop_duration_hours=24.0,
        n_time_steps=240,
        n_lat=36, n_lon=72,
        lat_bounds_deg=(-70.0, 70.0),
        min_elevation_deg=10.0,
        softness_deg=2.0,              # tau_cov (matches Exp 2 baseline)
        revisit_softness_deg=2.0,      # tau_rev (shared with tau_cov)
        revisit_logsumexp_temp=10.0,   # beta
        revisit_weight=0.1,            # lambda (matches Exp 2 calibration)
        revisit_reduce='mean',
        n_iterations=N_ITERATIONS,
        lr=1e-2,
        randomize_gmst=True,
        parameter_specs=ma_and_raan_specs(),
        per_plane_params=[IDX_NODEO],
    )
    if overrides:
        cfg_kwargs.update(overrides)
    return Config(**cfg_kwargs)


# ---- Optimizer subclass that supports loss-composition ablation ------------

class AblationOptimizer(ConstellationOptimizer):
    """ConstellationOptimizer with a selectable loss composition."""

    def __init__(self, config, loss_mode='combined'):
        assert loss_mode in ('combined', 'coverage_only', 'revisit_only')
        self.loss_mode = loss_mode
        super().__init__(config)

    def step(self):
        cfg = self.config
        self.optimizer.zero_grad()

        if cfg.randomize_gmst:
            gmst_offset = torch.rand(1).item() * 2 * math.pi
            gmst_tensor = self.gmst_tensor + gmst_offset
        else:
            gmst_tensor = self.gmst_tensor

        tle_elements_list = []
        for i, tle in enumerate(self.tles):
            elements = self.reparam_elements[i].to_elements()
            update_tle_from_elements(tle, elements)
            tle_elements_list.append(dsgp4.initialize_tle(tle, with_grad=True))

        _, cov_frac, mean_revisit = compute_loss(
            self.tles, self.tsinces, gmst_tensor, self.ground_ecef,
            min_el=cfg.min_elevation_deg, softness=cfg.softness_deg,
            revisit_tau=cfg.revisit_logsumexp_temp, revisit_weight=cfg.revisit_weight,
            ground_weights=self.ground_weights,
            revisit_reduce=cfg.revisit_reduce,
            revisit_spatial_tau=cfg.revisit_spatial_tau,
            ground_unit=self.ground_unit,
            revisit_softness=cfg.revisit_softness_deg,
        )

        if self.loss_mode == 'coverage_only':
            loss = -cov_frac
        elif self.loss_mode == 'revisit_only':
            loss = cfg.revisit_weight * mean_revisit
        else:
            loss = -cov_frac + cfg.revisit_weight * mean_revisit

        loss.backward()
        for i in range(len(self.tles)):
            eph_grad = tle_elements_list[i].grad
            if eph_grad is not None:
                self.reparam_elements[i].compute_z_grad(eph_grad)
        self._accumulate_per_plane_grads()
        self.optimizer.step()
        self._sync_per_plane_params()

        return {
            'loss': loss.item(),
            'coverage_pct': cov_frac.item() * 100,
            'revisit_min': mean_revisit.item(),
        }


# ---- Single-run harness ----------------------------------------------------

def run_ablation(name, overrides=None, loss_mode='combined', seed=42,
                 hard_eval_every=50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = make_base_config(overrides, seed=seed)
    opt = AblationOptimizer(config, loss_mode=loss_mode)

    t0 = time.time()
    hard_cov_hist, hard_rev_hist, eval_iters = [], [], []

    for it in range(config.n_iterations):
        opt.step()
        if it == 0 or (it + 1) % hard_eval_every == 0 or it == config.n_iterations - 1:
            hard = opt.evaluate_hard()
            hard_cov_hist.append(hard['hard_coverage_pct'])
            hard_rev_hist.append(hard['hard_revisit_min'])
            eval_iters.append(it)

    final = opt.evaluate_hard()
    elapsed = time.time() - t0
    print(f"  [{name}] final: cov={final['hard_coverage_pct']:.2f}%  "
          f"rev={final['hard_revisit_min']:.1f}min  ({elapsed:.1f}s)")

    # Try to free as much as possible between runs
    del opt
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        'name': name,
        'loss_mode': loss_mode,
        'seed': seed,
        'overrides': overrides or {},
        'hard_cov_pct': final['hard_coverage_pct'],
        'hard_revisit_min': final['hard_revisit_min'],
        'hard_cov_history': hard_cov_hist,
        'hard_revisit_history': hard_rev_hist,
        'eval_iters': eval_iters,
        'time_sec': elapsed,
    }


# ---- LaTeX table emitter ---------------------------------------------------

def _row(label, r):
    return (f"{label} & {r['hard_cov_pct']:.2f} "
            f"& {r['hard_revisit_min']:.1f} \\\\")


def write_latex_table(results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    walker = WALKER_REF

    # The Exp 2 defaults (combined loss, shared tau=2 deg, lambda=0.1,
    # beta=10 min) all return the same metrics as the Walker reference; we
    # collapse them into a single row to keep the table compact.
    default = results['loss_combined']

    lines = []
    lines.append("% Auto-generated by experiments/exp4_ablations.py")
    lines.append("\\begin{tabular}{lrr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Setting} & \\textbf{Hard cov [\\%]} & \\textbf{Hard revisit [min]} \\\\")
    lines.append("\\midrule")
    lines.append(f"Walker 24/6/1 reference / Exp 2 defaults & "
                 f"{default['hard_cov_pct']:.2f} & "
                 f"{default['hard_revisit_min']:.1f} \\\\")
    lines.append("\\midrule")

    # (a) Loss composition (defaults = combined, omitted)
    lines.append(_row("(a) Coverage only ($\\lambda = 0$)", results['loss_coverage_only']))
    lines.append(_row("(a) Revisit only ($-\\tilde{C}$ dropped)", results['loss_revisit_only']))

    # (b) Sigmoid temperature (default = shared, omitted)
    lines.append(_row("(b) Split $\\tau$ ($\\tau_{\\text{cov}}=3^{\\circ}$, $\\tau_{\\text{rev}}=1^{\\circ}$)",
                      results['tau_split']))
    lines.append(_row("(b) Split $\\tau$ inverted ($\\tau_{\\text{cov}}=1^{\\circ}$, $\\tau_{\\text{rev}}=3^{\\circ}$)",
                      results['tau_split_inverted']))

    # (c) Revisit weight lambda (default lambda=0.1 omitted)
    for lam in [0.01, 1.0]:
        lines.append(_row(f"(c) $\\lambda = {lam}$", results[f'lambda_{lam}']))

    # (d) LogSumExp temperature beta (default beta=10 omitted)
    for beta in [1.0, 100.0]:
        lines.append(_row(f"(d) $\\beta = {int(beta)}$ min", results[f'beta_{beta}']))

    # (e) Init sensitivity -- single row with mean+-std and range
    seeds = list(range(1, 11))
    cov_vals = [results[f'seed_{s}']['hard_cov_pct'] for s in seeds]
    rev_vals = [results[f'seed_{s}']['hard_revisit_min'] for s in seeds]
    mean_cov = float(np.mean(cov_vals)); std_cov = float(np.std(cov_vals))
    mean_rev = float(np.mean(rev_vals)); std_rev = float(np.std(rev_vals))
    lines.append(
        f"(e) Random RAAN+MA init, $n={len(seeds)}$ & "
        f"{mean_cov:.2f} $\\pm$ {std_cov:.2f} "
        f"({min(cov_vals):.2f}--{max(cov_vals):.2f}) & "
        f"{mean_rev:.1f} $\\pm$ {std_rev:.1f} "
        f"({min(rev_vals):.1f}--{max(rev_vals):.1f}) \\\\"
    )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(out_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote table: {out_path}")


# ---- Main ------------------------------------------------------------------

def _save(results, json_path):
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)


def build_tasks():
    """Produce (key, name, overrides, loss_mode, seed) tuples for all ablations."""
    tasks = []
    # A. Loss composition (all seed=42, combined config)
    for mode in ['coverage_only', 'revisit_only', 'combined']:
        tasks.append((f'loss_{mode}', f'loss_{mode}', {}, mode, 42))
    # B. Sigmoid temperature variants (shared baseline = loss_combined)
    tasks.append(('tau_split', 'tau_split',
                  {'softness_deg': 3.0, 'revisit_softness_deg': 1.0}, 'combined', 42))
    tasks.append(('tau_split_inverted', 'tau_split_inverted',
                  {'softness_deg': 1.0, 'revisit_softness_deg': 3.0}, 'combined', 42))
    # C. Lambda sweep
    for lam in [0.01, 0.1, 1.0]:
        tasks.append((f'lambda_{lam}', f'lambda_{lam}',
                      {'revisit_weight': lam}, 'combined', 42))
    # D. Beta sweep
    for beta in [1.0, 10.0, 100.0]:
        tasks.append((f'beta_{beta}', f'beta_{beta}',
                      {'revisit_logsumexp_temp': beta}, 'combined', 42))
    # E. Initialization sensitivity (fully random RAAN + MA, n=10 seeds)
    for seed in range(1, 11):
        rng = np.random.RandomState(seed)
        raans = rng.uniform(0, 360, size=6).tolist()
        tasks.append((f'seed_{seed}', f'seed_{seed}',
                      {'initial_raan_offsets_deg': raans}, 'combined', seed))
    return tasks


def run_single(key, n_threads=None):
    """Run a single ablation by key and write its result to a per-run JSON file.

    Invoked by the orchestrator as a separate subprocess. Writes to
    paper/data/ablations_runs/<key>.json.
    """
    if n_threads is not None:
        torch.set_num_threads(n_threads)

    all_tasks = {t[0]: t for t in build_tasks()}
    if key not in all_tasks:
        raise KeyError(f"unknown ablation key: {key}")
    _, name, overrides, loss_mode, seed = all_tasks[key]

    result = run_ablation(name, overrides=overrides, loss_mode=loss_mode, seed=seed)

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'paper', 'data',
                           'ablations_runs')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{key}.json'), 'w') as f:
        json.dump(result, f, indent=2)


def main(smoke_test=False, n_workers=None):
    if smoke_test:
        global N_ITERATIONS
        N_ITERATIONS = 30

    paper_root = os.path.join(os.path.dirname(__file__), '..', 'paper')
    data_dir = os.path.join(paper_root, 'data')
    tables_dir = os.path.join(paper_root, 'tables')
    runs_dir = os.path.join(data_dir, 'ablations_runs')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    json_path = os.path.join(data_dir, 'ablations.json')

    tasks = build_tasks()
    # Skip tasks whose per-run JSON already exists (resume / partial reruns).
    pre_existing = {t[0] for t in tasks
                    if os.path.exists(os.path.join(runs_dir, f'{t[0]}.json'))}
    if pre_existing:
        print(f"Skipping {len(pre_existing)} already-completed: "
              f"{sorted(pre_existing)}")
        tasks = [t for t in tasks if t[0] not in pre_existing]
    if n_workers is None:
        # PyTorch parallelism for this workload is flat past ~2 threads;
        # 7 workers x 2 threads on a 14-core machine maximises throughput.
        n_workers = min(7, len(tasks)) if tasks else 1

    threads_per_worker = max(1, (os.cpu_count() or 4) // n_workers)
    n_waves = (len(tasks) + n_workers - 1) // n_workers

    print(f"Launching {len(tasks)} ablations, {n_workers} parallel workers, "
          f"{threads_per_worker} threads each ({n_waves} waves)")
    print(f"Each run: {N_ITERATIONS} iterations\n")

    script = os.path.abspath(__file__)
    env = dict(os.environ)
    env['OMP_NUM_THREADS']      = str(threads_per_worker)
    env['MKL_NUM_THREADS']      = str(threads_per_worker)
    env['OPENBLAS_NUM_THREADS'] = str(threads_per_worker)

    smoke_arg = ['--smoke'] if smoke_test else []
    logs_dir = os.path.join(runs_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    pending = list(tasks)
    running = {}  # proc -> (key, t_start)
    completed = []
    t_all = time.time()

    def launch(key):
        log_path = os.path.join(logs_dir, f'{key}.log')
        log_f = open(log_path, 'w')
        proc = subprocess.Popen(
            [sys.executable, '-u', script, '--single', key, *smoke_arg],
            stdout=log_f, stderr=subprocess.STDOUT, env=env,
        )
        running[proc] = (key, time.time(), log_f)
        print(f"  [start] {key}")

    # Fill initial wave
    while pending and len(running) < n_workers:
        key = pending.pop(0)[0]
        launch(key)

    while running:
        # Poll
        done_now = [p for p in running if p.poll() is not None]
        for p in done_now:
            key, t0, log_f = running.pop(p)
            log_f.close()
            elapsed = time.time() - t0
            rc = p.returncode
            if rc != 0:
                print(f"  [FAIL] {key} (rc={rc}, {elapsed:.1f}s, see logs)")
            else:
                print(f"  [done] {key} ({elapsed:.1f}s)")
                completed.append(key)
            # Launch next from queue
            if pending:
                next_key = pending.pop(0)[0]
                launch(next_key)
        if running:
            time.sleep(0.5)

    # Collate per-run JSON files (include pre-existing as well as new)
    results = {
        'walker_reference': WALKER_REF,
        'n_iterations': N_ITERATIONS,
        'n_workers': n_workers,
        'total_wall_sec': time.time() - t_all,
    }
    all_keys = [t[0] for t in build_tasks()]
    results['n_tasks'] = len(all_keys)
    for key in all_keys:
        run_path = os.path.join(runs_dir, f'{key}.json')
        if os.path.exists(run_path):
            with open(run_path) as f:
                results[key] = json.load(f)

    _save(results, json_path)
    print(f"\nAll done in {results['total_wall_sec']/60:.1f} min.  "
          f"Wrote JSON: {json_path}")

    expected = set(all_keys)
    got = {k for k in results if isinstance(results.get(k), dict) and 'hard_cov_pct' in results[k]}
    missing = expected - got
    if missing:
        print(f"WARNING: missing results: {sorted(missing)}")
    else:
        write_latex_table(results, os.path.join(tables_dir, 'ablations.tex'))


if __name__ == '__main__':
    if '--smoke' in sys.argv:
        N_ITERATIONS = 30
    if '--single' in sys.argv:
        idx = sys.argv.index('--single')
        run_single(sys.argv[idx + 1])
    else:
        main(smoke_test='--smoke' in sys.argv)
