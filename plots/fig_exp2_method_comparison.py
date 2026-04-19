"""
Figure: Convergence of coverage and revisit vs function evaluations for
the four methods on the exp2 Walker recovery problem.

Emits two standalone PDFs (one per metric) so LaTeX can lay them out as a
subfigure pair, matching the ablations figure convention.

Method distinction is by color, not dashing, so the plots stay readable with
four curves. Gradient (Exp 2 baseline) keeps the ablations teal convention.
Markers are omitted and each curve is extended back to evaluation 0 with a
zero-order hold so all methods share a common start point.

Data sources:
  - Gradient: paper/data/ablations_runs/loss_combined.json (exp2 baseline run)
  - SA/GA/NG: experiments/exp2_baselines_data.pt
"""

import os
import json
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
plt.style.use(['science', 'no-latex', 'shreeyam.mplstyle'])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

ROOT = os.path.join(os.path.dirname(__file__), '..')


def zoh_to_zero(x, y):
    """Prepend (0, y[0]) so the curve extends back to evaluation 0."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x[0] > 0:
        x = np.concatenate([[0], x])
        y = np.concatenate([[y[0]], y])
    return x, y


def main():
    # ---- Gradient (exp2 baseline) ----
    grad_path = os.path.join(ROOT, 'paper', 'data', 'ablations_runs', 'loss_combined.json')
    with open(grad_path) as f:
        grad = json.load(f)
    grad_evals = np.array(grad['eval_iters'])
    grad_cov = np.array(grad['hard_cov_history'])
    grad_rev = np.array(grad['hard_revisit_history'])

    # ---- Baselines (SA/GA/NG) ----
    data_path = os.path.join(ROOT, 'experiments', 'exp2_baselines_data.pt')
    data = torch.load(data_path, weights_only=False)
    walker_cov, walker_rev, walker_fit, _, _ = data['walker']
    cfg = data['config']

    def best_hist(results, histories):
        if not results:
            return None
        fits = [r[4] for r in results]
        return histories[int(np.argmin(fits))]

    sa_hist = best_hist(data['sa_results'], data['sa_histories'])
    ga_hist = best_hist(data['ga_results'], data['ga_histories'])
    ng_hist = best_hist(data.get('ng_results', []), data.get('ng_histories', []))

    # x-axis: function evaluations
    # SA: probe (50) consumed before first main iter; main iters are 1-indexed
    # GA: 1 generation = pop_size evals; +pop_size for initial population eval
    # NG: 1 eval per tell
    sa_evals = np.arange(len(sa_hist['best_cov'])) + 50 if sa_hist else None
    ga_evals = np.arange(len(ga_hist['best_cov'])) * cfg['ga_pop'] + cfg['ga_pop'] if ga_hist else None
    ng_evals = np.arange(1, len(ng_hist['best_cov']) + 1) if ng_hist else None

    # ---- Method colors (gradient teal per ablations convention) ----
    GRAD_COLOR = colors[2]  # teal  -- exp 2 baseline
    SA_COLOR = colors[1]    # pink
    GA_COLOR = colors[3]    # orange
    NG_COLOR = colors[0]    # indigo

    x_candidates = [grad_evals[-1]]
    for e in (sa_evals, ga_evals, ng_evals):
        if e is not None:
            x_candidates.append(e[-1])
    x_right = max(x_candidates)

    def plot_metric(slug, ylabel, walker_val, walker_fmt,
                    grad_series, sa_series, ga_series, ng_series, legend_loc):
        fig, ax = plt.subplots(figsize=(4.0, 3.2))

        series = [
            ('Gradient (ours)', GRAD_COLOR, grad_evals, grad_series),
            ('Simulated annealing', SA_COLOR, sa_evals, sa_series),
            ('Genetic algorithm', GA_COLOR, ga_evals, ga_series),
            ('TwoPointsDE (nevergrad)', NG_COLOR, ng_evals, ng_series),
        ]
        for label, color, x, y in series:
            if x is None or y is None:
                continue
            x, y = zoh_to_zero(x, y)
            ax.plot(x, y, '-', color=color, lw=1.4, label=label)

        ax.axhline(walker_val, color='0.4', ls=':', lw=1.0, alpha=0.8,
                   label=f'Walker ({walker_fmt})')
        ax.set_xlabel('Function evaluations')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, x_right * 1.02)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=6.5, loc=legend_loc, framealpha=0.85)
        fig.tight_layout()

        out_png = os.path.join(ROOT, 'experiments',
                               f'exp2_method_comparison_{slug}.png')
        out_pdf = os.path.join(ROOT, 'paper', 'figures',
                               f'exp2_method_comparison_{slug}.pdf')
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        fig.savefig(out_pdf, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_pdf}')

    plot_metric('coverage', 'Hard coverage [%]',
                walker_cov, f'{walker_cov:.1f}%',
                grad_cov,
                sa_hist['best_cov'] if sa_hist else None,
                ga_hist['best_cov'] if ga_hist else None,
                ng_hist['best_cov'] if ng_hist else None,
                legend_loc='lower right')

    plot_metric('revisit', 'Hard mean max revisit [min]',
                walker_rev, f'{walker_rev:.0f} min',
                grad_rev,
                sa_hist['best_rev'] if sa_hist else None,
                ga_hist['best_rev'] if ga_hist else None,
                ng_hist['best_rev'] if ng_hist else None,
                legend_loc='upper right')

    # ---- Summary ----
    print('\n' + '=' * 60)
    print('  FINAL VALUES @ END OF BUDGET')
    print('=' * 60)
    print(f"  Walker:    cov={walker_cov:.2f}%, rev={walker_rev:.1f} min")
    print(f"  Gradient:  cov={grad_cov[-1]:.2f}%, rev={grad_rev[-1]:.1f} min "
          f"(@ {grad_evals[-1]} evals)")
    if sa_hist:
        print(f"  SA (best): cov={sa_hist['best_cov'][-1]:.2f}%, "
              f"rev={sa_hist['best_rev'][-1]:.1f} min (@ {sa_evals[-1]} evals)")
    if ga_hist:
        print(f"  GA (best): cov={ga_hist['best_cov'][-1]:.2f}%, "
              f"rev={ga_hist['best_rev'][-1]:.1f} min (@ {ga_evals[-1]} evals)")
    if ng_hist:
        print(f"  NG (best): cov={ng_hist['best_cov'][-1]:.2f}%, "
              f"rev={ng_hist['best_rev'][-1]:.1f} min (@ {ng_evals[-1]} evals)")
    print('=' * 60)


if __name__ == '__main__':
    main()
