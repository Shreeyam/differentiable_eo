"""
Ablations convergence figures.

Emits six standalone PDFs to paper/figures/, one per (panel, metric) pair:

  ablations_loss_revisit.pdf      ablations_loss_coverage.pdf
  ablations_lambda_revisit.pdf    ablations_lambda_coverage.pdf
  ablations_beta_revisit.pdf      ablations_beta_coverage.pdf

Each subfigure has no title; subcaption labels are added in LaTeX.

Reads paper/data/ablations.json (produced by experiments/exp4_ablations.py).
"""
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
plt.style.use(['science', 'no-latex', 'shreeyam.mplstyle'])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(ROOT, 'paper', 'data', 'ablations.json')
OUT_DIR = os.path.join(ROOT, 'paper', 'figures')

with open(DATA) as f:
    d = json.load(f)

WALKER_COV = d['walker_reference']['hard_cov_pct']
WALKER_REV = d['walker_reference']['hard_revisit_min']

# (slug, panel-series spec)
PANELS = [
    ('loss', [
        ('loss_coverage_only', 'Coverage only',  colors[0]),
        ('loss_revisit_only',  'Revisit only',   colors[1]),
        ('loss_combined',      'Combined (Exp 2)', colors[2]),
    ]),
    ('lambda', [
        ('lambda_0.01', r'$\lambda = 0.01$',         colors[0]),
        ('lambda_0.1',  r'$\lambda = 0.1$ (Exp 2)',  colors[2]),
        ('lambda_1.0',  r'$\lambda = 1.0$',          colors[3]),
    ]),
    ('beta', [
        ('beta_1.0',   r'$\beta = 1$ min',           colors[0]),
        ('beta_10.0',  r'$\beta = 10$ min (Exp 2)',  colors[2]),
        ('beta_100.0', r'$\beta = 100$ min',         colors[3]),
    ]),
]

METRICS = [
    # slug,     history key,            ylabel,                         walker,     walker label
    ('revisit',  'hard_revisit_history', 'Hard mean max revisit [min]', WALKER_REV, 'Walker'),
    ('coverage', 'hard_cov_history',     'Hard coverage [%]',            WALKER_COV, 'Walker'),
]


def plot_one(slug, series, metric_slug, hist_key, ylabel, walker, walker_label):
    fig, ax = plt.subplots(figsize=(2.67, 2.4))
    for key, label, color in series:
        v = d.get(key)
        if v is None:
            continue
        ax.plot(v['eval_iters'], v[hist_key],
                'o-', color=color, lw=1.2, markersize=2.5, label=label)
    ax.axhline(walker, color='0.4', ls='--', lw=1.0, alpha=0.7,
               label=f'{walker_label} ({walker:.1f})')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    # Per-panel legend anchors (None → use loc='best'); tuple → (loc, bbox_to_anchor)
    legend_overrides = {
        ('loss', 'revisit'): ('center right', (1.0, 0.35)),
    }
    override = legend_overrides.get((slug, metric_slug))
    if override is None:
        ax.legend(fontsize=6.5, loc='best', framealpha=0.85)
    else:
        loc, anchor = override
        ax.legend(fontsize=6.5, loc=loc, bbox_to_anchor=anchor, framealpha=0.85)
    ax.grid(True, alpha=0.2)
    if d[series[0][0]].get('eval_iters'):
        ax.set_xlim(0, max(d[series[0][0]]['eval_iters']))
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f'ablations_{slug}_{metric_slug}.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


os.makedirs(OUT_DIR, exist_ok=True)
for slug, series in PANELS:
    for mslug, hist_key, ylabel, walker, walker_label in METRICS:
        plot_one(slug, series, mslug, hist_key, ylabel, walker, walker_label)
