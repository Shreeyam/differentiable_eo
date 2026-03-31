"""
Figure 2: Continuous relaxations for differentiable constellation optimization.

Generates four separate PDFs at 4.2 x 3.5 cm each:
  (a) Soft sigmoid visibility model
  (b) Noisy-OR multi-satellite aggregation
  (c) Leaky integrator for revisit gap tracking
  (d) LogSumExp soft-maximum
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import scienceplots
import matplotlib.pyplot as plt

plt.style.use('science')

CM = 1 / 2.54  # cm to inches
W, H = 4.2 * CM, 3.5 * CM


def plot_soft_visibility(path):
    fig, ax = plt.subplots(figsize=(W, H))
    x = np.linspace(-15, 15, 300)
    sigmoid = 1.0 / (1.0 + np.exp(-x / 2.0))
    hard = np.where(x >= 0, 1.0, 0.0)
    ax.plot(x, hard, '--', lw=1.5, label='Step', color='#3F51B5')
    ax.plot(x, sigmoid, '-', lw=1.5, label=r'$\tau=2$', color='#E91E63')
    ax.set_xlabel(r'$\alpha - \alpha_{\min}$ [deg]')
    ax.set_ylabel(r'$c_{ij}$')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='lower right', fontsize=6)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def plot_noisy_or(path):
    fig, ax = plt.subplots(figsize=(W, H))
    c1 = np.linspace(0, 1, 100)
    for c2, color, label in [(0.0, '#009688', r'$c_2=0$'),
                              (0.3, '#E91E63', r'$c_2=0.3$'),
                              (0.7, '#3F51B5', r'$c_2=0.7$')][::-1]:
        ax.plot(c1, 1.0 - (1.0 - c1) * (1.0 - c2), color=color, lw=1.5, label=label)
    ax.set_xlabel(r'$c_1$')
    ax.set_ylabel(r'$1 - \prod_{i=1}^2 (1-c_i)$', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='lower right', fontsize=6)
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


def plot_leaky_integrator(path):
    fig, ax = plt.subplots(figsize=(W, H))
    t = np.arange(21)
    gap = np.array([0, 7, 14, 21, 28, 35,
                    0, 7, 14, 21, 28, 35, 42,
                    0, 7, 14, 21, 28, 35, 42, 49], dtype=float)
    coverage_t = np.array([6, 13])
    ax.plot(t, gap, '-', lw=1.5, color='#E91E63')
    ax.plot(coverage_t, [0, 0], '^', ms=5, label='Coverage event', color='#3F51B5')
    ax.set_xlabel('Time step')
    ax.set_ylabel(r'Gap $\Delta_j$ [min]')
    ax.set_xlim(0, 20)
    ax.set_ylim(-5, 85)
    ax.legend(fontsize=6, loc='upper left')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def plot_logsumexp(path):
    fig, ax = plt.subplots(figsize=(W, H))
    tau = np.linspace(1, 50, 200)
    values = np.array([20.0, 45.0, 30.0])
    lse = tau * np.log(np.sum(np.exp(values[:, None] / tau[None, :]), axis=0))
    ax.plot(tau, lse, '-', lw=1.5, label='LSE', color='#E91E63')
    ax.axhline(45, color='#3F51B5', ls='--', lw=1.5, label='True max')
    ax.set_xlabel(r'Temperature $\tau_r$')
    ax.set_ylabel(r'$\widetilde{\Delta}^{\max}$')
    ax.set_xlim(0.5, 50)
    ax.set_ylim(40, 110)
    ax.legend(loc='upper left', fontsize=6)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def main():
    plot_soft_visibility('plots/fig2a_soft_visibility.pdf')
    plot_noisy_or('plots/fig2b_noisy_or.pdf')
    plot_leaky_integrator('plots/fig2c_leaky_integrator.pdf')
    plot_logsumexp('plots/fig2d_logsumexp.pdf')


if __name__ == '__main__':
    main()


ml_colors = ['#3F51B5', '#E91E63', '#009688', '#FFC107', '#673AB7', '#03A9F4']