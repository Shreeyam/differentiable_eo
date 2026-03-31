"""
Generate before/after 3D constellation globe figures.

Uses the globe rendering module from differentiable_eo.
Prerequisite: run render_globe_blender.py first to produce globe_earth_layer.png
"""
import sys, os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from differentiable_eo.constants import R_EARTH
from differentiable_eo.globe import measure_ball_pixels, render_globe, setup_globe_axes

# ── Constellation parameters (same as exp2) ──────────────────────────────────
N_PLANES         = 6
N_SATS_PER_PLANE = 4
ALT_KM           = 550.0
INC_DEG          = 60.0
WALKER_F         = 1

BAD_RAANS    = [0.0, 30.0, 120.0, 200.0, 210.0, 300.0]
WALKER_RAANS = [i * 360.0 / N_PLANES for i in range(N_PLANES)]

rng = np.random.RandomState(42)
BAD_MAS = rng.uniform(0, 360, size=N_PLANES * N_SATS_PER_PLANE).tolist()

WALKER_MAS = []
for p in range(N_PLANES):
    for s in range(N_SATS_PER_PLANE):
        ma = (360.0 * s / N_SATS_PER_PLANE
              + WALKER_F * (360.0 / (N_PLANES * N_SATS_PER_PLANE)) * p) % 360
        WALKER_MAS.append(ma)

FIG_SIZE = 3.2
DPI = 300
R_ORBIT = R_EARTH + ALT_KM
LIM = R_ORBIT * 0.72

# ── Run ───────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
os.makedirs(out, exist_ok=True)

earth_layer = os.path.join(out, 'globe_earth_layer.png')
if not os.path.exists(earth_layer):
    print(f"ERROR: {earth_layer} not found. Run render_globe_blender.py first.")
    sys.exit(1)

ball_diam, ball_cx, ball_cy, frame_size = measure_ball_pixels(LIM, FIG_SIZE, DPI)

for name, raans, mas in [('initial', BAD_RAANS, BAD_MAS),
                          ('optimized', WALKER_RAANS, WALKER_MAS)]:
    fig = plt.figure(figsize=(FIG_SIZE, FIG_SIZE), facecolor='none')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax_earth, ax3d = setup_globe_axes(fig)

    render_globe(ax_earth, ax3d, raans, mas, earth_layer,
                 ball_diam, ball_cx, ball_cy, frame_size,
                 N_PLANES, N_SATS_PER_PLANE, ALT_KM, INC_DEG)

    save_path = os.path.join(out, f'globe_{name}.pdf')
    fig.savefig(save_path, format='pdf', dpi=DPI, transparent=True)

    svg_path = os.path.join(out, f'globe_{name}.svg')
    fig.savefig(svg_path, format='svg', dpi=DPI, transparent=True)

    plt.close()
    print(f"Saved: {save_path}")
