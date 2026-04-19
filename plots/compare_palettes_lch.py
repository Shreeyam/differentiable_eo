"""Compare old and new color palettes in CIELCh space."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from colorspacious import cspace_convert

old = ['#4354b5', '#cf175a', '#009688', '#db5c1d', '#18a5e7', '#f5b616']
new = ['#4a5de9', '#ea1a69', '#00c5b3', '#f46009', '#07acf8', '#fcbf07']
labels = ['Blue', 'Pink', 'Teal', 'Orange', 'Sky', 'Yellow']


def hex_to_lch(colors):
    rgb255 = np.array([to_rgb(c) for c in colors]) * 255
    lab = cspace_convert(rgb255, "sRGB255", "CIELab")
    L = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]
    C = np.sqrt(a**2 + b**2)
    H = np.degrees(np.arctan2(b, a)) % 360
    return L, C, H


L_old, C_old, H_old = hex_to_lch(old)
L_new, C_new, H_new = hex_to_lch(new)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Lightness
ax = axes[0]
x = np.arange(len(labels))
ax.bar(x - 0.18, L_old, 0.35, label='Old', color=old, edgecolor='k', linewidth=0.5)
ax.bar(x + 0.18, L_new, 0.35, label='New', color=new, edgecolor='k', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.set_ylabel('Lightness (L*)')
ax.set_ylim(0, 100)
ax.set_title('Lightness')
ax.legend()

# Chroma
ax = axes[1]
ax.bar(x - 0.18, C_old, 0.35, label='Old', color=old, edgecolor='k', linewidth=0.5)
ax.bar(x + 0.18, C_new, 0.35, label='New', color=new, edgecolor='k', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.set_ylabel('Chroma (C*)')
ax.set_title('Chroma')
ax.legend()

# Hue wheel
ax = axes[2]
ax.set_aspect('equal')
r_old, r_new = 1.0, 0.7
for i in range(len(labels)):
    theta_o = np.radians(H_old[i])
    theta_n = np.radians(H_new[i])
    ax.plot([0, r_old * np.cos(theta_o)], [0, r_old * np.sin(theta_o)],
            '-', color=old[i], lw=2, alpha=0.6)
    ax.scatter(r_old * np.cos(theta_o), r_old * np.sin(theta_o),
               s=120, color=old[i], edgecolors='k', linewidths=0.5, zorder=5)
    ax.plot([0, r_new * np.cos(theta_n)], [0, r_new * np.sin(theta_n)],
            '--', color=new[i], lw=2, alpha=0.6)
    ax.scatter(r_new * np.cos(theta_n), r_new * np.sin(theta_n),
               s=120, color=new[i], edgecolors='k', linewidths=0.5, zorder=5,
               marker='D')
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_title('Hue angle (H*)')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
# Add hue labels
for angle, name in [(0, '0° Red'), (90, '90° Yellow'), (180, '180° Green'), (270, '270° Blue')]:
    th = np.radians(angle)
    ax.text(1.2 * np.cos(th), 1.2 * np.sin(th), name, ha='center', va='center', fontsize=7)

fig.suptitle('Old (circles, solid) vs New (diamonds, dashed) palette in CIELCh', fontsize=12)
fig.tight_layout()
fig.savefig('plots/palette_comparison_lch.png', dpi=200, bbox_inches='tight')
print('Saved plots/palette_comparison_lch.png')

# Print values
print(f"\n{'Color':<8} {'L_old':>6} {'L_new':>6} {'C_old':>6} {'C_new':>6} {'H_old':>6} {'H_new':>6}")
for i in range(len(labels)):
    print(f"{labels[i]:<8} {L_old[i]:6.1f} {L_new[i]:6.1f} {C_old[i]:6.1f} {C_new[i]:6.1f} {H_old[i]:6.1f} {H_new[i]:6.1f}")
