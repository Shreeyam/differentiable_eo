"""Sweep 10 single-sat configs hunting for E/W banding in density plots."""
import sys, os, math
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from exp3_sweep import (run_optimization, make_gmst_array, extract_elements,
                        alt_from_no_kozai, sample_europe_targets,
                        IDX_INCLO, IDX_ECCO, IDX_NO_KOZAI, R_E, MU)

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dsgp4

geojson_path = os.path.join(os.path.dirname(__file__), 'europe.geojson')
ground_ecef, ground_weights, target_lats, target_lons = sample_europe_targets(
    geojson_path, n_points=300)
ground_unit = ground_ecef / torch.norm(ground_ecef, dim=-1, keepdim=True)
out_dir = os.path.join(os.path.dirname(__file__), 'sweep_results')
os.makedirs(out_dir, exist_ok=True)

CONFIGS = [
    # (name, min_el, excess_max, n_bins_lon, n_bins_lat, prop_hours, rev_weight)
    ("1sat_30el_10k",        30, 10000, 181, 91, 24, 2.0),
    ("1sat_40el_10k",        40, 10000, 181, 91, 24, 2.0),
    ("1sat_45el_10k",        45, 10000, 181, 91, 24, 2.0),
    ("1sat_50el_10k",        50, 10000, 181, 91, 24, 2.0),
    ("1sat_30el_5k",         30, 5000,  181, 91, 24, 2.0),
    ("1sat_30el_3k",         30, 3000,  181, 91, 24, 2.0),
    ("1sat_30el_10k_hires",  30, 10000, 361, 181, 24, 2.0),
    ("1sat_40el_10k_hires",  40, 10000, 361, 181, 24, 2.0),
    ("1sat_35el_10k",        35, 10000, 181, 91, 24, 2.0),
    ("1sat_30el_10k_heavyrev", 30, 10000, 181, 91, 24, 5.0),
]

for cfg_name, min_el, excess_max, n_lon, n_lat, prop_h, rev_w in CONFIGS:
    print(f"\n{'='*50}")
    print(f"  {cfg_name}: el={min_el}° excess={excess_max}km bins={n_lon}x{n_lat}")
    print(f"{'='*50}")

    prop_min = prop_h * 60
    tsinces = torch.linspace(0, prop_min, 240)
    gmst_tensor = torch.tensor(make_gmst_array(tsinces), dtype=torch.float64)

    tles, reparams = run_optimization(
        n_planes=1, n_sats_pp=1, min_el=min_el, perigee_bounds=(400, 800),
        excess_max=excess_max, n_iter=2000, revisit_weight=rev_w,
        ground_ecef=ground_ecef, ground_weights=ground_weights,
        ground_unit=ground_unit, tsinces=tsinces, gmst_tensor=gmst_tensor,
        prop_hours=prop_h)

    e = reparams[0].to_elements()
    inc = math.degrees(e[IDX_INCLO].item())
    ecc = e[IDX_ECCO].item()
    alt = alt_from_no_kozai(e[IDX_NO_KOZAI].item())
    a = R_E + alt
    T = 2*math.pi*math.sqrt(a**3/MU)/60
    revs = 23.9345*60/T
    rp = a*(1-ecc) - R_E
    ra = a*(1+ecc) - R_E
    print(f"  i={inc:.1f}° e={ecc:.4f} alt={alt:.0f}km T={T:.1f}min revs={revs:.3f}")
    print(f"  rp={rp:.0f}km ra={ra:.0f}km")

    # Compute density
    lon_bins = np.linspace(-180, 180, n_lon)
    lat_bins = np.linspace(-90, 90, n_lat)
    lon_c = (lon_bins[:-1]+lon_bins[1:])/2
    lat_c = (lat_bins[:-1]+lat_bins[1:])/2
    GL, GA = np.meshgrid(lon_c, lat_c)
    glr = np.radians(GA.ravel())
    glo = np.radians(GL.ravel())
    g_ecef = np.stack([R_E*np.cos(glr)*np.cos(glo),
                       R_E*np.cos(glr)*np.sin(glo),
                       R_E*np.sin(glr)], axis=-1)
    g_unit = g_ecef / np.linalg.norm(g_ecef, axis=-1, keepdims=True)

    dense_t = torch.linspace(0, prop_min, 2000)
    dense_g = make_gmst_array(dense_t)
    min_el_rad = np.radians(min_el)

    density = np.zeros(len(glr))
    for t in tles:
        dsgp4.initialize_tle(t, with_grad=False)
    step = max(1, len(dense_t)//500)  # more samples for better resolution
    for t_idx in range(0, len(dense_t), step):
        for tle in tles:
            try:
                state = dsgp4.propagate(tle, dense_t[t_idx:t_idx+1])
                pos_teme = (state[0,:] if state.dim()==2 else state[0,0,:]).detach().numpy()
            except:
                continue
            gmst = dense_g[t_idx]
            cg, sg = np.cos(gmst), np.sin(gmst)
            pos_ecef = np.array([cg*pos_teme[0]+sg*pos_teme[1],
                                -sg*pos_teme[0]+cg*pos_teme[1], pos_teme[2]])
            diff = pos_ecef - g_ecef
            dist = np.linalg.norm(diff, axis=-1)
            elev = np.arcsin(np.clip(np.sum(diff*g_unit, axis=-1)/dist, -1, 1))
            density += (elev >= min_el_rad).astype(float)

    density = density.reshape(GA.shape)

    # Plot
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='#cccccc', linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.coastlines(linewidth=0.3, color='#999999')

    im = ax.imshow(density, cmap='turbo', vmin=0, vmax=max(density.max(), 1),
                   alpha=0.7, extent=[-180,180,-90,90], origin='lower',
                   interpolation='nearest', transform=ccrs.PlateCarree(), zorder=2)
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    ax.scatter(target_lons, target_lats, s=3, c='white', edgecolors='k',
              linewidths=0.2, alpha=0.3, transform=ccrs.PlateCarree(), zorder=4)
    ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.3,
                xlocs=range(-180,181,60), ylocs=range(-90,91,30))
    ax.set_title(f'{cfg_name}\ni={inc:.1f}° e={ecc:.3f} alt={alt:.0f}km '
                 f'T={T:.0f}min {revs:.2f}rev/day', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'1sat_{cfg_name}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved 1sat_{cfg_name}.pdf")

print("\nDone!")
