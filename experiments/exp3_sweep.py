"""
Sweep 10 configurations to find a clear resonant ground track visualization.
Each config runs a quick optimization (500 iter) and produces a density plot.
"""
import sys, os, math, json
import numpy as np
import torch, dsgp4

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from differentiable_eo import (
    make_gmst_array, compute_loss, compute_hard_metrics, extract_elements,
    IDX_INCLO, IDX_MO, IDX_NODEO, IDX_ECCO, IDX_ARGPO, IDX_NO_KOZAI,
)
from differentiable_eo.constraints import (
    FixedConstraint, UnboundedConstraint, BoxConstraint,
    PeriapsisApoapsisConstraint, ReparameterizedElements,
)
from differentiable_eo.tle_utils import (
    no_kozai_from_alt, alt_from_no_kozai, make_tle, make_constellation,
    extract_elements, update_tle_from_elements,
)
from differentiable_eo.constants import EARTH_ROT_RAD_PER_MIN
from exp3_weighted_europe import sample_europe_targets

import cartopy.crs as ccrs
import cartopy.feature as cfeature

R_E = 6378.137
MU = 398600.4418

CONFIGS = [
    # name, n_planes, n_sats_pp, min_el, perigee_bounds, excess_max, n_iter, revisit_weight, plot_type
    ("1_raw_track_10el_2k",         2, 2, 10, (400,800), 2000, 2000, 1.0, "track"),
    ("2_raw_track_10el_11k",        2, 2, 10, (300,600), 11000, 2000, 1.0, "track"),
    ("3_density_nearest_10el_2k",   2, 2, 10, (400,800), 2000, 2000, 1.0, "density_nearest"),
    ("4_density_nearest_15el_2k",   2, 2, 15, (400,800), 2000, 2000, 1.0, "density_nearest"),
    ("5_hires_density_10el_2k",     2, 2, 10, (400,800), 2000, 2000, 1.0, "density_hires"),
    ("6_8sats_15el_2k",             4, 2, 15, (400,800), 2000, 2000, 1.0, "density_nearest"),
    ("7_4sats_1plane_10el_2k",      1, 4, 10, (400,800), 2000, 2000, 2.0, "track"),
    ("8_subsat_heatmap_10el_2k",    2, 2, 10, (400,800), 2000, 2000, 1.0, "subsat"),
    ("9_tight_LEO_10el_500",        2, 2, 10, (400,600), 500,  2000, 1.0, "density_nearest"),
    ("10_4sats_10el_2k_48h",        2, 2, 10, (400,800), 2000, 2000, 1.0, "track_48h"),
]


def make_specs(perigee_bounds, excess_max):
    coupled = PeriapsisApoapsisConstraint(
        perigee_bounds_km=perigee_bounds,
        excess_bounds_km=(0.0, excess_max),
    )
    return {
        IDX_INCLO: BoxConstraint(math.radians(30), math.radians(90)),
        IDX_NODEO: UnboundedConstraint(),
        IDX_MO: UnboundedConstraint(),
        IDX_ARGPO: UnboundedConstraint(),
        IDX_NO_KOZAI: coupled,
        IDX_ECCO: coupled,
        0: FixedConstraint(),
        1: FixedConstraint(),
        2: FixedConstraint(),
    }


def run_optimization(n_planes, n_sats_pp, min_el, perigee_bounds, excess_max,
                     n_iter, revisit_weight, ground_ecef, ground_weights,
                     ground_unit, tsinces, gmst_tensor, prop_hours):
    n_sats = n_planes * n_sats_pp
    specs = make_specs(perigee_bounds, excess_max)
    init_raans = [i * 360.0 / max(n_planes, 1) for i in range(n_planes)]

    rng = np.random.RandomState(123)
    tles = []
    reparams = []
    for p in range(n_planes):
        for s in range(n_sats_pp):
            tle = make_tle(math.radians(60.0), math.radians(init_raans[p]),
                           math.radians(rng.uniform(0, 360)), 550.0)
            dsgp4.initialize_tle(tle, with_grad=False)
            elems = extract_elements(tle)
            reparam = ReparameterizedElements(elems, specs)
            tles.append(tle)
            reparams.append(reparam)

    z_params = [r.optimizer_param for r in reparams if r.n_free > 0]
    optimizer = torch.optim.AdamW(z_params, lr=1e-2)

    # Per-plane param sharing
    per_plane = [IDX_NODEO, IDX_INCLO]
    # Add coupled params
    coupled_z_labels = ['coupled_rp', 'coupled_exc']

    torch.manual_seed(7)
    for iteration in range(n_iter):
        optimizer.zero_grad()

        tle_elem_list = []
        for i, tle in enumerate(tles):
            elements = reparams[i].to_elements()
            update_tle_from_elements(tle, elements)
            tle_elem_list.append(dsgp4.initialize_tle(tle, with_grad=True))

        gmst_offset = torch.rand(1).item() * 2 * math.pi
        gmst_cur = gmst_tensor + gmst_offset

        loss, cov, rev = compute_loss(
            tles, tsinces, gmst_cur, ground_ecef,
            min_el=min_el, softness=5.0,
            revisit_tau=10.0, revisit_weight=revisit_weight,
            ground_weights=ground_weights, revisit_reduce='mean',
            ground_unit=ground_unit, revisit_softness=2.0,
        )
        loss.backward()

        for i in range(n_sats):
            if tle_elem_list[i].grad is not None:
                reparams[i].compute_z_grad(tle_elem_list[i].grad)

        # Accumulate per-plane gradients
        for param_idx in per_plane + coupled_z_labels:
            for p in range(n_planes):
                lead = reparams[p * n_sats_pp]
                try:
                    z_pos = lead.free_indices.index(param_idx)
                except ValueError:
                    continue
                if lead._z.grad is None:
                    continue
                total = lead._z.grad[z_pos].clone()
                count = 1
                for s_i in range(1, n_sats_pp):
                    follower = reparams[p * n_sats_pp + s_i]
                    try:
                        fz = follower.free_indices.index(param_idx)
                    except ValueError:
                        continue
                    if follower._z.grad is not None:
                        total += follower._z.grad[fz]
                        count += 1
                        follower._z.grad[fz] = 0.0
                lead._z.grad[z_pos] = total / count

        optimizer.step()

        # Sync per-plane params
        for param_idx in per_plane + coupled_z_labels:
            for p in range(n_planes):
                lead = reparams[p * n_sats_pp]
                try:
                    z_pos = lead.free_indices.index(param_idx)
                except ValueError:
                    continue
                lead_val = lead._z.data[z_pos].item()
                for s_i in range(1, n_sats_pp):
                    follower = reparams[p * n_sats_pp + s_i]
                    try:
                        fz = follower.free_indices.index(param_idx)
                    except ValueError:
                        continue
                    follower._z.data[fz] = lead_val

        if iteration % 500 == 0:
            print(f"    iter {iteration}: loss={loss.item():.4f}")

    # Final state
    for i, tle in enumerate(tles):
        elements = reparams[i].to_elements()
        update_tle_from_elements(tle, elements)

    return tles, reparams


def compute_ground_tracks(tles, tsinces_dense, gmst_dense):
    for t in tles:
        dsgp4.initialize_tle(t, with_grad=False)
    all_lats, all_lons = [], []
    for tle in tles:
        try:
            state = dsgp4.propagate(tle, tsinces_dense)
            pos = state[:, 0, :].detach().numpy()
            r = np.linalg.norm(pos, axis=-1)
            lat = np.degrees(np.arcsin(np.clip(pos[:, 2] / r, -1, 1)))
            lon = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))
            lon_ecef = (lon - np.degrees(EARTH_ROT_RAD_PER_MIN * tsinces_dense.numpy())) % 360
            lon_ecef[lon_ecef > 180] -= 360
            all_lats.append(lat)
            all_lons.append(lon_ecef)
        except Exception:
            pass
    return np.concatenate(all_lats), np.concatenate(all_lons)


def make_plot(name, plot_type, tles, target_lats, target_lons, prop_hours, min_el, out_dir):
    prop_min = prop_hours * 60
    dense_t = torch.linspace(0, prop_min, 2000)
    dense_g = make_gmst_array(dense_t)

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='#cccccc', linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.coastlines(linewidth=0.3, color='#999999')

    if plot_type == "track" or plot_type == "track_48h":
        if plot_type == "track_48h":
            dense_t = torch.linspace(0, 48 * 60, 4000)
            dense_g = make_gmst_array(dense_t)
        lats, lons = compute_ground_tracks(tles, dense_t, dense_g)
        t_color = np.tile(np.linspace(0, 1, len(dense_t)), len(tles))[:len(lats)]
        ax.scatter(lons, lats, s=0.3, c=t_color, cmap='turbo', alpha=0.6,
                  transform=ccrs.PlateCarree(), zorder=2)

    elif plot_type in ("density_nearest", "density_hires"):
        if plot_type == "density_hires":
            lon_bins = np.linspace(-180, 180, 361)
            lat_bins = np.linspace(-90, 90, 181)
        else:
            lon_bins = np.linspace(-180, 180, 181)
            lat_bins = np.linspace(-90, 90, 91)

        lon_c = (lon_bins[:-1] + lon_bins[1:]) / 2
        lat_c = (lat_bins[:-1] + lat_bins[1:]) / 2
        GL, GA = np.meshgrid(lon_c, lat_c)
        grid_lr = np.radians(GA.ravel())
        grid_lo = np.radians(GL.ravel())
        g_ecef = np.stack([R_E*np.cos(grid_lr)*np.cos(grid_lo),
                          R_E*np.cos(grid_lr)*np.sin(grid_lo),
                          R_E*np.sin(grid_lr)], axis=-1)
        g_unit = g_ecef / np.linalg.norm(g_ecef, axis=-1, keepdims=True)

        density = np.zeros(len(grid_lr))
        min_el_rad = np.radians(min_el)
        for t in tles:
            dsgp4.initialize_tle(t, with_grad=False)
        step = max(1, len(dense_t) // 200)
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
        interp = 'nearest' if 'nearest' in plot_type else 'bilinear'
        im = ax.imshow(density, cmap='turbo', vmin=0, vmax=max(density.max(), 1),
                       alpha=0.7, extent=[-180,180,-90,90], origin='lower',
                       interpolation=interp, transform=ccrs.PlateCarree(), zorder=2)
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    elif plot_type == "subsat":
        lats, lons = compute_ground_tracks(tles, dense_t, dense_g)
        h, xedges, yedges = np.histogram2d(lons, lats,
            bins=[np.linspace(-180,180,361), np.linspace(-90,90,181)])
        lon_c = (xedges[:-1]+xedges[1:])/2
        lat_c = (yedges[:-1]+yedges[1:])/2
        GL, GA = np.meshgrid(lon_c, lat_c)
        im = ax.imshow(h.T, cmap='turbo', vmin=0, alpha=0.7,
                       extent=[-180,180,-90,90], origin='lower',
                       interpolation='nearest', transform=ccrs.PlateCarree(), zorder=2)
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    ax.scatter(target_lons, target_lats, s=3, c='white', edgecolors='k',
              linewidths=0.2, alpha=0.3, transform=ccrs.PlateCarree(), zorder=4)
    ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.3,
                xlocs=range(-180,181,60), ylocs=range(-90,91,30))

    # Print orbital info in title
    for t in tles:
        dsgp4.initialize_tle(t, with_grad=False)
    e0 = extract_elements(tles[0])
    inc = math.degrees(e0[IDX_INCLO].item())
    ecc = e0[IDX_ECCO].item()
    alt = alt_from_no_kozai(e0[IDX_NO_KOZAI].item())
    a = R_E + alt
    T_min = 2*math.pi*math.sqrt(a**3/MU)/60
    revs = 23.9345*60/T_min
    ax.set_title(f'{name}\ni={inc:.1f}° e={ecc:.3f} alt={alt:.0f}km  '
                 f'T={T_min:.0f}min {revs:.2f}rev/day', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'sweep_{name}.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    geojson_path = os.path.join(os.path.dirname(__file__), 'europe.geojson')
    ground_ecef, ground_weights, target_lats, target_lons = sample_europe_targets(
        geojson_path, n_points=300)
    ground_unit = ground_ecef / torch.norm(ground_ecef, dim=-1, keepdim=True)

    out_dir = os.path.join(os.path.dirname(__file__), 'sweep_results')
    os.makedirs(out_dir, exist_ok=True)

    for cfg in CONFIGS:
        name, n_planes, n_sats_pp, min_el, peri_bounds, excess_max, n_iter, rev_w, plot_type = cfg
        print(f"\n{'='*60}")
        print(f"  Config: {name}")
        print(f"  {n_planes}P x {n_sats_pp}S, el={min_el}°, peri={peri_bounds}, excess={excess_max}")
        print(f"{'='*60}")

        prop_hours = 48.0 if '48h' in plot_type else 24.0
        prop_min = prop_hours * 60
        tsinces = torch.linspace(0, prop_min, 240)
        gmst_tensor = torch.tensor(make_gmst_array(tsinces), dtype=torch.float64)

        tles, reparams = run_optimization(
            n_planes, n_sats_pp, min_el, peri_bounds, excess_max,
            n_iter, rev_w, ground_ecef, ground_weights, ground_unit,
            tsinces, gmst_tensor, prop_hours)

        make_plot(name, plot_type, tles, target_lats, target_lons,
                  prop_hours, min_el, out_dir)

        # Print geometry
        for p in range(n_planes):
            e = reparams[p * n_sats_pp].to_elements()
            inc = math.degrees(e[IDX_INCLO].item())
            ecc = e[IDX_ECCO].item()
            alt = alt_from_no_kozai(e[IDX_NO_KOZAI].item())
            a = R_E + alt
            T = 2*math.pi*math.sqrt(a**3/MU)/60
            revs = 23.9345*60/T
            print(f"  Plane {p}: i={inc:.1f}° e={ecc:.4f} alt={alt:.0f}km T={T:.1f}min revs={revs:.3f}")

    print(f"\nAll results saved to {out_dir}/")


if __name__ == '__main__':
    main()


def run_hail_mary():
    """2 sats, 1 plane, density plot, up to 10000km excess."""
    geojson_path = os.path.join(os.path.dirname(__file__), 'europe.geojson')
    ground_ecef, ground_weights, target_lats, target_lons = sample_europe_targets(
        geojson_path, n_points=300)
    ground_unit = ground_ecef / torch.norm(ground_ecef, dim=-1, keepdim=True)

    out_dir = os.path.join(os.path.dirname(__file__), 'sweep_results')
    os.makedirs(out_dir, exist_ok=True)

    prop_hours = 24.0
    prop_min = prop_hours * 60
    tsinces = torch.linspace(0, prop_min, 240)
    gmst_tensor = torch.tensor(make_gmst_array(tsinces), dtype=torch.float64)

    print("\n" + "=" * 60)
    print("  HAIL MARY: 2 sats, 1 plane, 10000km excess")
    print("=" * 60)

    tles, reparams = run_optimization(
        n_planes=1, n_sats_pp=2, min_el=10, perigee_bounds=(400, 800),
        excess_max=10000, n_iter=2000, revisit_weight=2.0,
        ground_ecef=ground_ecef, ground_weights=ground_weights,
        ground_unit=ground_unit, tsinces=tsinces, gmst_tensor=gmst_tensor,
        prop_hours=prop_hours)

    # Print geometry
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

    # Raw ground track
    make_plot("hailmary_track", "track", tles, target_lats, target_lons,
              prop_hours, 10, out_dir)

    # Sub-satellite heatmap
    make_plot("hailmary_subsat", "subsat", tles, target_lats, target_lons,
              prop_hours, 10, out_dir)

    # Density nearest
    make_plot("hailmary_density", "density_nearest", tles, target_lats, target_lons,
              prop_hours, 10, out_dir)


if __name__ == '__main__' and 'hail' in sys.argv[-1]:
    run_hail_mary()
