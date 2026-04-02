"""Hail mary: 2 sats, 1 plane, 10000km excess, Europe targets."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from exp3_sweep import run_optimization, make_plot, sample_europe_targets
from exp3_sweep import make_gmst_array, extract_elements, alt_from_no_kozai
from exp3_sweep import IDX_INCLO, IDX_ECCO, IDX_NO_KOZAI, R_E, MU
import torch, math, numpy as np

geojson_path = os.path.join(os.path.dirname(__file__), 'europe.geojson')
ground_ecef, ground_weights, target_lats, target_lons = sample_europe_targets(
    geojson_path, n_points=300)
ground_unit = ground_ecef / torch.norm(ground_ecef, dim=-1, keepdim=True)

out_dir = os.path.join(os.path.dirname(__file__), 'sweep_results')
os.makedirs(out_dir, exist_ok=True)

tsinces = torch.linspace(0, 24*60, 240)
gmst_tensor = torch.tensor(make_gmst_array(tsinces), dtype=torch.float64)

print("HAIL MARY: 2 sats, 1 plane, 10000km excess")
tles, reparams = run_optimization(
    n_planes=1, n_sats_pp=2, min_el=30, perigee_bounds=(400, 800),
    excess_max=10000, n_iter=2000, revisit_weight=2.0,
    ground_ecef=ground_ecef, ground_weights=ground_weights,
    ground_unit=ground_unit, tsinces=tsinces, gmst_tensor=gmst_tensor,
    prop_hours=24.0)

e = reparams[0].to_elements()
inc = math.degrees(e[IDX_INCLO].item())
ecc = e[IDX_ECCO].item()
alt = alt_from_no_kozai(e[IDX_NO_KOZAI].item())
a = R_E + alt
T = 2*math.pi*math.sqrt(a**3/MU)/60
revs = 23.9345*60/T
rp = a*(1-ecc) - R_E
ra = a*(1+ecc) - R_E
print(f"i={inc:.1f}° e={ecc:.4f} alt={alt:.0f}km T={T:.1f}min revs={revs:.3f}")
print(f"rp={rp:.0f}km ra={ra:.0f}km")

make_plot("hailmary_track", "track", tles, target_lats, target_lons, 24.0, 30, out_dir)
make_plot("hailmary_subsat", "subsat", tles, target_lats, target_lons, 24.0, 30, out_dir)
make_plot("hailmary_density", "density_nearest", tles, target_lats, target_lons, 24.0, 30, out_dir)
print("Done!")
