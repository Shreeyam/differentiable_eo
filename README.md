# Differentiable Earth Observation

Differentiable satellite constellation configuration via relaxed coverage and revisit objectives. Constellation geometry is optimized end-to-end by gradient descent through a differentiable SGP4 orbit propagator ([dSGP4](https://github.com/esa/dSGP4)).

**Paper:** [Differentiable Satellite Constellation Configuration via Relaxed Coverage and Revisit Objectives](https://arxiv.org/abs/2604.19062) (Kacker and Cahoy).

![Constellation optimization](experiments/exp2_animation.gif)

## Installation

```bash
pip3 install -e .
```

Requires Python >= 3.9, `torch`, `dsgp4`, `numpy`, `matplotlib`.

## Usage

```python
from differentiable_eo import Config, ConstellationOptimizer

config = Config(
    n_planes=4,
    n_sats_per_plane=3,
    target_alt_km=550.0,
    prop_duration_hours=24.0,
    n_iterations=2000,
)

opt = ConstellationOptimizer(config)
result = opt.run()

print(result.final_tles)
```

`Config` exposes the constellation size, propagation horizon, ground grid, coverage model (elevation threshold and sigmoid softness), revisit model (LogSumExp temperature, weight, mean/minimax reduction), and per-element constraint specs.

## Package layout

`differentiable_eo/`

- `config.py` — `Config` dataclass of all run parameters.
- `constraints.py` — reparameterized element constraints (fixed, box, periapsis/apoapsis).
- `coordinates.py` — TEME/ECEF transforms, elevation, ground grids, GMST.
- `coverage.py` — the four relaxations (`soft_coverage`, `noisy_or`, `leaky_integrator_step`, `logsumexp_soft_max`).
- `objective.py` — propagation and differentiable coverage/revisit loss.
- `optimize.py` — `ConstellationOptimizer` and `OptimizationResult`.
- `tle_utils.py` — TLE construction and element manipulation.
- `visualization.py`, `globe.py` — result plots and globe rendering.

## Experiments

`experiments/` reproduces the paper figures:

- `exp1_gradient_validation.py` — finite-difference check of analytic gradients.
- `exp2_walker_recovery.py` — Walker-Delta recovery from a suboptimal start.
- `exp2_baselines.py` — comparison against metaheuristic baselines.
- `exp3_loss_landscape.py`, `exp3_weighted_europe.py` — loss landscapes and regional targeting.
- `exp4_ablations.py` — relaxation and hyperparameter ablations.

## Citation

```bibtex
@article{kacker2026differentiable,
  title   = {Differentiable Satellite Constellation Configuration via Relaxed Coverage and Revisit Objectives},
  author  = {Kacker, Shreeyam and Cahoy, Kerri},
  journal = {arXiv preprint arXiv:2604.19062},
  year    = {2026}
}
```
