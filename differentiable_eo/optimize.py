"""ConstellationOptimizer: main optimization loop with reparameterized constraints."""

import math
import torch
import dsgp4
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable

from .config import Config
from .constraints import ReparameterizedElements, alt_from_no_kozai
from .coordinates import make_ground_grid, make_ground_grid_with_weights, make_gmst_array
from .tle_utils import make_constellation, extract_elements, update_tle_from_elements
from .objective import compute_loss, compute_hard_metrics


@dataclass
class OptimizationResult:
    """Results from a constellation optimization run."""
    loss_history: list = field(default_factory=list)
    cov_history: list = field(default_factory=list)
    revisit_history: list = field(default_factory=list)
    hard_cov_history: list = field(default_factory=list)
    hard_revisit_history: list = field(default_factory=list)
    hard_eval_iters: list = field(default_factory=list)
    initial_tles: list = field(default_factory=list)
    final_tles: list = field(default_factory=list)
    config: Optional[Config] = None


class ConstellationOptimizer:
    """
    Gradient-based constellation optimizer using dSGP4 with reparameterized constraints.

    Usage:
        config = Config()
        opt = ConstellationOptimizer(config)
        result = opt.run()
    """

    def __init__(self, config: Config):
        self.config = config
        self._setup()

    def _setup(self):
        cfg = self.config

        # Ground grid with cos(lat) area weights
        self.ground_ecef, self.ground_weights = make_ground_grid_with_weights(
            cfg.n_lat, cfg.n_lon, cfg.lat_bounds_deg)
        self.n_ground = self.ground_ecef.shape[0]

        # Time array
        prop_min = cfg.prop_duration_hours * 60
        self.tsinces = torch.linspace(0, prop_min, cfg.n_time_steps)
        self.gmst_array = make_gmst_array(self.tsinces)

        # Create initial constellation
        self.tles = make_constellation(
            cfg.n_planes, cfg.n_sats_per_plane,
            cfg.initial_inc_deg, cfg.initial_raan_offsets_deg,
            cfg.target_alt_km,
        )
        # Save a separate copy for comparison
        self.initial_tles = make_constellation(
            cfg.n_planes, cfg.n_sats_per_plane,
            cfg.initial_inc_deg, cfg.initial_raan_offsets_deg,
            cfg.target_alt_km,
        )

        # Initialize TLEs to get internal element values
        for tle in self.tles:
            dsgp4.initialize_tle(tle, with_grad=False)

        # Create reparameterized elements for each satellite
        self.reparam_elements = []
        for tle in self.tles:
            initial_elem = extract_elements(tle)
            reparam = ReparameterizedElements(initial_elem, cfg.parameter_specs)
            self.reparam_elements.append(reparam)

        # Create optimizer over z-parameters
        z_params = [re.optimizer_param for re in self.reparam_elements if re.n_free > 0]
        self.optimizer = torch.optim.AdamW(z_params, lr=cfg.lr)

    def step(self) -> dict:
        """
        Execute one optimization step.

        Returns dict with: loss, coverage_pct, revisit_min
        """
        cfg = self.config
        self.optimizer.zero_grad()

        # Optionally randomize Earth orientation to avoid systematic bias
        if cfg.randomize_gmst:
            gmst_offset = torch.rand(1).item() * 2 * math.pi
            gmst_array = [g + gmst_offset for g in self.gmst_array]
        else:
            gmst_array = self.gmst_array

        # 1. Map z -> elements -> write to TLEs -> initialize for autograd
        tle_elements_list = []
        for i, tle in enumerate(self.tles):
            elements = self.reparam_elements[i].to_elements()
            update_tle_from_elements(tle, elements)

        for tle in self.tles:
            tle_elements_list.append(dsgp4.initialize_tle(tle, with_grad=True))

        # 2. Forward pass
        loss, cov_frac, mean_revisit = compute_loss(
            self.tles, self.tsinces, gmst_array, self.ground_ecef,
            min_el=cfg.min_elevation_deg, softness=cfg.softness_deg,
            revisit_tau=cfg.revisit_logsumexp_temp, revisit_weight=cfg.revisit_weight,
            ground_weights=self.ground_weights,
            revisit_reduce=cfg.revisit_reduce,
            revisit_spatial_tau=cfg.revisit_spatial_tau,
        )

        # 3. Backward through dsgp4's ephemeral graph
        loss.backward()

        # 4. Chain rule: map ephemeral gradients through sigmoid to z-space
        for i in range(len(self.tles)):
            ephemeral_grad = tle_elements_list[i].grad
            if ephemeral_grad is not None:
                self.reparam_elements[i].compute_z_grad(ephemeral_grad)

        # 5. Adam step on z-parameters
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'coverage_pct': cov_frac.item() * 100,
            'revisit_min': mean_revisit.item(),
        }

    def evaluate_hard(self) -> dict:
        """Compute exact discrete metrics (non-differentiable)."""
        # Re-initialize TLEs for evaluation
        for i, tle in enumerate(self.tles):
            elements = self.reparam_elements[i].to_elements()
            update_tle_from_elements(tle, elements)
        for tle in self.tles:
            dsgp4.initialize_tle(tle, with_grad=False)

        with torch.no_grad():
            h_cov, h_rev = compute_hard_metrics(
                self.tles, self.tsinces, self.gmst_array, self.ground_ecef,
                min_el=self.config.min_elevation_deg,
                ground_weights=self.ground_weights,
                revisit_reduce=self.config.revisit_reduce,
            )
        return {'hard_coverage_pct': h_cov * 100, 'hard_revisit_min': h_rev}

    def get_current_elements(self) -> list:
        """Return current element tensors for all satellites."""
        return [re.to_elements().detach() for re in self.reparam_elements]

    def get_current_altitudes(self) -> list:
        """Return current altitudes (km) for all satellites."""
        elements = self.get_current_elements()
        return [alt_from_no_kozai(e[7].item()) for e in elements]

    def summary(self) -> str:
        """Print summary of current constellation state."""
        cfg = self.config
        lines = []
        elements = self.get_current_elements()
        for p in range(cfg.n_planes):
            idx = p * cfg.n_sats_per_plane
            e = elements[idx]
            alt = alt_from_no_kozai(e[7].item())
            lines.append(
                f"  Plane {p}: inc={math.degrees(e[5].item()):.2f}deg, "
                f"RAAN={math.degrees(e[8].item()) % 360:.2f}deg, "
                f"alt={alt:.1f}km"
            )
        return "\n".join(lines)

    def run(self, callback: Callable = None) -> OptimizationResult:
        """
        Run full optimization loop.

        Args:
            callback: optional function called each iteration with (iteration, step_result, optimizer)

        Returns:
            OptimizationResult with full histories
        """
        cfg = self.config
        result = OptimizationResult(config=cfg, initial_tles=self.initial_tles)

        print("=" * 65)
        print("  Differentiable Constellation Optimization via dSGP4")
        print("=" * 65)
        print(f"\nConstellation: {cfg.n_sats} sats ({cfg.n_planes}P x {cfg.n_sats_per_plane}S)")
        print(f"Grid: {cfg.n_lat}x{cfg.n_lon}={self.n_ground} pts | {cfg.n_time_steps} steps over {cfg.prop_duration_hours}h")
        print(f"Altitude: {cfg.target_alt_km} km | Initial inc: {cfg.initial_inc_deg}deg")

        # Free params summary
        from .constraints import BoxConstraint, UnboundedConstraint
        from .constants import ELEMENT_NAMES
        free_names = [ELEMENT_NAMES[idx] for idx, spec in cfg.parameter_specs.items()
                      if isinstance(spec, (BoxConstraint, UnboundedConstraint))]
        print(f"Free parameters: {', '.join(free_names)}")

        # Initial metrics
        for tle in self.tles:
            dsgp4.initialize_tle(tle, with_grad=False)
        with torch.no_grad():
            _, init_cov, init_rev = compute_loss(
                self.tles, self.tsinces, self.gmst_array, self.ground_ecef,
                cfg.min_elevation_deg, cfg.softness_deg,
                cfg.revisit_logsumexp_temp, cfg.revisit_weight,
                ground_weights=self.ground_weights,
                revisit_reduce=cfg.revisit_reduce,
                revisit_spatial_tau=cfg.revisit_spatial_tau,
            )
        print(f"\nInitial coverage: {init_cov.item()*100:.2f}%")
        print(f"Initial revisit:  {init_rev.item():.1f} min")

        print(f"\n{'It':>4} {'Cov%':>7} {'Revisit':>8} {'HardCov%':>9} {'HardRev':>8} {'Loss':>8}")
        print("-" * 52)

        for iteration in range(cfg.n_iterations):
            step_result = self.step()

            result.loss_history.append(step_result['loss'])
            result.cov_history.append(step_result['coverage_pct'])
            result.revisit_history.append(step_result['revisit_min'])

            if callback:
                callback(iteration, step_result, self)

            if iteration % 20 == 0 or iteration == cfg.n_iterations - 1:
                hard = self.evaluate_hard()
                result.hard_cov_history.append(hard['hard_coverage_pct'])
                result.hard_revisit_history.append(hard['hard_revisit_min'])
                result.hard_eval_iters.append(iteration)

                print(f"{iteration:4d} {step_result['coverage_pct']:6.2f}% "
                      f"{step_result['revisit_min']:7.1f}m "
                      f"{hard['hard_coverage_pct']:8.2f}% "
                      f"{hard['hard_revisit_min']:7.1f}m "
                      f"{step_result['loss']:7.4f}")

        # Final summary
        print(f"\n{'=' * 52}")
        print(f"Coverage:  {result.cov_history[0]:.2f}% -> {result.cov_history[-1]:.2f}%")
        print(f"Revisit:   {result.revisit_history[0]:.1f} min -> {result.revisit_history[-1]:.1f} min")
        print(f"Hard cov:  {result.hard_cov_history[0]:.2f}% -> {result.hard_cov_history[-1]:.2f}%")
        print(f"Hard rev:  {result.hard_revisit_history[0]:.1f} min -> {result.hard_revisit_history[-1]:.1f} min")
        print(f"\nFinal constellation:")
        print(self.summary())

        # Write final state to TLEs for result
        for i, tle in enumerate(self.tles):
            elements = self.reparam_elements[i].to_elements()
            update_tle_from_elements(tle, elements)
        result.final_tles = self.tles

        return result
