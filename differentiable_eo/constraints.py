"""
Reparameterized constraint system for TLE elements.

Instead of clamping parameters after each optimizer step, we reparameterize
bounded parameters via sigmoid so the optimizer works in unconstrained z-space:

    element = lower + (upper - lower) * sigmoid(z)

This makes constraints automatic (sigmoid output is always in (0,1)) and
provides smooth gradients everywhere (no gradient clipping at bounds).
"""

import math
import torch

from .constants import (
    N_ELEMENTS, IDX_BSTAR, IDX_NDOT, IDX_NDDOT, IDX_ECCO,
    IDX_ARGPO, IDX_INCLO, IDX_MO, IDX_NO_KOZAI, IDX_NODEO,
    MU_EARTH, R_EARTH,
)


class FixedConstraint:
    """Parameter held at its initial value (no optimization)."""
    pass


class UnboundedConstraint:
    """Parameter optimized directly without reparameterization. Use for periodic angles."""
    pass


class BoxConstraint:
    """Parameter bounded within [lower, upper] via sigmoid reparameterization."""

    def __init__(self, lower: float, upper: float):
        if lower >= upper:
            raise ValueError(f"lower ({lower}) must be < upper ({upper})")
        self.lower = lower
        self.upper = upper


def no_kozai_from_alt(alt_km: float) -> float:
    """Convert altitude (km) to no_kozai (rad/min) for SGP4."""
    a = R_EARTH + alt_km
    n_rad_s = math.sqrt(MU_EARTH / a**3)
    return n_rad_s * 60.0  # rad/min


def alt_from_no_kozai(no_kozai_rad_per_min: float) -> float:
    """Convert no_kozai (rad/min) to altitude (km)."""
    n_rad_s = no_kozai_rad_per_min / 60.0
    if n_rad_s <= 0:
        return 0.0
    return (MU_EARTH / n_rad_s**2) ** (1.0 / 3.0) - R_EARTH


def default_parameter_specs() -> dict:
    """Default specs: only inc, ma, raan are free (backward compatible with original)."""
    return {
        IDX_BSTAR: FixedConstraint(),
        IDX_NDOT: FixedConstraint(),
        IDX_NDDOT: FixedConstraint(),
        IDX_ECCO: FixedConstraint(),
        IDX_ARGPO: FixedConstraint(),
        IDX_INCLO: BoxConstraint(0.05, 3.09),        # ~3 to ~177 deg
        IDX_MO: UnboundedConstraint(),
        IDX_NO_KOZAI: FixedConstraint(),
        IDX_NODEO: UnboundedConstraint(),
    }


def specs_with_altitude(alt_bounds_km=(400.0, 600.0), **overrides) -> dict:
    """Like default_parameter_specs but also frees altitude (no_kozai) within bounds."""
    specs = default_parameter_specs()
    lo_nk = no_kozai_from_alt(alt_bounds_km[1])  # higher alt = lower n
    hi_nk = no_kozai_from_alt(alt_bounds_km[0])  # lower alt = higher n
    specs[IDX_NO_KOZAI] = BoxConstraint(lo_nk, hi_nk)
    for idx, constraint in overrides.items():
        specs[idx] = constraint
    return specs


class ReparameterizedElements:
    """
    Manages a set of TLE elements with sigmoid reparameterization for bounded params.

    For each BoxConstraint element:
        z = logit((x - lo) / (hi - lo))     (init: map element to unconstrained space)
        x = lo + (hi - lo) * sigmoid(z)      (forward: map back to element space)

    Fixed elements are stored as constants and not optimized.
    The optimizer tracks `z` (the unconstrained parameters).
    """

    def __init__(self, initial_elements: torch.Tensor, specs: dict):
        """
        Args:
            initial_elements: [9] tensor of initial TLE element values
            specs: dict mapping element index -> FixedConstraint or BoxConstraint
        """
        assert initial_elements.shape == (N_ELEMENTS,)
        self.specs = specs
        self.fixed_values = initial_elements.detach().clone()

        # Build z tensor for free parameters (BoxConstraint + UnboundedConstraint)
        self.free_indices = []
        self.free_types = []  # 'box' or 'unbounded'
        self.bounds_lower = []
        self.bounds_upper = []
        z_values = []

        for idx in range(N_ELEMENTS):
            spec = specs.get(idx, FixedConstraint())
            if isinstance(spec, BoxConstraint):
                self.free_indices.append(idx)
                self.free_types.append('box')
                self.bounds_lower.append(spec.lower)
                self.bounds_upper.append(spec.upper)
                # Map initial value to unconstrained space via logit
                x = initial_elements[idx].item()
                x_clamped = max(spec.lower + 1e-6, min(x, spec.upper - 1e-6))
                t = (x_clamped - spec.lower) / (spec.upper - spec.lower)
                z_values.append(math.log(t / (1.0 - t)))  # logit
            elif isinstance(spec, UnboundedConstraint):
                self.free_indices.append(idx)
                self.free_types.append('unbounded')
                self.bounds_lower.append(0.0)  # unused placeholder
                self.bounds_upper.append(0.0)  # unused placeholder
                z_values.append(initial_elements[idx].item())

        self.n_free = len(self.free_indices)
        if self.n_free > 0:
            self._z = torch.tensor(z_values, dtype=torch.float64, requires_grad=True)
            self._lower = torch.tensor(self.bounds_lower, dtype=torch.float64)
            self._upper = torch.tensor(self.bounds_upper, dtype=torch.float64)
        else:
            self._z = torch.tensor([], dtype=torch.float64, requires_grad=True)
            self._lower = torch.tensor([], dtype=torch.float64)
            self._upper = torch.tensor([], dtype=torch.float64)

    @property
    def optimizer_param(self) -> torch.Tensor:
        """The unconstrained parameter tensor that the optimizer should track."""
        return self._z

    def to_elements(self) -> torch.Tensor:
        """Map from z-space back to element-space. Returns [9] tensor."""
        elements = self.fixed_values.clone()
        if self.n_free > 0:
            sig = torch.sigmoid(self._z)
            for i, idx in enumerate(self.free_indices):
                if self.free_types[i] == 'box':
                    elements[idx] = self._lower[i] + (self._upper[i] - self._lower[i]) * sig[i]
                else:  # unbounded
                    elements[idx] = self._z[i]
        return elements

    def compute_z_grad(self, ephemeral_grad: torch.Tensor):
        """
        Manually compute gradient for z via chain rule.

        For BoxConstraint:  dL/dz = dL/dx * (hi - lo) * sig(z) * (1 - sig(z))
        For UnboundedConstraint: dL/dz = dL/dx  (identity)

        Args:
            ephemeral_grad: [9] gradient tensor from dsgp4's ephemeral computation graph
        """
        if self.n_free == 0:
            return

        sig = torch.sigmoid(self._z.detach())
        grad = torch.zeros(self.n_free, dtype=torch.float64)

        for i, idx in enumerate(self.free_indices):
            dl_dx = ephemeral_grad[idx]
            if self.free_types[i] == 'box':
                dx_dz = (self._upper[i] - self._lower[i]) * sig[i] * (1.0 - sig[i])
                grad[i] = dl_dx * dx_dz
            else:  # unbounded: gradient passes through directly
                grad[i] = dl_dx

        self._z.grad = grad
