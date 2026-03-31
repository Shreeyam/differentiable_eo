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


class PeriapsisApoapsisConstraint:
    """Coupled constraint on eccentricity and altitude via (perigee_alt, excess_alt).

    Parameterizes orbital shape as:
        z_rp  → perigee_alt  = rp_lo + (rp_hi - rp_lo) * sigmoid(z_rp)
        z_exc → excess_alt   = exc_lo + (exc_hi - exc_lo) * sigmoid(z_exc)
        apogee_alt = perigee_alt + excess_alt

    Then converts to TLE elements:
        a = R_earth + (perigee_alt + apogee_alt) / 2
        e = (apogee_alt - perigee_alt) / (2 * a)  ... more precisely (ra - rp) / (ra + rp)
        no_kozai = sqrt(mu / a^3) * 60  [rad/min]

    This guarantees perigee ≥ rp_lo and apogee ≥ perigee (e ≥ 0).

    Replaces both IDX_NO_KOZAI and IDX_ECCO in the element vector.
    """

    def __init__(self, perigee_bounds_km=(400.0, 600.0), excess_bounds_km=(0.0, 1500.0)):
        self.rp_lo, self.rp_hi = perigee_bounds_km
        self.exc_lo, self.exc_hi = excess_bounds_km
        # Element indices this constraint controls
        self.element_indices = (IDX_NO_KOZAI, IDX_ECCO)


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
            specs: dict mapping element index -> FixedConstraint, BoxConstraint,
                   UnboundedConstraint, or PeriapsisApoapsisConstraint
        """
        assert initial_elements.shape == (N_ELEMENTS,)
        self.specs = specs
        self.fixed_values = initial_elements.detach().clone()

        # Check for coupled periapsis/apoapsis constraint
        self.coupled = None
        coupled_indices = set()
        for idx, spec in specs.items():
            if isinstance(spec, PeriapsisApoapsisConstraint):
                self.coupled = spec
                coupled_indices.update(spec.element_indices)

        # Build z tensor for free parameters (BoxConstraint + UnboundedConstraint)
        self.free_indices = []
        self.free_types = []  # 'box', 'unbounded', or 'coupled_rp'/'coupled_exc'
        self.bounds_lower = []
        self.bounds_upper = []
        z_values = []

        for idx in range(N_ELEMENTS):
            if idx in coupled_indices:
                continue  # handled separately below
            spec = specs.get(idx, FixedConstraint())
            if isinstance(spec, BoxConstraint):
                self.free_indices.append(idx)
                self.free_types.append('box')
                self.bounds_lower.append(spec.lower)
                self.bounds_upper.append(spec.upper)
                x = initial_elements[idx].item()
                x_clamped = max(spec.lower + 1e-6, min(x, spec.upper - 1e-6))
                t = (x_clamped - spec.lower) / (spec.upper - spec.lower)
                z_values.append(math.log(t / (1.0 - t)))
            elif isinstance(spec, UnboundedConstraint):
                self.free_indices.append(idx)
                self.free_types.append('unbounded')
                self.bounds_lower.append(0.0)
                self.bounds_upper.append(0.0)
                z_values.append(initial_elements[idx].item())

        # Add coupled periapsis/apoapsis z-parameters at the end
        if self.coupled is not None:
            c = self.coupled
            # Derive initial perigee/apogee from current no_kozai and ecco
            nk = initial_elements[IDX_NO_KOZAI].item()
            ecc = initial_elements[IDX_ECCO].item()
            a_km = (MU_EARTH / (nk / 60.0) ** 2) ** (1.0 / 3.0)
            rp_km = a_km * (1.0 - ecc) - R_EARTH
            ra_km = a_km * (1.0 + ecc) - R_EARTH
            excess_km = ra_km - rp_km

            # z for perigee altitude
            self.free_indices.append('coupled_rp')
            self.free_types.append('coupled_rp')
            self.bounds_lower.append(c.rp_lo)
            self.bounds_upper.append(c.rp_hi)
            rp_clamped = max(c.rp_lo + 1e-3, min(rp_km, c.rp_hi - 1e-3))
            t_rp = (rp_clamped - c.rp_lo) / (c.rp_hi - c.rp_lo)
            z_values.append(math.log(t_rp / (1.0 - t_rp)))

            # z for excess altitude (apogee - perigee)
            self.free_indices.append('coupled_exc')
            self.free_types.append('coupled_exc')
            self.bounds_lower.append(c.exc_lo)
            self.bounds_upper.append(c.exc_hi)
            exc_clamped = max(c.exc_lo + 1e-3, min(excess_km, c.exc_hi - 1e-3))
            t_exc = (exc_clamped - c.exc_lo) / (c.exc_hi - c.exc_lo)
            z_values.append(math.log(t_exc / (1.0 - t_exc)))

            self._coupled_z_start = len(z_values) - 2  # index into _z

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
                elif self.free_types[i] == 'unbounded':
                    elements[idx] = self._z[i]
                # coupled types handled below

        # Coupled periapsis/apoapsis → (no_kozai, ecco)
        if self.coupled is not None:
            sig = torch.sigmoid(self._z)
            i_rp = self._coupled_z_start
            i_exc = self._coupled_z_start + 1
            c = self.coupled
            rp_km = c.rp_lo + (c.rp_hi - c.rp_lo) * sig[i_rp]
            exc_km = c.exc_lo + (c.exc_hi - c.exc_lo) * sig[i_exc]
            ra_km = rp_km + exc_km
            # Convert to orbital elements
            rp = R_EARTH + rp_km
            ra = R_EARTH + ra_km
            a = (rp + ra) / 2.0
            e = (ra - rp) / (ra + rp)
            nk = torch.sqrt(torch.tensor(MU_EARTH, dtype=torch.float64) / a ** 3) * 60.0
            elements[IDX_NO_KOZAI] = nk
            elements[IDX_ECCO] = e

        return elements

    def compute_z_grad(self, ephemeral_grad: torch.Tensor):
        """
        Manually compute gradient for z via chain rule.

        For BoxConstraint:  dL/dz = dL/dx * (hi - lo) * sig(z) * (1 - sig(z))
        For UnboundedConstraint: dL/dz = dL/dx  (identity)
        For coupled: chain rule through (rp, exc) → (no_kozai, ecco) Jacobian

        Args:
            ephemeral_grad: [9] gradient tensor from dsgp4's ephemeral computation graph
        """
        if self.n_free == 0:
            return

        sig = torch.sigmoid(self._z.detach())
        grad = torch.zeros(self.n_free, dtype=torch.float64)

        for i, idx in enumerate(self.free_indices):
            if self.free_types[i] == 'box':
                dl_dx = ephemeral_grad[idx]
                dx_dz = (self._upper[i] - self._lower[i]) * sig[i] * (1.0 - sig[i])
                grad[i] = dl_dx * dx_dz
            elif self.free_types[i] == 'unbounded':
                grad[i] = ephemeral_grad[idx]
            # coupled types handled below

        # Coupled: chain rule dL/d(z_rp, z_exc) via Jacobian
        if self.coupled is not None:
            c = self.coupled
            i_rp = self._coupled_z_start
            i_exc = self._coupled_z_start + 1
            s_rp = sig[i_rp].item()
            s_exc = sig[i_exc].item()

            # Current values
            rp_km = c.rp_lo + (c.rp_hi - c.rp_lo) * s_rp
            exc_km = c.exc_lo + (c.exc_hi - c.exc_lo) * s_exc
            ra_km = rp_km + exc_km
            rp = R_EARTH + rp_km
            ra = R_EARTH + ra_km
            a = (rp + ra) / 2.0

            # Upstream gradients from TLE elements
            dl_dnk = ephemeral_grad[IDX_NO_KOZAI].item()
            dl_de = ephemeral_grad[IDX_ECCO].item()

            # d(no_kozai)/d(a) = -1.5 * sqrt(mu) * a^(-5/2) * 60
            dnk_da = -1.5 * math.sqrt(MU_EARTH) * a ** (-2.5) * 60.0
            # d(a)/d(rp) = 0.5,  d(a)/d(ra) = 0.5
            # d(e)/d(rp) = -2*ra / (ra+rp)^2,  d(e)/d(ra) = 2*rp / (ra+rp)^2
            de_drp = -2.0 * ra / (ra + rp) ** 2
            de_dra = 2.0 * rp / (ra + rp) ** 2

            # d(rp_km)/d(z_rp) = (rp_hi - rp_lo) * s_rp * (1 - s_rp)
            drp_dz = (c.rp_hi - c.rp_lo) * s_rp * (1.0 - s_rp)
            # d(exc_km)/d(z_exc) = (exc_hi - exc_lo) * s_exc * (1 - s_exc)
            dexc_dz = (c.exc_hi - c.exc_lo) * s_exc * (1.0 - s_exc)

            # rp = R_earth + rp_km, so d(rp)/d(rp_km) = 1
            # ra = R_earth + rp_km + exc_km, so d(ra)/d(rp_km) = 1, d(ra)/d(exc_km) = 1

            # dL/d(rp_km) = dL/dnk * dnk/da * 0.5 + dL/de * de_drp
            #             + dL/dnk * dnk/da * 0.5 + dL/de * de_dra  (via ra depends on rp_km too)
            dl_drp_km = dl_dnk * dnk_da * 0.5 + dl_de * de_drp \
                      + dl_dnk * dnk_da * 0.5 + dl_de * de_dra  # ra also depends on rp_km
            # Simplify: dl_drp_km = dl_dnk * dnk_da + dl_de * (de_drp + de_dra)
            dl_drp_km = dl_dnk * dnk_da + dl_de * (de_drp + de_dra)

            # dL/d(exc_km) = dL/dnk * dnk/da * 0.5 + dL/de * de_dra
            dl_dexc_km = dl_dnk * dnk_da * 0.5 + dl_de * de_dra

            grad[i_rp] = dl_drp_km * drp_dz
            grad[i_exc] = dl_dexc_km * dexc_dz

        self._z.grad = grad
