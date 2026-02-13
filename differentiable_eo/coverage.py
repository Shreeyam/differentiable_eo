"""Differentiable coverage and revisit gap primitives."""

import torch


def soft_coverage(elevation_deg: torch.Tensor, min_el: float, temperature: float) -> torch.Tensor:
    """Soft (differentiable) coverage indicator via sigmoid."""
    return torch.sigmoid((elevation_deg - min_el) / temperature)


def hard_coverage(elevation_deg: torch.Tensor, min_el: float) -> torch.Tensor:
    """Binary coverage indicator (not differentiable)."""
    return (elevation_deg >= min_el).float()


def noisy_or(cov_per_sat: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """P(any satellite covers) = 1 - prod(1 - cov_i) along given dimension."""
    return 1.0 - torch.prod(1.0 - cov_per_sat, dim=dim)


def leaky_integrator_step(gap: torch.Tensor, dt: float, coverage: torch.Tensor) -> torch.Tensor:
    """One step of leaky integrator for revisit gap tracking.

    gap[t] = (gap[t-1] + dt) * (1 - coverage[t])
    Resets toward 0 when covered, grows by dt when not.
    """
    return (gap + dt) * (1.0 - coverage)


def logsumexp_soft_max(gap_stack: torch.Tensor, tau: float, dim: int = 0) -> torch.Tensor:
    """Differentiable soft-max via LogSumExp.

    Approximates max(gap) as tau * log(sum(exp(gap/tau))).
    """
    return tau * torch.logsumexp(gap_stack / tau, dim=dim)
