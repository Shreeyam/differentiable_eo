"""Loss computation: propagation, coverage, and revisit objectives."""

import torch
import dsgp4

from .coordinates import teme_to_ecef, compute_elevation
from .coverage import soft_coverage, hard_coverage, noisy_or, leaky_integrator_step, logsumexp_soft_max


def propagate_constellation(tles: list, tsinces: torch.Tensor) -> torch.Tensor:
    """
    Propagate all TLEs and return stacked positions.

    Returns:
        [N_sat, T, 3] position tensor in TEME frame (km)
    """
    n_time = len(tsinces)
    all_positions = []
    for tle in tles:
        try:
            state = dsgp4.propagate(tle, tsinces)
            all_positions.append(state[:, 0, :])
        except Exception:
            all_positions.append(torch.zeros(n_time, 3))
    return torch.stack(all_positions)


def compute_loss(tles: list, tsinces: torch.Tensor, gmst_array: list,
                 ground_ecef: torch.Tensor, min_el: float, softness: float,
                 revisit_tau: float, revisit_weight: float,
                 ground_weights: torch.Tensor = None,
                 revisit_reduce: str = 'mean',
                 revisit_spatial_tau: float = None):
    """
    Propagate constellation and compute differentiable coverage + revisit loss.

    Args:
        tles: list of initialized TLE objects
        tsinces: [T] time array in minutes
        gmst_array: list of GMST values (one per time step)
        ground_ecef: [N_ground, 3] ground point positions
        min_el: minimum elevation angle (deg)
        softness: sigmoid temperature (deg)
        revisit_tau: LogSumExp temperature for temporal max (minutes)
        revisit_weight: weight of revisit loss vs coverage loss
        ground_weights: optional [N_ground] weights for non-uniform targets
        revisit_reduce: 'mean' for weighted mean over ground points,
                        'max' for LogSumExp soft-max (minimax) over ground points
        revisit_spatial_tau: LogSumExp temperature for spatial max (minutes),
                             only used when revisit_reduce='max'. Defaults to revisit_tau.

    Returns:
        (loss, coverage_fraction, revisit_gap_minutes)
    """
    n_time = len(tsinces)
    n_ground = ground_ecef.shape[0]
    dt = (tsinces[-1] - tsinces[0]).item() / (n_time - 1)

    all_positions = propagate_constellation(tles, tsinces)

    total_coverage = torch.tensor(0.0)
    gap = torch.zeros(n_ground)
    all_gaps = []

    for t_idx in range(n_time):
        sat_teme = all_positions[:, t_idx, :]
        sat_ecef = teme_to_ecef(sat_teme, gmst_array[t_idx])
        elevation = compute_elevation(sat_ecef, ground_ecef)
        cov = soft_coverage(elevation, min_el, softness)
        any_covered = noisy_or(cov, dim=0)

        if ground_weights is not None:
            total_coverage = total_coverage + (any_covered * ground_weights).sum()
        else:
            total_coverage = total_coverage + any_covered.sum()

        gap = leaky_integrator_step(gap, dt, any_covered)
        all_gaps.append(gap.clone())

    if ground_weights is not None:
        coverage_frac = total_coverage / (n_time * ground_weights.sum())
    else:
        coverage_frac = total_coverage / (n_time * n_ground)

    gap_stack = torch.stack(all_gaps, dim=0)
    soft_max_gap = logsumexp_soft_max(gap_stack, revisit_tau, dim=0)  # [N_ground]

    if revisit_reduce == 'max':
        # Minimax: soft-max over ground points too
        spatial_tau = revisit_spatial_tau if revisit_spatial_tau is not None else revisit_tau
        revisit_metric = logsumexp_soft_max(soft_max_gap.unsqueeze(0), spatial_tau, dim=1).squeeze()
    elif ground_weights is not None:
        revisit_metric = (soft_max_gap * ground_weights).sum() / ground_weights.sum()
    else:
        revisit_metric = soft_max_gap.mean()

    loss = -coverage_frac + revisit_weight * revisit_metric
    return loss, coverage_frac, revisit_metric


def compute_hard_metrics(tles: list, tsinces: torch.Tensor, gmst_array: list,
                         ground_ecef: torch.Tensor, min_el: float,
                         ground_weights: torch.Tensor = None,
                         revisit_reduce: str = 'mean'):
    """
    Compute exact discrete coverage and revisit metrics (non-differentiable).

    Args:
        ground_weights: optional [N_ground] cos(lat) area weights
        revisit_reduce: 'mean' for weighted mean, 'max' for worst-case ground point

    Returns:
        (hard_coverage_fraction, hard_revisit_minutes)
    """
    n_time = len(tsinces)
    n_ground = ground_ecef.shape[0]
    dt = (tsinces[-1] - tsinces[0]).item() / (n_time - 1)

    all_positions = propagate_constellation(tles, tsinces)

    total_coverage = 0.0
    gap = torch.zeros(n_ground)
    max_gap = torch.zeros(n_ground)

    for t_idx in range(n_time):
        sat_teme = all_positions[:, t_idx, :]
        sat_ecef = teme_to_ecef(sat_teme, gmst_array[t_idx])
        elevation = compute_elevation(sat_ecef, ground_ecef)
        cov = hard_coverage(elevation, min_el)
        any_covered = (cov.sum(dim=0) > 0).float()
        if ground_weights is not None:
            total_coverage += (any_covered * ground_weights).sum().item()
        else:
            total_coverage += any_covered.sum().item()
        gap = (gap + dt) * (1.0 - any_covered)
        max_gap = torch.maximum(max_gap, gap)

    if ground_weights is not None:
        hard_cov_frac = total_coverage / (n_time * ground_weights.sum().item())
    else:
        hard_cov_frac = total_coverage / (n_time * n_ground)

    if revisit_reduce == 'max':
        hard_revisit = max_gap.max().item()
    elif ground_weights is not None:
        hard_revisit = (max_gap * ground_weights).sum().item() / ground_weights.sum().item()
    else:
        hard_revisit = max_gap.mean().item()
    return hard_cov_frac, hard_revisit
