"""Loss computation: propagation, coverage, and revisit objectives."""

import torch
import dsgp4

from .coordinates import teme_to_ecef, teme_to_ecef_batch, compute_elevation, compute_elevation_batch
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


def compute_loss(tles: list, tsinces: torch.Tensor, gmst_array,
                 ground_ecef: torch.Tensor, min_el: float, softness: float,
                 revisit_tau: float, revisit_weight: float,
                 ground_weights: torch.Tensor = None,
                 revisit_reduce: str = 'mean',
                 revisit_spatial_tau: float = None,
                 ground_unit: torch.Tensor = None,
                 revisit_softness: float = None):
    """
    Propagate constellation and compute differentiable coverage + revisit loss.

    Vectorized: ECEF transform, elevation, sigmoid, and noisy-OR are batched
    across all timesteps. Only the leaky integrator loops over time (sequential).

    Args:
        tles: list of initialized TLE objects
        tsinces: [T] time array in minutes
        gmst_array: [T] GMST tensor or list of GMST values
        ground_ecef: [N_ground, 3] ground point positions
        min_el: minimum elevation angle (deg)
        softness: sigmoid temperature (deg) for coverage objective
        revisit_tau: LogSumExp temperature for temporal max (minutes)
        revisit_weight: weight of revisit loss vs coverage loss
        ground_weights: optional [N_ground] weights for non-uniform targets
        revisit_reduce: 'mean' for weighted mean over ground points,
                        'max' for LogSumExp soft-max (minimax) over ground points
        revisit_spatial_tau: LogSumExp temperature for spatial max (minutes),
                             only used when revisit_reduce='max'. Defaults to revisit_tau.
        ground_unit: optional [N_ground, 3] precomputed unit vectors for ground points
        revisit_softness: separate sigmoid temperature (deg) for revisit integrator.
                          Defaults to softness if not specified.

    Returns:
        (loss, coverage_fraction, revisit_gap_minutes)
    """
    n_time = len(tsinces)
    n_ground = ground_ecef.shape[0]
    dt = (tsinces[-1] - tsinces[0]).item() / (n_time - 1)

    # Propagate: [N_sat, T, 3]
    all_positions = propagate_constellation(tles, tsinces)

    # Transpose to [T, N_sat, 3] for batch operations
    all_positions_t = all_positions.transpose(0, 1)

    # Batch ECEF transform: [T, N_sat, 3]
    if isinstance(gmst_array, torch.Tensor):
        gmst_tensor = gmst_array.to(dtype=all_positions_t.dtype)
    else:
        gmst_tensor = torch.tensor(gmst_array, dtype=all_positions_t.dtype)
    all_ecef = teme_to_ecef_batch(all_positions_t, gmst_tensor)

    # Batch elevation: [T, N_sat, N_ground]
    all_elevation = compute_elevation_batch(all_ecef, ground_ecef, ground_unit=ground_unit)

    # Batch soft coverage + noisy-OR: [T, N_ground]
    all_cov = soft_coverage(all_elevation, min_el, softness)  # [T, N_sat, N_ground]
    all_any_covered = noisy_or(all_cov, dim=1)  # [T, N_ground]

    # Coverage: sum over time and ground
    if ground_weights is not None:
        total_coverage = (all_any_covered * ground_weights.unsqueeze(0)).sum()
        coverage_frac = total_coverage / (n_time * ground_weights.sum())
    else:
        total_coverage = all_any_covered.sum()
        coverage_frac = total_coverage / (n_time * n_ground)

    # Revisit integrator: use separate softness if specified
    if revisit_softness is not None and revisit_softness != softness:
        rev_cov = soft_coverage(all_elevation, min_el, revisit_softness)
        rev_any_covered = noisy_or(rev_cov, dim=1)
    else:
        rev_any_covered = all_any_covered

    # Leaky integrator: pre-allocate and write directly
    gap_stack = torch.zeros(n_time, n_ground, dtype=rev_any_covered.dtype)
    gap = torch.zeros(n_ground, dtype=rev_any_covered.dtype)
    for t_idx in range(n_time):
        gap = leaky_integrator_step(gap, dt, rev_any_covered[t_idx])
        gap_stack[t_idx] = gap

    # Revisit: LogSumExp soft-max over time
    soft_max_gap = logsumexp_soft_max(gap_stack, revisit_tau, dim=0)  # [N_ground]

    if revisit_reduce == 'max':
        spatial_tau = revisit_spatial_tau if revisit_spatial_tau is not None else revisit_tau
        revisit_metric = logsumexp_soft_max(soft_max_gap.unsqueeze(0), spatial_tau, dim=1).squeeze()
    elif ground_weights is not None:
        revisit_metric = (soft_max_gap * ground_weights).sum() / ground_weights.sum()
    else:
        revisit_metric = soft_max_gap.mean()

    loss = -coverage_frac + revisit_weight * revisit_metric
    return loss, coverage_frac, revisit_metric


def compute_hard_metrics(tles: list, tsinces: torch.Tensor, gmst_array,
                         ground_ecef: torch.Tensor, min_el: float,
                         ground_weights: torch.Tensor = None,
                         revisit_reduce: str = 'mean',
                         ground_unit: torch.Tensor = None):
    """
    Compute exact discrete coverage and revisit metrics (non-differentiable).

    Vectorized: ECEF transform and elevation batched across timesteps.

    Args:
        ground_weights: optional [N_ground] cos(lat) area weights
        revisit_reduce: 'mean' for weighted mean, 'max' for worst-case ground point
        ground_unit: optional [N_ground, 3] precomputed unit vectors for ground points

    Returns:
        (hard_coverage_fraction, hard_revisit_minutes)
    """
    n_time = len(tsinces)
    n_ground = ground_ecef.shape[0]
    dt = (tsinces[-1] - tsinces[0]).item() / (n_time - 1)

    # Propagate: [N_sat, T, 3] -> [T, N_sat, 3]
    all_positions = propagate_constellation(tles, tsinces)
    all_positions_t = all_positions.transpose(0, 1)

    # Batch ECEF + elevation
    if isinstance(gmst_array, torch.Tensor):
        gmst_tensor = gmst_array.to(dtype=all_positions_t.dtype)
    else:
        gmst_tensor = torch.tensor(gmst_array, dtype=all_positions_t.dtype)
    all_ecef = teme_to_ecef_batch(all_positions_t, gmst_tensor)
    all_elevation = compute_elevation_batch(all_ecef, ground_ecef, ground_unit=ground_unit)

    # Hard coverage: [T, N_sat, N_ground] -> [T, N_ground]
    all_cov = hard_coverage(all_elevation, min_el)  # [T, N_sat, N_ground]
    all_any_covered = (all_cov.sum(dim=1) > 0).float()  # [T, N_ground]

    # Coverage fraction
    if ground_weights is not None:
        total_coverage = (all_any_covered * ground_weights.unsqueeze(0)).sum().item()
        hard_cov_frac = total_coverage / (n_time * ground_weights.sum().item())
    else:
        total_coverage = all_any_covered.sum().item()
        hard_cov_frac = total_coverage / (n_time * n_ground)

    # Gap tracking (sequential)
    gap = torch.zeros(n_ground)
    max_gap = torch.zeros(n_ground)
    for t_idx in range(n_time):
        gap = (gap + dt) * (1.0 - all_any_covered[t_idx])
        max_gap = torch.maximum(max_gap, gap)

    if revisit_reduce == 'max':
        hard_revisit = max_gap.max().item()
    elif ground_weights is not None:
        hard_revisit = (max_gap * ground_weights).sum().item() / ground_weights.sum().item()
    else:
        hard_revisit = max_gap.mean().item()
    return hard_cov_frac, hard_revisit
