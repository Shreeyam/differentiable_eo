"""Coordinate transforms and ground grid generation."""

import math
import torch

from .constants import R_EARTH, EARTH_ROT_RAD_PER_MIN


def gmst_at_epoch(epoch_year: int, epoch_days: float) -> float:
    """Greenwich Mean Sidereal Time at a given epoch (radians)."""
    y = epoch_year
    jd = 367 * y - int(7 * (y + int(10 / 12)) / 4) + int(275 / 9) + epoch_days + 1721013.5
    T = (jd - 2451545.0) / 36525.0
    gmst_deg = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T**2
    return math.radians(gmst_deg % 360)


def teme_to_ecef(positions: torch.Tensor, gmst_rad: float) -> torch.Tensor:
    """Convert TEME positions to ECEF. Supports batched [..., 3] inputs."""
    cos_g = math.cos(gmst_rad)
    sin_g = math.sin(gmst_rad)
    x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
    return torch.stack([cos_g * x + sin_g * y, -sin_g * x + cos_g * y, z], dim=-1)


def teme_to_ecef_batch(positions: torch.Tensor, gmst_rads: torch.Tensor) -> torch.Tensor:
    """
    Batch TEME-to-ECEF conversion for all timesteps at once.

    Args:
        positions: [T, N_sat, 3] TEME positions
        gmst_rads: [T] GMST angles in radians

    Returns:
        [T, N_sat, 3] ECEF positions
    """
    cos_g = torch.cos(gmst_rads)  # [T]
    sin_g = torch.sin(gmst_rads)  # [T]
    x = positions[..., 0]  # [T, N_sat]
    y = positions[..., 1]
    z = positions[..., 2]
    # Broadcast: [T, 1] * [T, N_sat]
    cos_g = cos_g.unsqueeze(1)
    sin_g = sin_g.unsqueeze(1)
    return torch.stack([cos_g * x + sin_g * y, -sin_g * x + cos_g * y, z], dim=-1)


def compute_elevation(sat_ecef: torch.Tensor, ground_ecef: torch.Tensor) -> torch.Tensor:
    """
    Elevation angle (deg) from ground to satellite.

    Args:
        sat_ecef: [N_sat, 3] satellite ECEF positions
        ground_ecef: [N_ground, 3] ground point ECEF positions

    Returns:
        [N_sat, N_ground] elevation angles in degrees
    """
    rel = sat_ecef.unsqueeze(1) - ground_ecef.unsqueeze(0)
    ground_unit = ground_ecef / torch.norm(ground_ecef, dim=-1, keepdim=True)
    rel_dist = torch.norm(rel, dim=-1)
    sin_el = (rel * ground_unit.unsqueeze(0)).sum(dim=-1) / (rel_dist + 1e-10)
    return torch.rad2deg(torch.asin(torch.clamp(sin_el, -1.0, 1.0)))


def compute_elevation_batch(sat_ecef: torch.Tensor, ground_ecef: torch.Tensor,
                            ground_unit: torch.Tensor = None) -> torch.Tensor:
    """
    Batch elevation angle computation for all timesteps at once.

    Args:
        sat_ecef: [T, N_sat, 3] satellite ECEF positions
        ground_ecef: [N_ground, 3] ground point ECEF positions
        ground_unit: [N_ground, 3] precomputed unit vectors (optional, avoids recomputation)

    Returns:
        [T, N_sat, N_ground] elevation angles in degrees
    """
    # sat_ecef: [T, N_sat, 1, 3] - ground_ecef: [1, 1, N_ground, 3]
    rel = sat_ecef.unsqueeze(2) - ground_ecef.unsqueeze(0).unsqueeze(0)  # [T, N_sat, N_ground, 3]
    if ground_unit is None:
        ground_unit = ground_ecef / torch.norm(ground_ecef, dim=-1, keepdim=True)
    rel_dist = torch.norm(rel, dim=-1)  # [T, N_sat, N_ground]
    # dot product: rel . ground_unit
    sin_el = (rel * ground_unit.unsqueeze(0).unsqueeze(0)).sum(dim=-1) / (rel_dist + 1e-10)
    return torch.rad2deg(torch.asin(torch.clamp(sin_el, -1.0, 1.0)))


def make_ground_grid(n_lat: int, n_lon: int, lat_bounds_deg: tuple = (-70.0, 70.0)) -> torch.Tensor:
    """
    Create a uniform lat/lon ground grid in ECEF coordinates.

    Returns:
        [N_ground, 3] ECEF positions on the Earth's surface
    """
    lat_deg = torch.linspace(lat_bounds_deg[0], lat_bounds_deg[1], n_lat)
    lon_deg = torch.linspace(-180, 180, n_lon + 1)[:-1]
    lat_mesh, lon_mesh = torch.meshgrid(lat_deg, lon_deg, indexing='ij')
    lat_rad = torch.deg2rad(lat_mesh.flatten())
    lon_rad = torch.deg2rad(lon_mesh.flatten())
    return torch.stack([
        R_EARTH * torch.cos(lat_rad) * torch.cos(lon_rad),
        R_EARTH * torch.cos(lat_rad) * torch.sin(lon_rad),
        R_EARTH * torch.sin(lat_rad),
    ], dim=-1)


def make_ground_grid_with_weights(n_lat: int, n_lon: int, lat_bounds_deg: tuple = (-70.0, 70.0)):
    """
    Create a uniform lat/lon ground grid with cos(lat) area weights.

    Returns:
        (ground_ecef [N_ground, 3], weights [N_ground]) where weights = cos(lat)
    """
    lat_deg = torch.linspace(lat_bounds_deg[0], lat_bounds_deg[1], n_lat)
    lon_deg = torch.linspace(-180, 180, n_lon + 1)[:-1]
    lat_mesh, lon_mesh = torch.meshgrid(lat_deg, lon_deg, indexing='ij')
    lat_rad = torch.deg2rad(lat_mesh.flatten())
    lon_rad = torch.deg2rad(lon_mesh.flatten())
    ground_ecef = torch.stack([
        R_EARTH * torch.cos(lat_rad) * torch.cos(lon_rad),
        R_EARTH * torch.cos(lat_rad) * torch.sin(lon_rad),
        R_EARTH * torch.sin(lat_rad),
    ], dim=-1)
    weights = torch.cos(lat_rad)
    return ground_ecef, weights


def make_gmst_array(tsinces: torch.Tensor, epoch_year: int = 2024, epoch_days: float = 1.0) -> list:
    """Compute GMST for each propagation time step."""
    gmst_0 = gmst_at_epoch(epoch_year, epoch_days)
    return [gmst_0 + EARTH_ROT_RAD_PER_MIN * t.item() for t in tsinces]


def make_gmst_tensor(tsinces: torch.Tensor, epoch_year: int = 2024, epoch_days: float = 1.0) -> torch.Tensor:
    """Compute GMST as a tensor for batch operations."""
    gmst_0 = gmst_at_epoch(epoch_year, epoch_days)
    return gmst_0 + EARTH_ROT_RAD_PER_MIN * tsinces.to(dtype=torch.float64)
