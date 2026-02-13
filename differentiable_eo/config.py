"""Configuration dataclass for constellation optimization."""

from dataclasses import dataclass, field
from typing import Optional

from .constraints import default_parameter_specs


@dataclass
class Config:
    # Constellation
    n_planes: int = 4
    n_sats_per_plane: int = 3
    target_alt_km: float = 550.0

    # Initial (deliberately suboptimal) configuration
    initial_inc_deg: float = 30.0
    initial_raan_offsets_deg: Optional[list] = None
    initial_ma_offsets_deg: Optional[list] = None  # flat list of MAs per sat; None = even spacing

    # Propagation
    prop_duration_hours: float = 24.0
    n_time_steps: int = 240

    # Ground grid
    n_lat: int = 36
    n_lon: int = 72
    lat_bounds_deg: tuple = (-70.0, 70.0)

    # Coverage model
    min_elevation_deg: float = 10.0
    softness_deg: float = 2.0

    # Revisit model
    revisit_logsumexp_temp: float = 10.0
    revisit_weight: float = 0.005
    revisit_reduce: str = 'mean'  # 'mean' or 'max' (minimax over ground points)
    revisit_spatial_tau: float = None  # LSE temp for spatial max; defaults to revisit_logsumexp_temp

    # Optimization
    n_iterations: int = 2000
    lr: float = 2e-3
    randomize_gmst: bool = True
    # Constraint specs (which elements are free/bounded)
    parameter_specs: dict = field(default_factory=default_parameter_specs)
    # Element indices shared per plane (e.g. [IDX_NODEO] for shared RAAN)
    per_plane_params: list = field(default_factory=list)

    def __post_init__(self):
        if self.initial_raan_offsets_deg is None:
            self.initial_raan_offsets_deg = [0, 40, 100, 200]

    @property
    def n_sats(self) -> int:
        return self.n_planes * self.n_sats_per_plane
