"""TLE creation, update, and constellation generation utilities."""

import math
import torch
import dsgp4

from .constants import MU_EARTH, R_EARTH, N_ELEMENTS


def mean_motion_from_alt(alt_km: float) -> float:
    """Convert altitude (km) to mean motion (rad/s) for TLE dict."""
    a = R_EARTH + alt_km
    return math.sqrt(MU_EARTH / a**3)


def alt_from_no_kozai(no_kozai_rad_per_min: float) -> float:
    """Convert no_kozai (rad/min) to altitude (km)."""
    n_rad_s = no_kozai_rad_per_min / 60.0
    if n_rad_s <= 0:
        return 0.0
    return (MU_EARTH / n_rad_s**2) ** (1.0 / 3.0) - R_EARTH


def no_kozai_from_alt(alt_km: float) -> float:
    """Convert altitude (km) to no_kozai (rad/min)."""
    a = R_EARTH + alt_km
    return math.sqrt(MU_EARTH / a**3) * 60.0


def make_tle(inc_rad: float, raan_rad: float, ma_rad: float, alt_km: float,
             sat_id: int = 99999, ecc: float = 0.001, bstar: float = 1e-5) -> dsgp4.tle.TLE:
    """Create a TLE object from orbital elements."""
    return dsgp4.tle.TLE(dict(
        satellite_catalog_number=sat_id, classification='U',
        international_designator='24001A',
        epoch_year=2024, epoch_days=1.0, ephemeris_type=0,
        element_number=999, revolution_number_at_epoch=1,
        mean_motion=mean_motion_from_alt(alt_km),
        mean_motion_first_derivative=0.0, mean_motion_second_derivative=0.0,
        eccentricity=ecc, inclination=inc_rad, argument_of_perigee=0.0,
        raan=raan_rad, mean_anomaly=ma_rad, b_star=bstar,
    ))


def extract_elements(tle) -> torch.Tensor:
    """Extract [9] element tensor from an initialized TLE."""
    return torch.tensor([
        tle._bstar, tle._ndot, tle._nddot, tle._ecco, tle._argpo,
        tle._inclo, tle._mo, tle._no_kozai, tle._nodeo,
    ], dtype=torch.float64)


def update_tle_from_elements(tle, elements: torch.Tensor):
    """Write element values into TLE instance attributes (not _data dict)."""
    tle._bstar = elements[0].detach()
    tle._ndot = elements[1].detach()
    tle._nddot = elements[2].detach()
    tle._ecco = elements[3].detach()
    tle._argpo = elements[4].detach()
    tle._inclo = elements[5].detach()
    tle._mo = elements[6].detach()
    tle._no_kozai = elements[7].detach()
    tle._nodeo = elements[8].detach()


def make_constellation(n_planes: int, n_sats_per_plane: int, inc_deg: float,
                       raan_offsets_deg: list, alt_km: float,
                       ecc: float = 0.001, bstar: float = 1e-5,
                       phase_offset_f: int = 0) -> list:
    """
    Create a list of TLEs for a Walker-like constellation.

    Walker T/P/F convention: satellite s in plane p gets
        MA = 2*pi*s/S + F * (2*pi/T) * p
    where S = sats_per_plane, T = total sats. F=0 gives no inter-plane offset.
    """
    total_sats = n_planes * n_sats_per_plane
    tles = []
    for p in range(n_planes):
        raan = math.radians(raan_offsets_deg[p])
        for s in range(n_sats_per_plane):
            ma = 2 * math.pi * s / n_sats_per_plane + phase_offset_f * (2 * math.pi / total_sats) * p
            tles.append(make_tle(
                inc_rad=math.radians(inc_deg),
                raan_rad=raan, ma_rad=ma,
                alt_km=alt_km, ecc=ecc, bstar=bstar,
                sat_id=40000 + p * n_sats_per_plane + s,
            ))
    return tles
