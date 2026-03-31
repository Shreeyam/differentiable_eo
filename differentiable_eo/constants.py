"""Physical constants and TLE element indexing."""

# Earth parameters
MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.137  # km
EARTH_ROT_RAD_PER_MIN = 7.2921159e-5 * 60  # rad/min

# TLE element indices (order used by dsgp4 after sgp4init)
IDX_BSTAR = 0
IDX_NDOT = 1
IDX_NDDOT = 2
IDX_ECCO = 3
IDX_ARGPO = 4
IDX_INCLO = 5
IDX_MO = 6
IDX_NO_KOZAI = 7 
IDX_NODEO = 8

N_ELEMENTS = 9

ELEMENT_NAMES = [
    "bstar", "ndot", "nddot", "ecco", "argpo",
    "inclo", "mo", "no_kozai", "nodeo",
]
