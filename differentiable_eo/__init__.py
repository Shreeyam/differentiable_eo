"""differentiable_eo: Differentiable satellite constellation optimization via dSGP4."""

from .config import Config
from .constants import (
    MU_EARTH, R_EARTH, EARTH_ROT_RAD_PER_MIN,
    IDX_BSTAR, IDX_NDOT, IDX_NDDOT, IDX_ECCO, IDX_ARGPO,
    IDX_INCLO, IDX_MO, IDX_NO_KOZAI, IDX_NODEO,
    N_ELEMENTS, ELEMENT_NAMES,
)
from .constraints import (
    FixedConstraint, UnboundedConstraint, BoxConstraint, ReparameterizedElements,
    default_parameter_specs, specs_with_altitude,
)
from .coordinates import (
    gmst_at_epoch, teme_to_ecef, compute_elevation,
    make_ground_grid, make_ground_grid_with_weights, make_gmst_array,
)
from .coverage import (
    soft_coverage, hard_coverage, noisy_or, leaky_integrator_step, logsumexp_soft_max,
)
from .tle_utils import (
    mean_motion_from_alt, alt_from_no_kozai, no_kozai_from_alt,
    make_tle, extract_elements, update_tle_from_elements, make_constellation,
)
from .objective import propagate_constellation, compute_loss, compute_hard_metrics
from .optimize import ConstellationOptimizer, OptimizationResult
from .visualization import plot_optimization_results, plot_loss_landscape, compute_coverage_map, compute_revisit_map
