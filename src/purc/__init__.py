"""
pyPURC - Perturbed Utility Route Choice (PURC) implementation

Based on Fosgerau, Paulsen, Rasmussen (Transportation Research Part C, 2022).

Modules:
- core: Core optimization functions and data structures
- solver: Main PURC solver with entropy and quadratic perturbations
- projector: Active-set projection and projector matrices
- estimation: Projected KKT OLS parameter estimation
- graph: Network construction and incidence matrices
- io: I/O functions for networks, results, and fixtures
- utils: Utility functions
"""

# Core functionality
from .core import solve_purc, verify_kkt, PURCResult
from .solver import solve_purc_entropy, solve_purc_quadratic
from .projector import build_active_set_matrix, compute_projector_matrix, verify_projector_identities
from .estimation import estimate_projected_kkt_ols, ProjectedKKTResult
from .graph import build_incidence_matrix, create_demand_vector, ODDisconnected, load_toy_network
from .io import load_network_json, save_network_json, save_results_json, load_results_json
from .osm import osm_to_multigraph, create_od_demand_vector, extract_network_matrices, osm_network_summary

__version__ = "0.1.0"
__all__ = [
    # Core
    "solve_purc",
    "verify_kkt", 
    "PURCResult",
    
    # Solver
    "solve_purc_entropy", 
    "solve_purc_quadratic",
    
    # Projector
    "build_active_set_matrix",
    "compute_projector_matrix", 
    "verify_projector_identities",
    
    # Estimation
    "estimate_projected_kkt_ols",
    "ProjectedKKTResult",
    
    # Graph
    "build_incidence_matrix",
    "create_demand_vector", 
    "ODDisconnected",
    "load_toy_network",
    
    # I/O
    "load_network_json",
    "save_network_json", 
    "save_results_json",
    "load_results_json",
    
    # OSM Integration
    "osm_to_multigraph",
    "create_od_demand_vector",
    "extract_network_matrices", 
    "osm_network_summary"
]