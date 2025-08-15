"""
Numerical utilities and validation functions for PURC

This module provides shared utilities for numerical operations and validation.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple


# Numerical tolerances from CLAUDE.md
ATOL_GOLD = 1e-10
ATOL_KKT = 1e-10  
ATOL_FEASIBILITY = 1e-12


def validate_network_structure(incidence_matrix: sp.csr_matrix, demand_vector: np.ndarray) -> bool:
    """
    Validate basic network structure and demand vector.
    
    Args:
        incidence_matrix: Network incidence matrix A
        demand_vector: Demand vector b
        
    Returns:
        True if structure is valid
        
    Raises:
        ValueError: If validation fails
    """
    A = incidence_matrix
    b = demand_vector
    
    # Check dimensions
    if A.shape[0] != len(b):
        raise ValueError(f"Dimension mismatch: A has {A.shape[0]} nodes, b has {len(b)} elements")
    
    # Check that demand sums to zero
    if abs(np.sum(b)) > 1e-15:
        raise ValueError(f"Demand vector sum is {np.sum(b)}, should be zero")
    
    # Check incidence matrix structure (each column should have at most one +1 and one -1)
    for j in range(A.shape[1]):
        col = A[:, j].toarray().flatten()
        nonzero_vals = col[col != 0]
        if len(nonzero_vals) > 2:
            raise ValueError(f"Link {j} connects to more than 2 nodes")
        if not np.all(np.abs(nonzero_vals) == 1):
            raise ValueError(f"Link {j} has invalid incidence values: {nonzero_vals}")
    
    return True


def check_feasibility(x: np.ndarray, incidence_matrix: sp.csr_matrix, demand_vector: np.ndarray) -> Tuple[float, float]:
    """
    Check primal feasibility of a solution.
    
    Args:
        x: Flow vector
        incidence_matrix: Network incidence matrix A
        demand_vector: Demand vector b
        
    Returns:
        Tuple of (constraint_violation, non_negativity_violation)
    """
    # Check flow conservation: Ax = b
    constraint_violation = np.linalg.norm(incidence_matrix @ x - demand_vector, np.inf)
    
    # Check non-negativity: x >= 0
    non_negativity_violation = -np.min(x)  # Positive if any x_i < 0
    
    return constraint_violation, non_negativity_violation


def compute_route_flows_from_link_flows(link_flows: np.ndarray, 
                                      incidence_matrix: sp.csr_matrix,
                                      demand_vector: np.ndarray) -> dict:
    """
    Decompose link flows into route flows (conceptual).
    
    This is mainly for analysis purposes. In practice, PURC works directly with link flows.
    
    Args:
        link_flows: Optimal link flows x
        incidence_matrix: Network incidence matrix A  
        demand_vector: Demand vector b
        
    Returns:
        Dictionary with analysis information
    """
    # Find active links
    active_links = np.where(link_flows > ATOL_FEASIBILITY)[0]
    
    return {
        'active_links': active_links,
        'num_active_links': len(active_links),
        'total_flow': np.sum(link_flows),
        'flow_distribution': {
            'mean': np.mean(link_flows),
            'std': np.std(link_flows),
            'max': np.max(link_flows),
            'min': np.min(link_flows)
        }
    }