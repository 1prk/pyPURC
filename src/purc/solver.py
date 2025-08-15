"""
PURC Solver Module - Build A, construct b, solve (entropy & quadratic)

Implements the main PURC optimization problem per SOLVER_SPEC.md:
- Minimize f(x) = -U(x) subject to LinearConstraint(A, b, b) and Bounds(0, +inf)
- Use trust-constr method with proper LinearConstraint
- KKT verification per KKT_VERIFICATION.md

Paper References:
- §2: Setup and traveler problem (Eq. 1-2)
- §2.1: KKT conditions
- §2.5: Entropy perturbation (Eq. 8)
"""

import numpy as np
import scipy.sparse as sp
import scipy.optimize as opt
from scipy.optimize import LinearConstraint
from typing import Tuple, Optional, Dict, Any
import warnings

from .core import (
    PURCResult, compute_objective, compute_objective_gradient,
    compute_dual_variables, compute_kkt_residuals
)


def build_incidence_matrix(
    nodes: list,
    edges: list
) -> Tuple[sp.csr_matrix, Dict[str, int], Dict[int, int]]:
    """
    Build incidence matrix A from network topology.
    
    For link e = (i->j): A[i, e] = -1, A[j, e] = +1, else 0
    Paper Reference: §2 "Setup", incidence definition
    
    Args:
        nodes: List of node names
        edges: List of edge dictionaries with 'tail', 'head', 'id'
        
    Returns:
        Tuple of (incidence_matrix, node_mapping, edge_mapping)
    """
    # Create stable node mapping
    node_mapping = {node: idx for idx, node in enumerate(sorted(nodes))}
    n_nodes = len(nodes)
    
    # Create edge mapping sorted by edge ID
    edges_sorted = sorted(edges, key=lambda x: x['id'])
    edge_mapping = {edge['id']: idx for idx, edge in enumerate(edges_sorted)}
    n_edges = len(edges)
    
    # Build incidence matrix
    row_indices = []
    col_indices = []
    data = []
    
    for edge in edges_sorted:
        edge_idx = edge_mapping[edge['id']]
        tail_idx = node_mapping[edge['tail']]
        head_idx = node_mapping[edge['head']]
        
        # Tail: outflow (-1)
        row_indices.append(tail_idx)
        col_indices.append(edge_idx)
        data.append(-1.0)
        
        # Head: inflow (+1)
        row_indices.append(head_idx)
        col_indices.append(edge_idx)
        data.append(+1.0)
    
    incidence_matrix = sp.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_nodes, n_edges)
    )
    
    return incidence_matrix, node_mapping, edge_mapping


def construct_demand_vector(
    node_mapping: Dict[str, int],
    origin: str,
    destination: str
) -> np.ndarray:
    """
    Construct demand vector b for O-D pair.
    
    Demand b has -1 at origin, +1 at destination, zeros elsewhere; sum(b) = 0
    Paper Reference: §2 "Setup", flow conservation constraints
    
    Args:
        node_mapping: Mapping from node names to indices
        origin: Origin node name
        destination: Destination node name
        
    Returns:
        Demand vector b
    """
    n_nodes = len(node_mapping)
    b = np.zeros(n_nodes)
    
    b[node_mapping[origin]] = -1.0
    b[node_mapping[destination]] = +1.0
    
    # Verify sum is zero
    assert abs(np.sum(b)) < 1e-15, "Demand vector must sum to zero"
    
    return b


def solve_purc_entropy(
    incidence_matrix: sp.csr_matrix,
    demand_vector: np.ndarray,
    lengths: np.ndarray,
    utilities: np.ndarray,
    initial_guess: Optional[np.ndarray] = None,
    options: Optional[Dict[str, Any]] = None
) -> PURCResult:
    """
    Solve PURC optimization problem with entropy perturbation.
    
    Maximize: U(x) = ℓᵀ(u ∘ x) - ℓᵀF(x)
    Subject to: Ax = b, x ≥ 0
    
    Uses trust-constr method per SOLVER_SPEC.md
    
    Args:
        incidence_matrix: Network incidence matrix A
        demand_vector: Demand vector b
        lengths: Link lengths ℓ > 0
        utilities: Link utility rates u (negative values)
        initial_guess: Initial flow vector (optional)
        options: Solver options (optional)
        
    Returns:
        PURCResult with optimal flows and diagnostics
    """
    A = incidence_matrix.tocsr()
    b = demand_vector
    n_nodes, n_edges = A.shape
    
    # Validate inputs per HARD_CONSTRAINTS.md
    if len(b) != n_nodes:
        raise ValueError(f"Demand vector size {len(b)} != number of nodes {n_nodes}")
    if len(lengths) != n_edges:
        raise ValueError(f"Lengths size {len(lengths)} != number of edges {n_edges}")
    if len(utilities) != n_edges:
        raise ValueError(f"Utilities size {len(utilities)} != number of edges {n_edges}")
    if np.any(lengths <= 0):
        raise ValueError("All lengths must be positive")
    if not np.all(utilities <= 0):
        warnings.warn("Utilities should be negative (representing costs)")
    
    # Initial point: strictly positive feasible x0 per SOLVER_SPEC.md
    if initial_guess is None:
        try:
            # Solve least squares for A x = b, then clip to >= 1e-8
            x0_ls = sp.linalg.lstsq(A.toarray(), b)[0]
            initial_guess = np.maximum(x0_ls, 1e-8)
            
            # Verify feasibility
            constraint_violation = np.linalg.norm(A @ initial_guess - b, np.inf)
            if constraint_violation > 1e-6:
                # Fallback: uniform positive values
                total_demand = np.sum(np.abs(b))
                if total_demand > 0:
                    initial_guess = np.full(n_edges, total_demand / n_edges + 1e-8)
                else:
                    initial_guess = np.full(n_edges, 1e-8)
                    
        except Exception:
            # Ultimate fallback
            total_demand = np.sum(np.abs(b))
            if total_demand > 0:
                initial_guess = np.full(n_edges, total_demand / n_edges + 1e-8)
            else:
                initial_guess = np.full(n_edges, 1e-8)
    
    # Define optimization problem (minimize negative of objective)
    def objective_fun(x):
        return -compute_objective(x, lengths, utilities)
    
    def objective_grad(x):
        return -compute_objective_gradient(x, lengths, utilities)
    
    # Linear constraint: Ax = b using LinearConstraint per SOLVER_SPEC.md
    linear_constraint = LinearConstraint(A.toarray(), b, b)
    
    # Bounds: x ≥ 0
    bounds = [(0, None) for _ in range(n_edges)]
    
    # Trust-constr options per SOLVER_SPEC.md
    trust_constr_options = {
        'gtol': 1e-12,
        'xtol': 1e-12,
        'maxiter': 2000,
        'verbose': 0,
        'disp': False
    }
    if options:
        trust_constr_options.update(options)
    
    # Solve using trust-constr method
    try:
        result = opt.minimize(
            fun=objective_fun,
            x0=initial_guess,
            method='trust-constr',
            jac=objective_grad,
            bounds=bounds,
            constraints=linear_constraint,
            options=trust_constr_options
        )
        x_opt = result.x
        success = result.success
        iterations = result.nit
        
    except Exception as e:
        # Fallback to SLSQP if trust-constr fails
        warnings.warn(f"trust-constr failed ({e}), falling back to SLSQP")
        
        # Dict constraint for SLSQP
        constraints = {
            'type': 'eq',
            'fun': lambda x: A @ x - b,
            'jac': lambda x: A.toarray()
        }
        
        slsqp_options = {
            'ftol': 1e-12,
            'maxiter': 1000,
            'disp': False
        }
        
        try:
            result = opt.minimize(
                fun=objective_fun,
                x0=initial_guess,
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                constraints=constraints,
                options=slsqp_options
            )
            x_opt = result.x
            success = result.success
            iterations = result.nit
        except Exception as e2:
            # Ultimate fallback
            warnings.warn(f"All optimization methods failed: {e2}")
            x_opt = initial_guess
            success = False
            iterations = 0
    
    # Compute dual variables and reduced costs for KKT verification
    try:
        dual_vars, reduced_costs = compute_dual_variables(
            x_opt, A, b, lengths, utilities
        )
    except Exception as e:
        warnings.warn(f"Dual variable computation failed: {e}")
        dual_vars = np.zeros(n_nodes)
        reduced_costs = np.zeros(n_edges)
    
    # Compute diagnostics
    obj_value = compute_objective(x_opt, lengths, utilities)
    constraint_viol = np.linalg.norm(A @ x_opt - b, np.inf)
    
    kkt_stat_res, kkt_comp_res = compute_kkt_residuals(
        x_opt, dual_vars, reduced_costs, A, b, lengths, utilities
    )
    
    return PURCResult(
        flows=x_opt,
        dual_variables=dual_vars,
        reduced_costs=reduced_costs,
        success=success,
        iterations=iterations,
        objective_value=obj_value,
        constraint_violation=constraint_viol,
        kkt_stationarity_residual=kkt_stat_res,
        kkt_complementarity_residual=kkt_comp_res
    )


def solve_purc_quadratic(
    incidence_matrix: sp.csr_matrix,
    demand_vector: np.ndarray,
    lengths: np.ndarray,
    utilities: np.ndarray,
    regularization: float = 1e-6,
    **kwargs
) -> PURCResult:
    """
    Solve PURC optimization problem with quadratic regularization.
    
    Maximize: U(x) = ℓᵀ(u ∘ x) - (λ/2) * ||x||²
    Subject to: Ax = b, x ≥ 0
    
    This provides an alternative to entropy regularization for comparison.
    
    Args:
        incidence_matrix: Network incidence matrix A
        demand_vector: Demand vector b
        lengths: Link lengths ℓ > 0
        utilities: Link utility rates u (negative values)
        regularization: Quadratic regularization parameter λ > 0
        **kwargs: Additional arguments passed to solve_purc_entropy
        
    Returns:
        PURCResult with optimal flows and diagnostics
    """
    # Modify objective to use quadratic instead of entropy perturbation
    def quadratic_objective(x, lengths, utilities):
        utility_term = np.dot(lengths, utilities * x)
        regularization_term = 0.5 * regularization * np.dot(x, x)
        return utility_term - regularization_term
    
    def quadratic_gradient(x, lengths, utilities):
        return lengths * utilities - regularization * x
    
    # Replace the objective functions temporarily
    import purc.core as core_module
    original_obj = core_module.compute_objective
    original_grad = core_module.compute_objective_gradient
    
    try:
        core_module.compute_objective = quadratic_objective
        core_module.compute_objective_gradient = quadratic_gradient
        
        result = solve_purc_entropy(
            incidence_matrix, demand_vector, lengths, utilities, **kwargs
        )
        
    finally:
        # Restore original functions
        core_module.compute_objective = original_obj
        core_module.compute_objective_gradient = original_grad
    
    return result
