"""
Core PURC solver implementation with entropy perturbation

Based on Fosgerau, Paulsen, Rasmussen (Transportation Research Part C, 2022).
Paper References:
- §2 Eq. (2): Objective function U(x) = ℓᵀ(u ∘ x) - ℓᵀF(x)
- §2.5 Eq. (8): F(x) = (1+x)ln(1+x) - x (entropy perturbation)
- §2.1: KKT conditions for numerical verification
"""

import numpy as np
import scipy.sparse as sp
import scipy.optimize as opt
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings


@dataclass
class PURCResult:
    """Results from PURC optimization."""
    flows: np.ndarray
    dual_variables: np.ndarray
    reduced_costs: np.ndarray
    success: bool
    iterations: int
    objective_value: float
    constraint_violation: float
    kkt_stationarity_residual: float
    kkt_complementarity_residual: float


def entropy_function(x: np.ndarray) -> np.ndarray:
    """
    Entropy perturbation function: F(x) = (1+x)ln(1+x) - x
    
    Paper Reference: §2.5 Eq. (8), Assumption 1
    Properties: F(0)=0, F'(0)=0, strict convexity on ℝ₊
    
    Args:
        x: Flow vector (non-negative)
        
    Returns:
        Entropy function values F(x)
    """
    # Use np.log1p for numerical stability: log1p(x) = ln(1+x)
    # Handle x=0 case: (1+0)*ln(1+0) - 0 = 1*0 - 0 = 0
    return (1 + x) * np.log1p(x) - x


def entropy_gradient(x: np.ndarray) -> np.ndarray:
    """
    Gradient of entropy function: F'(x) = ln(1+x)
    
    Paper Reference: §2.5, first derivative of F(x)
    
    Args:
        x: Flow vector (non-negative)
        
    Returns:
        Gradient F'(x)
    """
    return np.log1p(x)


def entropy_hessian_diag(x: np.ndarray) -> np.ndarray:
    """
    Diagonal of Hessian matrix: F''(x) = (1+x)^(-1)
    
    Paper Reference: §2.5, second derivative of F(x)
    Used for Newton-type optimization methods.
    
    Args:
        x: Flow vector (non-negative)
        
    Returns:
        Diagonal elements of Hessian F''(x)
    """
    return 1.0 / (1.0 + x)


def compute_objective(x: np.ndarray, lengths: np.ndarray, utilities: np.ndarray) -> float:
    """
    Compute PURC objective function value.
    
    U(x) = l^T(u * x) - l^T*F(x)
    Paper Reference: §2 Eq. (2)
    Paper states: "all components of u are negative" (representing costs)
    
    Args:
        x: Flow vector
        lengths: Link lengths l > 0
        utilities: Link utility rates u (negative values, representing costs)
        
    Returns:
        Objective function value
    """
    utility_term = np.dot(lengths, utilities * x)
    perturbation_term = np.dot(lengths, entropy_function(x))
    return utility_term - perturbation_term


def compute_objective_gradient(x: np.ndarray, lengths: np.ndarray, utilities: np.ndarray) -> np.ndarray:
    """
    Compute gradient of PURC objective function.
    
    grad U(x) = l * (u - F'(x))
    Paper Reference: First-order conditions in §2.1
    
    Args:
        x: Flow vector
        lengths: Link lengths l > 0
        utilities: Link utility rates u (negative values, representing costs)
        
    Returns:
        Gradient vector
    """
    return lengths * (utilities - entropy_gradient(x))


def solve_purc(
    incidence_matrix: sp.csr_matrix,
    demand_vector: np.ndarray,
    lengths: np.ndarray,
    utilities: np.ndarray,
    method: str = 'entropy',
    initial_guess: Optional[np.ndarray] = None,
    options: Optional[Dict[str, Any]] = None
) -> PURCResult:
    """
    Solve the PURC optimization problem.
    
    Maximize: U(x) = ℓᵀ(u ∘ x) - ℓᵀF(x)
    Subject to: Ax = b, x ≥ 0
    
    Paper Reference: §2 Eq. (1-2), traveler problem formulation
    
    Args:
        incidence_matrix: Network incidence matrix A
        demand_vector: Demand vector b  
        lengths: Link lengths ℓ > 0
        utilities: Link utility rates u = Zβ
        method: Perturbation method ('entropy' only supported)
        initial_guess: Initial flow vector (optional)
        options: Solver options (optional)
        
    Returns:
        PURCResult with optimal flows and diagnostics
    """
    if method != 'entropy':
        raise ValueError("Only 'entropy' method is currently supported")
    
    # Validate inputs
    A = incidence_matrix.tocsr()
    b = demand_vector
    n_nodes, n_edges = A.shape
    
    if len(b) != n_nodes:
        raise ValueError(f"Demand vector size {len(b)} != number of nodes {n_nodes}")
    if len(lengths) != n_edges:
        raise ValueError(f"Lengths size {len(lengths)} != number of edges {n_edges}")
    if len(utilities) != n_edges:
        raise ValueError(f"Utilities size {len(utilities)} != number of edges {n_edges}")
    if np.any(lengths <= 0):
        raise ValueError("All lengths must be positive")
    
    # Default options for SLSQP solver
    default_options = {
        'ftol': 1e-12,
        'maxiter': 1000,
        'disp': False
    }
    if options:
        default_options.update(options)
    
    # Initial point: strictly positive feasible x0 per SOLVER_SPEC.md
    if initial_guess is None:
        try:
            # Solve least squares for A x = b, then clip to >= 1e-8
            x0_ls = sp.linalg.lstsq(A.toarray(), b)[0]
            initial_guess = np.maximum(x0_ls, 1e-8)
            
            # Verify feasibility and adjust if needed
            constraint_violation = np.linalg.norm(A @ initial_guess - b, np.inf)
            if constraint_violation > 1e-6:
                # If not feasible, use a simple feasible solution
                # Find a single path and put minimal flow everywhere else
                initial_guess = np.full(n_edges, 1e-8)
                # Add extra flow on first few edges to satisfy demand
                total_demand = np.sum(np.abs(b))
                if total_demand > 0:
                    extra_flow = total_demand / min(3, n_edges)  # Spread over first few edges
                    initial_guess[:min(3, n_edges)] += extra_flow
                    
        except Exception:
            # Ultimate fallback: uniform positive values
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
    
    # Equality constraints: Ax = b
    def constraint_fun(x):
        return A @ x - b
    
    def constraint_jac(x):
        return A.toarray()
    
    constraints = {
        'type': 'eq',
        'fun': constraint_fun,
        'jac': constraint_jac
    }
    
    # Bounds: x ≥ 0
    bounds = [(0, None) for _ in range(n_edges)]
    
    # Use trust-constr method as specified in SOLVER_SPEC.md
    from scipy.optimize import LinearConstraint
    
    # Create proper LinearConstraint instead of dict constraint
    linear_constraint = LinearConstraint(A.toarray(), b, b)
    
    # Trust-constr options per SOLVER_SPEC.md
    trust_constr_options = {
        'gtol': 1e-12,
        'xtol': 1e-12, 
        'maxiter': 2000,
        'verbose': 0,
        'disp': False
    }
    
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
        try:
            result = opt.minimize(
                fun=objective_fun,
                x0=initial_guess,
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                constraints=constraints,
                options=default_options
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


def compute_dual_variables(
    x: np.ndarray,
    incidence_matrix: sp.csr_matrix,
    demand_vector: np.ndarray,
    lengths: np.ndarray,
    utilities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dual variables (node multipliers) and reduced costs from optimal flows.
    
    Follows KKT_VERIFICATION.md specification exactly:
    1) g = l * (u - log(1 + x))
    2) E_plus = { i : x[i] > eps } with eps = 1e-12
    3) Solve (A_T * lambda)[E_plus] = -g[E_plus] by least squares
    4) s = -(g + A_T * lambda)
    
    Args:
        x: Optimal flow vector
        incidence_matrix: Network incidence matrix A
        demand_vector: Demand vector b
        lengths: Link lengths ℓ
        utilities: Link utility rates u
        
    Returns:
        Tuple of (dual variables λ, reduced costs s)
    """
    A = incidence_matrix.tocsr()
    eps = 1e-12
    
    # Step 1: g = l * (u - log(1 + x))
    g = lengths * (utilities - np.log1p(x))
    
    # Step 2: E_plus = { i : x[i] > eps }
    E_plus = np.where(x > eps)[0]
    
    # Step 3: Solve (A_T * lambda)[E_plus] = -g[E_plus] by least squares
    if len(E_plus) > 0:
        A_T_active = A.T[E_plus, :]  # Select rows corresponding to active variables
        g_active = g[E_plus]
        
        try:
            # Solve A_T_active @ lambda = -g_active by least squares
            lambda_vars, _, _, _ = np.linalg.lstsq(
                A_T_active.toarray(), -g_active, rcond=None
            )
        except:
            # Fallback if lstsq fails
            lambda_vars = np.zeros(A.shape[0])
    else:
        lambda_vars = np.zeros(A.shape[0])
    
    # Step 4: s = -(g + A_T * lambda)
    reduced_costs = -(g + A.T @ lambda_vars)
    
    return lambda_vars, reduced_costs


def compute_kkt_residuals(
    x: np.ndarray,
    dual_vars: np.ndarray,
    reduced_costs: np.ndarray,
    incidence_matrix: sp.csr_matrix,
    demand_vector: np.ndarray,
    lengths: np.ndarray,
    utilities: np.ndarray
) -> Tuple[float, float]:
    """
    Compute KKT residuals following KKT_VERIFICATION.md specification.
    
    Checks:
    - Dual feasibility: min(s) >= -1e-12
    - Complementarity: max_i |x[i] * s[i]| <= 1e-12
    - Stationarity on E_zero: max_i |min(s[i], 0)| <= 1e-10
    
    Args:
        x: Flow vector
        dual_vars: Node multipliers λ
        reduced_costs: Reduced costs s
        incidence_matrix: Network incidence matrix A
        demand_vector: Demand vector b
        lengths: Link lengths ℓ
        utilities: Link utility rates u
        
    Returns:
        Tuple of (stationarity residual, complementarity residual)
    """
    eps = 1e-12
    
    # Find active and inactive sets
    E_plus = np.where(x > eps)[0]
    E_zero = np.where(x <= eps)[0]
    
    # Stationarity check on E_zero (inactive variables)
    # For inactive variables, we need s[i] >= 0, so check max |min(s[i], 0)|
    if len(E_zero) > 0:
        stationarity_residual = np.max(np.abs(np.minimum(reduced_costs[E_zero], 0.0)))
    else:
        stationarity_residual = 0.0
    
    # Complementarity residual: max_i |x[i] * s[i]|
    # For inactive variables (x <= eps), treat x as exactly 0 for complementarity
    x_comp = np.where(x > eps, x, 0.0)
    complementarity_residual = np.max(np.abs(x_comp * reduced_costs))
    
    return stationarity_residual, complementarity_residual


def verify_kkt(
    x: np.ndarray,
    dual_vars: np.ndarray,
    reduced_costs: np.ndarray,
    incidence_matrix: sp.csr_matrix,
    demand_vector: np.ndarray,
    lengths: np.ndarray,
    utilities: np.ndarray,
    atol: float = 1e-10
) -> bool:
    """
    Verify KKT conditions for solution optimality.
    
    Args:
        x: Flow vector
        dual_vars: Node multipliers λ
        reduced_costs: Reduced costs s
        incidence_matrix: Network incidence matrix A
        demand_vector: Demand vector b
        lengths: Link lengths ℓ
        utilities: Link utility rates u
        atol: Absolute tolerance for residuals
        
    Returns:
        True if KKT conditions satisfied within tolerance
    """
    A = incidence_matrix.tocsr()
    
    # Check primal feasibility
    constraint_violation = np.linalg.norm(A @ x - demand_vector, np.inf)
    if constraint_violation > 1e-12:
        return False
    
    # Check non-negativity
    if np.any(x < -1e-12) or np.any(reduced_costs < -1e-12):
        return False
    
    # Check KKT residuals
    stat_res, comp_res = compute_kkt_residuals(
        x, dual_vars, reduced_costs, A, demand_vector, lengths, utilities
    )
    
    return stat_res < atol and comp_res < 1e-12