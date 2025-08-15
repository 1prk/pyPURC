"""
PURC Estimation Module - Projected KKT OLS

Implements projected KKT OLS per PROJECTOR_REGRESSION.md:
- y = P(l * F'(x_hat)), elementwise operations on vectors
- W = P(l * Z)
- Estimate beta by OLS; robust SE allowed

Paper References:
- §2.2-2.3: Projection and OLS
- Projected KKT regression for parameter estimation
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings

from .projector import (
    build_active_set_matrix,
    compute_pseudoinverse_matrix,
    compute_projector_matrix,
    compute_projected_gradient,
    compute_projected_features,
    verify_projector_identities
)


@dataclass
class ProjectedKKTResult:
    """Results from projected KKT OLS estimation."""
    beta: np.ndarray
    standard_errors: np.ndarray
    t_statistics: np.ndarray
    p_values: np.ndarray
    r_squared: float
    residuals: np.ndarray
    fitted_values: np.ndarray
    active_indices: np.ndarray
    projector_matrix: np.ndarray
    projector_diagnostics: Dict[str, float]


def estimate_projected_kkt_ols(
    flows: np.ndarray,
    lengths: np.ndarray,
    features: np.ndarray,
    incidence_matrix,
    robust_se: bool = True,
    eps: float = 1e-12
) -> ProjectedKKTResult:
    """
    Estimate parameters using projected KKT OLS.
    
    Per PROJECTOR_REGRESSION.md:
    1. Build B after solving by dropping links with x[i] <= eps
    2. Compute projector P = I - B A_T C where C = (B A_T)^+
    3. Regression: y = P(l * F'(x_hat)), W = P(l * Z)
    4. Estimate beta by OLS; robust SE allowed
    
    Args:
        flows: Optimal flow vector x_hat
        lengths: Link lengths l
        features: Feature matrix Z (n_edges × n_features)
        incidence_matrix: Network incidence matrix A
        robust_se: Whether to compute robust standard errors
        eps: Threshold for active variables
        
    Returns:
        ProjectedKKTResult with estimation results
    """
    n_edges, n_features = features.shape
    
    # Step 1: Build active-set matrix B
    B, active_indices = build_active_set_matrix(flows, incidence_matrix, eps)
    
    # Step 2: Compute projector matrix P
    C = compute_pseudoinverse_matrix(B, incidence_matrix)
    P = compute_projector_matrix(B, incidence_matrix, C)
    
    # Verify projector identities
    projector_passed, projector_diagnostics = verify_projector_identities(
        B, incidence_matrix, C, P
    )
    
    if not projector_passed:
        warnings.warn("Projector identity verification failed")
    
    # Step 3: Compute projected variables
    # y = P(l * F'(x_hat))
    y = compute_projected_gradient(flows, lengths, P)
    
    # W = P(l * Z)
    W = compute_projected_features(features, lengths, P)
    
    # Step 4: OLS regression y = W β + ε
    try:
        # Solve normal equations: W^T W β = W^T y
        WTW = W.T @ W
        WTy = W.T @ y
        
        # Add regularization for numerical stability
        regularization = 1e-12 * np.trace(WTW) / n_features
        WTW_reg = WTW + regularization * np.eye(n_features)
        
        beta = np.linalg.solve(WTW_reg, WTy)
        
    except np.linalg.LinAlgError:
        # Fallback to least squares
        beta, residuals_norm, rank, s = np.linalg.lstsq(W, y, rcond=None)
    
    # Compute fitted values and residuals
    fitted_values = W @ beta
    residuals = y - fitted_values
    
    # Compute R-squared
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum(residuals ** 2)
    
    if ss_tot > 1e-15:
        r_squared = 1 - ss_res / ss_tot
    else:
        r_squared = 0.0
    
    # Compute standard errors
    if robust_se:
        standard_errors = compute_robust_standard_errors(W, residuals, beta)
    else:
        standard_errors = compute_standard_errors(W, residuals, n_features)
    
    # Compute t-statistics and p-values
    t_statistics = np.zeros_like(beta)
    p_values = np.ones_like(beta)
    
    for i in range(n_features):
        if standard_errors[i] > 1e-15:
            t_statistics[i] = beta[i] / standard_errors[i]
            # Two-tailed p-value (approximate using normal distribution)
            p_values[i] = 2 * (1 - _normal_cdf(abs(t_statistics[i])))
    
    return ProjectedKKTResult(
        beta=beta,
        standard_errors=standard_errors,
        t_statistics=t_statistics,
        p_values=p_values,
        r_squared=r_squared,
        residuals=residuals,
        fitted_values=fitted_values,
        active_indices=active_indices,
        projector_matrix=P,
        projector_diagnostics=projector_diagnostics
    )


def compute_standard_errors(
    W: np.ndarray,
    residuals: np.ndarray,
    n_features: int
) -> np.ndarray:
    """
    Compute standard OLS standard errors.
    
    SE(β) = σ sqrt(diag((W^T W)^(-1)))
    where σ² = RSS / (n - k)
    
    Args:
        W: Design matrix (n_edges × n_features)
        residuals: Regression residuals
        n_features: Number of features
        
    Returns:
        Standard errors
    """
    n_obs = len(residuals)
    
    # Estimate residual variance
    if n_obs > n_features:
        sigma_squared = np.sum(residuals ** 2) / (n_obs - n_features)
    else:
        sigma_squared = np.sum(residuals ** 2) / max(1, n_obs - 1)
    
    try:
        # Compute (W^T W)^(-1)
        WTW = W.T @ W
        WTW_inv = np.linalg.inv(WTW + 1e-12 * np.eye(n_features))
        
        # Standard errors
        standard_errors = np.sqrt(sigma_squared * np.diag(WTW_inv))
        
    except np.linalg.LinAlgError:
        # Fallback: use diagonal approximation
        WTW_diag = np.sum(W ** 2, axis=0)
        standard_errors = np.sqrt(sigma_squared / np.maximum(WTW_diag, 1e-15))
    
    return standard_errors


def compute_robust_standard_errors(
    W: np.ndarray,
    residuals: np.ndarray,
    beta: np.ndarray
) -> np.ndarray:
    """
    Compute heteroscedasticity-robust standard errors (White/Huber).
    
    SE(β) = sqrt(diag((W^T W)^(-1) W^T Ω W (W^T W)^(-1)))
    where Ω = diag(e_i²)
    
    Args:
        W: Design matrix (n_edges × n_features)
        residuals: Regression residuals
        beta: Estimated coefficients
        
    Returns:
        Robust standard errors
    """
    n_obs, n_features = W.shape
    
    try:
        # Compute (W^T W)^(-1)
        WTW = W.T @ W
        WTW_inv = np.linalg.inv(WTW + 1e-12 * np.eye(n_features))
        
        # Compute meat matrix: W^T Ω W where Ω = diag(e_i²)
        Omega = np.diag(residuals ** 2)
        meat = W.T @ Omega @ W
        
        # Sandwich estimator: (W^T W)^(-1) meat (W^T W)^(-1)
        sandwich = WTW_inv @ meat @ WTW_inv
        
        # Standard errors
        robust_se = np.sqrt(np.diag(sandwich))
        
    except np.linalg.LinAlgError:
        # Fallback to non-robust standard errors
        warnings.warn("Robust standard error computation failed, using non-robust")
        robust_se = compute_standard_errors(W, residuals, n_features)
    
    return robust_se


def _normal_cdf(x: float) -> float:
    """
    Approximate standard normal CDF using error function.
    
    Φ(x) ≈ 0.5 * (1 + erf(x/√2))
    
    Args:
        x: Value to evaluate CDF at
        
    Returns:
        Approximate CDF value
    """
    # Simple approximation for p-values
    # For more accurate p-values, would use scipy.stats.norm.cdf
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def compute_marginal_effects(
    result: ProjectedKKTResult,
    feature_names: Optional[list] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute marginal effects and summary statistics.
    
    Args:
        result: ProjectedKKTResult from estimation
        feature_names: Optional feature names for labeling
        
    Returns:
        Dictionary with marginal effects and statistics
    """
    n_features = len(result.beta)
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    marginal_effects = {}
    
    for i, name in enumerate(feature_names):
        marginal_effects[name] = {
            'coefficient': result.beta[i],
            'standard_error': result.standard_errors[i],
            't_statistic': result.t_statistics[i],
            'p_value': result.p_values[i],
            'significant_5pct': bool(result.p_values[i] < 0.05),
            'significant_1pct': bool(result.p_values[i] < 0.01)
        }
    
    return marginal_effects


def summarize_estimation_results(
    result: ProjectedKKTResult,
    feature_names: Optional[list] = None
) -> str:
    """
    Create a summary string of estimation results.
    
    Args:
        result: ProjectedKKTResult from estimation
        feature_names: Optional feature names for labeling
        
    Returns:
        Summary string
    """
    marginal_effects = compute_marginal_effects(result, feature_names)
    
    summary = []
    summary.append("Projected KKT OLS Estimation Results")
    summary.append("=" * 40)
    summary.append(f"R-squared: {result.r_squared:.4f}")
    summary.append(f"Active variables: {len(result.active_indices)} / {len(result.residuals)}")
    summary.append("")
    
    summary.append("Coefficients:")
    summary.append("-" * 60)
    summary.append(f"{'Variable':<15} {'Coeff':<10} {'SE':<10} {'t-stat':<8} {'p-value':<8}")
    summary.append("-" * 60)
    
    for name, stats in marginal_effects.items():
        significance = "***" if stats['significant_1pct'] else ("**" if stats['significant_5pct'] else "")
        summary.append(
            f"{name:<15} {stats['coefficient']:<10.4f} {stats['standard_error']:<10.4f} "
            f"{stats['t_statistic']:<8.2f} {stats['p_value']:<8.3f} {significance}"
        )
    
    summary.append("-" * 60)
    summary.append("Significance: *** p<0.01, ** p<0.05")
    
    return "\n".join(summary)
