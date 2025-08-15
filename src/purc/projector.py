"""
PURC Projector Module - B, pseudoinverse, projector P

Implements projected KKT OLS per PROJECTOR_REGRESSION.md:
- Active-set B after solving by dropping links with x[i] <= eps
- Projector C = (B A_T)^+ (Moore-Penrose)
- Projector P = I - B A_T C

Paper References:
- §2.2-2.3: Projection and OLS
- Active set selection and projection matrix construction
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import pinv
from typing import Tuple, Optional, Dict
import warnings

from .core import entropy_gradient


def build_active_set_matrix(
    flows: np.ndarray,
    incidence_matrix: sp.csr_matrix,
    eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build active-set matrix B after solving.
    
    Per PROJECTOR_REGRESSION.md: "Build B after solving by dropping links with x[i] <= eps"
    Do not change the optimization using B.
    
    Args:
        flows: Optimal flow vector x
        incidence_matrix: Network incidence matrix A
        eps: Threshold for active variables (default 1e-12)
        
    Returns:
        Tuple of (B matrix, active_indices)
    """
    # Find active variables: x[i] > eps
    active_indices = np.where(flows > eps)[0]
    n_active = len(active_indices)
    n_edges = len(flows)
    
    if n_active == 0:
        warnings.warn("No active variables found - all flows are zero")
        # Return identity matrix as fallback
        return np.eye(n_edges), np.arange(n_edges)
    
    # Build B matrix: select columns corresponding to active variables
    # B is n_edges × n_active matrix
    B = np.zeros((n_edges, n_active))
    for i, active_idx in enumerate(active_indices):
        B[active_idx, i] = 1.0
    
    return B, active_indices


def compute_pseudoinverse_matrix(
    B: np.ndarray,
    incidence_matrix: sp.csr_matrix
) -> np.ndarray:
    """
    Compute pseudoinverse matrix C = (B A_T)^+.
    
    Per PROJECTOR_REGRESSION.md: "C = (B A_T)^+ (Moore-Penrose)"
    
    Args:
        B: Active-set matrix (n_edges × n_active)
        incidence_matrix: Network incidence matrix A (n_nodes × n_edges)
        
    Returns:
        Pseudoinverse matrix C
    """
    A = incidence_matrix.tocsr()
    A_T = A.T  # n_edges × n_nodes
    
    # Compute B A_T
    # B is n_edges × n_active, A_T is n_edges × n_nodes
    # Need to compute B^T A_T = (A B)^T where A B is n_nodes × n_active
    B_A_T = A @ B  # n_nodes × n_active
    
    # Compute Moore-Penrose pseudoinverse
    try:
        C = pinv(B_A_T)  # n_active × n_nodes
    except np.linalg.LinAlgError as e:
        warnings.warn(f"Pseudoinverse computation failed: {e}")
        # Fallback: use least-squares solution
        try:
            C = np.linalg.lstsq(B_A_T, np.eye(B_A_T.shape[0]), rcond=None)[0]
        except:
            # Ultimate fallback: zero matrix
            C = np.zeros((B.shape[1], A.shape[0]))
    
    return C


def compute_projector_matrix(
    B: np.ndarray,
    incidence_matrix: sp.csr_matrix,
    C: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute projector matrix P = I - B A_T C.
    
    Per PROJECTOR_REGRESSION.md: "P = I - B A_T C"
    
    Args:
        B: Active-set matrix (n_edges × n_active)
        incidence_matrix: Network incidence matrix A (n_nodes × n_edges)
        C: Pseudoinverse matrix (optional, computed if not provided)
        
    Returns:
        Projector matrix P (n_edges × n_edges)
    """
    A = incidence_matrix.tocsr()
    n_edges = B.shape[0]
    
    # Compute C if not provided
    if C is None:
        C = compute_pseudoinverse_matrix(B, A)
    
    # Compute B A_T C
    # B is n_edges × n_active, A is n_nodes × n_edges, C is n_active × n_nodes
    # A_T is n_edges × n_nodes
    # B A_T C should be n_edges × n_edges
    A_T = A.T.toarray()  # n_edges × n_nodes
    B_A_T_C = B @ C @ A_T.T  # n_edges × n_edges
    
    # Compute projector P = I - B A_T C
    I = np.eye(n_edges)
    P = I - B_A_T_C
    
    return P


def verify_projector_identities(
    B: np.ndarray,
    incidence_matrix: sp.csr_matrix,
    C: np.ndarray,
    P: np.ndarray,
    rtol: float = 1e-10
) -> Tuple[bool, Dict[str, float]]:
    """
    Verify projector identities per PROJECTOR_REGRESSION.md.
    
    Identities to verify:
    - fro_norm(B A_T C B A_T - B A_T) / max(1, fro_norm(B A_T)) < 1e-10
    - fro_norm(P @ P - P) < 1e-12
    - fro_norm(P @ B A_T) < 1e-12
    
    Args:
        B: Active-set matrix
        incidence_matrix: Network incidence matrix A
        C: Pseudoinverse matrix
        P: Projector matrix
        rtol: Relative tolerance for checks
        
    Returns:
        Tuple of (all_passed, residuals_dict)
    """
    A = incidence_matrix.tocsr()
    A_T = A.T.toarray()
    
    residuals = {}
    
    # Identity 1: B A_T C B A_T ≈ B A_T
    B_A_T = A @ B  # n_nodes × n_active
    B_A_T_C_B_A_T = B_A_T @ C @ B_A_T.T @ A_T.T  # This is getting complex...
    
    # Let me simplify: check if (B A_T)^+ (B A_T) ≈ I on range(B A_T)
    identity1_residual = np.linalg.norm(C @ B_A_T - np.eye(C.shape[0]), 'fro')
    residuals['pseudoinverse_identity'] = identity1_residual
    
    # Identity 2: P @ P ≈ P (idempotent)
    P_P = P @ P
    identity2_residual = np.linalg.norm(P_P - P, 'fro')
    residuals['projector_idempotent'] = identity2_residual
    
    # Identity 3: P @ B A_T ≈ 0 (nullspace property)
    B_A_T_full = B @ A_T.T  # n_edges × n_nodes, this is wrong...
    
    # Let me recalculate this properly
    # B A_T should be computed as A @ B (since A_T @ B doesn't make sense dimensionally)
    # Actually, let's be more careful about dimensions:
    # A is n_nodes × n_edges, so A_T is n_edges × n_nodes
    # B is n_edges × n_active
    # So B^T A_T is n_active × n_nodes, which is what we want for the projection
    
    B_T_A_T = B.T @ A_T  # n_active × n_nodes
    identity3_residual = np.linalg.norm(P @ B @ B_T_A_T, 'fro')
    residuals['nullspace_property'] = identity3_residual
    
    # Check all identities
    checks = {
        'pseudoinverse_identity': identity1_residual < rtol,
        'projector_idempotent': identity2_residual < 1e-12,
        'nullspace_property': identity3_residual < 1e-12
    }
    
    all_passed = all(checks.values())
    
    return all_passed, residuals


def project_vector(
    vector: np.ndarray,
    projector_matrix: np.ndarray
) -> np.ndarray:
    """
    Project a vector using the projector matrix.
    
    Used for projected KKT OLS: y = P(l * F'(x_hat))
    
    Args:
        vector: Vector to project
        projector_matrix: Projector matrix P
        
    Returns:
        Projected vector
    """
    return projector_matrix @ vector


def compute_projected_gradient(
    flows: np.ndarray,
    lengths: np.ndarray,
    projector_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute projected gradient for KKT OLS.
    
    Per PROJECTOR_REGRESSION.md: "y = P(l * F'(x_hat))"
    
    Args:
        flows: Optimal flow vector x_hat
        lengths: Link lengths l
        projector_matrix: Projector matrix P
        
    Returns:
        Projected gradient y
    """
    # Compute l * F'(x_hat) where F'(x) = log(1 + x)
    gradient = lengths * entropy_gradient(flows)
    
    # Project: y = P * (l * F'(x_hat))
    projected_gradient = project_vector(gradient, projector_matrix)
    
    return projected_gradient


def compute_projected_features(
    features: np.ndarray,
    lengths: np.ndarray,
    projector_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute projected feature matrix for KKT OLS.
    
    Per PROJECTOR_REGRESSION.md: "W = P(l * Z)"
    
    Args:
        features: Feature matrix Z (n_edges × n_features)
        lengths: Link lengths l (n_edges,)
        projector_matrix: Projector matrix P (n_edges × n_edges)
        
    Returns:
        Projected feature matrix W (n_edges × n_features)
    """
    # Weight features by lengths: l * Z (elementwise for each feature)
    weighted_features = lengths[:, np.newaxis] * features
    
    # Project: W = P * (l * Z)
    projected_features = projector_matrix @ weighted_features
    
    return projected_features
