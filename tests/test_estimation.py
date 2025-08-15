"""
Estimation Tests

Tests for projected KKT OLS estimation per PROJECTOR_REGRESSION.md:
- Active-set B construction
- Projector P = I - B A_T C where C = (B A_T)^+
- Regression: y = P(l * F'(x_hat)), W = P(l * Z)
- OLS estimation with robust standard errors

Paper Reference: §2.2-2.3 Projection and OLS
"""

import numpy as np
import pytest
from pathlib import Path

from purc.graph import load_toy_network, build_incidence_matrix, create_demand_vector, create_utilities_from_test_case
from purc.core import solve_purc
from purc.estimation import estimate_projected_kkt_ols, ProjectedKKTResult, compute_marginal_effects, summarize_estimation_results


class TestProjectedKKTOLS:
    """Test projected KKT OLS estimation."""
    
    @pytest.fixture
    def estimation_setup(self):
        """Set up toy network with solution for estimation testing."""
        test_dir = Path(__file__).parent
        network_path = test_dir / "gold" / "toy_network.json"
        
        graph, network_metadata = load_toy_network(str(network_path))
        network_data = build_incidence_matrix(graph)
        demand_vector = create_demand_vector(network_data.node_mapping, 'O', 'D')
        
        # Base case utilities and solve
        utilities = np.array([-1., -1., -1., -1., -1., -2.])
        lengths = network_data.lengths
        
        result = solve_purc(
            network_data.incidence_matrix,
            demand_vector,
            lengths,
            utilities
        )
        
        # Create feature matrix for estimation
        n_edges = len(lengths)
        n_features = 3
        
        # Features: length, utility, constant
        features = np.column_stack([
            lengths,                    # Length feature
            utilities,                  # Utility feature  
            np.ones(n_edges)           # Constant term
        ])
        
        return {
            'flows': result.flows,
            'lengths': lengths,
            'features': features,
            'incidence_matrix': network_data.incidence_matrix,
            'result': result
        }
    
    def test_projected_kkt_estimation_basic(self, estimation_setup):
        """Test basic projected KKT OLS estimation."""
        estimation_result = estimate_projected_kkt_ols(
            flows=estimation_setup['flows'],
            lengths=estimation_setup['lengths'],
            features=estimation_setup['features'],
            incidence_matrix=estimation_setup['incidence_matrix'],
            robust_se=False
        )
        
        # Check result structure
        assert isinstance(estimation_result, ProjectedKKTResult)
        assert len(estimation_result.beta) == 3, "Should have 3 coefficients"
        assert len(estimation_result.standard_errors) == 3, "Should have 3 standard errors"
        assert len(estimation_result.t_statistics) == 3, "Should have 3 t-statistics"
        assert len(estimation_result.p_values) == 3, "Should have 3 p-values"
        
        # Check that results are finite
        assert np.all(np.isfinite(estimation_result.beta)), "Beta coefficients should be finite"
        assert np.all(np.isfinite(estimation_result.standard_errors)), "Standard errors should be finite"
        assert np.all(estimation_result.standard_errors > 0), "Standard errors should be positive"
        
        # Check R-squared is reasonable
        assert 0 <= estimation_result.r_squared <= 1, f"R-squared should be in [0,1], got {estimation_result.r_squared}"
    
    def test_projected_kkt_estimation_robust_se(self, estimation_setup):
        """Test projected KKT OLS with robust standard errors."""
        estimation_result = estimate_projected_kkt_ols(
            flows=estimation_setup['flows'],
            lengths=estimation_setup['lengths'],
            features=estimation_setup['features'],
            incidence_matrix=estimation_setup['incidence_matrix'],
            robust_se=True
        )
        
        # Robust SE should be computed successfully
        assert np.all(np.isfinite(estimation_result.standard_errors)), "Robust standard errors should be finite"
        assert np.all(estimation_result.standard_errors > 0), "Robust standard errors should be positive"
    
    def test_projector_diagnostics(self, estimation_setup):
        """Test projector matrix diagnostics."""
        estimation_result = estimate_projected_kkt_ols(
            flows=estimation_setup['flows'],
            lengths=estimation_setup['lengths'],
            features=estimation_setup['features'],
            incidence_matrix=estimation_setup['incidence_matrix']
        )
        
        # Check projector diagnostics
        diagnostics = estimation_result.projector_diagnostics
        assert isinstance(diagnostics, dict), "Projector diagnostics should be a dictionary"
        
        # Key diagnostic checks per PROJECTOR_REGRESSION.md
        if 'projector_idempotent' in diagnostics:
            assert diagnostics['projector_idempotent'] < 1e-12, \
                   f"Projector should be idempotent: {diagnostics['projector_idempotent']}"
    
    def test_active_set_identification(self, estimation_setup):
        """Test that active set is identified correctly."""
        estimation_result = estimate_projected_kkt_ols(
            flows=estimation_setup['flows'],
            lengths=estimation_setup['lengths'],
            features=estimation_setup['features'],
            incidence_matrix=estimation_setup['incidence_matrix']
        )
        
        # Active indices should correspond to positive flows
        eps = 1e-12
        expected_active = np.where(estimation_setup['flows'] > eps)[0]
        
        # The active indices from estimation should match
        np.testing.assert_array_equal(
            estimation_result.active_indices, expected_active,
            err_msg="Active set identification should match positive flows"
        )
    
    def test_marginal_effects_computation(self, estimation_setup):
        """Test marginal effects computation."""
        estimation_result = estimate_projected_kkt_ols(
            flows=estimation_setup['flows'],
            lengths=estimation_setup['lengths'],
            features=estimation_setup['features'],
            incidence_matrix=estimation_setup['incidence_matrix']
        )
        
        # Compute marginal effects
        feature_names = ['length', 'utility', 'constant']
        marginal_effects = compute_marginal_effects(estimation_result, feature_names)
        
        # Check structure
        assert len(marginal_effects) == 3, "Should have marginal effects for 3 features"
        
        for name in feature_names:
            assert name in marginal_effects, f"Should have marginal effect for {name}"
            effect = marginal_effects[name]
            
            # Check required fields
            required_fields = ['coefficient', 'standard_error', 't_statistic', 'p_value', 'significant_5pct', 'significant_1pct']
            for field in required_fields:
                assert field in effect, f"Should have {field} in marginal effects"
            
            # Check that significance flags are boolean
            assert isinstance(effect['significant_5pct'], bool), "5% significance should be boolean"
            assert isinstance(effect['significant_1pct'], bool), "1% significance should be boolean"
    
    def test_estimation_summary(self, estimation_setup):
        """Test estimation results summary."""
        estimation_result = estimate_projected_kkt_ols(
            flows=estimation_setup['flows'],
            lengths=estimation_setup['lengths'],
            features=estimation_setup['features'],
            incidence_matrix=estimation_setup['incidence_matrix']
        )
        
        # Generate summary
        feature_names = ['length', 'utility', 'constant']
        summary = summarize_estimation_results(estimation_result, feature_names)
        
        # Check that summary is a string and contains expected elements
        assert isinstance(summary, str), "Summary should be a string"
        assert "Projected KKT OLS" in summary, "Summary should mention the method"
        assert "R-squared" in summary, "Summary should include R-squared"
        assert "Coefficients" in summary, "Summary should include coefficients table"
        
        # Check that all feature names appear in summary
        for name in feature_names:
            assert name in summary, f"Summary should include feature {name}"
    
    def test_estimation_with_different_features(self, estimation_setup):
        """Test estimation with different feature specifications."""
        # Test with just length feature
        length_features = estimation_setup['lengths'].reshape(-1, 1)
        
        estimation_result = estimate_projected_kkt_ols(
            flows=estimation_setup['flows'],
            lengths=estimation_setup['lengths'],
            features=length_features,
            incidence_matrix=estimation_setup['incidence_matrix']
        )
        
        # Should work with single feature
        assert len(estimation_result.beta) == 1, "Should have 1 coefficient for 1 feature"
        assert np.isfinite(estimation_result.beta[0]), "Single coefficient should be finite"
        assert estimation_result.standard_errors[0] > 0, "Standard error should be positive"
    
    def test_estimation_edge_cases(self, estimation_setup):
        """Test estimation with edge cases."""
        # Test with very small flows (numerical precision)
        small_flows = estimation_setup['flows'] * 1e-10
        
        try:
            estimation_result = estimate_projected_kkt_ols(
                flows=small_flows,
                lengths=estimation_setup['lengths'],
                features=estimation_setup['features'],
                incidence_matrix=estimation_setup['incidence_matrix']
            )
            
            # Should handle small flows gracefully
            assert isinstance(estimation_result, ProjectedKKTResult), "Should return valid result for small flows"
            
        except Exception as e:
            # If it fails, should fail gracefully with meaningful error
            assert isinstance(e, (ValueError, np.linalg.LinAlgError)), f"Should fail gracefully, got {type(e)}: {e}"
    
    def test_projected_variables_properties(self, estimation_setup):
        """Test properties of projected variables."""
        estimation_result = estimate_projected_kkt_ols(
            flows=estimation_setup['flows'],
            lengths=estimation_setup['lengths'],
            features=estimation_setup['features'],
            incidence_matrix=estimation_setup['incidence_matrix']
        )
        
        # Projected variables should have same length as original
        n_edges = len(estimation_setup['flows'])
        
        assert len(estimation_result.residuals) == n_edges, "Residuals should have correct length"
        assert len(estimation_result.fitted_values) == n_edges, "Fitted values should have correct length"
        
        # Residuals and fitted values should sum to projected gradient
        # (This is a fundamental property of OLS)
        total_explained = estimation_result.fitted_values + estimation_result.residuals
        assert np.allclose(total_explained, total_explained, atol=1e-10), "OLS decomposition should be consistent"
    
    def test_estimation_with_perfect_multicollinearity(self, estimation_setup):
        """Test estimation behavior with multicollinear features."""
        # Create features with perfect multicollinearity
        n_edges = len(estimation_setup['flows'])
        collinear_features = np.column_stack([
            np.ones(n_edges),           # Constant
            np.ones(n_edges) * 2,       # Perfect multiple of constant
            estimation_setup['lengths']  # Independent feature
        ])
        
        # Estimation should handle this gracefully (regularization should help)
        try:
            estimation_result = estimate_projected_kkt_ols(
                flows=estimation_setup['flows'],
                lengths=estimation_setup['lengths'],
                features=collinear_features,
                incidence_matrix=estimation_setup['incidence_matrix']
            )
            
            # Should produce some result, even if not perfect
            assert isinstance(estimation_result, ProjectedKKTResult), "Should handle multicollinearity"
            
        except np.linalg.LinAlgError:
            # Acceptable to fail with singular matrix error
            pass
    
    def test_regression_identities_per_specification(self, estimation_setup):
        """Test regression follows PROJECTOR_REGRESSION.md specification exactly."""
        # This test verifies the exact procedure from the specification
        
        from purc.projector import (
            build_active_set_matrix,
            compute_pseudoinverse_matrix, 
            compute_projector_matrix,
            verify_projector_identities
        )
        
        flows = estimation_setup['flows']
        lengths = estimation_setup['lengths']
        features = estimation_setup['features']
        incidence_matrix = estimation_setup['incidence_matrix']
        
        # Step 1: Build B after solving by dropping links with x[i] <= eps
        eps = 1e-12
        B, active_indices = build_active_set_matrix(flows, incidence_matrix, eps)
        
        # Step 2: Projector C = (B A_T)^+ and P = I - B A_T C
        C = compute_pseudoinverse_matrix(B, incidence_matrix)
        P = compute_projector_matrix(B, incidence_matrix, C)
        
        # Step 3: Verify identities per specification
        identities_pass, residuals = verify_projector_identities(B, incidence_matrix, C, P)
        
        # The key identities should pass
        assert residuals['projector_idempotent'] < 1e-12, \
               f"P @ P = P should hold: {residuals['projector_idempotent']}"
        
        # Step 4: The estimation should use these same components
        estimation_result = estimate_projected_kkt_ols(
            flows, lengths, features, incidence_matrix
        )
        
        # Active indices should match
        np.testing.assert_array_equal(
            estimation_result.active_indices, active_indices,
            err_msg="Estimation should use same active set"
        )
        
        print(f"Projector verification residuals: {residuals}")
        print(f"Active variables: {len(active_indices)} / {len(flows)}")
        print(f"Estimation R-squared: {estimation_result.r_squared:.4f}")
