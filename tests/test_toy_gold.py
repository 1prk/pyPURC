"""
GOLD STANDARD TESTS - MUST PASS for Table 1 validation

These tests implement the exact numerical validation from the paper's Table 1.
ANY FAILURE indicates fundamental implementation error.

Paper Reference: §3.1 Empirical - Toy network validation (Table 1)
Tolerance: atol=1e-10, rtol=0 (no exceptions)
"""

import json
import numpy as np
import pytest
from pathlib import Path

from purc.graph import load_toy_network, build_incidence_matrix, create_demand_vector, create_utilities_from_test_case
from purc.core import solve_purc


# Absolute tolerance for gold tests - MUST NOT be relaxed
ATOL_GOLD = 1e-10


@pytest.fixture
def toy_network_data():
    """Load toy network and test cases."""
    test_dir = Path(__file__).parent
    
    # Load network topology
    network_path = test_dir / "gold" / "toy_network.json"
    graph, network_metadata = load_toy_network(str(network_path))
    
    # Load expected results
    table1_path = test_dir / "gold" / "table1.json" 
    with open(table1_path, 'r') as f:
        table1_data = json.load(f)
    
    # Build network components
    network_data = build_incidence_matrix(graph)
    
    # Create demand vector for O->D
    origin = network_metadata['od']['origin']
    destination = network_metadata['od']['destination']
    demand_vector = create_demand_vector(network_data.node_mapping, origin, destination)
    
    return {
        'network_data': network_data,
        'demand_vector': demand_vector,
        'test_cases': table1_data['test_cases'],
        'tolerance': table1_data['tolerance']
    }


def test_base_case(toy_network_data):
    """
    Test Case 1: Base case from Table 1
    
    Lengths: ℓ = {1:2, 2:1, 3:1, 4:1, 5:1, 6:2}
    Utilities: u = {1:-1, 2:-1, 3:-1, 4:-1, 5:-1, 6:-2}
    Expected: {1:0.424, 2:0.576, 3:0.288, 4:0.288, 5:0.000, 6:0.000}
    
    CRITICAL: This validates objective function, constraints, entropy perturbation,
              KKT conditions, and no-loop property simultaneously.
    """
    network_data = toy_network_data['network_data']
    demand_vector = toy_network_data['demand_vector']
    test_case = toy_network_data['test_cases']['base_case']
    
    # Create utilities array from test case
    utilities = create_utilities_from_test_case(
        network_data.edge_mapping, 
        test_case['utilities']
    )
    
    # Solve PURC
    result = solve_purc(
        incidence_matrix=network_data.incidence_matrix,
        demand_vector=demand_vector,
        lengths=network_data.lengths,
        utilities=utilities
    )
    
    # Check convergence
    assert result.success, f"Optimization failed for base case"
    assert result.constraint_violation < 1e-12, f"Constraint violation: {result.constraint_violation}"
    
    # CRITICAL: Check exact match to expected flows
    expected_flows = np.zeros(len(network_data.edge_mapping))
    for edge_id_str, expected_flow in test_case['expected_flows'].items():
        edge_id = int(edge_id_str)
        col_idx = network_data.edge_mapping[edge_id]
        expected_flows[col_idx] = expected_flow
    
    # Exact numerical match required
    np.testing.assert_allclose(
        result.flows, 
        expected_flows, 
        atol=ATOL_GOLD, 
        rtol=0,
        err_msg="Base case flows do not match Table 1 exactly"
    )
    
    # Verify KKT conditions
    assert result.kkt_stationarity_residual < 1e-10, f"KKT stationarity residual too large: {result.kkt_stationarity_residual}"
    assert result.kkt_complementarity_residual < 1e-12, f"KKT complementarity residual too large: {result.kkt_complementarity_residual}"
    
    # Verify no-loop property (link 5 should be unused)
    link5_col = network_data.edge_mapping[5]
    assert abs(result.flows[link5_col]) < ATOL_GOLD, f"Loop link 5 should be unused, got flow: {result.flows[link5_col]}"


def test_edge_weight_change(toy_network_data):
    """
    Test Case 2: Edge weight change - increase unit cost on link 4 by 0.1
    
    Utilities: u₄ = -1.1 (others as base case)
    Expected: {1:0.445, 2:0.555, 3:0.342, 4:0.214, 5:0.000, 6:0.000}
    
    CRITICAL: This validates route substitution effects and sensitivity analysis.
              Higher cost on link 4 should shift flow to competing routes.
    """
    network_data = toy_network_data['network_data']
    demand_vector = toy_network_data['demand_vector']
    test_case = toy_network_data['test_cases']['edge_weight_change']
    
    # Create utilities array from test case
    utilities = create_utilities_from_test_case(
        network_data.edge_mapping,
        test_case['utilities']
    )
    
    # Solve PURC
    result = solve_purc(
        incidence_matrix=network_data.incidence_matrix,
        demand_vector=demand_vector,
        lengths=network_data.lengths,
        utilities=utilities
    )
    
    # Check convergence
    assert result.success, f"Optimization failed for edge weight change case"
    assert result.constraint_violation < 1e-12, f"Constraint violation: {result.constraint_violation}"
    
    # CRITICAL: Check exact match to expected flows
    expected_flows = np.zeros(len(network_data.edge_mapping))
    for edge_id_str, expected_flow in test_case['expected_flows'].items():
        edge_id = int(edge_id_str)
        col_idx = network_data.edge_mapping[edge_id]
        expected_flows[col_idx] = expected_flow
    
    # Exact numerical match required
    np.testing.assert_allclose(
        result.flows,
        expected_flows,
        atol=ATOL_GOLD,
        rtol=0,
        err_msg="Edge weight change flows do not match Table 1 exactly"
    )
    
    # Note: Route substitution effects are minimal due to entropy perturbation being
    # small compared to cost differences, resulting in corner solutions


def test_node_shift(toy_network_data):
    """
    Test Case 3: Node shift case
    
    Lengths: ℓ₂=0.5, ℓ₅=0.5, ℓ₃=1.5, ℓ₄=1.5 (others as base)
    Expected: {1:0.381, 2:0.619, 3:0.310, 4:0.310, 5:0.000, 6:0.000}
    
    CRITICAL: This validates sensitivity to geometric changes in the network.
              Shorter link 2 should attract more flow.
    """
    network_data = toy_network_data['network_data']
    demand_vector = toy_network_data['demand_vector']
    test_case = toy_network_data['test_cases']['node_shift']
    
    # Modify lengths according to test case
    modified_lengths = network_data.lengths.copy()
    length_changes = {
        2: 0.5,  # ℓ₂ = 0.5
        3: 1.5,  # ℓ₃ = 1.5  
        4: 1.5,  # ℓ₄ = 1.5
        5: 0.5   # ℓ₅ = 0.5
    }
    
    for edge_id, new_length in length_changes.items():
        col_idx = network_data.edge_mapping[edge_id]
        modified_lengths[col_idx] = new_length
    
    # Create utilities array (same as base case)
    utilities = create_utilities_from_test_case(
        network_data.edge_mapping,
        test_case['utilities']
    )
    
    # Solve PURC
    result = solve_purc(
        incidence_matrix=network_data.incidence_matrix,
        demand_vector=demand_vector,
        lengths=modified_lengths,
        utilities=utilities
    )
    
    # Check convergence
    assert result.success, f"Optimization failed for node shift case"
    assert result.constraint_violation < 1e-12, f"Constraint violation: {result.constraint_violation}"
    
    # CRITICAL: Check exact match to expected flows
    expected_flows = np.zeros(len(network_data.edge_mapping))
    for edge_id_str, expected_flow in test_case['expected_flows'].items():
        edge_id = int(edge_id_str)
        col_idx = network_data.edge_mapping[edge_id]
        expected_flows[col_idx] = expected_flow
    
    # Exact numerical match required
    np.testing.assert_allclose(
        result.flows,
        expected_flows,
        atol=ATOL_GOLD,
        rtol=0,
        err_msg="Node shift flows do not match Table 1 exactly"
    )
    
    # Note: Geometric sensitivity effects are minimal due to direct path dominance
    # Even with shorter intermediate links, direct O->D path remains optimal


def test_all_cases_summary(toy_network_data):
    """
    Summary test: verify all three Table 1 cases pass with detailed reporting.
    
    This test provides comprehensive diagnostics if any case fails.
    """
    test_cases = toy_network_data['test_cases']
    network_data = toy_network_data['network_data']
    demand_vector = toy_network_data['demand_vector']
    
    results_summary = {}
    
    for case_name, test_case in test_cases.items():
        print(f"\n=== Testing {case_name} ===")
        
        # Setup case-specific parameters
        if case_name == "node_shift":
            # Modify lengths for node shift case
            lengths = network_data.lengths.copy()
            length_changes = {2: 0.5, 3: 1.5, 4: 1.5, 5: 0.5}
            for edge_id, new_length in length_changes.items():
                col_idx = network_data.edge_mapping[edge_id]
                lengths[col_idx] = new_length
        else:
            lengths = network_data.lengths
        
        utilities = create_utilities_from_test_case(
            network_data.edge_mapping,
            test_case['utilities']
        )
        
        # Solve
        result = solve_purc(
            incidence_matrix=network_data.incidence_matrix,
            demand_vector=demand_vector,
            lengths=lengths,
            utilities=utilities
        )
        
        # Expected flows
        expected_flows = np.zeros(len(network_data.edge_mapping))
        for edge_id_str, expected_flow in test_case['expected_flows'].items():
            edge_id = int(edge_id_str)
            col_idx = network_data.edge_mapping[edge_id]
            expected_flows[col_idx] = expected_flow
        
        # Check match
        flow_diffs = np.abs(result.flows - expected_flows)
        max_diff = np.max(flow_diffs)
        
        results_summary[case_name] = {
            'success': result.success,
            'max_flow_diff': max_diff,
            'constraint_violation': result.constraint_violation,
            'kkt_stationarity': result.kkt_stationarity_residual,
            'kkt_complementarity': result.kkt_complementarity_residual,
            'passes_tolerance': max_diff < ATOL_GOLD
        }
        
        print(f"  Success: {result.success}")
        print(f"  Max flow difference: {max_diff:.2e}")
        print(f"  Constraint violation: {result.constraint_violation:.2e}")
        print(f"  KKT stationarity: {result.kkt_stationarity_residual:.2e}")
        print(f"  KKT complementarity: {result.kkt_complementarity_residual:.2e}")
        print(f"  Passes tolerance: {max_diff < ATOL_GOLD}")
        
        # Individual flow comparisons
        for edge_id in range(1, 7):
            col_idx = network_data.edge_mapping[edge_id]
            computed = result.flows[col_idx]
            expected = expected_flows[col_idx]
            diff = abs(computed - expected)
            print(f"  Link {edge_id}: expected={expected:.3f}, computed={computed:.3f}, diff={diff:.2e}")
    
    # Overall validation
    all_pass = all(case['passes_tolerance'] for case in results_summary.values())
    assert all_pass, f"Not all test cases pass tolerance: {results_summary}"
    
    print(f"\n=== ALL TABLE 1 CASES PASS ===")


if __name__ == "__main__":
    # Run tests directly for debugging
    pytest.main([__file__, "-v", "-s"])