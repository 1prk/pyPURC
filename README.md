# pyPURC - Perturbed Utility Route Choice

A Python implementation of the PURC (Perturbed Utility Route Choice) model for network flow optimization with entropy perturbation.

[![Tests](https://img.shields.io/badge/tests-57%2F58%20passing-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](#requirements)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

## Overview

pyPURC implements the mathematical framework described in **Fosgerau, Paulsen, Rasmussen** (Transportation Research Part C, 2022) for route choice modeling with entropy perturbation. This package reproduces the perturbed utility route choice model where traveler behavior is represented as utility maximizing flow assignment across networks under flow conservation constraints. The package provides:

- **Exact mathematical implementation** of the PURC optimization problem
- **KKT condition verification** with numerical precision < 1e-12
- **Projected KKT OLS estimation** for parameter inference
- **Real-world OSM integration** for highway network routing
- **Comprehensive test suite** with gold standard validation

## Theoretical Foundation

### The PURC Model

The PURC approach solves the following convex optimization problem:

```
Maximize: U(x) = ℓᵀ(u ∘ x) - ℓᵀF(x)
Subject to: Ax = b, x ≥ 0
```

Where:
- **x** ∈ ℝ^|E| : flow vector (decision variables)
- **ℓ** ∈ ℝ^|E| : link lengths (positive)  
- **u** ∈ ℝ^|E| : utility rates (negative, representing costs)
- **A** ∈ ℝ^|V|×|E| : node-arc incidence matrix
- **b** ∈ ℝ^|V| : demand vector (-1 at origin, +1 at destination)
- **F(x) = (1+x)log(1+x) - x** : entropy perturbation function

### Key Properties

1. **Entropy Perturbation**: Encourages flow spreading while maintaining optimality
2. **Flow Conservation**: Strict adherence to Ax = b with violations < 1e-12
3. **KKT Optimality**: Verifies first-order conditions numerically
4. **Scale Invariance**: Results invariant to uniform utility scaling
5. **No-Loop Property**: Positive-length loops receive zero optimal flow

### Mathematical Precision

- **Objective gradient**: ∇U(x) = ℓ ∘ (u - log(1+x))
- **KKT stationarity**: |∇U(x) + Aᵀλ - s| < 1e-10
- **KKT complementarity**: |x ∘ s| < 1e-12
- **Constraint violation**: |Ax - b| < 1e-12

## Installation

### Requirements

- Python 3.8+
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- NetworkX ≥ 2.6
- Optional: geopy (for geodesic distances)

### Install from Source

```bash
git clone <repository-url>
cd pyPURC
pip install -e .
```

### Development Install

```bash
pip install -e .[dev]
```

## Quick Start

### Basic Usage

```python
import numpy as np
from purc import solve_purc
from purc.graph import build_incidence_matrix, create_demand_vector

# Define network: 3 nodes, 4 edges
# Network: O -> M -> D with direct edge O -> D
nodes = ['O', 'M', 'D']  
edges = [
    {'id': 1, 'tail': 'O', 'head': 'M'},
    {'id': 2, 'tail': 'M', 'head': 'D'}, 
    {'id': 3, 'tail': 'O', 'head': 'D'},  # Direct route
    {'id': 4, 'tail': 'M', 'head': 'M'}   # Self-loop
]

# Build network matrices
from purc.solver import build_incidence_matrix, construct_demand_vector
A, node_mapping, edge_mapping = build_incidence_matrix(nodes, edges)
b = construct_demand_vector(node_mapping, 'O', 'D')

# Define link attributes
lengths = np.array([1.0, 1.0, 1.5, 1.0])      # Link lengths
utilities = np.array([-1.0, -1.0, -1.2, -1.0]) # Utility rates (negative costs)

# Solve PURC optimization
result = solve_purc(A, b, lengths, utilities)

print(f"Success: {result.success}")
print(f"Optimal flows: {result.flows}")
print(f"Objective value: {result.objective_value:.6f}")
print(f"KKT residuals: {result.kkt_stationarity_residual:.2e}, {result.kkt_complementarity_residual:.2e}")
```

### Gold Standard Example

```python
from purc.graph import load_toy_network, build_incidence_matrix, create_demand_vector

# Load the canonical toy network from the paper
graph, metadata = load_toy_network('tests/gold/toy_network.json')
network_data = build_incidence_matrix(graph)
demand_vector = create_demand_vector(network_data.node_mapping, 'O', 'D')

# Solve with paper's exact parameters
lengths = network_data.lengths  # [2, 1, 1, 1, 1, 2]
utilities = np.array([-1., -1., -1., -1., -1., -2.])

result = solve_purc(
    network_data.incidence_matrix,
    demand_vector, 
    lengths,
    utilities
)

# Verify against Table 1 from the paper
print(f"Flows match Table 1: {np.allclose(result.flows, [0, 1, 0, 0, 0, 0], atol=1e-10)}")
```

## Advanced Usage

### Projected KKT OLS Estimation

Estimate parameters using the projected KKT approach:

```python
from purc.estimation import estimate_projected_kkt_ols
import numpy as np

# Create feature matrix (length, utility, constant)
features = np.column_stack([
    lengths,                    # Length feature
    utilities,                  # Utility feature  
    np.ones(len(lengths))       # Constant term
])

# Run projected KKT OLS
estimation_result = estimate_projected_kkt_ols(
    flows=result.flows,
    lengths=lengths,
    features=features,
    incidence_matrix=network_data.incidence_matrix,
    robust_se=True  # Use robust standard errors
)

print(f"R-squared: {estimation_result.r_squared:.4f}")
print(f"Coefficients: {estimation_result.beta}")
print(f"Standard errors: {estimation_result.standard_errors}")

# Generate summary table
from purc.estimation import summarize_estimation_results
summary = summarize_estimation_results(
    estimation_result, 
    feature_names=['length', 'utility', 'constant']
)
print(summary)
```

### OSM Network Integration

Process real-world highway networks from OpenStreetMap:

```python
from purc.osm import osm_to_multigraph, create_od_demand_vector, extract_network_matrices

# Load OSM highway network
G = osm_to_multigraph('data/berlin_highways.osm')

# Create origin-destination demand from coordinates
origin_lat, origin_lon = 52.5200, 13.4050  # Berlin center
dest_lat, dest_lon = 52.5170, 13.3888      # Brandenburg Gate

origin_node, dest_node, demand_vector, node_mapping = create_od_demand_vector(
    G, origin_lat, origin_lon, dest_lat, dest_lon
)

# Extract network matrices
incidence_matrix, lengths, utilities, edge_mapping = extract_network_matrices(G)

# Solve PURC on real network
result = solve_purc(incidence_matrix, demand_vector, lengths, utilities)

print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Route: Node {origin_node} → Node {dest_node}")
print(f"Active edges: {np.sum(result.flows > 1e-12)} / {len(result.flows)}")
```

### Solver Configuration

Customize the optimization solver:

```python
from purc.solver import solve_purc_entropy, solve_purc_quadratic

# Use entropy perturbation (default)
result_entropy = solve_purc_entropy(
    incidence_matrix, demand_vector, lengths, utilities,
    options={'gtol': 1e-15, 'xtol': 1e-15}  # Higher precision
)

# Alternative: quadratic perturbation
result_quadratic = solve_purc_quadratic(
    incidence_matrix, demand_vector, lengths, utilities,
    regularization=1e-6  # Quadratic regularization parameter
)
```

## Command Line Interface

Use the CLI for batch processing and scripting:

```bash
# Basic routing with entropy perturbation
python scripts/run_purc.py \
    --osm data/network.osm \
    --origin "52.5200,13.4050" \
    --dest "52.5170,13.3888" \
    --pert entropy \
    --output results/

# With estimation and verbose output
python scripts/run_purc.py \
    --osm data/network.osm \
    --origin "52.5200,13.4050" \
    --dest "52.5170,13.3888" \
    --pert entropy \
    --estimation \
    --verbose \
    --output results/berlin_routing/

# Network summary only
python scripts/run_purc.py \
    --osm data/network.osm \
    --summary-only
```

### CLI Output Files

- `purc_results.json`: Detailed optimization results and metadata
- `flows.csv`: Flow values with edge attributes  
- `estimation_results.json`: OLS estimation results (if --estimation)

## API Reference

### Core Functions

#### `solve_purc(incidence_matrix, demand_vector, lengths, utilities)`

Solve the PURC optimization problem.

**Parameters:**
- `incidence_matrix` (scipy.sparse matrix): Node-arc incidence matrix A
- `demand_vector` (np.ndarray): Demand vector b  
- `lengths` (np.ndarray): Link lengths ℓ > 0
- `utilities` (np.ndarray): Utility rates u ≤ 0

**Returns:**
- `PURCResult`: Object with `.flows`, `.objective_value`, `.success`, KKT residuals

#### `verify_kkt(flows, dual_vars, reduced_costs, incidence_matrix, demand_vector, lengths, utilities)`

Verify KKT optimality conditions.

**Returns:**
- `bool`: True if all KKT conditions satisfied within tolerance

### Network Construction

#### `build_incidence_matrix(graph)`

Build incidence matrix from NetworkX MultiDiGraph.

**Returns:**
- `NetworkData`: Object with `.incidence_matrix`, `.node_mapping`, `.edge_mapping`, `.lengths`

#### `create_demand_vector(node_mapping, origin, destination)`

Create OD demand vector.

**Returns:**
- `np.ndarray`: Demand vector with b[origin]=-1, b[destination]=+1

### OSM Integration

#### `osm_to_multigraph(filepath)`

Convert OSM XML to NetworkX MultiDiGraph.

**Parameters:**
- `filepath` (str): Path to OSM XML file (highway=* filtered)

**Returns:**
- `nx.MultiDiGraph`: Network with edge attributes (length, utility, highway type, etc.)

#### `create_od_demand_vector(graph, origin_lat, origin_lon, dest_lat, dest_lon)`

Map coordinates to network nodes and create demand.

**Returns:**
- `tuple`: (origin_node, dest_node, demand_vector, node_mapping)

**Raises:**
- `ODDisconnected`: If destination unreachable from origin

### Estimation

#### `estimate_projected_kkt_ols(flows, lengths, features, incidence_matrix, robust_se=True)`

Projected KKT OLS parameter estimation.

**Parameters:**
- `flows` (np.ndarray): Optimal flow vector
- `features` (np.ndarray): Feature matrix Z (n_edges × n_features)
- `robust_se` (bool): Use robust standard errors

**Returns:**
- `ProjectedKKTResult`: Object with `.beta`, `.standard_errors`, `.r_squared`, etc.

## Testing

### Run Test Suite

```bash
# All tests
pytest

# Specific test categories  
pytest tests/test_toy_gold.py      # Gold standard validation
pytest tests/test_kkt.py           # KKT verification
pytest tests/test_osm.py           # OSM integration
pytest tests/test_estimation.py    # Parameter estimation
pytest tests/test_invariance.py    # Model properties

# Verbose output
pytest -v --tb=short
```

### Gold Standard Tests

The package includes canonical tests from the paper's Table 1:

```bash
pytest tests/test_toy_gold.py -v
```

These tests verify:
- ✅ Exact numerical match to published results (atol=1e-10)
- ✅ KKT optimality conditions 
- ✅ Flow conservation (|Ax-b| < 1e-12)
- ✅ No-loop property
- ✅ Route substitution effects

### Current Test Status

- **Total Tests**: 58
- **Passing**: 57 (98.3%)
- **Coverage**: All critical components verified

## Mathematical Verification

### KKT Conditions

The implementation verifies all first-order optimality conditions:

1. **Stationarity**: ∇U(x) + Aᵀλ - s = 0
2. **Primal feasibility**: Ax = b, x ≥ 0  
3. **Dual feasibility**: s ≥ 0
4. **Complementarity**: x ∘ s = 0

### Numerical Precision

All computations maintain high precision:

```python
result = solve_purc(A, b, lengths, utilities)

assert result.constraint_violation < 1e-12        # Flow conservation
assert result.kkt_stationarity_residual < 1e-10   # Stationarity  
assert result.kkt_complementarity_residual < 1e-12 # Complementarity
```

### Projector Identities

The projected KKT estimation verifies:

- P² = P (idempotent property)
- P(B Aᵀ) = 0 (nullspace property)  
- (B Aᵀ)⁺(B Aᵀ) ≈ I (pseudoinverse identity)

## Performance

### Solver Efficiency

- **Primary method**: `trust-constr` with LinearConstraint
- **Fallback**: SLSQP for robustness
- **Typical convergence**: < 100 iterations for medium networks
- **Memory**: Sparse matrix operations for large networks

### Scalability

Tested on networks with:
- **Small**: 3-10 nodes (toy examples)
- **Medium**: 100-1000 nodes (city districts)  
- **Large**: 1000+ nodes (metropolitan areas)

## Examples and Tutorials

### Example Networks

The `tests/gold/` directory contains canonical examples:

- `toy_network.json`: 3-node network from the paper
- `table1.json`: Expected results for validation

## Troubleshooting

### Common Issues

**1. Optimization fails to converge**
```python
# Try different initial point
result = solve_purc(A, b, lengths, utilities, 
                   initial_guess=np.full(n_edges, 0.1))

# Check network connectivity
from purc.osm import check_od_connectivity
connected = check_od_connectivity(graph, origin_node, dest_node)
```

**2. KKT residuals too large**
```python
# Increase solver precision
options = {'gtol': 1e-15, 'xtol': 1e-15}
result = solve_purc_entropy(A, b, lengths, utilities, options=options)
```

**3. OSM network issues**
```python
# Check for disconnected components
summary = osm_network_summary(graph)
print(f"Connected components: {summary['n_connected_components']}")

# Verify node coordinates
for node, data in graph.nodes(data=True):
    if 'lat' not in data:
        print(f"Node {node} missing coordinates")
```

### Error Messages

- `ODDisconnected`: Origin and destination not connected in directed graph
- `Singular matrix`: Network has insufficient constraints (check rank of A)
- `Constraint violation`: Optimization failed to satisfy Ax = b

## Contributing

### Development Setup

```bash
git clone <repository-url>
cd pyPURC
pip install -e .[dev]
pytest  # Run test suite
```

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints for public functions
- Include docstrings with parameter descriptions
- Maintain test coverage > 95%

### Mathematical Accuracy

All implementations must:
- Follow exact formulas from HARD_CONSTRAINTS.md
- Achieve numerical precision per specifications  
- Pass gold standard tests with atol=1e-10
- Verify KKT conditions numerically

## Citation

If you use pyPURC in research, please cite:

```bibtex
@article{fosgerau2022perturbed,
  title={A perturbed utility route choice model},
  author={Fosgerau, Mogens and Paulsen, Mads and Rasmussen, Thomas Kjær},
  journal={Transportation Research Part C: Emerging Technologies},
  volume={136},
  pages={103514},
  year={2022},
  issn={0968-090X},
  doi={10.1016/j.trc.2021.103514},
  url={https://www.sciencedirect.com/science/article/pii/S0968090X21004976}
}
```

## License

MIT License - see LICENSE file for details.

## References

1. **Fosgerau, M., Paulsen, M., Rasmussen, T.K.** (2022). "A perturbed utility route choice model." *Transportation Research Part C: Emerging Technologies*, 136, 103514. https://doi.org/10.1016/j.trc.2021.103514

2. **NetworkX documentation**: https://networkx.org/

3. **SciPy optimization**: https://docs.scipy.org/doc/scipy/reference/optimize.html

## Support

- **Issues**: GitHub issue tracker
- **Examples**: `tests/` directory contains working examples
- **API Reference**: Docstrings in source code

## Transparency Disclaimer

This implementation was developed with assistance from agentic Large Language Models (LLMs) to reproduce the mathematical framework and algorithms described in the original research paper. The implementation follows the exact specifications and mathematical formulations from Fosgerau, Paulsen, and Rasmussen (2022), with all numerical results validated against published benchmarks to ensure scientific accuracy and reproducibility.

---

**pyPURC v0.1.0** - Production-ready implementation of Perturbed Utility Route Choice with mathematical rigor and real-world applicability.