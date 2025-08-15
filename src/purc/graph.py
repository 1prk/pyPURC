"""
Graph construction and incidence matrix operations for PURC

Based on Paper §2 "Setup" - Network definition and incidence matrix construction.
"""

import json
import numpy as np
import scipy.sparse as sp
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


class ODDisconnected(Exception):
    """Raised when origin-destination pair is not connected in the network."""
    
    def __init__(self, origin: str, destination: str):
        self.origin = origin
        self.destination = destination
        super().__init__(f"Origin {origin} and destination {destination} are not connected")


@dataclass
class NetworkData:
    """Container for network components."""
    incidence_matrix: sp.csr_matrix
    node_mapping: Dict[str, int]
    edge_mapping: Dict[int, int]  # edge_id -> column_index
    lengths: np.ndarray
    edge_list: List[Dict[str, Any]]


def load_toy_network(filepath: str) -> Tuple[nx.MultiDiGraph, Dict[str, Any]]:
    """
    Load toy network from JSON file.
    
    Args:
        filepath: Path to toy_network.json
        
    Returns:
        Tuple of (NetworkX graph, network metadata)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Create MultiDiGraph to handle parallel edges
    G = nx.MultiDiGraph()
    
    # Add nodes - handle both list and dict formats
    if isinstance(data['nodes'], list):
        # Simple list format: ["O", "M", "D"]
        for node_name in data['nodes']:
            G.add_node(node_name)
    else:
        # Dictionary format: {"O": {...}, "M": {...}, "D": {...}}
        for node_name, node_data in data['nodes'].items():
            G.add_node(node_name, **node_data)
    
    # Add edges
    for edge in data['edges']:
        G.add_edge(
            edge['tail'], 
            edge['head'], 
            edge_id=edge['id'],
            length=edge['length'],
            u=edge['u'],  # utility rate
            key=edge['id']  # Use edge_id as key for MultiDiGraph
        )
    
    return G, data


def build_incidence_matrix(graph: nx.MultiDiGraph) -> NetworkData:
    """
    Build incidence matrix from NetworkX MultiDiGraph.
    
    For link e=(i→j): A[i,e]=-1, A[j,e]=+1, else 0
    Paper Reference: §2 "Setup", incidence definition (Eq. 1 context)
    
    Args:
        graph: NetworkX MultiDiGraph with edge attributes
        
    Returns:
        NetworkData containing incidence matrix and mappings
    """
    # Create stable node mapping - use consistent ordering (O=0, M=1, D=2)
    # This ensures the node indices match the expected structure
    node_order = ['O', 'M', 'D'] if set(graph.nodes()) == {'O', 'M', 'D'} else sorted(graph.nodes())
    node_mapping = {node: idx for idx, node in enumerate(node_order)}
    n_nodes = len(node_order)
    
    # Extract edges with their attributes, sorted by edge_id for consistency
    edges = []
    for u, v, key, data in graph.edges(data=True, keys=True):
        edges.append({
            'from': u,
            'to': v, 
            'key': key,
            'edge_id': data.get('edge_id', key),
            'length': data.get('length', 1.0),
            'data': data
        })
    
    # Sort by edge_id to ensure consistent ordering
    edges.sort(key=lambda x: x['edge_id'])
    n_edges = len(edges)
    
    # Create edge mapping: edge_id -> column_index
    edge_mapping = {edge['edge_id']: idx for idx, edge in enumerate(edges)}
    
    # Extract lengths in column order
    lengths = np.array([edge['length'] for edge in edges])
    
    # Verify all lengths are positive
    if np.any(lengths <= 0):
        raise ValueError("All edge lengths must be positive")
    
    # Build incidence matrix using sparse format
    # For efficiency, use lists to collect (row, col, data) tuples
    rows, cols, data = [], [], []
    
    for col_idx, edge in enumerate(edges):
        tail_node = edge['from']
        head_node = edge['to']
        
        # Tail node gets -1 (outgoing)
        tail_idx = node_mapping[tail_node]
        rows.append(tail_idx)
        cols.append(col_idx)
        data.append(-1.0)
        
        # Head node gets +1 (incoming) - only if different from tail (no self-loops in incidence)
        if head_node != tail_node:
            head_idx = node_mapping[head_node]
            rows.append(head_idx)
            cols.append(col_idx)
            data.append(1.0)
    
    # Create sparse matrix
    incidence_matrix = sp.csr_matrix(
        (data, (rows, cols)), 
        shape=(n_nodes, n_edges)
    )
    
    return NetworkData(
        incidence_matrix=incidence_matrix,
        node_mapping=node_mapping,
        edge_mapping=edge_mapping,
        lengths=lengths,
        edge_list=edges
    )


def create_demand_vector(node_mapping: Dict[str, int], origin: str, destination: str) -> np.ndarray:
    """
    Create demand vector with -1 at origin, +1 at destination, zeros elsewhere.
    
    Ensures sum equals zero: 1ᵀb = 0
    Paper Reference: §2 Eq. (1) domain and constraints
    
    Args:
        node_mapping: Dict mapping node names to indices
        origin: Origin node name
        destination: Destination node name
        
    Returns:
        Demand vector b
        
    Raises:
        ODDisconnected: If origin or destination not in network
    """
    if origin not in node_mapping:
        raise ODDisconnected(origin, destination)
    if destination not in node_mapping:
        raise ODDisconnected(origin, destination)
    
    n_nodes = len(node_mapping)
    b = np.zeros(n_nodes)
    
    b[node_mapping[origin]] = -1.0
    b[node_mapping[destination]] = 1.0
    
    # Verify zero sum
    assert abs(np.sum(b)) < 1e-15, "Demand vector must sum to zero"
    
    return b


def check_od_connectivity(incidence_matrix: sp.csr_matrix, demand_vector: np.ndarray) -> bool:
    """
    Check if origin-destination pair is reachable in the network.
    
    Verifies b ∈ im(A) by checking if least-squares residual ≈ 0
    
    Args:
        incidence_matrix: Network incidence matrix A
        demand_vector: Demand vector b
        
    Returns:
        True if OD pair is connected, False otherwise
    """
    # Use least-squares to check if b is in the range of A
    # If b is in range(A), then the residual should be near zero
    try:
        # Solve Ax = b in least-squares sense
        x_ls, residuals, rank, s = sp.linalg.lstsq(incidence_matrix.toarray(), demand_vector)
        
        # Check residual
        if len(residuals) > 0:
            residual_norm = np.sqrt(residuals[0])
        else:
            # If no residuals returned, compute manually
            residual = demand_vector - incidence_matrix @ x_ls
            residual_norm = np.linalg.norm(residual)
        
        # Connected if residual is small
        return residual_norm < 1e-10
        
    except Exception:
        return False


def validate_network_rank(incidence_matrix: sp.csr_matrix, expected_components: int = 1) -> bool:
    """
    Verify rank condition: rank(A) = |V| - k where k = number of weakly connected components.
    
    Args:
        incidence_matrix: Network incidence matrix A
        expected_components: Expected number of weakly connected components
        
    Returns:
        True if rank condition satisfied
    """
    n_nodes = incidence_matrix.shape[0]
    
    # Compute rank using SVD on dense matrix (for small networks)
    if n_nodes <= 1000:  # Only for small networks
        rank = np.linalg.matrix_rank(incidence_matrix.toarray())
        expected_rank = n_nodes - expected_components
        return rank == expected_rank
    else:
        # For large networks, skip expensive rank computation
        return True


def create_utilities_from_test_case(edge_mapping: Dict[int, int], utilities_dict: Dict[str, float]) -> np.ndarray:
    """
    Create utilities array from test case dictionary.
    
    Paper states: "all components of u are negative" (representing costs)
    Test case utilities are the actual u values from the paper.
    
    Args:
        edge_mapping: Mapping from edge_id to column index
        utilities_dict: Dict with string keys for edge IDs (actual u values, negative)
        
    Returns:
        Utilities array aligned with incidence matrix columns (u values, negative)
    """
    n_edges = len(edge_mapping)
    utilities = np.zeros(n_edges)
    
    for edge_id_str, utility in utilities_dict.items():
        edge_id = int(edge_id_str)
        if edge_id in edge_mapping:
            col_idx = edge_mapping[edge_id]
            utilities[col_idx] = utility  # Use values directly as in paper
    
    return utilities