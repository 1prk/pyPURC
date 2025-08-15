"""
PURC I/O Module - Read/write networks and fixtures

Handles loading and saving of:
- Network topologies (JSON, GraphML, OSM XML)
- Test fixtures and gold standards
- Estimation results and diagnostics
- Configuration parameters

Paper References:
- Network data formats for reproducible research
- OSM integration per OSM_INTEGRATION.md
"""

import json
import pickle
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings

import numpy as np
import networkx as nx

from .graph import load_toy_network
from .core import PURCResult
from .estimation import ProjectedKKTResult


def load_network_json(filepath: Union[str, Path]) -> Tuple[nx.MultiDiGraph, Dict[str, Any]]:
    """
    Load network from JSON file.
    
    Supports the toy network format used in gold tests and general network formats.
    
    Args:
        filepath: Path to JSON network file
        
    Returns:
        Tuple of (NetworkX graph, metadata)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Network file not found: {filepath}")
    
    try:
        # Try toy network format first
        return load_toy_network(str(filepath))
    except Exception as e:
        # Try general network format
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        G = nx.MultiDiGraph()
        
        # Add nodes
        if 'nodes' in data:
            if isinstance(data['nodes'], list):
                for node in data['nodes']:
                    if isinstance(node, str):
                        G.add_node(node)
                    else:
                        G.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
            elif isinstance(data['nodes'], dict):
                for node_id, node_data in data['nodes'].items():
                    G.add_node(node_id, **node_data)
        
        # Add edges
        if 'edges' in data:
            for edge in data['edges']:
                G.add_edge(
                    edge['source'] if 'source' in edge else edge['tail'],
                    edge['target'] if 'target' in edge else edge['head'],
                    key=edge.get('id', edge.get('key')),
                    **{k: v for k, v in edge.items() if k not in ['source', 'target', 'tail', 'head', 'id', 'key']}
                )
        
        return G, data


def save_network_json(
    graph: nx.MultiDiGraph,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save network to JSON file.
    
    Args:
        graph: NetworkX MultiDiGraph
        filepath: Output file path
        metadata: Optional metadata to include
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract nodes
    nodes = []
    for node, data in graph.nodes(data=True):
        node_dict = {'id': node}
        node_dict.update(data)
        nodes.append(node_dict)
    
    # Extract edges
    edges = []
    for u, v, key, data in graph.edges(data=True, keys=True):
        edge_dict = {
            'source': u,
            'target': v,
            'key': key
        }
        edge_dict.update(data)
        edges.append(edge_dict)
    
    # Create output data
    output_data = {
        'nodes': nodes,
        'edges': edges
    }
    
    if metadata:
        output_data.update(metadata)
    
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)


def load_osm_network(filepath: Union[str, Path]) -> nx.MultiDiGraph:
    """
    Load network from OSM XML file.
    
    Per OSM_INTEGRATION.md: Highway OSM -> NetworkX MultiDiGraph
    
    Args:
        filepath: Path to OSM XML file
        
    Returns:
        NetworkX MultiDiGraph
    """
    try:
        import osmnx as ox
        
        # Load OSM data
        G = ox.graph_from_xml(str(filepath))
        
        # Convert to MultiDiGraph if needed
        if not isinstance(G, nx.MultiDiGraph):
            G = nx.MultiDiGraph(G)
        
        # Add required attributes
        for u, v, key, data in G.edges(data=True, keys=True):
            # Ensure length attribute
            if 'length' not in data:
                # Calculate length from geometry or coordinates
                if 'geometry' in data:
                    # Use geometry length
                    try:
                        from shapely.geometry import LineString
                        geom = data['geometry']
                        if isinstance(geom, LineString):
                            data['length'] = geom.length
                        else:
                            data['length'] = 1.0  # Fallback
                    except ImportError:
                        data['length'] = 1.0
                else:
                    # Calculate from node coordinates
                    try:
                        import geopy.distance
                        node_u = G.nodes[u]
                        node_v = G.nodes[v]
                        if 'y' in node_u and 'x' in node_u and 'y' in node_v and 'x' in node_v:
                            coord_u = (node_u['y'], node_u['x'])
                            coord_v = (node_v['y'], node_v['x'])
                            data['length'] = geopy.distance.geodesic(coord_u, coord_v).meters
                        else:
                            data['length'] = 1.0
                    except ImportError:
                        data['length'] = 1.0
            
            # Add default utility if not present
            if 'u' not in data:
                data['u'] = -1.0  # Default negative utility
        
        return G
        
    except ImportError:
        raise ImportError("OSM loading requires osmnx package: pip install osmnx")


def save_results_json(
    result: Union[PURCResult, ProjectedKKTResult],
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save PURC or estimation results to JSON file.
    
    Args:
        result: PURCResult or ProjectedKKTResult
        filepath: Output file path
        metadata: Optional metadata to include
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert result to dictionary
    if isinstance(result, PURCResult):
        result_dict = {
            'type': 'PURCResult',
            'flows': result.flows.tolist(),
            'dual_variables': result.dual_variables.tolist(),
            'reduced_costs': result.reduced_costs.tolist(),
            'success': result.success,
            'iterations': result.iterations,
            'objective_value': float(result.objective_value),
            'constraint_violation': float(result.constraint_violation),
            'kkt_stationarity_residual': float(result.kkt_stationarity_residual),
            'kkt_complementarity_residual': float(result.kkt_complementarity_residual)
        }
    elif isinstance(result, ProjectedKKTResult):
        result_dict = {
            'type': 'ProjectedKKTResult',
            'beta': result.beta.tolist(),
            'standard_errors': result.standard_errors.tolist(),
            't_statistics': result.t_statistics.tolist(),
            'p_values': result.p_values.tolist(),
            'r_squared': float(result.r_squared),
            'residuals': result.residuals.tolist(),
            'fitted_values': result.fitted_values.tolist(),
            'active_indices': result.active_indices.tolist(),
            'projector_diagnostics': result.projector_diagnostics
        }
    else:
        raise ValueError(f"Unsupported result type: {type(result)}")
    
    if metadata:
        result_dict['metadata'] = metadata
    
    with open(filepath, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)


def load_results_json(filepath: Union[str, Path]) -> Union[PURCResult, ProjectedKKTResult]:
    """
    Load PURC or estimation results from JSON file.
    
    Args:
        filepath: Path to results JSON file
        
    Returns:
        PURCResult or ProjectedKKTResult
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    result_type = data.get('type', 'PURCResult')
    
    if result_type == 'PURCResult':
        return PURCResult(
            flows=np.array(data['flows']),
            dual_variables=np.array(data['dual_variables']),
            reduced_costs=np.array(data['reduced_costs']),
            success=data['success'],
            iterations=data['iterations'],
            objective_value=data['objective_value'],
            constraint_violation=data['constraint_violation'],
            kkt_stationarity_residual=data['kkt_stationarity_residual'],
            kkt_complementarity_residual=data['kkt_complementarity_residual']
        )
    elif result_type == 'ProjectedKKTResult':
        return ProjectedKKTResult(
            beta=np.array(data['beta']),
            standard_errors=np.array(data['standard_errors']),
            t_statistics=np.array(data['t_statistics']),
            p_values=np.array(data['p_values']),
            r_squared=data['r_squared'],
            residuals=np.array(data['residuals']),
            fitted_values=np.array(data['fitted_values']),
            active_indices=np.array(data['active_indices']),
            projector_matrix=np.eye(len(data['residuals'])),  # Cannot save/load large matrices in JSON
            projector_diagnostics=data['projector_diagnostics']
        )
    else:
        raise ValueError(f"Unknown result type: {result_type}")


def save_results_pickle(
    result: Union[PURCResult, ProjectedKKTResult],
    filepath: Union[str, Path]
) -> None:
    """
    Save results to pickle file (preserves all numpy arrays exactly).
    
    Args:
        result: PURCResult or ProjectedKKTResult
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(result, f)


def load_results_pickle(filepath: Union[str, Path]) -> Union[PURCResult, ProjectedKKTResult]:
    """
    Load results from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        PURCResult or ProjectedKKTResult
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def export_flows_csv(
    flows: np.ndarray,
    edge_mapping: Dict[int, int],
    filepath: Union[str, Path],
    edge_attributes: Optional[Dict[int, Dict[str, Any]]] = None
) -> None:
    """
    Export flows to CSV file.
    
    Args:
        flows: Flow vector
        edge_mapping: Mapping from edge IDs to array indices
        filepath: Output CSV file path
        edge_attributes: Optional edge attributes to include
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Create reverse mapping
    index_to_edge = {idx: edge_id for edge_id, idx in edge_mapping.items()}
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['edge_id', 'flow']
        if edge_attributes:
            # Get all possible attribute names
            all_attrs = set()
            for attrs in edge_attributes.values():
                all_attrs.update(attrs.keys())
            header.extend(sorted(all_attrs))
        
        writer.writerow(header)
        
        # Data rows
        for idx, flow in enumerate(flows):
            edge_id = index_to_edge.get(idx, idx)
            row = [edge_id, flow]
            
            if edge_attributes and edge_id in edge_attributes:
                attrs = edge_attributes[edge_id]
                for attr_name in sorted(all_attrs):
                    row.append(attrs.get(attr_name, ''))
            
            writer.writerow(row)


def load_test_fixture(fixture_name: str) -> Dict[str, Any]:
    """
    Load test fixture from tests/gold/ directory.
    
    Args:
        fixture_name: Name of fixture file (without extension)
        
    Returns:
        Fixture data dictionary
    """
    # Look for fixture in tests/gold/
    fixture_path = Path(__file__).parent.parent.parent / 'tests' / 'gold' / f'{fixture_name}.json'
    
    if not fixture_path.exists():
        raise FileNotFoundError(f"Test fixture not found: {fixture_path}")
    
    with open(fixture_path, 'r') as f:
        return json.load(f)


def save_test_fixture(
    data: Dict[str, Any],
    fixture_name: str
) -> None:
    """
    Save test fixture to tests/gold/ directory.
    
    Args:
        data: Fixture data dictionary
        fixture_name: Name of fixture file (without extension)
    """
    fixture_path = Path(__file__).parent.parent.parent / 'tests' / 'gold' / f'{fixture_name}.json'
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(fixture_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def create_network_summary(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    """
    Create summary statistics for a network.
    
    Args:
        graph: NetworkX MultiDiGraph
        
    Returns:
        Summary statistics dictionary
    """
    summary = {
        'n_nodes': graph.number_of_nodes(),
        'n_edges': graph.number_of_edges(),
        'n_connected_components': nx.number_weakly_connected_components(graph),
        'is_strongly_connected': nx.is_strongly_connected(graph),
        'average_degree': np.mean([d for n, d in graph.degree()]),
        'average_in_degree': np.mean([d for n, d in graph.in_degree()]),
        'average_out_degree': np.mean([d for n, d in graph.out_degree()])
    }
    
    # Edge attributes summary
    edge_attrs = set()
    for u, v, key, data in graph.edges(data=True, keys=True):
        edge_attrs.update(data.keys())
    
    summary['edge_attributes'] = list(edge_attrs)
    
    # Node attributes summary
    node_attrs = set()
    for node, data in graph.nodes(data=True):
        node_attrs.update(data.keys())
    
    summary['node_attributes'] = list(node_attrs)
    
    return summary
