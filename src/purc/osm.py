"""
OSM Integration Module - Highway OSM -> NetworkX MultiDiGraph

Implements OSM integration per OSM_INTEGRATION.md:
- Input: .osm XML filtered to highway=*
- Graph: NetworkX MultiDiGraph with consecutive node pairs
- oneway=yes -> forward only; oneway=-1 -> reverse only; otherwise both directions
- Preserve parallel edges as distinct edges

Paper References:
- Real-world network integration for PURC routing
- OSM highway network processing
"""

import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

try:
    import geopy.distance
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False
    warnings.warn("geopy not available - using simple distance calculation")


@dataclass
class OSMEdge:
    """OSM edge with attributes."""
    way_id: int
    from_node: int
    to_node: int
    length: float
    highway_type: str
    maxspeed: Optional[float]
    oneway: str
    geometry: Optional[Any] = None


class ODDisconnected(Exception):
    """Raised when origin-destination pair is not connected in the network."""
    
    def __init__(self, origin: str, destination: str):
        self.origin = origin
        self.destination = destination
        super().__init__(f"Origin {origin} and destination {destination} are not connected")


def parse_osm_xml(filepath: str) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parse OSM XML file and extract nodes and ways.
    
    Args:
        filepath: Path to OSM XML file
        
    Returns:
        Tuple of (nodes_dict, ways_list)
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # Extract nodes
    nodes = {}
    for node in root.findall('node'):
        node_id = int(node.get('id'))
        lat = float(node.get('lat'))
        lon = float(node.get('lon'))
        
        nodes[node_id] = {
            'lat': lat,
            'lon': lon,
            'y': lat,  # NetworkX convention
            'x': lon   # NetworkX convention
        }
    
    # Extract ways (only highway ways)
    ways = []
    for way in root.findall('way'):
        # Check if it's a highway
        highway_tag = None
        oneway_tag = 'no'
        maxspeed_tag = None
        
        for tag in way.findall('tag'):
            key = tag.get('k')
            value = tag.get('v')
            
            if key == 'highway':
                highway_tag = value
            elif key == 'oneway':
                oneway_tag = value
            elif key == 'maxspeed':
                try:
                    # Handle different maxspeed formats
                    if value.endswith(' mph'):
                        maxspeed_tag = float(value[:-4]) * 1.609344  # Convert to km/h
                    elif value.endswith(' km/h'):
                        maxspeed_tag = float(value[:-5])
                    else:
                        maxspeed_tag = float(value)  # Assume km/h
                except:
                    maxspeed_tag = None
        
        # Only process highway ways
        if highway_tag is not None:
            # Extract node references
            node_refs = []
            for nd in way.findall('nd'):
                node_refs.append(int(nd.get('ref')))
            
            if len(node_refs) >= 2:  # Need at least 2 nodes
                ways.append({
                    'id': int(way.get('id')),
                    'highway': highway_tag,
                    'oneway': oneway_tag,
                    'maxspeed': maxspeed_tag,
                    'nodes': node_refs
                })
    
    return nodes, ways


def calculate_distance(node1: Dict[str, Any], node2: Dict[str, Any]) -> float:
    """
    Calculate geodesic distance between two nodes.
    
    Args:
        node1: First node with 'lat', 'lon'
        node2: Second node with 'lat', 'lon'
        
    Returns:
        Distance in meters
    """
    if HAS_GEOPY:
        coord1 = (node1['lat'], node1['lon'])
        coord2 = (node2['lat'], node2['lon'])
        return geopy.distance.geodesic(coord1, coord2).meters
    else:
        # Simple Haversine distance calculation
        lat1, lon1 = np.radians(node1['lat']), np.radians(node1['lon'])
        lat2, lon2 = np.radians(node2['lat']), np.radians(node2['lon'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in meters
        R = 6371000
        return R * c


def get_free_flow_speed(highway_type: str, maxspeed: Optional[float] = None) -> float:
    """
    Get free-flow speed for highway type.
    
    Args:
        highway_type: OSM highway type
        maxspeed: Posted speed limit (km/h)
        
    Returns:
        Free-flow speed in km/h
    """
    # Default speeds by highway type (km/h)
    default_speeds = {
        'motorway': 120,
        'trunk': 100,
        'primary': 80,
        'secondary': 60,
        'tertiary': 50,
        'residential': 30,
        'service': 20,
        'unclassified': 40,
        'motorway_link': 80,
        'trunk_link': 60,
        'primary_link': 50,
        'secondary_link': 40,
        'tertiary_link': 30
    }
    
    # Use maxspeed if available, otherwise default
    if maxspeed is not None:
        return maxspeed
    else:
        return default_speeds.get(highway_type, 50)  # Default 50 km/h


def create_edge_features(edge: OSMEdge) -> Dict[str, float]:
    """
    Create feature vector for an edge.
    
    Per OSM_INTEGRATION.md: "Features Z: free-flow time = l_e / v_free from maxspeed or defaults; 
    highway-class dummies; optional curvature/grade"
    
    Args:
        edge: OSM edge
        
    Returns:
        Feature dictionary
    """
    free_flow_speed = get_free_flow_speed(edge.highway_type, edge.maxspeed)
    free_flow_time = edge.length / (free_flow_speed * 1000 / 3600)  # Convert km/h to m/s
    
    features = {
        'length': edge.length,
        'free_flow_time': free_flow_time,
        'free_flow_speed': free_flow_speed,
    }
    
    # Highway class dummies
    highway_classes = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service']
    for highway_class in highway_classes:
        features[f'highway_{highway_class}'] = 1.0 if edge.highway_type == highway_class else 0.0
    
    return features


def osm_to_multigraph(filepath: str) -> nx.MultiDiGraph:
    """
    Convert OSM XML to NetworkX MultiDiGraph.
    
    Per OSM_INTEGRATION.md:
    - Graph: NetworkX MultiDiGraph. For each way, split into consecutive node pairs
    - oneway=yes -> forward only; oneway=-1 -> reverse only; otherwise both directions
    - Preserve parallel edges as distinct edges
    
    Args:
        filepath: Path to OSM XML file
        
    Returns:
        NetworkX MultiDiGraph
    """
    nodes, ways = parse_osm_xml(filepath)
    
    # Create MultiDiGraph
    G = nx.MultiDiGraph()
    
    # Add nodes with coordinates
    for node_id, node_data in nodes.items():
        G.add_node(node_id, **node_data)
    
    # Process ways and create edges
    edge_id = 0
    
    for way in ways:
        way_id = way['id']
        highway_type = way['highway']
        oneway = way['oneway']
        maxspeed = way['maxspeed']
        node_list = way['nodes']
        
        # Split way into consecutive node pairs
        for i in range(len(node_list) - 1):
            from_node = node_list[i]
            to_node = node_list[i + 1]
            
            # Skip if nodes don't exist in graph
            if from_node not in nodes or to_node not in nodes:
                continue
            
            # Calculate edge length
            length = calculate_distance(nodes[from_node], nodes[to_node])
            
            if length <= 0:
                continue  # Skip zero-length edges
            
            # Create edge object
            osm_edge = OSMEdge(
                way_id=way_id,
                from_node=from_node,
                to_node=to_node,
                length=length,
                highway_type=highway_type,
                maxspeed=maxspeed,
                oneway=oneway
            )
            
            # Create features
            features = create_edge_features(osm_edge)
            
            # Default utility (negative cost)
            default_utility = -features['free_flow_time']
            
            # Add edges based on oneway tag
            if oneway == 'yes':
                # Forward direction only
                edge_attrs = {
                    'edge_id': edge_id,
                    'way_id': way_id,
                    'length': length,
                    'u': default_utility,
                    'highway': highway_type,
                    'oneway': oneway,
                    'maxspeed': maxspeed,
                    **features
                }
                G.add_edge(from_node, to_node, key=edge_id, **edge_attrs)
                edge_id += 1
                
            elif oneway == '-1' or oneway == 'reverse':
                # Reverse direction only
                edge_attrs = {
                    'edge_id': edge_id,
                    'way_id': way_id,
                    'length': length,
                    'u': default_utility,
                    'highway': highway_type,
                    'oneway': oneway,
                    'maxspeed': maxspeed,
                    **features
                }
                G.add_edge(to_node, from_node, key=edge_id, **edge_attrs)
                edge_id += 1
                
            else:
                # Both directions (default)
                # Forward direction
                edge_attrs_fwd = {
                    'edge_id': edge_id,
                    'way_id': way_id,
                    'length': length,
                    'u': default_utility,
                    'highway': highway_type,
                    'oneway': 'no',
                    'maxspeed': maxspeed,
                    **features
                }
                G.add_edge(from_node, to_node, key=edge_id, **edge_attrs_fwd)
                edge_id += 1
                
                # Reverse direction
                edge_attrs_rev = {
                    'edge_id': edge_id,
                    'way_id': way_id,
                    'length': length,
                    'u': default_utility,
                    'highway': highway_type,
                    'oneway': 'no',
                    'maxspeed': maxspeed,
                    **features
                }
                G.add_edge(to_node, from_node, key=edge_id, **edge_attrs_rev)
                edge_id += 1
    
    return G


def find_nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """
    Find nearest node to given coordinates.
    
    Args:
        G: NetworkX MultiDiGraph
        lat: Latitude
        lon: Longitude
        
    Returns:
        Node ID of nearest node
    """
    min_distance = float('inf')
    nearest_node = None
    
    target_coord = {'lat': lat, 'lon': lon}
    
    for node_id, node_data in G.nodes(data=True):
        if 'lat' in node_data and 'lon' in node_data:
            distance = calculate_distance(node_data, target_coord)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
    
    return nearest_node


def check_od_connectivity(G: nx.MultiDiGraph, origin: int, destination: int) -> bool:
    """
    Check if destination is reachable from origin.
    
    Per OSM_INTEGRATION.md: "If destination not reachable from origin in the directed graph, 
    raise ODDisconnected before solving"
    
    Args:
        G: NetworkX MultiDiGraph
        origin: Origin node ID
        destination: Destination node ID
        
    Returns:
        True if connected, False otherwise
    """
    try:
        return nx.has_path(G, origin, destination)
    except nx.NodeNotFound:
        return False


def create_od_demand_vector(
    G: nx.MultiDiGraph,
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float
) -> Tuple[int, int, np.ndarray, Dict[int, int]]:
    """
    Create origin-destination demand vector from coordinates.
    
    Per OSM_INTEGRATION.md: "Map origin/destination to nearest graph nodes; 
    build b with b[o] = -1, b[d] = +1; sum(b) = 0"
    
    Args:
        G: NetworkX MultiDiGraph
        origin_lat: Origin latitude
        origin_lon: Origin longitude
        dest_lat: Destination latitude
        dest_lon: Destination longitude
        
    Returns:
        Tuple of (origin_node, dest_node, demand_vector, node_mapping)
    """
    # Find nearest nodes
    origin_node = find_nearest_node(G, origin_lat, origin_lon)
    dest_node = find_nearest_node(G, dest_lat, dest_lon)
    
    if origin_node is None or dest_node is None:
        raise ValueError("Could not find nearest nodes for origin/destination")
    
    # Check connectivity
    if not check_od_connectivity(G, origin_node, dest_node):
        raise ODDisconnected(str(origin_node), str(dest_node))
    
    # Create node mapping
    nodes = sorted(G.nodes())
    node_mapping = {node: idx for idx, node in enumerate(nodes)}
    
    # Create demand vector
    n_nodes = len(nodes)
    demand_vector = np.zeros(n_nodes)
    demand_vector[node_mapping[origin_node]] = -1.0
    demand_vector[node_mapping[dest_node]] = +1.0
    
    return origin_node, dest_node, demand_vector, node_mapping


def extract_network_matrices(G: nx.MultiDiGraph) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Extract incidence matrix, lengths, and utilities from MultiDiGraph.
    
    Per OSM_INTEGRATION.md: "Build A with columns = edges (tail=-1, head=+1) as SciPy CSR/CSC.
    Align arrays: columns of A, entries of l, and rows of Z share identical edge ordering"
    
    Args:
        G: NetworkX MultiDiGraph
        
    Returns:
        Tuple of (incidence_matrix, lengths, utilities, edge_mapping)
    """
    nodes = sorted(G.nodes())
    node_mapping = {node: idx for idx, node in enumerate(nodes)}
    n_nodes = len(nodes)
    
    # Extract edges in consistent order
    edges = []
    for u, v, key, data in G.edges(data=True, keys=True):
        edges.append((u, v, key, data))
    
    # Sort by edge_id for consistent ordering
    edges.sort(key=lambda x: x[3].get('edge_id', x[2]))
    
    n_edges = len(edges)
    edge_mapping = {edges[i][3].get('edge_id', i): i for i in range(n_edges)}
    
    # Build incidence matrix
    incidence_matrix = np.zeros((n_nodes, n_edges))
    lengths = np.zeros(n_edges)
    utilities = np.zeros(n_edges)
    
    for col, (u, v, key, data) in enumerate(edges):
        # Incidence: tail=-1, head=+1
        incidence_matrix[node_mapping[u], col] = -1.0
        incidence_matrix[node_mapping[v], col] = +1.0
        
        # Extract attributes
        lengths[col] = data.get('length', 1.0)
        utilities[col] = data.get('u', -1.0)
    
    return incidence_matrix, lengths, utilities, edge_mapping


def osm_network_summary(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """
    Create summary statistics for OSM network.
    
    Args:
        G: NetworkX MultiDiGraph
        
    Returns:
        Summary statistics dictionary
    """
    summary = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'n_connected_components': nx.number_weakly_connected_components(G),
        'is_strongly_connected': nx.is_strongly_connected(G),
        'bbox': None
    }
    
    # Calculate bounding box
    lats = [data.get('lat', 0) for _, data in G.nodes(data=True) if 'lat' in data]
    lons = [data.get('lon', 0) for _, data in G.nodes(data=True) if 'lon' in data]
    
    if lats and lons:
        summary['bbox'] = {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons)
        }
    
    # Highway type distribution
    highway_types = {}
    for u, v, key, data in G.edges(data=True, keys=True):
        highway = data.get('highway', 'unknown')
        highway_types[highway] = highway_types.get(highway, 0) + 1
    
    summary['highway_types'] = highway_types
    
    return summary
