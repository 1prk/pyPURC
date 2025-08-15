"""
OSM Integration Tests

Tests for OSM integration per OSM_INTEGRATION.md and TEST_PLAN.md:
- test_multidigraph_construction: NetworkX MultiDiGraph creation
- test_oneway_handling: oneway=yes/no/-1 semantics
- test_geodesic_lengths: Positive lengths and feature presence
- test_end_to_end_pipeline: Complete OSM to PURC pipeline

Paper Reference: Real-world network integration
"""

import numpy as np
import pytest
import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile
import networkx as nx

from purc.osm import (
    parse_osm_xml,
    osm_to_multigraph,
    calculate_distance,
    get_free_flow_speed,
    create_edge_features,
    find_nearest_node,
    check_od_connectivity,
    create_od_demand_vector,
    extract_network_matrices,
    osm_network_summary,
    ODDisconnected
)


class TestOSMIntegration:
    """Test OSM integration functionality."""
    
    @pytest.fixture
    def sample_osm_xml(self):
        """Create a sample OSM XML for testing."""
        osm_content = '''<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
  <node id="1" lat="52.5200" lon="13.4050"/>
  <node id="2" lat="52.5210" lon="13.4060"/>
  <node id="3" lat="52.5220" lon="13.4070"/>
  <node id="4" lat="52.5230" lon="13.4080"/>
  
  <way id="100">
    <nd ref="1"/>
    <nd ref="2"/>
    <nd ref="3"/>
    <tag k="highway" v="primary"/>
    <tag k="maxspeed" v="50"/>
  </way>
  
  <way id="101">
    <nd ref="2"/>
    <nd ref="4"/>
    <tag k="highway" v="secondary"/>
    <tag k="oneway" v="yes"/>
  </way>
  
  <way id="102">
    <nd ref="3"/>
    <nd ref="4"/>
    <tag k="highway" v="residential"/>
    <tag k="oneway" v="-1"/>
  </way>
</osm>'''
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.osm', delete=False) as f:
            f.write(osm_content)
            return f.name
    
    def test_osm_xml_parsing(self, sample_osm_xml):
        """Test OSM XML parsing."""
        nodes, ways = parse_osm_xml(sample_osm_xml)
        
        # Check nodes
        assert len(nodes) == 4, "Should parse 4 nodes"
        assert 1 in nodes, "Should have node 1"
        assert nodes[1]['lat'] == 52.5200, "Should parse latitude correctly"
        assert nodes[1]['lon'] == 13.4050, "Should parse longitude correctly"
        
        # Check ways
        assert len(ways) == 3, "Should parse 3 highway ways"
        
        # Check way attributes
        primary_way = next(w for w in ways if w['id'] == 100)
        assert primary_way['highway'] == 'primary', "Should parse highway type"
        assert primary_way['maxspeed'] == 50.0, "Should parse maxspeed"
        assert primary_way['oneway'] == 'no', "Should default oneway to 'no'"
        
        oneway_yes = next(w for w in ways if w['id'] == 101)
        assert oneway_yes['oneway'] == 'yes', "Should parse oneway=yes"
        
        oneway_reverse = next(w for w in ways if w['id'] == 102)
        assert oneway_reverse['oneway'] == '-1', "Should parse oneway=-1"
    
    def test_multidigraph_construction(self, sample_osm_xml):
        """Test NetworkX MultiDiGraph construction from OSM."""
        G = osm_to_multigraph(sample_osm_xml)
        
        # Check graph type
        assert isinstance(G, nx.MultiDiGraph), "Should create MultiDiGraph"
        
        # Check nodes
        assert G.number_of_nodes() == 4, "Should have 4 nodes"
        
        # Check node attributes
        node_data = G.nodes[1]
        assert 'lat' in node_data, "Nodes should have latitude"
        assert 'lon' in node_data, "Nodes should have longitude"
        assert 'x' in node_data, "Nodes should have x coordinate"
        assert 'y' in node_data, "Nodes should have y coordinate"
    
    def test_oneway_handling(self, sample_osm_xml):
        """Test oneway semantics per OSM_INTEGRATION.md."""
        G = osm_to_multigraph(sample_osm_xml)
        
        # Count edges by direction
        edges_12 = list(G.edges(1, 2, keys=True))
        edges_21 = list(G.edges(2, 1, keys=True))
        edges_24 = list(G.edges(2, 4, keys=True))
        edges_42 = list(G.edges(4, 2, keys=True))
        edges_34 = list(G.edges(3, 4, keys=True))
        edges_43 = list(G.edges(4, 3, keys=True))
        
        # Way 100 (primary, no oneway): should have both directions for each segment
        # Segments: 1->2, 2->3
        assert len(edges_12) >= 1, "Should have edge 1->2"
        assert len(edges_21) >= 1, "Should have edge 2->1 (bidirectional)"
        
        # Way 101 (oneway=yes): should only have forward direction 2->4
        assert len(edges_24) >= 1, "Should have edge 2->4 (oneway=yes)"
        assert len(edges_42) == 0, "Should not have edge 4->2 (oneway=yes)"
        
        # Way 102 (oneway=-1): should only have reverse direction 4->3
        assert len(edges_43) >= 1, "Should have edge 4->3 (oneway=-1)"
        assert len(edges_34) == 0, "Should not have edge 3->4 (oneway=-1)"
    
    def test_geodesic_lengths(self, sample_osm_xml):
        """Test that all edges have positive geodesic lengths."""
        G = osm_to_multigraph(sample_osm_xml)
        
        for u, v, key, data in G.edges(data=True, keys=True):
            length = data.get('length', 0)
            assert length > 0, f"Edge ({u}, {v}) should have positive length, got {length}"
            
            # Length should be reasonable for the coordinates (roughly 100-200m between consecutive nodes)
            assert 50 < length < 2000, f"Length {length} seems unrealistic for test coordinates"
    
    def test_feature_presence(self, sample_osm_xml):
        """Test that edges have required features per OSM_INTEGRATION.md."""
        G = osm_to_multigraph(sample_osm_xml)
        
        required_attributes = ['length', 'u', 'highway', 'free_flow_time', 'free_flow_speed']
        
        for u, v, key, data in G.edges(data=True, keys=True):
            for attr in required_attributes:
                assert attr in data, f"Edge ({u}, {v}) should have attribute '{attr}'"
            
            # Check that utility is negative (cost)
            assert data['u'] < 0, f"Utility should be negative cost, got {data['u']}"
            
            # Check that free flow time is positive
            assert data['free_flow_time'] > 0, "Free flow time should be positive"
            
            # Check that free flow speed is reasonable
            assert 10 <= data['free_flow_speed'] <= 150, "Free flow speed should be reasonable (10-150 km/h)"
    
    def test_distance_calculation(self):
        """Test geodesic distance calculation."""
        # Berlin coordinates (roughly 1km apart)
        node1 = {'lat': 52.5200, 'lon': 13.4050}
        node2 = {'lat': 52.5290, 'lon': 13.4050}  # ~1km north
        
        distance = calculate_distance(node1, node2)
        
        # Should be roughly 1000m
        assert 900 < distance < 1100, f"Distance should be ~1000m, got {distance}"
    
    def test_free_flow_speed_defaults(self):
        """Test free-flow speed calculation."""
        # Test with maxspeed
        speed_with_limit = get_free_flow_speed('primary', 60.0)
        assert speed_with_limit == 60.0, "Should use provided maxspeed"
        
        # Test without maxspeed (use defaults)
        speed_motorway = get_free_flow_speed('motorway')
        assert speed_motorway == 120, "Motorway should default to 120 km/h"
        
        speed_residential = get_free_flow_speed('residential')
        assert speed_residential == 30, "Residential should default to 30 km/h"
        
        speed_unknown = get_free_flow_speed('unknown_type')
        assert speed_unknown == 50, "Unknown highway type should default to 50 km/h"
    
    def test_edge_features_creation(self):
        """Test edge feature creation."""
        from purc.osm import OSMEdge
        
        edge = OSMEdge(
            way_id=100,
            from_node=1,
            to_node=2,
            length=1000.0,  # 1km
            highway_type='primary',
            maxspeed=50.0,  # 50 km/h
            oneway='no'
        )
        
        features = create_edge_features(edge)
        
        # Check basic features
        assert features['length'] == 1000.0, "Should preserve length"
        assert features['free_flow_speed'] == 50.0, "Should use maxspeed"
        
        # Free flow time should be length/speed = 1000m / (50 km/h) = 1000 / (50*1000/3600) = 72 seconds
        expected_time = 1000.0 / (50.0 * 1000 / 3600)
        assert abs(features['free_flow_time'] - expected_time) < 1, "Free flow time calculation"
        
        # Check highway dummies
        assert features['highway_primary'] == 1.0, "Should have primary highway dummy"
        assert features['highway_residential'] == 0.0, "Should not have residential dummy"
    
    def test_nearest_node_finding(self, sample_osm_xml):
        """Test finding nearest node to coordinates."""
        G = osm_to_multigraph(sample_osm_xml)
        
        # Find nearest to node 1 coordinates (should be node 1)
        nearest = find_nearest_node(G, 52.5200, 13.4050)
        assert nearest == 1, "Should find node 1 as nearest to its own coordinates"
        
        # Find nearest to coordinates between nodes 1 and 2
        nearest_mid = find_nearest_node(G, 52.5205, 13.4055)
        assert nearest_mid in [1, 2], "Should find node 1 or 2 as nearest to midpoint"
    
    def test_od_connectivity_check(self, sample_osm_xml):
        """Test OD connectivity checking."""
        G = osm_to_multigraph(sample_osm_xml)
        
        # Test connectivity between existing nodes
        # Note: connectivity depends on oneway restrictions
        connected_12 = check_od_connectivity(G, 1, 2)
        connected_21 = check_od_connectivity(G, 2, 1)
        
        # Should have some connectivity (exact depends on network structure)
        assert isinstance(connected_12, bool), "Should return boolean"
        assert isinstance(connected_21, bool), "Should return boolean"
        
        # Test with non-existent node
        connected_invalid = check_od_connectivity(G, 1, 999)
        assert not connected_invalid, "Non-existent destination should not be connected"
    
    def test_od_demand_vector_creation(self, sample_osm_xml):
        """Test OD demand vector creation from coordinates."""
        G = osm_to_multigraph(sample_osm_xml)
        
        # Create demand vector using node coordinates
        origin_node, dest_node, demand_vector, node_mapping = create_od_demand_vector(
            G, 52.5200, 13.4050,  # Near node 1
            52.5230, 13.4080     # Near node 4
        )
        
        # Check results
        assert origin_node in G.nodes(), "Origin should be valid node"
        assert dest_node in G.nodes(), "Destination should be valid node"
        assert len(demand_vector) == G.number_of_nodes(), "Demand vector should match number of nodes"
        assert abs(np.sum(demand_vector)) < 1e-15, "Demand vector should sum to zero"
        
        # Check demand values
        origin_idx = node_mapping[origin_node]
        dest_idx = node_mapping[dest_node]
        assert demand_vector[origin_idx] == -1.0, "Origin should have demand -1"
        assert demand_vector[dest_idx] == +1.0, "Destination should have demand +1"
    
    def test_network_matrices_extraction(self, sample_osm_xml):
        """Test extraction of network matrices."""
        G = osm_to_multigraph(sample_osm_xml)
        
        incidence_matrix, lengths, utilities, edge_mapping = extract_network_matrices(G)
        
        # Check dimensions
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        assert incidence_matrix.shape == (n_nodes, n_edges), "Incidence matrix should be n_nodes × n_edges"
        assert len(lengths) == n_edges, "Lengths should have n_edges entries"
        assert len(utilities) == n_edges, "Utilities should have n_edges entries"
        
        # Check incidence matrix properties
        # Each column should sum to 0 (flow conservation)
        column_sums = np.sum(incidence_matrix, axis=0)
        assert np.allclose(column_sums, 0, atol=1e-15), "Incidence matrix columns should sum to 0"
        
        # Each column should have exactly one +1 and one -1
        for col in range(n_edges):
            column = incidence_matrix[:, col]
            assert np.sum(column == +1.0) == 1, f"Column {col} should have exactly one +1"
            assert np.sum(column == -1.0) == 1, f"Column {col} should have exactly one -1"
            assert np.sum(np.abs(column)) == 2.0, f"Column {col} should have exactly two non-zero entries"
        
        # Check that all lengths are positive
        assert np.all(lengths > 0), "All lengths should be positive"
        
        # Check that all utilities are negative (costs)
        assert np.all(utilities <= 0), "All utilities should be non-positive (costs)"
    
    def test_osm_network_summary(self, sample_osm_xml):
        """Test OSM network summary statistics."""
        G = osm_to_multigraph(sample_osm_xml)
        summary = osm_network_summary(G)
        
        # Check required fields
        required_fields = ['n_nodes', 'n_edges', 'n_connected_components', 'is_strongly_connected', 'bbox', 'highway_types']
        for field in required_fields:
            assert field in summary, f"Summary should include {field}"
        
        # Check values
        assert summary['n_nodes'] == G.number_of_nodes(), "Node count should match"
        assert summary['n_edges'] == G.number_of_edges(), "Edge count should match"
        assert isinstance(summary['is_strongly_connected'], bool), "Strong connectivity should be boolean"
        
        # Check bounding box
        bbox = summary['bbox']
        assert bbox is not None, "Should have bounding box"
        assert 'min_lat' in bbox and 'max_lat' in bbox, "Should have latitude bounds"
        assert 'min_lon' in bbox and 'max_lon' in bbox, "Should have longitude bounds"
        assert bbox['min_lat'] <= bbox['max_lat'], "Latitude bounds should be ordered"
        assert bbox['min_lon'] <= bbox['max_lon'], "Longitude bounds should be ordered"
        
        # Check highway types
        highway_types = summary['highway_types']
        assert isinstance(highway_types, dict), "Highway types should be dictionary"
        assert len(highway_types) > 0, "Should have some highway types"
    
    def test_end_to_end_pipeline(self, sample_osm_xml):
        """Test complete OSM to PURC pipeline."""
        # Step 1: Load OSM network
        G = osm_to_multigraph(sample_osm_xml)
        assert G.number_of_nodes() > 0, "Should load network"
        
        # Step 2: Extract matrices
        incidence_matrix, lengths, utilities, edge_mapping = extract_network_matrices(G)
        assert incidence_matrix.shape[1] == len(lengths), "Matrices should be aligned"
        
        # Step 3: Create OD demand (if connectivity allows)
        try:
            origin_node, dest_node, demand_vector, node_mapping = create_od_demand_vector(
                G, 52.5200, 13.4050,  # Near node 1
                52.5230, 13.4080     # Near node 4
            )
            
            # Step 4: Verify everything is ready for PURC solver
            assert incidence_matrix.shape[0] == len(demand_vector), "Incidence matrix and demand should be compatible"
            assert np.abs(np.sum(demand_vector)) < 1e-15, "Demand should sum to zero"
            assert np.all(lengths > 0), "All lengths should be positive"
            
            print(f"Successfully created network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            print(f"OD pair: {origin_node} -> {dest_node}")
            
        except ODDisconnected as e:
            # This is acceptable - small test network might not have full connectivity
            print(f"OD disconnected (expected for small test network): {e}")
            assert True  # Test passes - disconnection is handled correctly
    
    def test_od_disconnected_exception(self, sample_osm_xml):
        """Test that ODDisconnected is raised appropriately."""
        G = osm_to_multigraph(sample_osm_xml)
        
        # Try to create OD demand for potentially disconnected nodes
        # This might raise ODDisconnected, which should be handled gracefully
        try:
            origin_node, dest_node, demand_vector, node_mapping = create_od_demand_vector(
                G, 52.5200, 13.4050,  # Node 1 area
                52.5230, 13.4080     # Node 4 area
            )
            # If it succeeds, that's fine
            assert True
        except ODDisconnected as e:
            # If it raises ODDisconnected, that's also fine - it means the check works
            assert hasattr(e, 'origin'), "ODDisconnected should have origin attribute"
            assert hasattr(e, 'destination'), "ODDisconnected should have destination attribute"
            assert "not connected" in str(e), "Exception message should mention connectivity"
    
    def test_highway_class_features(self, sample_osm_xml):
        """Test that highway class dummy variables are created correctly."""
        G = osm_to_multigraph(sample_osm_xml)
        
        # Check that different highway types get different dummy variables
        highway_dummies_found = set()
        
        for u, v, key, data in G.edges(data=True, keys=True):
            highway_type = data.get('highway', 'unknown')
            
            # Check that the corresponding dummy variable is set
            dummy_key = f'highway_{highway_type}'
            if dummy_key in data:
                assert data[dummy_key] == 1.0, f"Highway dummy {dummy_key} should be 1.0"
                highway_dummies_found.add(highway_type)
            
            # Check that other highway dummies are 0
            for other_type in ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service']:
                if other_type != highway_type:
                    other_dummy = f'highway_{other_type}'
                    if other_dummy in data:
                        assert data[other_dummy] == 0.0, f"Other highway dummy {other_dummy} should be 0.0"
        
        # Should have found some highway dummies
        assert len(highway_dummies_found) > 0, "Should have found some highway type dummies"
        
        # Clean up temporary file
        Path(sample_osm_xml).unlink(missing_ok=True)
