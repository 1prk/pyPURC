#!/usr/bin/env python3
"""
PURC CLI Script - Command-line interface for PURC routing

Usage per RUNBOOK.md:
python scripts/run_purc.py --osm map.osm.xml --origin "lat,lon" --dest "lat,lon" --pert entropy

Implements complete OSM-to-PURC pipeline:
1. Load OSM network
2. Create OD demand from coordinates  
3. Solve PURC optimization
4. Output results and diagnostics

Paper Reference: Real-world PURC routing application
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from purc.osm import (
    osm_to_multigraph, 
    create_od_demand_vector, 
    extract_network_matrices,
    osm_network_summary,
    ODDisconnected
)
from purc.core import solve_purc
from purc.solver import solve_purc_entropy, solve_purc_quadratic
from purc.io import save_results_json, export_flows_csv
from purc.estimation import estimate_projected_kkt_ols


def parse_coordinates(coord_str: str) -> Tuple[float, float]:
    """
    Parse coordinate string "lat,lon" to tuple.
    
    Args:
        coord_str: Coordinate string like "52.5200,13.4050"
        
    Returns:
        Tuple of (latitude, longitude)
    """
    try:
        parts = coord_str.strip().split(',')
        if len(parts) != 2:
            raise ValueError("Coordinates must be in format 'lat,lon'")
        
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        
        # Basic validation
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude {lat} must be in range [-90, 90]")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Longitude {lon} must be in range [-180, 180]")
        
        return lat, lon
        
    except ValueError as e:
        raise ValueError(f"Invalid coordinate format '{coord_str}': {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PURC routing with entropy perturbation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_purc.py --osm berlin.osm --origin "52.5200,13.4050" --dest "52.5170,13.3888" --pert entropy
  python scripts/run_purc.py --osm map.osm.xml --origin "lat,lon" --dest "lat,lon" --pert quadratic --output results/
        """
    )
    
    # Required arguments per RUNBOOK.md
    parser.add_argument('--osm', required=True, 
                       help='OSM XML file path (highway=* filtered)')
    parser.add_argument('--origin', required=True,
                       help='Origin coordinates as "lat,lon"')
    parser.add_argument('--dest', required=True, 
                       help='Destination coordinates as "lat,lon"')
    parser.add_argument('--pert', choices=['entropy', 'quadratic'], default='entropy',
                       help='Perturbation type (default: entropy)')
    
    # Optional arguments
    parser.add_argument('--output', '-o', default='.',
                       help='Output directory for results (default: current directory)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--estimation', action='store_true',
                       help='Run projected KKT OLS estimation')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only show network summary, do not solve')
    
    args = parser.parse_args()
    
    try:
        # Parse coordinates
        origin_lat, origin_lon = parse_coordinates(args.origin)
        dest_lat, dest_lon = parse_coordinates(args.dest)
        
        if args.verbose:
            print(f"PURC Routing CLI")
            print(f"OSM file: {args.osm}")
            print(f"Origin: ({origin_lat:.6f}, {origin_lon:.6f})")
            print(f"Destination: ({dest_lat:.6f}, {dest_lon:.6f})")
            print(f"Perturbation: {args.pert}")
            print()
        
        # Step 1: Load OSM network
        if args.verbose:
            print("Loading OSM network...")
        start_time = time.time()
        
        G = osm_to_multigraph(args.osm)
        
        load_time = time.time() - start_time
        if args.verbose:
            print(f"Loaded in {load_time:.2f}s")
        
        # Network summary
        summary = osm_network_summary(G)
        print(f"Network: {summary['n_nodes']} nodes, {summary['n_edges']} edges")
        print(f"Connected components: {summary['n_connected_components']}")
        print(f"Strongly connected: {summary['is_strongly_connected']}")
        
        if summary['bbox']:
            bbox = summary['bbox']
            print(f"Bounding box: ({bbox['min_lat']:.4f}, {bbox['min_lon']:.4f}) to ({bbox['max_lat']:.4f}, {bbox['max_lon']:.4f})")
        
        if summary['highway_types']:
            print("Highway types:", ', '.join(f"{k}:{v}" for k, v in sorted(summary['highway_types'].items())))
        
        if args.summary_only:
            return 0
        
        print()
        
        # Step 2: Create OD demand
        if args.verbose:
            print("Creating OD demand vector...")
        
        try:
            origin_node, dest_node, demand_vector, node_mapping = create_od_demand_vector(
                G, origin_lat, origin_lon, dest_lat, dest_lon
            )
            
            print(f"Origin mapped to node: {origin_node}")
            print(f"Destination mapped to node: {dest_node}")
            
        except ODDisconnected as e:
            print(f"ERROR: {e}")
            print("The origin and destination are not connected in the road network.")
            return 1
        
        # Step 3: Extract network matrices
        if args.verbose:
            print("Extracting network matrices...")
        
        incidence_matrix, lengths, utilities, edge_mapping = extract_network_matrices(G)
        
        print(f"Incidence matrix: {incidence_matrix.shape}")
        print(f"Edge attributes: lengths={len(lengths)}, utilities={len(utilities)}")
        print()
        
        # Step 4: Solve PURC optimization
        if args.verbose:
            print(f"Solving PURC with {args.pert} perturbation...")
        start_time = time.time()
        
        if args.pert == 'entropy':
            result = solve_purc(incidence_matrix, demand_vector, lengths, utilities)
        elif args.pert == 'quadratic':
            result = solve_purc_quadratic(incidence_matrix, demand_vector, lengths, utilities)
        else:
            raise ValueError(f"Unknown perturbation type: {args.pert}")
        
        solve_time = time.time() - start_time
        
        # Step 5: Report results
        print(f"Optimization Results:")
        print(f"Success: {result.success}")
        print(f"Iterations: {result.iterations}")
        print(f"Solve time: {solve_time:.2f}s")
        print(f"Objective value: {result.objective_value:.6f}")
        print(f"Constraint violation: {result.constraint_violation:.2e}")
        print(f"KKT stationarity residual: {result.kkt_stationarity_residual:.2e}")
        print(f"KKT complementarity residual: {result.kkt_complementarity_residual:.2e}")
        print()
        
        if not result.success:
            print("WARNING: Optimization did not converge successfully")
            return 1
        
        # Flow statistics
        active_flows = result.flows[result.flows > 1e-12]
        print(f"Flow statistics:")
        print(f"Total flow: {np.sum(result.flows):.6f}")
        print(f"Active edges: {len(active_flows)} / {len(result.flows)}")
        if len(active_flows) > 0:
            print(f"Flow range: [{np.min(active_flows):.6f}, {np.max(active_flows):.6f}]")
        print()
        
        # Step 6: Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / "purc_results.json"
        save_results_json(result, results_file, metadata={
            'osm_file': str(args.osm),
            'origin': {'lat': origin_lat, 'lon': origin_lon, 'node': origin_node},
            'destination': {'lat': dest_lat, 'lon': dest_lon, 'node': dest_node},
            'perturbation': args.pert,
            'network_summary': summary,
            'solve_time': solve_time
        })
        
        # Export flows to CSV
        flows_file = output_dir / "flows.csv"
        edge_attributes = {}
        for u, v, key, data in G.edges(data=True, keys=True):
            edge_id = data.get('edge_id', key)
            edge_attributes[edge_id] = {
                'from_node': u,
                'to_node': v,
                'way_id': data.get('way_id', ''),
                'highway': data.get('highway', ''),
                'length': data.get('length', 0),
                'utility': data.get('u', 0)
            }
        
        export_flows_csv(result.flows, edge_mapping, flows_file, edge_attributes)
        
        print(f"Results saved:")
        print(f"- Detailed results: {results_file}")
        print(f"- Flow CSV: {flows_file}")
        
        # Step 7: Optional estimation
        if args.estimation:
            print("\nRunning projected KKT OLS estimation...")
            
            try:
                # Create feature matrix from edge attributes
                n_edges = len(lengths)
                features = np.column_stack([
                    lengths,                    # Length feature
                    utilities,                  # Utility feature
                    np.ones(n_edges)           # Constant term
                ])
                
                estimation_result = estimate_projected_kkt_ols(
                    flows=result.flows,
                    lengths=lengths,
                    features=features,
                    incidence_matrix=incidence_matrix,
                    robust_se=True
                )
                
                print(f"Estimation Results:")
                print(f"R-squared: {estimation_result.r_squared:.4f}")
                print(f"Active variables: {len(estimation_result.active_indices)} / {n_edges}")
                
                # Save estimation results
                estimation_file = output_dir / "estimation_results.json"
                save_results_json(estimation_result, estimation_file)
                print(f"- Estimation results: {estimation_file}")
                
            except Exception as e:
                print(f"Estimation failed: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
        print("\nPURC routing completed successfully!")
        return 0
        
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        return 1
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
