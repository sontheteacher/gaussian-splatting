#!/usr/bin/env python3
"""
Test script: Build k-NN graph on a sphere and compute discrete exponential map.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.research_utils import (
    build_connectivity, 
    estimate_normals_svd_simple,
    compute_discrete_exponential_map,
    adjacency_to_edges,
    save_to_ply
)

def main():
    print("\n" + "="*70)
    print("SPHERE k-NN GRAPH TEST")
    print("="*70)
    
    # Create a small sphere with Fibonacci spiral sampling
    n_points = 50000
    print(f"\n[1] Creating sphere with {n_points} points...")
    
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    vertices = []
    
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)
        theta = golden_angle * i
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    print(f"    ✓ Sphere vertices created: {vertices.shape}")
    
    # Build k-NN connectivity
    print(f"\n[2] Building k-NN connectivity graph...")
    k_neighbors = 10
    adjacency, avg_radius = build_connectivity(vertices, k=k_neighbors, prune_factor=2.0, use_gpu=True)
    print(f"    ✓ Connectivity built")
    print(f"    ✓ Average radius: {avg_radius:.6f}")
    
    # Convert adjacency to edges
    print(f"\n[3] Converting adjacency list to edges...")
    edges = adjacency_to_edges(adjacency)
    print(f"    ✓ Edges created: {edges.shape}")
    print(f"    ✓ Total edges: {len(edges)}")
    
    # Estimate normals from k-NN connectivity
    print(f"\n[4] Estimating normals using SVD on k-NN neighborhoods...")
    normals = estimate_normals_svd_simple(
        vertices, 
        adjacency, 
        k_neighbors=k_neighbors,
        y_up=True,
        use_gpu=True,
        return_confidences=False
    )
    print(f"    ✓ Normals estimated: {normals.shape}")
    print(f"    ✓ Normal sample: {normals[0]} (should be normalized)")
    
    # Compute discrete exponential map
    print(f"\n[5] Computing discrete exponential map...")
    exp_map = compute_discrete_exponential_map(
        vertices,
        edges,
        normals,
        root_vertex=0,  # Use north pole as root
        local_coordinates=True
    )
    print(f"    ✓ Exponential map computed: {exp_map.shape}")
    print(f"    ✓ Exp map range U: [{exp_map[:,0].min():.3f}, {exp_map[:,0].max():.3f}]")
    print(f"    ✓ Exp map range V: [{exp_map[:,1].min():.3f}, {exp_map[:,1].max():.3f}]")
    
    # Verify exp_map is saved to PLY
    output_path = "test_sphere_knn.ply"
    print(f"\n[6] Saving to PLY with exponential map coordinates...")
    save_to_ply(
        vertices,
        normals,
        edges,
        output_path,
        confidences=None,
        exp_map=exp_map
    )
    print(f"    ✓ Saved to: {output_path}")
    
    # Verify file exists
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"    ✓ File size: {file_size} bytes")
    
    print(f"\n" + "="*70)
    print("✅ TEST COMPLETE - Ready to visualize!")
    print("="*70)
    print(f"\nTo visualize: python scripts/visualization_script/visualize_expmap.py {output_path}")
    print()

if __name__ == "__main__":
    main()
