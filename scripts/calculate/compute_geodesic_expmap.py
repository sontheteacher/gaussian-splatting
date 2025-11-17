#!/usr/bin/env python3
"""
Compute geodesic exponential map using potpourri3d heat method.

This script loads a point cloud PLY file and computes the geodesic exponential map
from a root point using the continuous heat method (via potpourri3d). The exponential
map provides a 2D embedding that preserves geodesic distances.

Unlike the discrete exponential map (which requires connectivity), this method:
- Uses the continuous heat method on the point cloud
- Does not require pre-computed edges/connectivity
- Works directly from 3D point positions
- Preserves geodesic distances more accurately for smooth surfaces

Usage:
    python scripts/calculate/compute_geodesic_expmap.py input.ply output.ply
    python scripts/calculate/compute_geodesic_expmap.py input.ply output.ply --root-vertex 42

Output:
    PLY file with vertices containing:
    - x, y, z: original 3D coordinates
    - nx, ny, nz: estimated normals (using local PCA)
    - exp_u, exp_v: geodesic exponential map coordinates

The output can be visualized using:
    python scripts/visualization_script/visualize_expmap.py output.ply
"""

import argparse
import numpy as np
import os
import sys
import time
from typing import Optional

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("[ERROR] plyfile is required. Install with: pip install plyfile")
    sys.exit(1)

try:
    import potpourri3d as pp3d
except ImportError:
    print("[ERROR] potpourri3d is required. Install with: pip install potpourri3d")
    sys.exit(1)

from utils.research_utils import compute_geodesic_exp_map
from scene.gaussian_model import load_gaussian_model


def load_ply_points(ply_path: str) -> np.ndarray:
    """
    Load point coordinates from PLY file using Gaussian model loader.
    
    Args:
        ply_path: Path to PLY file
        
    Returns:
        vertices: (N, 3) array of point coordinates
    """
    vertices, gaussian_model = load_gaussian_model(ply_path)
    
    print(f"[INFO] Loaded {len(vertices)} Gaussian centers from {ply_path}")
    return vertices


def estimate_normals_simple(points: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Estimate normals using simple k-NN PCA.
    
    Args:
        points: (N, 3) point coordinates
        k: Number of nearest neighbors
        
    Returns:
        normals: (N, 3) normal vectors
    """
    from scipy.spatial import KDTree
    
    print(f"[INFO] Estimating normals using {k}-NN PCA...")
    n_points = len(points)
    normals = np.zeros((n_points, 3))
    
    # Build KDTree
    tree = KDTree(points)
    
    # Compute normals
    for i in range(n_points):
        # Find k nearest neighbors
        _, neighbor_indices = tree.query(points[i], k=k+1)  # +1 to include self
        neighbor_indices = neighbor_indices[1:]  # Exclude self
        
        # Center neighborhood
        neighborhood = points[neighbor_indices] - points[i]
        
        # PCA: normal is direction of smallest variance
        cov = neighborhood.T @ neighborhood
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]  # Smallest eigenvalue
        
        # Orient outward (away from centroid)
        centroid = np.mean(points, axis=0)
        if np.dot(normal, points[i] - centroid) < 0:
            normal = -normal
        
        normals[i] = normal
    
    # Normalize
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    print(f"[INFO] Normals estimated")
    return normals


def save_ply_with_expmap(output_path: str, 
                         vertices: np.ndarray,
                         normals: np.ndarray,
                         exp_map: np.ndarray):
    """
    Save point cloud with exponential map to PLY file.
    
    Args:
        output_path: Output PLY file path
        vertices: (N, 3) point coordinates
        normals: (N, 3) normal vectors
        exp_map: (N, 2) exponential map coordinates
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    n_points = len(vertices)
    
    # Create structured array
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('exp_u', 'f4'), ('exp_v', 'f4')
    ]
    
    vertex_data = np.empty(n_points, dtype=dtype)
    vertex_data['x'] = vertices[:, 0]
    vertex_data['y'] = vertices[:, 1]
    vertex_data['z'] = vertices[:, 2]
    vertex_data['nx'] = normals[:, 0]
    vertex_data['ny'] = normals[:, 1]
    vertex_data['nz'] = normals[:, 2]
    vertex_data['exp_u'] = exp_map[:, 0]
    vertex_data['exp_v'] = exp_map[:, 1]
    
    # Create PLY element
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    
    # Write PLY file
    PlyData([vertex_element], text=False).write(output_path)
    
    print(f"[INFO] Saved {n_points} vertices with exponential map to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute geodesic exponential map using potpourri3d heat method"
    )
    parser.add_argument(
        'input_ply',
        help='Input PLY file with point cloud'
    )
    parser.add_argument(
        'output_ply',
        help='Output PLY file (will include exp_u, exp_v coordinates)'
    )
    parser.add_argument(
        '--root-vertex',
        type=int,
        default=None,
        help='Root vertex index (default: closest to centroid)'
    )
    parser.add_argument(
        '--k-neighbors',
        type=int,
        default=10,
        help='Number of neighbors for normal estimation (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.isfile(args.input_ply):
        print(f"[ERROR] Input file not found: {args.input_ply}")
        return 1
    
    print("\n" + "="*70)
    print("GEODESIC EXPONENTIAL MAP COMPUTATION")
    print("="*70)
    print(f"Input:  {args.input_ply}")
    print(f"Output: {args.output_ply}")
    if args.root_vertex is not None:
        print(f"Root vertex: {args.root_vertex}")
    else:
        print(f"Root vertex: auto (closest to centroid)")
    print("="*70 + "\n")
    
    try:
        # Load point cloud
        start_time = time.time()
        vertices = load_ply_points(args.input_ply)
        
        # Estimate normals (simple k-NN PCA)
        normals = estimate_normals_simple(vertices, k=args.k_neighbors)
        
        # Compute geodesic exponential map
        exp_map = compute_geodesic_exp_map(
            vertices,
            root_vertex=args.root_vertex,
            north_tangent=None
        )
        
        # Save results
        save_ply_with_expmap(args.output_ply, vertices, normals, exp_map)
        
        elapsed = time.time() - start_time
        
        # Print summary
        print("\n" + "="*70)
        print("✅ COMPUTATION COMPLETE")
        print("="*70)
        print(f"Total time: {elapsed:.2f}s")
        print(f"Output saved to: {args.output_ply}")
        print(f"\nVisualize with:")
        print(f"  python scripts/visualization_script/visualize_expmap.py {args.output_ply}")
        print("="*70 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n[INFO] Computation interrupted by user")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Computation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
