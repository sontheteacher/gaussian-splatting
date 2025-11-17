#!/usr/bin/env python3
"""
Visualize exponential map coordinates from PLY files using Polyscope.

This script loads a PLY file containing exponential map coordinates (exp_u, exp_v)
and visualizes them using Polyscope. The exponential map provides a 2D embedding
of the 3D surface that preserves geodesic distances from a root point.

Usage:
    python scripts/visualization_script/visualize_exp_map.py path/to/processed_scene.ply
    
The PLY file should contain vertices with:
- x, y, z: 3D coordinates
- nx, ny, nz: normal vectors
- exp_u, exp_v: exponential map coordinates (if computed)

Features:
- 3D point cloud visualization with normals
- 2D exponential map visualization
- Root vertex highlighting
- Interactive controls for exploring the mapping
"""

import argparse
import numpy as np
import os
import sys
from typing import Optional, Tuple

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import polyscope as ps
    import polyscope.imgui as psim
except ImportError:
    print("[ERROR] polyscope is required for visualization.")
    print("Install with: pip install polyscope")
    sys.exit(1)

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("[ERROR] plyfile is required for loading PLY files.")
    print("Install with: pip install plyfile")
    sys.exit(1)


def load_ply_with_exp_map(ply_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load PLY file with vertices, normals, and exponential map coordinates.
    
    Args:
        ply_path: Path to PLY file
        
    Returns:
        Tuple of (vertices, normals, exp_map, edges)
        exp_map is None if not present in file
        edges is None if not present in file
    """
    with open(ply_path, 'rb') as f:
        plydata = PlyData.read(f)
    
    # Extract vertex data
    vertex_element = plydata['vertex']
    vertex_data = vertex_element.data
    n_vertices = len(vertex_data)
    
    # Extract coordinates and normals
    vertices = np.column_stack([
        vertex_data['x'],
        vertex_data['y'], 
        vertex_data['z']
    ])
    
    normals = np.column_stack([
        vertex_data['nx'],
        vertex_data['ny'],
        vertex_data['nz']
    ])
    
    # Check for exponential map coordinates
    exp_map = None
    field_names = vertex_data.dtype.names
    if field_names is not None and 'exp_u' in field_names and 'exp_v' in field_names:
        exp_map = np.column_stack([
            vertex_data['exp_u'],
            vertex_data['exp_v']
        ])
        print(f"[INFO] Loaded exponential map coordinates for {n_vertices} vertices")
    else:
        if field_names is not None:
            print(f"[DEBUG] Available fields: {list(field_names)}")
        print(f"[WARN] No exponential map coordinates found in PLY file")
    
    # Check for edges
    edges = None
    if 'edge' in plydata:
        edge_element = plydata['edge']
        edge_data = edge_element.data
        edges = np.column_stack([
            edge_data['vertex1'],
            edge_data['vertex2']
        ])
        print(f"[INFO] Loaded {len(edges)} edges")
    else:
        print(f"[WARN] No edges found in PLY file")
    
    print(f"[INFO] Loaded {n_vertices} vertices from {ply_path}")
    return vertices, normals, exp_map, edges


def find_root_vertex(vertices: np.ndarray, exp_map: np.ndarray) -> int:
    """
    Find the root vertex (closest to origin in exponential map space).
    
    Args:
        vertices: 3D vertex positions
        exp_map: 2D exponential map coordinates
        
    Returns:
        Index of root vertex
    """
    if exp_map is None:
        # Fallback: return center vertex
        centroid = np.mean(vertices, axis=0)
        distances = np.linalg.norm(vertices - centroid, axis=1)
        return np.argmin(distances)
    
    # Find vertex closest to origin in exp map space
    distances_to_origin = np.linalg.norm(exp_map, axis=1)
    root_idx = np.argmin(distances_to_origin)
    return root_idx


def visualize_exp_map(ply_path: str, point_size: float = 0.01):
    """
    Visualize exponential map using Polyscope.
    
    Args:
        ply_path: Path to PLY file
        point_size: Size of points in visualization
    """
    # Load data
    vertices, normals, exp_map, edges = load_ply_with_exp_map(ply_path)
    
    if exp_map is None:
        print("[ERROR] No exponential map coordinates found. Cannot visualize exp map.")
        return
    
    # Diagnostics: validate shapes and show sample values
    n_vertices = len(vertices)
    if exp_map.shape[0] != n_vertices:
        print(f"[ERROR] exp_map length ({exp_map.shape[0]}) does not match vertex count ({n_vertices})")
        return

    print(f"[DIAG] exp_map shape: {exp_map.shape}")
    print(f"[DIAG] exp_u range: [{exp_map[:,0].min():.6f}, {exp_map[:,0].max():.6f}], mean={exp_map[:,0].mean():.6f}")
    print(f"[DIAG] exp_v range: [{exp_map[:,1].min():.6f}, {exp_map[:,1].max():.6f}], mean={exp_map[:,1].mean():.6f}")
    sample_n = min(5, n_vertices)
    print("[DIAG] sample vertex -> exp_map (first %d):" % sample_n)
    for i in range(sample_n):
        print(f"  idx={i}: v=({vertices[i,0]:.4f},{vertices[i,1]:.4f},{vertices[i,2]:.4f}) -> exp=({exp_map[i,0]:.6f},{exp_map[i,1]:.6f})")

    # Find exp-map root (closest to origin in exp map space) and scene centroid root
    exp_root_idx = find_root_vertex(vertices, exp_map)
    centroid = np.mean(vertices, axis=0)
    centroid_idx = int(np.argmin(np.linalg.norm(vertices - centroid, axis=1)))
    print(f"[INFO] exp-map root index (closest to exp origin): {exp_root_idx}")
    print(f"[INFO] scene centroid index (closest to centroid): {centroid_idx}")
    print(f"[INFO] centroid position (3D): ({centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f})")

    # Use the centroid vertex as the highlighted center point (user requested)
    root_idx = centroid_idx
    
    # Initialize Polyscope
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_front_dir("z_front")
    
    # Create 3D point cloud
    pc_3d = ps.register_point_cloud("3D Point Cloud", vertices)
    pc_3d.add_vector_quantity("normals", normals, enabled=False, color=(0.2, 0.5, 0.8))
    
    # Add exponential map coordinates as scalar quantities on the 3D points
    # Scale coordinates for better visualization of small ranges
    exp_u_range = exp_map[:, 0].max() - exp_map[:, 0].min()
    exp_v_range = exp_map[:, 1].max() - exp_map[:, 1].min()
    scale_factor = max(10.0, 1.0 / max(exp_u_range, exp_v_range, 1e-6))
    
    print(f"[INFO] Applying scale factor {scale_factor:.1f}x for better visualization")
    
    # pc_3d.add_scalar_quantity("exp_u", exp_map[:, 0], enabled=False, cmap="viridis")
    # pc_3d.add_scalar_quantity("exp_v", exp_map[:, 1], enabled=False, cmap="plasma")
    # add the scaled coordinates
    pc_3d.add_scalar_quantity("exp_u", exp_map[:, 0] * scale_factor, enabled=False, cmap="viridis")
    pc_3d.add_scalar_quantity("exp_v", exp_map[:, 1] * scale_factor, enabled=False, cmap="plasma")

    # Color by distance from root in exp map space (this shows geodesic distance)
    distances_from_root = np.linalg.norm(exp_map - exp_map[root_idx], axis=1)
    # pc_3d.add_scalar_quantity("geodesic_distance", distances_from_root, enabled=True, cmap="plasma")

    # Add scaled distance to uniform [0, 1] range 
    scaled_distances = (distances_from_root - distances_from_root.min()) / (distances_from_root.max() - distances_from_root.min())
    pc_3d.add_scalar_quantity("scaled_distance", scaled_distances, enabled=False, cmap="plasma")

    
    # Add discrete distance bands for clearer visualization
    max_dist = distances_from_root.max()
    if max_dist > 0:
        n_bands = 20
        distance_bands = np.floor(distances_from_root * n_bands / max_dist)
        pc_3d.add_scalar_quantity("distance_bands", distance_bands, enabled=False, cmap="Set1")
    
    # Highlight root vertex in 3D (centroid selection)
    root_colors = np.zeros((len(vertices), 3))
    root_colors[root_idx] = [1.0, 0.0, 0.0]  # Red for root/centroid
    pc_3d.add_color_quantity("root_vertex", root_colors, enabled=False)

    # Also register a single-point marker for the centroid so it's visually prominent
    try:
        centroid_pos = vertices[root_idx:root_idx+1]
        centroid_marker = ps.register_point_cloud("🔴 CENTER POINT", centroid_pos)
        centroid_marker.add_color_quantity("red", np.array([[1.0, 0.0, 0.0]]), enabled=True)
        # Make it larger and more visible
        print(f"[INFO] Added prominent red marker for center point at index {root_idx}")
    except Exception as e:
        print(f"[WARN] Could not create center marker: {e}")
        pass
    
    # Add edges if available
    if edges is not None:
        # Filter edges to only include valid ones
        valid_edges = edges[(edges[:, 0] < len(vertices)) & (edges[:, 1] < len(vertices))]
        
        if len(valid_edges) > 0:
            # Add edges to 3D visualization with thin lines
            curve_network = ps.register_curve_network("3D Edges", vertices, valid_edges, enabled=False)
            curve_network.set_radius(0.0001, relative=False)  # Set thin radius for clean visualization
            print(f"[INFO] Added {len(valid_edges)} edges to visualization (thin lines)")
    
    # Print statistics
    print(f"\n📊 EXPONENTIAL MAP STATISTICS:")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Root vertex: {root_idx}")
    print(f"  Root position (3D): ({vertices[root_idx, 0]:.3f}, {vertices[root_idx, 1]:.3f}, {vertices[root_idx, 2]:.3f})")
    print(f"  Root position (2D): ({exp_map[root_idx, 0]:.3f}, {exp_map[root_idx, 1]:.3f})")
    print(f"  Exp map range U: [{exp_map[:, 0].min():.3f}, {exp_map[:, 0].max():.3f}]")
    print(f"  Exp map range V: [{exp_map[:, 1].min():.3f}, {exp_map[:, 1].max():.3f}]")
    print(f"  Max distance from root: {distances_from_root.max():.3f}")
    
    # Instructions
    print(f"\n🎮 VISUALIZATION CONTROLS:")
    print(f"  - 3D Point Cloud with integrated exponential map:")
    print(f"    * 'geodesic_distance' shows distance from root (enabled by default)")
    print(f"    * 'distance_bands' shows discrete distance levels for clearer patterns")
    print(f"    * 'exp_u' and 'exp_v' show raw exponential map coordinates")
    print(f"    * 'exp_u_scaled' and 'exp_v_scaled' show scaled coordinates (better contrast)")
    print(f"    * 'root_vertex' highlights the root point in red")
    print(f"    * 'normals' shows surface normal vectors")
    print(f"    * '3D Edges' shows connectivity graph (disabled by default)")
    print(f"  - Use mouse to rotate, zoom, and pan")
    print(f"  - Press ESC or close window to exit")
    
    # Show visualization
    ps.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize exponential map coordinates from PLY files"
    )
    parser.add_argument(
        'ply_path',
        help='Path to PLY file with exponential map coordinates'
    )
    parser.add_argument(
        '--point-size',
        type=float,
        default=0.01,
        help='Size of points in visualization (default: 0.01)'
    )
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Validate input
    if not os.path.isfile(args.ply_path):
        print(f"[ERROR] PLY file not found: {args.ply_path}", file=sys.stderr)
        return 1
    
    try:
        visualize_exp_map(args.ply_path, args.point_size)
        return 0
        
    except KeyboardInterrupt:
        print("\n[INFO] Visualization interrupted by user")
        return 0
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())