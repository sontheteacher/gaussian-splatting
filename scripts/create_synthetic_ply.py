#!/usr/bin/env python3
"""
Generate synthetic PLY file with known exponential map coordinates for testing.
Creates a simple grid or spiral pattern with clear exponential map structure.
"""
import numpy as np
import os


def create_synthetic_expmap_ply(output_path: str, pattern: str = "spiral", n_points: int = 1000):
    """
    Create a synthetic PLY file with exponential map coordinates.
    
    Args:
        output_path: Path to save PLY file
        pattern: "spiral", "grid", "radial", or "sphere"
        n_points: Number of points to generate
    """
    print(f"Creating synthetic PLY with {n_points} points, pattern: {pattern}")
    
    if pattern == "spiral":
        # Create spiral in 3D space with corresponding 2D exp map
        t = np.linspace(0, 4*np.pi, n_points)
        radius = np.linspace(0.1, 2.0, n_points)
        
        # 3D spiral
        vertices = np.column_stack([
            radius * np.cos(t),
            radius * np.sin(t), 
            t * 0.1  # Height increases with angle
        ])
        
        # 2D exponential map - polar coordinates
        exp_map = np.column_stack([
            radius * np.cos(t * 0.5),  # Slower rotation in exp space
            radius * np.sin(t * 0.5)
        ])
        
        # Center point is at t=0
        center_idx = 0
        
    elif pattern == "grid":
        # Create 3D grid with 2D exponential map
        side = int(np.sqrt(n_points))
        n_points = side * side  # Adjust to perfect square
        
        x = np.linspace(-1, 1, side)
        y = np.linspace(-1, 1, side)
        X, Y = np.meshgrid(x, y)
        
        vertices = np.column_stack([
            X.flatten(),
            Y.flatten(),
            0.1 * (X.flatten()**2 + Y.flatten()**2)  # Paraboloid height
        ])
        
        # 2D exp map is just the XY projection scaled
        exp_map = np.column_stack([
            X.flatten() * 0.8,
            Y.flatten() * 0.8
        ])
        
        # Center point is closest to origin
        center_idx = np.argmin(X.flatten()**2 + Y.flatten()**2)
        
    elif pattern == "radial":
        # Create radial pattern from center
        angles = np.random.uniform(0, 2*np.pi, n_points)
        radii = np.random.uniform(0, 1.5, n_points)
        
        # Set first point as center
        angles[0] = 0
        radii[0] = 0
        
        vertices = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles),
            radii * 0.2  # Height proportional to radius
        ])
        
        # Exponential map preserves radial structure but with distortion
        exp_radii = radii * 0.7
        exp_angles = angles * 1.2  # Slight angular distortion
        
        exp_map = np.column_stack([
            exp_radii * np.cos(exp_angles),
            exp_radii * np.sin(exp_angles)
        ])
        
        center_idx = 0
        
    elif pattern == "sphere":
        # Create a unit sphere using spherical coordinates
        print(f"   Creating sphere with {n_points} points using Fibonacci spiral sampling")
        
        # Use Fibonacci spiral for uniform sampling on sphere
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians
        
        vertices = []
        
        for i in range(n_points):
            # Fibonacci spiral sampling for uniform distribution on sphere
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)
            
            theta = golden_angle * i  # Golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            vertices.append([x, y, z])
        
        vertices = np.array(vertices)
        
        # Exponential map: use stereographic projection from north pole
        # This is a standard conformal mapping from sphere to plane
        exp_map = []
        north_pole = np.array([0, 1, 0])  # North pole at (0,1,0)
        
        for i in range(n_points):
            v = vertices[i]
            
            if abs(v[1] - 1.0) < 1e-6:  # Very close to north pole
                # North pole maps to origin in stereographic projection
                exp_u, exp_v = 0.0, 0.0
            else:
                # Stereographic projection formula
                denom = 1 - v[1]  # 1 - y coordinate  
                exp_u = v[0] / denom
                exp_v = v[2] / denom
            
            exp_map.append([exp_u, exp_v])
        
        exp_map = np.array(exp_map)
        
        # Find point closest to north pole as center
        distances_to_north = np.linalg.norm(vertices - north_pole, axis=1)
        center_idx = np.argmin(distances_to_north)
        
        print(f"   Center point (closest to north pole): index {center_idx}")
        print(f"   Exp map range: U=[{exp_map[:,0].min():.3f}, {exp_map[:,0].max():.3f}]")
        print(f"                  V=[{exp_map[:,1].min():.3f}, {exp_map[:,1].max():.3f}]")
        
    else:
        raise ValueError(f"Unknown pattern: {pattern}. Use 'spiral', 'grid', 'radial', or 'sphere'")
    
    # Generate normals based on pattern
    if pattern == "sphere":
        # For sphere, normals are just the vertex positions (outward pointing)
        normals = vertices.copy()
    else:
        # For other patterns, generate normals (pointing up with some variation)
        normals = np.zeros_like(vertices)
    normals[:, 2] = 1.0  # Base normal pointing up
    # Add some variation
    normals[:, 0] += np.random.normal(0, 0.1, n_points)
    normals[:, 1] += np.random.normal(0, 0.1, n_points)
    # Normalize
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    # Generate confidence values
    confidences = np.random.uniform(0.5, 1.0, n_points)
    confidences[center_idx] = 1.0  # Center has max confidence
    
    # Generate simple edge connectivity (each point connects to nearest neighbors)
    from scipy.spatial import KDTree
    tree = KDTree(vertices)
    edges = []
    k_neighbors = min(8, n_points - 1)
    
    for i in range(n_points):
        _, neighbors = tree.query(vertices[i], k=k_neighbors+1)  # +1 because it includes self
        neighbors = neighbors[1:]  # Remove self
        for j in neighbors:
            if i < j:  # Avoid duplicate edges
                edges.append([i, j])
    
    edges = np.array(edges)
    
    # Write PLY file
    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property float confidence\n")
        f.write("property float exp_u\n")
        f.write("property float exp_v\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")
        
        # Vertices
        for i in range(n_points):
            f.write(f"{vertices[i,0]:.6f} {vertices[i,1]:.6f} {vertices[i,2]:.6f} ")
            f.write(f"{normals[i,0]:.6f} {normals[i,1]:.6f} {normals[i,2]:.6f} ")
            f.write(f"{confidences[i]:.6f} ")
            f.write(f"{exp_map[i,0]:.6f} {exp_map[i,1]:.6f}\n")
        
        # Edges
        for edge in edges:
            f.write(f"{edge[0]} {edge[1]}\n")
    
    print(f"Created synthetic PLY: {output_path}")
    print(f"  Points: {n_points}")
    print(f"  Edges: {len(edges)}")
    print(f"  Center point index: {center_idx}")
    print(f"  Center 3D position: ({vertices[center_idx,0]:.3f}, {vertices[center_idx,1]:.3f}, {vertices[center_idx,2]:.3f})")
    print(f"  Center exp coords: ({exp_map[center_idx,0]:.3f}, {exp_map[center_idx,1]:.3f})")
    print(f"  Exp map range: U=[{exp_map[:,0].min():.3f}, {exp_map[:,0].max():.3f}], V=[{exp_map[:,1].min():.3f}, {exp_map[:,1].max():.3f}]")
    
    return center_idx


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='./test_synthetic.ply', help='Output PLY file')
    parser.add_argument('--pattern', choices=['spiral', 'grid', 'radial', 'sphere'], default='spiral', help='Pattern type')
    parser.add_argument('--points', type=int, default=1000, help='Number of points')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    center_idx = create_synthetic_expmap_ply(args.output, args.pattern, args.points)
    
    print(f"\\n✅ Synthetic PLY created successfully!")
    print(f"To visualize: python scripts/visualization_script/visualize_expmap.py {args.output}")


if __name__ == '__main__':
    main()