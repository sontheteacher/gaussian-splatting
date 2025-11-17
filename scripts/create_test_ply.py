#!/usr/bin/env python3
"""
Generate a synthetic PLY file with exponential map coordinates for testing visualization.
Creates a simple geometric pattern (grid, circle, or spiral) with known exp_map coordinates.
"""
import numpy as np
import os


def create_grid_pattern(size=20, spacing=0.1):
    """Create a 2D grid pattern in 3D space with exponential map coordinates."""
    # Create 2D grid
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # 3D vertices (grid on XY plane)
    vertices = np.column_stack([
        X.flatten(),
        Y.flatten(), 
        np.zeros(size * size)  # Z = 0 for simplicity
    ])
    
    # Normals pointing up
    normals = np.tile([0, 0, 1], (len(vertices), 1))
    
    # Exponential map coordinates = scaled XY coordinates
    # Center at (0,0) maps to exp_map (0,0)
    exp_map = np.column_stack([
        X.flatten() * 0.5,  # Scale down for realistic ranges
        Y.flatten() * 0.5
    ])
    
    # Confidences (random but reasonable)
    confidences = 0.5 + 0.5 * np.random.random(len(vertices))
    
    return vertices, normals, exp_map, confidences


def create_circle_pattern(n_points=200, radius=1.0):
    """Create a circular pattern with radial exponential map coordinates."""
    # Create points in a circle
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # Add some radial variation
    radii = radius * (0.3 + 0.7 * np.random.random(n_points))
    
    # 3D vertices
    vertices = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles),
        0.1 * np.random.random(n_points)  # Small Z variation
    ])
    
    # Normals pointing up with slight random variation
    normals = np.column_stack([
        0.1 * np.random.random(n_points) - 0.05,
        0.1 * np.random.random(n_points) - 0.05,
        np.ones(n_points)
    ])
    # Normalize normals
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    # Exponential map: polar coordinates
    exp_map = np.column_stack([
        radii * np.cos(angles) * 0.3,  # Scale for realistic ranges
        radii * np.sin(angles) * 0.3
    ])
    
    # Add center point
    center_vertex = np.array([[0, 0, 0]])
    center_normal = np.array([[0, 0, 1]])
    center_exp = np.array([[0, 0]])
    
    vertices = np.vstack([center_vertex, vertices])
    normals = np.vstack([center_normal, normals])
    exp_map = np.vstack([center_exp, exp_map])
    
    confidences = np.ones(len(vertices)) * 0.8
    confidences[0] = 1.0  # Center has high confidence
    
    return vertices, normals, exp_map, confidences


def create_spiral_pattern(n_points=300, turns=3):
    """Create a spiral pattern with exp_map coordinates following the spiral."""
    # Create spiral
    t = np.linspace(0, turns * 2 * np.pi, n_points)
    radius = np.linspace(0.1, 1.0, n_points)
    
    # 3D vertices
    vertices = np.column_stack([
        radius * np.cos(t),
        radius * np.sin(t),
        0.2 * t / (turns * 2 * np.pi)  # Slight Z variation along spiral
    ])
    
    # Normals
    normals = np.tile([0, 0, 1], (len(vertices), 1))
    
    # Exponential map: parametric along spiral
    # Map spiral parameter t to 2D coordinates
    u = (t / (turns * 2 * np.pi)) * 0.8 - 0.4  # [-0.4, 0.4]
    v = np.sin(t * 4) * 0.3  # Oscillating component
    
    exp_map = np.column_stack([u, v])
    
    # Add center point at start
    center_vertex = np.array([[0, 0, 0]])
    center_normal = np.array([[0, 0, 1]])
    center_exp = np.array([[-0.4, 0]])  # Start of spiral in exp space
    
    vertices = np.vstack([center_vertex, vertices])
    normals = np.vstack([center_normal, normals])
    exp_map = np.vstack([center_exp, exp_map])
    
    confidences = np.ones(len(vertices)) * 0.7
    
    return vertices, normals, exp_map, confidences


def create_simple_edges(n_vertices, pattern='grid', grid_size=20):
    """Create simple edge connectivity."""
    edges = []
    
    if pattern == 'grid':
        # Grid connectivity
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                # Connect to right neighbor
                if j < grid_size - 1:
                    edges.append([idx, idx + 1])
                # Connect to bottom neighbor
                if i < grid_size - 1:
                    edges.append([idx, idx + grid_size])
    
    elif pattern in ['circle', 'spiral']:
        # Connect each point to a few neighbors
        for i in range(1, n_vertices):  # Skip center point
            # Connect to center
            edges.append([0, i])
            # Connect to next few points
            for j in range(1, min(4, n_vertices - i)):
                if i + j < n_vertices:
                    edges.append([i, i + j])
    
    return np.array(edges, dtype=np.int32) if edges else np.empty((0, 2), dtype=np.int32)


def save_test_ply(vertices, normals, exp_map, confidences, edges, output_path):
    """Save test data to PLY format."""
    n_vertices = len(vertices)
    n_edges = len(edges)
    
    # PLY header
    header = f"""ply
format ascii 1.0
element vertex {n_vertices}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float confidence
property float exp_u
property float exp_v
element edge {n_edges}
property int vertex1
property int vertex2
end_header
"""
    
    with open(output_path, 'w') as f:
        f.write(header)
        
        # Write vertices
        for i in range(n_vertices):
            v = vertices[i]
            n = normals[i]
            conf = confidences[i]
            exp_u, exp_v = exp_map[i]
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f} {conf:.6f} {exp_u:.6f} {exp_v:.6f}\n")
        
        # Write edges
        for edge in edges:
            f.write(f"{edge[0]} {edge[1]}\n")
    
    print(f"Saved test PLY with {n_vertices} vertices and {n_edges} edges to: {output_path}")


def create_test_ply(pattern='circle', output_path='test_expmap.ply'):
    """Create a test PLY file with the specified pattern."""
    print(f"Creating {pattern} pattern...")
    
    if pattern == 'grid':
        vertices, normals, exp_map, confidences = create_grid_pattern(size=20)
        edges = create_simple_edges(len(vertices), 'grid', 20)
    elif pattern == 'circle':
        vertices, normals, exp_map, confidences = create_circle_pattern(n_points=200)
        edges = create_simple_edges(len(vertices), 'circle')
    elif pattern == 'spiral':
        vertices, normals, exp_map, confidences = create_spiral_pattern(n_points=300)
        edges = create_simple_edges(len(vertices), 'spiral')
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    save_test_ply(vertices, normals, exp_map, confidences, edges, output_path)
    
    # Print statistics
    print(f"\nTest PLY Statistics:")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Edges: {len(edges)}")
    print(f"  exp_u range: [{exp_map[:,0].min():.6f}, {exp_map[:,0].max():.6f}]")
    print(f"  exp_v range: [{exp_map[:,1].min():.6f}, {exp_map[:,1].max():.6f}]")
    print(f"  3D bounds: X=[{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}]")
    print(f"             Y=[{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}]")
    print(f"             Z=[{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")
    
    return output_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test PLY files with exponential map coordinates")
    parser.add_argument('--pattern', choices=['grid', 'circle', 'spiral'], default='circle',
                       help='Pattern to generate (default: circle)')
    parser.add_argument('--output', default='test_expmap.ply',
                       help='Output PLY file path (default: test_expmap.ply)')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    ply_path = create_test_ply(args.pattern, args.output)
    
    print(f"\n✅ Test PLY created: {ply_path}")
    print(f"\nTo visualize:")
    print(f"  python scripts/visualization_script/visualize_expmap.py {ply_path}")