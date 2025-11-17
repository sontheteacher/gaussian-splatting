import numpy as np
import polyscope as ps
import sys
import os

# Add the parent directory to the path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import utils.discrete_exp_map as dem
import utils.graphics_utils as utils

def test_point_cloud_vs_mesh():
    """Demonstrate when to use generate_knn_edges vs mesh edges"""
    
    print("=== Point Cloud vs Mesh Connectivity Test ===")
    
    # Case 1: Point cloud (no mesh) - NEEDS generate_knn_edges
    print("\n1. Point Cloud Case (needs generate_knn_edges):")
    
    # Create a random point cloud on a sphere
    n_points = 500
    np.random.seed(42)
    
    # Random points on sphere surface
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    
    V_cloud = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta), 
        np.cos(phi)
    ])
    
    # Compute normals (for sphere, normals = positions)
    N_cloud = V_cloud.copy()
    
    # Generate connectivity using k-nearest neighbors
    k = 8  # Connect each point to 8 nearest neighbors
    max_radius = 0.3  # Don't connect points that are too far apart
    
    print(f"Building k-NN edges for {n_points} points...")
    E_cloud = dem.generate_knn_edges(V_cloud, k, max_radius)
    print(f"Generated {len(E_cloud)} edges")
    
    # Compute exponential map
    root_idx = 0
    exp_map_cloud = dem.discrete_exp_map(V_cloud, E_cloud, N_cloud, root_idx, add_locally=True)
    
    # Visualize
    ps.init()
    
    # Show point cloud with edges
    ps_cloud = ps.register_point_cloud('point_cloud', V_cloud)
    ps_cloud.add_scalar_quantity('exp_map_distance', np.linalg.norm(exp_map_cloud, axis=1), cmap='viridis')
    
    # Show the k-NN edges as a network
    ps_edges = ps.register_curve_network('knn_edges', V_cloud, E_cloud)
    
    # Mark root
    root_colors = np.zeros(len(V_cloud))
    root_colors[root_idx] = 1
    ps_cloud.add_scalar_quantity('root', root_colors, cmap='reds')
    
    print("Visualization: Point cloud connected via k-NN edges")
    
    # Case 2: Compare with mesh connectivity
    print("\n2. Mesh Case (uses gpy.edges):")
    
    import gpytoolbox as gpy
    
    # Create mesh version of similar sphere
    V_mesh, F_mesh = gpy.icosphere(3)  # Creates ~640 vertices
    N_mesh = gpy.per_vertex_normals(V_mesh, F_mesh)
    E_mesh = gpy.edges(F_mesh)  # Uses mesh topology, not k-NN
    
    exp_map_mesh = dem.discrete_exp_map(V_mesh, E_mesh, N_mesh, 0, add_locally=True)
    
    # Add mesh to visualization
    offset = np.array([3, 0, 0])  # Offset to separate from point cloud
    ps_mesh = ps.register_surface_mesh('mesh', V_mesh + offset, F_mesh)
    ps_mesh.add_scalar_quantity('exp_map_distance', np.linalg.norm(exp_map_mesh, axis=1), cmap='viridis')
    ps_mesh.add_parameterization_quantity('exp_map', exp_map_mesh)
    
    print(f"Mesh has {len(E_mesh)} edges from triangulation topology")
    
    ps.show()
    
    return E_cloud, E_mesh

def gaussian_splatting_example():
    """Example of using generate_knn_edges for Gaussian splatting centers"""
    
    print("\n=== Gaussian Splatting Use Case ===")
    
    # Simulate Gaussian splat centers (typically not a mesh)
    n_splats = 200
    
    # Create splats on a bumpy surface
    x = np.random.uniform(-2, 2, n_splats)
    y = np.random.uniform(-2, 2, n_splats)
    z = 0.3 * np.sin(x) * np.cos(y) + 0.1 * np.random.randn(n_splats)  # Bumpy surface
    
    V_splats = np.column_stack([x, y, z])
    
    # Estimate normals (in practice, these would come from your Gaussian fitting)
    # Simple finite difference approximation
    N_splats = np.zeros_like(V_splats)
    for i in range(n_splats):
        # Find nearby points for normal estimation
        dists = np.linalg.norm(V_splats - V_splats[i], axis=1)
        nearby = np.argsort(dists)[1:4]  # 3 nearest neighbors
        
        if len(nearby) >= 2:
            v1 = V_splats[nearby[0]] - V_splats[i]
            v2 = V_splats[nearby[1]] - V_splats[i]
            normal = np.cross(v1, v2)
            if np.linalg.norm(normal) > 1e-6:
                N_splats[i] = normal / np.linalg.norm(normal)
            else:
                N_splats[i] = [0, 0, 1]  # Default up
        else:
            N_splats[i] = [0, 0, 1]
    
    # Build connectivity for exponential map computation
    k_neighbors = 6  # For splats, fewer connections often work better
    max_connect_dist = 0.5  # Don't connect distant splats
    
    E_splats = dem.generate_knn_edges(V_splats, k_neighbors, max_connect_dist)
    
    print(f"Built connectivity for {n_splats} Gaussian splat centers")
    print(f"Generated {len(E_splats)} connections")
    print(f"Average connections per splat: {2*len(E_splats)/n_splats:.1f}")
    
    # Compute exponential map from a central splat
    center_idx = np.argmin(np.linalg.norm(V_splats, axis=1))  # Find center-most splat
    exp_map_splats = dem.discrete_exp_map(V_splats, E_splats, N_splats, center_idx, add_locally=True)
    
    # Visualize
    ps.init()
    
    ps_splats = ps.register_point_cloud('gaussian_splats', V_splats)
    ps_splats.add_scalar_quantity('exp_map_u', exp_map_splats[:, 0], cmap='coolwarm')
    ps_splats.add_scalar_quantity('exp_map_v', exp_map_splats[:, 1], cmap='coolwarm') 
    ps_splats.add_scalar_quantity('exp_map_distance', np.linalg.norm(exp_map_splats, axis=1), cmap='viridis')
    
    # Show connectivity
    ps_connections = ps.register_curve_network('splat_connections', V_splats, E_splats)
    
    # Mark center splat
    center_marker = np.zeros(len(V_splats))
    center_marker[center_idx] = 1
    ps_splats.add_scalar_quantity('center_splat', center_marker, cmap='reds')
    
    print("This shows how you'd use exponential map for Gaussian splat parameterization")
    ps.show()

def compare_connectivity_methods():
    """Compare different ways of building connectivity"""
    
    print("\n=== Connectivity Methods Comparison ===")
    
    # Same point set, different connectivity
    import gpytoolbox as gpy
    V, F = gpy.icosphere(2)  # Small mesh for comparison
    N = gpy.per_vertex_normals(V, F)
    
    # Method 1: Mesh edges (ground truth)
    E_mesh = gpy.edges(F)
    
    # Method 2: k-NN with small k
    E_knn_small = dem.generate_knn_edges(V, k=4, max_radius=np.inf)
    
    # Method 3: k-NN with large k
    E_knn_large = dem.generate_knn_edges(V, k=12, max_radius=np.inf)
    
    # Method 4: k-NN with distance limit
    avg_edge_length = np.mean([np.linalg.norm(V[e[1]] - V[e[0]]) for e in E_mesh])
    E_knn_limited = dem.generate_knn_edges(V, k=8, max_radius=1.2*avg_edge_length)
    
    print(f"Mesh topology: {len(E_mesh)} edges")
    print(f"k-NN (k=4): {len(E_knn_small)} edges") 
    print(f"k-NN (k=12): {len(E_knn_large)} edges")
    print(f"k-NN limited: {len(E_knn_limited)} edges")
    
    # Compute exponential maps with different connectivities
    root_idx = 0
    
    exp_mesh = dem.discrete_exp_map(V, E_mesh, N, root_idx, add_locally=True)
    exp_knn_small = dem.discrete_exp_map(V, E_knn_small, N, root_idx, add_locally=True)
    exp_knn_large = dem.discrete_exp_map(V, E_knn_large, N, root_idx, add_locally=True)
    exp_knn_limited = dem.discrete_exp_map(V, E_knn_limited, N, root_idx, add_locally=True)
    
    # Compare results
    dist_mesh = np.linalg.norm(exp_mesh, axis=1)
    dist_knn_small = np.linalg.norm(exp_knn_small, axis=1)
    dist_knn_large = np.linalg.norm(exp_knn_large, axis=1)
    dist_knn_limited = np.linalg.norm(exp_knn_limited, axis=1)
    
    print(f"\nCorrelation with mesh result:")
    print(f"k-NN (k=4): {np.corrcoef(dist_mesh, dist_knn_small)[0,1]:.3f}")
    print(f"k-NN (k=12): {np.corrcoef(dist_mesh, dist_knn_large)[0,1]:.3f}")
    print(f"k-NN limited: {np.corrcoef(dist_mesh, dist_knn_limited)[0,1]:.3f}")

if __name__ == "__main__":
    print("Demonstrating when to use generate_knn_edges...")
    
    # Main comparison
    # E_cloud, E_mesh = test_point_cloud_vs_mesh()
    
    # Gaussian splatting example  
    gaussian_splatting_example()
    
    # Connectivity comparison
    # compare_connectivity_methods()
    
    print("\n" + "="*50)
    print("WHEN TO USE generate_knn_edges:")
    print("="*50)
    print("✅ Point clouds (no mesh connectivity)")
    print("✅ Gaussian splat centers") 
    print("✅ Sparse/irregular point sets")
    print("✅ When you need to build neighborhood graphs")
    print("\nWHEN NOT TO USE:")
    print("❌ When you already have mesh faces (use gpy.edges instead)")
    print("❌ Regular grids (use grid connectivity)")
    print("❌ When topology is known a priori")