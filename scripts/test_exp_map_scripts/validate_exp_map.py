import gpytoolbox as gpy
import polyscope as ps
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import utils.discrete_exp_map as dem
import utils.graphics_utils as utils

def validate_exponential_map():
    """Comprehensive validation of discrete exponential map implementation"""
    
    # Test 1: Simple sphere - should create circular patterns
    print("=== Test 1: Sphere (should show radial patterns) ===")
    V, F = gpy.icosphere(4)  # Higher resolution
    N = gpy.per_vertex_normals(V, F)
    E = gpy.edges(F)
    
    root_idx = 0
    exp_map = dem.discrete_exp_map(V, E, N, root_idx, add_locally=True)
    
    # Compute distances and angles in the exp map
    exp_distances = np.linalg.norm(exp_map, axis=1)
    exp_angles = np.arctan2(exp_map[:, 1], exp_map[:, 0])
    
    # Test properties that should hold:
    print(f"Root vertex exp_map coords: {exp_map[root_idx]} (should be ~[0,0])")
    print(f"Max distance in exp_map: {np.max(exp_distances):.4f}")
    print(f"Mean distance in exp_map: {np.mean(exp_distances):.4f}")
    
    # Visualize
    ps.init()
    ps_mesh = ps.register_surface_mesh('sphere', V, F)
    ps_mesh.add_parameterization_quantity('exp_map', exp_map)
    ps_mesh.add_scalar_quantity('exp_distance', exp_distances, cmap='viridis')
    ps_mesh.add_scalar_quantity('exp_angle', exp_angles, cmap='hsv')
    
    # Mark the root
    root_indicator = np.zeros(len(V))
    root_indicator[root_idx] = 1
    ps_mesh.add_scalar_quantity('root_vertex', root_indicator, cmap='reds')
    
    print("Check visualization: Should see smooth radial gradients from red root vertex")
    
    # Test 2: Flat plane - should preserve distances better
    print("\n=== Test 2: Flat Grid (should preserve distances) ===")
    V_flat, F_flat = gpy.regular_square_mesh(20)
    V_flat = np.column_stack([V_flat, np.zeros(len(V_flat))])  # Make it 3D
    N_flat = np.tile([0, 0, 1], (len(V_flat), 1))  # All normals pointing up
    E_flat = gpy.edges(F_flat)
    
    root_flat = len(V_flat) // 2  # Center vertex
    exp_map_flat = dem.discrete_exp_map(V_flat, E_flat, N_flat, root_flat, add_locally=True)
    
    ps_mesh_flat = ps.register_surface_mesh('flat_grid', V_flat, F_flat)
    ps_mesh_flat.add_parameterization_quantity('exp_map_flat', exp_map_flat)
    
    # Test 3: Distance preservation check
    print("\n=== Test 3: Distance Preservation Analysis ===")
    
    # Pick a few vertices and compare 3D distances vs exp_map distances
    test_vertices = [1, 10, 50, 100] if len(V) > 100 else [1, len(V)//4, len(V)//2]
    
    for v_idx in test_vertices:
        # 3D distance on surface (approximate)
        euclidean_dist = np.linalg.norm(V[v_idx] - V[root_idx])
        exp_map_dist = np.linalg.norm(exp_map[v_idx] - exp_map[root_idx])
        
        print(f"Vertex {v_idx}: 3D euclidean={euclidean_dist:.4f}, exp_map={exp_map_dist:.4f}, ratio={exp_map_dist/euclidean_dist:.4f}")
    
    # Test 4: Symmetry check (for symmetric meshes)
    print("\n=== Test 4: Symmetry Check ===")
    
    # Find vertices at similar distances from root
    distances_3d = np.linalg.norm(V - V[root_idx], axis=1)
    similar_dist_mask = np.abs(distances_3d - np.median(distances_3d)) < 0.1
    similar_vertices = np.where(similar_dist_mask)[0]
    
    if len(similar_vertices) > 3:
        exp_distances_similar = np.linalg.norm(exp_map[similar_vertices], axis=1)
        print(f"Vertices at similar 3D distances have exp_map distances: mean={np.mean(exp_distances_similar):.4f}, std={np.std(exp_distances_similar):.4f}")
        print(f"Low std deviation indicates good distance preservation")
    
    ps.show()
    
    return exp_map, exp_distances, exp_angles

def create_dartboard_visualization():
    """Create a dartboard pattern to visualize the exp map quality"""
    V, F = gpy.icosphere(3)
    N = gpy.per_vertex_normals(V, F)
    E = gpy.edges(F)
    
    root_idx = 0
    exp_map = dem.discrete_exp_map(V, E, N, root_idx, add_locally=True)
    
    # Create dartboard pattern
    distances = np.linalg.norm(exp_map, axis=1)
    angles = np.arctan2(exp_map[:, 1], exp_map[:, 0])
    
    # Radial rings
    max_dist = np.max(distances)
    num_rings = 8
    ring_size = max_dist / num_rings
    ring_pattern = np.floor(distances / ring_size) % 2
    
    # Angular sectors  
    num_sectors = 12
    sector_size = 2 * np.pi / num_sectors
    sector_pattern = np.floor((angles + np.pi) / sector_size) % 2
    
    # Combine patterns
    dartboard = (ring_pattern + sector_pattern) % 2
    
    ps.init()
    ps_mesh = ps.register_surface_mesh('dartboard', V, F)
    ps_mesh.add_scalar_quantity('dartboard_pattern', dartboard, cmap='coolwarm')
    
    # Mark root
    root_indicator = np.zeros(len(V))
    root_indicator[root_idx] = 1
    ps_mesh.add_scalar_quantity('root', root_indicator, cmap='reds')
    
    print("Dartboard pattern should show:")
    print("- Smooth alternating rings radiating from root")
    print("- Regular angular sectors")
    print("- No distortion or irregular patterns")
    
    ps.show()

if __name__ == "__main__":
    print("Running comprehensive exponential map validation...")
    validate_exponential_map()
    print("\nCreating dartboard visualization...")
    create_dartboard_visualization()