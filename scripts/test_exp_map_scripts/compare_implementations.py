import gpytoolbox as gpy
import polyscope as ps
import numpy as np
import potpourri3d as pp3d
import sys
import os

# Add the parent directory to the path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import utils.discrete_exp_map as dem

def compare_with_potpourri3d():
    """Compare our implementation with potpourri3d's heat method"""
    
    print("=== Comparing with Potpourri3d Heat Method ===")
    
    # Create test mesh
    V, F = gpy.icosphere(3)
    N = gpy.per_vertex_normals(V, F)
    E = gpy.edges(F)
    root_idx = 0
    
    # Our implementation
    print("Computing our discrete exp map...")
    our_exp_map = dem.discrete_exp_map(V, E, N, root_idx, add_locally=True)
    our_distances = np.linalg.norm(our_exp_map, axis=1)
    
    # Potpourri3d implementation (heat method for geodesic distances)
    print("Computing potpourri3d heat method...")
    solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
    pp3d_distances = solver.compute_distance(root_idx)
    
    # Also get log map from potpourri3d
    try:
        pp3d_log_map = solver.compute_log_map(root_idx)
        pp3d_log_distances = np.linalg.norm(pp3d_log_map, axis=1)
    except:
        print("Log map not available in this potpourri3d version")
        pp3d_log_map = None
        pp3d_log_distances = None
    
    # Compare distance fields
    print(f"\nDistance Field Comparison:")
    print(f"Our max distance: {np.max(our_distances):.4f}")
    print(f"Potpourri3d max distance: {np.max(pp3d_distances):.4f}")
    print(f"Correlation coefficient: {np.corrcoef(our_distances, pp3d_distances)[0,1]:.4f}")
    
    # Find problematic vertices (large distance differences)
    distance_diff = np.abs(our_distances - pp3d_distances)
    relative_error = distance_diff / (pp3d_distances + 1e-8)
    
    print(f"Mean relative error: {np.mean(relative_error):.4f}")
    print(f"Max relative error: {np.max(relative_error):.4f}")
    print(f"Vertices with >20% error: {np.sum(relative_error > 0.2)}/{len(V)}")
    
    # Visualization
    ps.init()
    
    # Our result
    ps_our = ps.register_surface_mesh('our_implementation', V, F)
    ps_our.add_parameterization_quantity('our_exp_map', our_exp_map)
    ps_our.add_scalar_quantity('our_distances', our_distances, cmap='viridis')
    
    # Potpourri3d result
    ps_pp3d = ps.register_surface_mesh('potpourri3d', V + np.array([3, 0, 0]), F)
    ps_pp3d.add_scalar_quantity('pp3d_distances', pp3d_distances, cmap='viridis')
    
    if pp3d_log_map is not None:
        ps_pp3d.add_parameterization_quantity('pp3d_log_map', pp3d_log_map)
    
    # Error visualization
    ps_error = ps.register_surface_mesh('error_analysis', V + np.array([6, 0, 0]), F)
    ps_error.add_scalar_quantity('relative_error', relative_error, cmap='reds')
    ps_error.add_scalar_quantity('absolute_diff', distance_diff, cmap='plasma')
    
    # Mark root on all
    root_marker = np.zeros(len(V))
    root_marker[root_idx] = 1
    ps_our.add_scalar_quantity('root', root_marker, cmap='reds')
    ps_pp3d.add_scalar_quantity('root', root_marker, cmap='reds')
    ps_error.add_scalar_quantity('root', root_marker, cmap='reds')
    
    print(f"\nVisualization shows:")
    print(f"Left: Our implementation")
    print(f"Middle: Potpourri3d reference") 
    print(f"Right: Error analysis (red = high error)")
    
    ps.show()
    
    return our_distances, pp3d_distances, relative_error

def compare_with_geometry_central():
    """Try to use geometry-central if available"""
    try:
        # This requires geometry-central Python bindings
        import geometry_central.surface as gcs
        print("Geometry-central is available!")
        # Implementation would go here
    except ImportError:
        print("Geometry-central not available. Install with:")
        print("pip install geometry-central")

def test_against_simple_cases():
    """Test against analytically known cases"""
    
    print("=== Testing Against Analytical Cases ===")
    
    # Test 1: Flat plane - should preserve Euclidean distances exactly
    print("\nTest 1: Flat Plane (should match Euclidean)")
    
    # Create a flat grid
    n_pts = 10
    x = np.linspace(-1, 1, n_pts)
    y = np.linspace(-1, 1, n_pts)
    X, Y = np.meshgrid(x, y)
    V_flat = np.column_stack([X.ravel(), Y.ravel(), np.zeros(n_pts*n_pts)])
    
    # Delaunay triangulation
    from scipy.spatial import Delaunay
    points_2d = V_flat[:, :2]
    tri = Delaunay(points_2d)
    F_flat = tri.simplices
    
    N_flat = np.tile([0, 0, 1], (len(V_flat), 1))
    E_flat = gpy.edges(F_flat)
    
    # Root at center
    center_idx = np.argmin(np.linalg.norm(V_flat[:, :2], axis=1))
    
    exp_map_flat = dem.discrete_exp_map(V_flat, E_flat, N_flat, center_idx, add_locally=True)
    
    # Compare with true 2D coordinates (shifted to center)
    true_2d = V_flat[:, :2] - V_flat[center_idx, :2]
    
    # Error analysis
    coordinate_error = np.linalg.norm(exp_map_flat - true_2d, axis=1)
    print(f"Flat plane coordinate error - mean: {np.mean(coordinate_error):.6f}, max: {np.max(coordinate_error):.6f}")
    
    # Visualize
    ps.init()
    ps_flat = ps.register_surface_mesh('flat_test', V_flat, F_flat)
    ps_flat.add_parameterization_quantity('computed_exp_map', exp_map_flat)
    ps_flat.add_parameterization_quantity('true_coordinates', true_2d)
    ps_flat.add_scalar_quantity('coordinate_error', coordinate_error, cmap='reds')
    
    print("For flat surfaces, error should be very small (< 1e-4)")
    
    ps.show()

def benchmark_performance():
    """Compare performance with reference implementations"""
    import time
    
    print("=== Performance Benchmark ===")
    
    mesh_sizes = [3, 4, 5]  # icosphere subdivision levels
    
    for level in mesh_sizes:
        V, F = gpy.icosphere(level)
        N = gpy.per_vertex_normals(V, F)
        E = gpy.edges(F)
        
        print(f"\nMesh: {len(V)} vertices, {len(F)} faces")
        
        # Our implementation
        start = time.time()
        our_result = dem.discrete_exp_map(V, E, N, 0, add_locally=True)
        our_time = time.time() - start
        
        # Potpourri3d
        start = time.time()
        solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
        pp3d_result = solver.compute_distance(0)
        pp3d_time = time.time() - start
        
        print(f"Our implementation: {our_time:.3f}s")
        print(f"Potpourri3d heat: {pp3d_time:.3f}s")
        print(f"Speed ratio: {our_time/pp3d_time:.2f}x")

if __name__ == "__main__":
    print("Running comprehensive comparison tests...")
    
    # Main comparison
    our_dist, ref_dist, errors = compare_with_potpourri3d()
    
    # Analytical tests
    test_against_simple_cases()
    
    # Performance comparison
    benchmark_performance()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print("- Check correlation coefficient (should be > 0.95)")
    print("- Check mean relative error (should be < 0.1)")
    print("- Flat plane test should have very low error")
    print("- Visual inspection should show similar patterns")