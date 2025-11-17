#!/usr/bin/env python3
"""
Test script for exponential map computation and saving integrity.
Verifies that exp_map coordinates are computed correctly and saved properly to PLY.
"""
import sys
import os
import numpy as np
import time

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.research_utils import (
    build_knn_direct,
    estimate_normals_svd_knn,
    adjacency_to_edges,
    compute_discrete_exponential_map,
    save_to_ply
)
from scene.gaussian_model import load_gaussian_model
from scripts.visualization_script.visualize_expmap import load_ply_with_exp_map


def test_expmap_integrity(ply_path: str, output_dir: str = "./test_output"):
    """
    Test exponential map computation and saving integrity.
    
    Args:
        ply_path: Path to input Gaussian PLY file
        output_dir: Directory for test outputs
    """
    print("="*60)
    print("EXPONENTIAL MAP INTEGRITY TEST")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load and process scene
    print("\n1. Loading and processing scene...")
    vertices, gaussian_model = load_gaussian_model(ply_path)
    print(f"   Loaded {len(vertices)} vertices")
    
    # Build k-NN and estimate normals
    knn_indices, knn_distances, avg_radius = build_knn_direct(vertices, k=15, use_gpu=True)
    normals, confidences = estimate_normals_svd_knn(
        vertices, knn_indices, knn_distances, k_neighbors=15, y_up=True, 
        use_gpu=True, return_confidences=True, sigma=avg_radius
    )
    
    # Convert to edges for exp_map computation
    def knn_to_adjacency(knn_indices):
        n_points = len(knn_indices)
        adjacency = [[] for _ in range(n_points)]
        for i, neighbors in enumerate(knn_indices):
            for neighbor_idx in neighbors:
                if neighbor_idx >= 0 and neighbor_idx != i:
                    adjacency[i].append(int(neighbor_idx))
        return adjacency
    
    adjacency_lists = knn_to_adjacency(knn_indices)
    edges = adjacency_to_edges(adjacency_lists)
    print(f"   Built {len(edges)} edges from k-NN graph")
    
    # Step 2: Compute exponential map
    print("\n2. Computing exponential map...")
    
    # Test with different root vertices
    centroid = np.mean(vertices, axis=0)
    centroid_idx = np.argmin(np.linalg.norm(vertices - centroid, axis=1))
    
    test_roots = [
        ("centroid", centroid_idx),
        ("vertex_0", 0),
        ("vertex_100", min(100, len(vertices)-1)),
        ("random", np.random.randint(0, len(vertices)))
    ]
    
    exp_maps = {}
    
    for root_name, root_idx in test_roots:
        print(f"   Computing with root: {root_name} (idx={root_idx})")
        start_time = time.time()
        
        exp_map = compute_discrete_exponential_map(
            vertices=vertices,
            edges=edges,
            normals=normals,
            root_vertex=root_idx,
            local_coordinates=True
        )
        
        compute_time = time.time() - start_time
        
        # Analyze the results
        unique_coords = len(np.unique(exp_map.view(np.void)))
        zero_coords = np.sum(np.all(exp_map == 0, axis=1))
        coord_range_u = exp_map[:, 0].max() - exp_map[:, 0].min()
        coord_range_v = exp_map[:, 1].max() - exp_map[:, 1].min()
        
        exp_maps[root_name] = {
            'exp_map': exp_map,
            'root_idx': root_idx,
            'compute_time': compute_time,
            'unique_coords': unique_coords,
            'zero_coords': zero_coords,
            'range_u': coord_range_u,
            'range_v': coord_range_v
        }
        
        print(f"     Time: {compute_time:.2f}s")
        print(f"     Unique coords: {unique_coords}/{len(vertices)}")
        print(f"     Zero coords: {zero_coords}")
        print(f"     Range U: {coord_range_u:.6f}, V: {coord_range_v:.6f}")
    
    # Step 3: Save and reload test
    print("\n3. Testing save/reload integrity...")
    
    best_root = max(exp_maps.keys(), key=lambda k: exp_maps[k]['unique_coords'])
    print(f"   Using best result: {best_root}")
    
    best_exp_map = exp_maps[best_root]['exp_map']
    test_ply_path = os.path.join(output_dir, "test_expmap.ply")
    
    # Save with exp_map
    save_to_ply(vertices, normals, edges, test_ply_path, confidences, best_exp_map)
    print(f"   Saved to: {test_ply_path}")
    
    # Reload and compare
    loaded_vertices, loaded_normals, loaded_exp_map, loaded_edges = load_ply_with_exp_map(test_ply_path)
    
    # Verify integrity
    vertices_match = np.allclose(vertices, loaded_vertices, rtol=1e-5)
    normals_match = np.allclose(normals, loaded_normals, rtol=1e-5)
    exp_map_match = np.allclose(best_exp_map, loaded_exp_map, rtol=1e-5)
    
    print(f"   Vertices match: {vertices_match}")
    print(f"   Normals match: {normals_match}")
    print(f"   Exp_map match: {exp_map_match}")
    
    if not exp_map_match:
        print("   ERROR: Exponential map data corruption detected!")
        diff = np.abs(best_exp_map - loaded_exp_map)
        print(f"   Max difference: {diff.max():.8f}")
        print(f"   Mean difference: {diff.mean():.8f}")
    
    # Step 4: Generate analysis report
    print("\n4. Analysis Report:")
    print("-" * 40)
    
    for root_name, data in exp_maps.items():
        print(f"Root: {root_name}")
        print(f"  Uniqueness: {data['unique_coords']}/{len(vertices)} ({100*data['unique_coords']/len(vertices):.1f}%)")
        print(f"  Coordinate ranges: U={data['range_u']:.6f}, V={data['range_v']:.6f}")
        print(f"  Zero coordinates: {data['zero_coords']}")
        print(f"  Computation time: {data['compute_time']:.2f}s")
        print()
    
    # Step 5: Recommendations
    print("5. Recommendations:")
    print("-" * 40)
    
    best_uniqueness = max(data['unique_coords'] for data in exp_maps.values())
    if best_uniqueness < len(vertices) * 0.9:
        print("⚠️  Low coordinate uniqueness detected. Issues:")
        print("   - Discrete exp_map algorithm may have convergence problems")
        print("   - Graph connectivity issues (isolated vertices)")
        print("   - Numerical precision limits")
    
    max_range = max(max(data['range_u'], data['range_v']) for data in exp_maps.values())
    if max_range < 0.1:
        print("⚠️  Very small coordinate ranges detected. For visualization:")
        print("   - Scale coordinates by 10-100x for better color mapping")
        print("   - Use higher precision colormap")
        print("   - Consider using discrete colormap instead of continuous")
    
    return {
        'exp_maps': exp_maps,
        'best_root': best_root,
        'integrity_passed': exp_map_match,
        'test_ply_path': test_ply_path
    }


def create_visualization_test(test_ply_path: str):
    """
    Create a test visualization script that shows different aspects of the exp_map.
    """
    script_content = f'''#!/usr/bin/env python3
"""
Generated test visualization for exponential map.
Shows multiple views with different scalings and color mappings.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import polyscope as ps
    HAS_POLYSCOPE = True
except ImportError:
    HAS_POLYSCOPE = False
    import matplotlib.pyplot as plt

from scripts.visualization_script.visualize_expmap import load_ply_with_exp_map

def main():
    ply_path = "{test_ply_path}"
    V, N, EMap, E = load_ply_with_exp_map(ply_path)
    
    if EMap is None:
        print("No exp_map found!")
        return
    
    print(f"Loaded {{len(V)}} vertices")
    print(f"Exp_map range: U=[{{EMap[:,0].min():.6f}}, {{EMap[:,0].max():.6f}}]")
    print(f"               V=[{{EMap[:,1].min():.6f}}, {{EMap[:,1].max():.6f}}]")
    
    # Find root (closest to origin in exp space)
    root_idx = np.argmin(np.linalg.norm(EMap, axis=1))
    distances = np.linalg.norm(EMap - EMap[root_idx], axis=1)
    
    if HAS_POLYSCOPE:
        # Polyscope visualization
        ps.init()
        ps.set_up_dir("y_up")
        
        # Main point cloud
        pc = ps.register_point_cloud("Points", V)
        
        # Original coordinates (scaled for visibility)
        scale_factor = 10.0  # Scale up small coordinates
        pc.add_scalar_quantity("exp_u_scaled", EMap[:,0] * scale_factor, enabled=False)
        pc.add_scalar_quantity("exp_v_scaled", EMap[:,1] * scale_factor, enabled=False)
        
        # Distance from root
        pc.add_scalar_quantity("distance_from_root", distances, enabled=True)
        
        # Discrete distance bands
        max_dist = distances.max()
        bands = np.floor(distances / (max_dist / 10))  # 10 bands
        pc.add_scalar_quantity("distance_bands", bands, enabled=False)
        
        # Root highlighting
        root_colors = np.zeros((len(V), 3))
        root_colors[root_idx] = [1, 0, 0]
        pc.add_color_quantity("root", root_colors, enabled=False)
        
        print("\\nPolyscope controls:")
        print("- Toggle between different scalar quantities")
        print("- 'distance_from_root' shows geodesic-like distances")
        print("- 'distance_bands' shows discrete distance levels")
        print("- 'exp_u_scaled' and 'exp_v_scaled' show scaled coordinates")
        
        ps.show()
    
    else:
        # Matplotlib fallback
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 2D exp_map scatter
        scatter = axes[0,0].scatter(EMap[:,0], EMap[:,1], c=distances, s=1, cmap='plasma')
        axes[0,0].plot(EMap[root_idx,0], EMap[root_idx,1], 'r*', markersize=10, label='Root')
        axes[0,0].set_title('Exp Map 2D (colored by distance)')
        axes[0,0].set_xlabel('exp_u')
        axes[0,0].set_ylabel('exp_v')
        axes[0,0].legend()
        plt.colorbar(scatter, ax=axes[0,0])
        
        # Histogram of distances
        axes[0,1].hist(distances, bins=50, alpha=0.7)
        axes[0,1].set_title('Distance Distribution')
        axes[0,1].set_xlabel('Distance from root')
        axes[0,1].set_ylabel('Count')
        
        # U coordinate distribution
        axes[1,0].hist(EMap[:,0], bins=50, alpha=0.7, label='exp_u')
        axes[1,0].set_title('U Coordinate Distribution')
        axes[1,0].set_xlabel('exp_u value')
        axes[1,0].set_ylabel('Count')
        
        # V coordinate distribution  
        axes[1,1].hist(EMap[:,1], bins=50, alpha=0.7, label='exp_v', color='orange')
        axes[1,1].set_title('V Coordinate Distribution')
        axes[1,1].set_xlabel('exp_v value')
        axes[1,1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
'''
    
    script_path = os.path.dirname(test_ply_path).replace('\\', '/') + "/test_visualization.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_ply', help='Input Gaussian PLY file')
    parser.add_argument('--output-dir', default='./test_output', help='Output directory')
    args = parser.parse_args()
    
    # Run integrity test
    results = test_expmap_integrity(args.input_ply, args.output_dir)
    
    # Create visualization test
    if results['integrity_passed']:
        viz_script = create_visualization_test(results['test_ply_path'])
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Visualization test script created: {viz_script}")
        print(f"\nTo run visualization test:")
        print(f"   python {viz_script}")
    else:
        print(f"\n❌ Integrity test failed - data corruption detected")