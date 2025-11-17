#!/usr/bin/env python3
"""
Wrapper script for k-NN based Gaussian scene processing.

This script provides the same command-line interface as the original
process_gaussian_scene.py but calls functions from utils.research_utils.

Enhanced Gaussian Scene Processing with SVD Normal Estimation

This script provides a complete pipeline for processing Gaussian scenes:
1. Load Gaussian splats from PLY using existing load_gaussian_model
2. Build k-NN connectivity using BVH (for unstructured point clouds)
3. Estimate normals using SVD (direction of least variance)
4. Compute discrete exponential maps
5. Compatible with existing visualization scripts

Usage:
    python scripts/process_gaussian_scene.py input.ply --k 15 --output results/
    
Then visualize with existing script:
    python visualization_script/visualize_normals.py results/processed_scene.ply
"""

import argparse
import numpy as np
import os
import sys
import time
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from utils.research_utils import (
        build_knn_direct,
        estimate_normals_svd_knn,
        geobrush_smoothing,
        compute_discrete_exponential_map,
        adjacency_to_edges,
        save_to_ply,
        sample_points_from_gaussians
    )
    from scene.gaussian_model import load_gaussian_model
except Exception as e:
    print(f"[ERROR] Could not import from utils.research_utils: {e}")
    raise


def process_gaussian_scene(ply_path: str,
                          k: int = 15,
                          prune_factor: float = 2.0,
                          smooth_iterations: int = 0,
                          smooth_epsilon: float = 1e-6,
                          smooth_sparse: bool = False,
                          output_dir: str = None,
                          y_up: bool = True,
                          compute_exp_map: bool = False,
                          root_vertex: Optional[int] = None,
                          local_coords: bool = True,
                          n_samples_per_gaussian: int = 2,
                          opacity_threshold: float = 0.1) -> dict:
    """
    Complete pipeline to process a Gaussian scene PLY file using k-NN connectivity.
    
    This function orchestrates the full processing pipeline:
    1. Load Gaussian scene from PLY
    2. Optional: Sample additional points using reparametrization trick
    3. Build k-NN connectivity with adaptive radius pruning
    4. Estimate normals using SVD on neighborhoods
    5. Optional: Apply geobrush smoothing to normals
    6. Optional: Compute discrete exponential maps
    7. Save results to various formats
    
    Args:
        ply_path: Path to input PLY file
        k: Number of neighbors for connectivity graph
        prune_factor: Multiplier for edge pruning (default: 2.0)
        smooth_iterations: Number of geobrush smoothing iterations (default: 0, no smoothing)
                          1 = light smoothing, 3-5 = moderate, 10+ = heavy
        smooth_epsilon: Epsilon for smoothing weight calculation (default: 1e-6)
        smooth_sparse: Use sparse matrix smoothing (faster for iterations >= 5)
        output_dir: Directory to save results (optional)
        y_up: If True, orient normals to point in +Y direction (default: True)
        compute_exp_map: If True, compute discrete exponential map (default: False)
        root_vertex: Root vertex for exponential map (default: None = centroid)
        local_coords: Use local coordinate system for exp map (default: True)
        n_samples_per_gaussian: Number of points to sample per Gaussian (default: 2)
                               Set to 0 or 1 to disable sampling
        opacity_threshold: Minimum opacity to include Gaussian in sampling (default: 0.1)
        
    Returns:
        Dictionary with processed data:
        {
            'vertices': np.ndarray,
            'gaussian_model': GaussianModel,
            'normals': np.ndarray,
            'edges': np.ndarray,
            'avg_radius': float,
            'stats': dict
        }
    """
    total_start = time.time()
    timings = {}
    
    # Step 1: Load Gaussian scene
    step_start = time.time()
    original_vertices, gaussian_model = load_gaussian_model(ply_path)
    timings['load'] = time.time() - step_start
    
    print(f"[INFO] Loaded {len(original_vertices)} Gaussian centers")
    
    # Step 2: Sample additional points using reparametrization trick
    if n_samples_per_gaussian > 1:
        step_start = time.time()
        
        # Get Gaussian parameters
        scales = gaussian_model.get_scaling.detach().cpu().numpy()
        rotations = gaussian_model.get_rotation.detach().cpu().numpy()
        opacities = gaussian_model.get_opacity.detach().cpu().numpy()
        
        # Sample points using GPU-optimized reparametrization (keeps data on GPU!)
        vertices = sample_points_from_gaussians(
            original_vertices,
            scales,
            rotations,
            opacities,
            n_samples_per_gaussian=n_samples_per_gaussian,
            opacity_threshold=opacity_threshold,
            use_gpu=False,
            return_torch=False  # Keep on GPU for k-NN!
        )
        
        timings['sampling'] = time.time() - step_start
        print(f"[INFO] Point cloud size: {len(original_vertices)} -> {len(vertices)} ({len(vertices)/len(original_vertices):.1f}x)")
        print(f"[INFO] ⚡ Sampled points kept on GPU to avoid CPU transfer overhead")
    else:
        vertices = original_vertices
        gaussian_indices = np.arange(len(vertices))
        timings['sampling'] = 0.0
        print(f"[INFO] Reparametrization sampling disabled (n_samples={n_samples_per_gaussian})")
    
    # Step 3: ULTRA-FAST k-NN computation (direct indices, no adjacency conversion!)
    step_start = time.time()
    knn_indices, knn_distances, avg_radius = build_knn_direct(vertices, k=k, use_gpu=True)
    timings['connectivity'] = time.time() - step_start
    
    # Step 4: ULTRA-FAST normal estimation (direct from k-NN indices!)
    step_start = time.time()
    normals, confidences = estimate_normals_svd_knn(
        vertices, knn_indices, knn_distances, k_neighbors=k, y_up=y_up, 
        use_gpu=True, return_confidences=True, sigma=avg_radius, prune_factor=prune_factor
    )
    timings['normal_estimation'] = time.time() - step_start
    
    # Convert vertices back to NumPy if it's a tensor (after all GPU operations are done)
    import torch
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
        print(f"[INFO] ⚡ Converted vertices back to NumPy after GPU pipeline completed")
    
    # Helper function to convert k-NN indices to adjacency lists when needed
    def knn_to_adjacency(knn_indices):
        """Convert k-NN indices to adjacency lists"""
        n_points = len(knn_indices)
        adjacency = [[] for _ in range(n_points)]
        for i, neighbors in enumerate(knn_indices):
            for neighbor_idx in neighbors:
                if neighbor_idx >= 0 and neighbor_idx != i:  # Valid neighbor, not self
                    adjacency[i].append(int(neighbor_idx))
        return adjacency
    
    # Step 5: Optional smoothing (convert k-NN to edges for smoothing function)
    if smooth_iterations > 0:
        step_start = time.time()
        adjacency_lists = knn_to_adjacency(knn_indices)
        edges = adjacency_to_edges(adjacency_lists)  # Convert when needed
        normals = geobrush_smoothing(vertices, normals, edges, 
                                     iterations=smooth_iterations,
                                     epsilon=smooth_epsilon,
                                     use_sparse=smooth_sparse)
        timings['smoothing'] = time.time() - step_start
    else:
        timings['smoothing'] = 0.0
        edges = None
    
    # Step 6: Compute statistics (create edges for stats if not already created)
    if edges is None:
        adjacency_lists = knn_to_adjacency(knn_indices)
        edges = adjacency_to_edges(adjacency_lists)
    
    stats = {
        'n_vertices': len(vertices),
        'n_original_gaussians': len(original_vertices),
        'n_sampled_points': len(vertices) - len(original_vertices) if n_samples_per_gaussian > 1 else 0,
        'sampling_ratio': len(vertices) / len(original_vertices) if n_samples_per_gaussian > 1 else 1.0,
        'n_edges': len(edges),
        'avg_connections': 2 * len(edges) / len(vertices),
        'avg_radius': avg_radius,
        'bbox_min': vertices.min(axis=0).tolist(),
        'bbox_max': vertices.max(axis=0).tolist(),
    }
    
    # Step 7: Optionally compute discrete exponential map
    discrete_exp_map = None
    if compute_exp_map:
        step_start = time.time()
        discrete_exp_map = compute_discrete_exponential_map(
            vertices=vertices,
            edges=edges,
            normals=normals,
            root_vertex=root_vertex,
            local_coordinates=local_coords
        )
        timings['exp_map'] = time.time() - step_start
    else:
        timings['exp_map'] = 0.0
    
    # Prepare result (ensure adjacency_lists exists for result)
    if 'adjacency_lists' not in locals():
        adjacency_lists = knn_to_adjacency(knn_indices)
    
    result = {
        'vertices': vertices,
        'original_vertices': original_vertices,
        'gaussian_indices': gaussian_indices,  # Maps sampled points back to source Gaussians
        'gaussian_model': gaussian_model,
        'normals': normals,
        'confidences': confidences,
        'knn_indices': knn_indices,  # Store k-NN indices for ultra-fast method
        'knn_distances': knn_distances,  # Store k-NN distances
        'adjacency_lists': adjacency_lists,  # Store adjacency lists for compatibility
        'edges': edges,  # Store edges for compatibility
        'avg_radius': avg_radius,
        'discrete_exp_map': discrete_exp_map,  # Store discrete exponential map if computed
        'stats': stats
    }
    
    # Save results if output directory specified
    if output_dir:
        step_start = time.time()
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the Gaussian model
        model_file = os.path.join(output_dir, 'gaussian_model.ply')
        gaussian_model.save_ply(model_file)
        
        # Save processed scene as PLY with vertices, normals, edges, confidence, and exp_map
        processed_ply_file = os.path.join(output_dir, 'processed_scene.ply')
        save_to_ply(vertices, normals, edges, processed_ply_file, confidences, discrete_exp_map)
        
        timings['save'] = time.time() - step_start
    else:
        timings['save'] = 0.0
    
    timings['total'] = time.time() - total_start
    
    # Add timings to stats and result
    stats['timings'] = timings
    result['timings'] = timings
    
    # Print timing summary only
    print(f"\n⏱️  TIMING SUMMARY:")
    print(f"  Load Gaussian model:     {timings['load']:>8.3f}s")
    print(f"  Sample points (reparam): {timings['sampling']:>8.3f}s")
    print(f"  Build k-NN connectivity: {timings['connectivity']:>8.3f}s")
    print(f"  Estimate normals (SVD):  {timings['normal_estimation']:>8.3f}s")
    print(f"  Smoothing:               {timings['smoothing']:>8.3f}s")
    print(f"  Discrete exp map:        {timings['exp_map']:>8.3f}s")
    print(f"  Save results:            {timings['save']:>8.3f}s")
    print(f"  {'-'*30}")
    print(f"  Total:                   {timings['total']:>8.3f}s")
    
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process Gaussian scene with k-NN connectivity and SVD normals"
    )
    parser.add_argument(
        'input_ply',
        help='Path to input Gaussian scene PLY file'
    )
    parser.add_argument(
        'output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=15,
        help='Number of nearest neighbors for connectivity (default: 15)'
    )
    parser.add_argument(
        '--prune-factor',
        type=float,
        default=2.0,
        help='Multiplier for edge pruning threshold (default: 2.0)'
    )
    parser.add_argument(
        '--smooth',
        type=int,
        default=0,
        dest='smooth_iterations',
        help='Number of geobrush smoothing iterations (default: 0, no smoothing). '
             'Values: 0=none, 1-2=light, 3-5=moderate, 10+=heavy'
    )
    parser.add_argument(
        '--smooth-epsilon',
        type=float,
        default=1e-6,
        help='Epsilon for smoothing weight calculation (default: 1e-6)'
    )
    parser.add_argument(
        '--smooth-sparse',
        action='store_true',
        help='Use sparse matrix smoothing (faster for --smooth >= 5)'
    )
    parser.add_argument(
        '--compute-exp-map',
        action='store_true',
        help='Also compute discrete exponential map'
    )
    parser.add_argument(
        '--root-vertex',
        type=int,
        default=None,
        help='Root vertex for exponential map (default: centroid)'
    )
    parser.add_argument(
        '--local-coords',
        action='store_true',
        default=True,
        help='Use local coordinate system for exponential map (default: True)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1,
        dest='n_samples_per_gaussian',
        help='Number of points to sample per Gaussian using reparametrization trick (default: 1 = disabled). '
             'Set to 2 or higher to enable sampling and densify the point cloud.'
    )
    parser.add_argument(
        '--opacity-threshold',
        type=float,
        default=0.1,
        help='Minimum opacity for Gaussians to be included in sampling (default: 0.1)'
    )
    parser.add_argument(
        '--y-up',
        action='store_true',
        default=True,
        help='Orient normals to point in +Y direction (default: True for Polyscope)'
    )
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Validate inputs
    if not os.path.isfile(args.input_ply):
        print(f"[ERROR] Input file not found: {args.input_ply}", file=sys.stderr)
        return 1
    
    if args.k < 3:
        print(f"[ERROR] k must be at least 3, got {args.k}", file=sys.stderr)
        return 1
    
    try:
        # Process scene using local process_gaussian_scene function
        print(f"Processing {args.input_ply} with k={args.k}, prune_factor={args.prune_factor}")
        if args.n_samples_per_gaussian > 1:
            print(f"Reparametrization sampling enabled: {args.n_samples_per_gaussian} samples per Gaussian")
        if args.smooth_iterations > 0:
            print(f"Smoothing enabled: {args.smooth_iterations} iteration(s)")
        if args.compute_exp_map:
            print(f"Exponential map computation enabled")
        result = process_gaussian_scene(
            ply_path=args.input_ply,
            k=args.k,
            prune_factor=args.prune_factor,
            smooth_iterations=args.smooth_iterations,
            smooth_epsilon=args.smooth_epsilon,
            smooth_sparse=args.smooth_sparse,
            output_dir=args.output,
            y_up=args.y_up,
            compute_exp_map=args.compute_exp_map,
            root_vertex=args.root_vertex,
            local_coords=args.local_coords,
            n_samples_per_gaussian=args.n_samples_per_gaussian,
            opacity_threshold=args.opacity_threshold
        )
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Original Gaussians: {result['stats']['n_original_gaussians']}")
        if result['stats']['n_sampled_points'] > 0:
            print(f"Sampled points: {result['stats']['n_sampled_points']}")
            print(f"Total vertices: {result['stats']['n_vertices']} ({result['stats']['sampling_ratio']:.1f}x)")
        else:
            print(f"Total vertices: {result['stats']['n_vertices']} (no sampling)")
        print(f"Edges: {result['stats']['n_edges']}")
        print(f"Average connections per vertex: {result['stats']['avg_connections']:.1f}")
        print(f"Average radius: {result['avg_radius']:.6f}")
        print(f"Total time: {result['timings']['total']:.2f}s")
        
        if args.output:
            print(f"\nResults saved to: {args.output}")
            print("  - gaussian_model.ply (Gaussian model)")
            print("  - processed_scene.ply (vertices + normals + edges for visualization)")
            if args.compute_exp_map:
                print("  - processed_scene.ply includes discrete exponential map coordinates (exp_u, exp_v)")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected failure: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
