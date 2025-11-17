#!/usr/bin/env python3
"""
Gaussian Normal Estimation Script

Estimates normals from Gaussian splat parameters (rotation and scale).
Uses the rotation matrix and scale vector to find the normal direction
(column corresponding to smallest scale), then averages over k-NN neighborhoods.

Usage:
    python scripts/gaussian_normal_estimation.py input.ply output_dir/ [options]

Examples:
    # Basic usage
    python scripts/gaussian_normal_estimation.py data/scene.ply results/
    
    # With more neighbors
    python scripts/gaussian_normal_estimation.py data/scene.ply results/ --k 20
    
    # With smoothing
    python scripts/gaussian_normal_estimation.py data/scene.ply results/ --k 15 --smooth 5
"""

import sys
import os
import argparse
import numpy as np
import time

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scene.gaussian_model import GaussianModel, load_gaussian_model
from utils.research_utils import (
    build_connectivity,
    gaussian_normal_estimation,
    geobrush_smoothing,
    save_ply_with_normals,
    save_to_ply
)


def process_gaussian_normal_estimation(
    ply_path: str,
    k: int = 15,
    prune_factor: float = 2.0,
    smooth_iterations: int = 0,
    smooth_epsilon: float = 1e-6,
    smooth_sparse: bool = False,
    output_dir: str | None = None,
    y_up: bool = True
) -> dict:
    """
    Process Gaussian scene to estimate normals from Gaussian rotation and scale.
    
    Pipeline:
    1. Load Gaussian model
    2. Build k-NN connectivity
    3. Estimate normals from rotation/scale parameters
    4. Optional: Apply smoothing
    5. Optional: Save results
    
    Args:
        ply_path: Path to input Gaussian PLY file
        k: Number of nearest neighbors for connectivity
        prune_factor: Multiplier for edge pruning (edges > prune_factor * avg_radius removed)
        smooth_iterations: Number of smoothing iterations (0 = no smoothing)
        smooth_epsilon: Epsilon for smoothing weight calculation
        smooth_sparse: Use sparse matrix smoothing (faster for iterations >= 5)
        output_dir: Output directory (if None, no files saved)
        y_up: Orient normals to point in +Y direction
        
    Returns:
        Dictionary with 'vertices', 'normals', 'edges', 'scales', 'rotations', etc.
    """
    # Track timing for each step
    timings = {}
    total_start = time.time()
    
    # 1. Load Gaussian model
    step_start = time.time()
    _, gaussian = load_gaussian_model(ply_path)
    
    vertices = gaussian.get_xyz.detach().cpu().numpy().astype(np.float64)
    scales = gaussian.get_scaling.detach().cpu().numpy()  # (N, 3)
    rotations = gaussian.get_rotation.detach().cpu().numpy()  # (N, 4) quaternions
    
    timings['load'] = time.time() - step_start
    
    # 2. Build k-NN connectivity
    step_start = time.time()
    edges, avg_radius = build_connectivity(vertices, k=k, prune_factor=prune_factor, use_gpu=True)
    timings['connectivity'] = time.time() - step_start
    
    # 3. Estimate normals from Gaussian parameters
    step_start = time.time()
    normals = gaussian_normal_estimation(
        vertices=vertices,
        scales=scales,
        rotations=rotations,
        edges=edges,
        y_up=y_up
    )
    timings['normal_estimation'] = time.time() - step_start
    
    # 4. Optional smoothing
    if smooth_iterations > 0:
        step_start = time.time()
        normals = geobrush_smoothing(
            vertices=vertices,
            normals=normals,
            edges=edges,
            iterations=smooth_iterations,
            epsilon=smooth_epsilon,
            use_sparse=smooth_sparse
        )
        timings['smoothing'] = time.time() - step_start
    else:
        timings['smoothing'] = 0.0
    
    # 5. Save results
    if output_dir:
        step_start = time.time()
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Gaussian model with normals
        gaussian_ply_file = os.path.join(output_dir, 'gaussian_model.ply')
        save_ply_with_normals(gaussian_ply_file, vertices, normals, 
                             confidences=None, gaussian_model=gaussian)
        
        # Save processed scene (vertices + normals + edges)
        processed_ply_file = os.path.join(output_dir, 'processed_scene.ply')
        save_to_ply(vertices, normals, edges, processed_ply_file)
        
        timings['save'] = time.time() - step_start
    else:
        timings['save'] = 0.0
    
    timings['total'] = time.time() - total_start
    
    # Print timing summary only
    print(f"\n⏱️  TIMING SUMMARY:")
    print(f"  Load Gaussian model:     {timings['load']:>8.3f}s")
    print(f"  Build k-NN connectivity: {timings['connectivity']:>8.3f}s")
    print(f"  Estimate normals:        {timings['normal_estimation']:>8.3f}s")
    print(f"  Smoothing:               {timings['smoothing']:>8.3f}s")
    print(f"  Save results:            {timings['save']:>8.3f}s")
    print(f"  {'-'*30}")
    print(f"  Total:                   {timings['total']:>8.3f}s")
    
    return {
        'vertices': vertices,
        'normals': normals,
        'edges': edges,
        'scales': scales,
        'rotations': rotations,
        'avg_radius': avg_radius,
        'timings': timings
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate normals from Gaussian rotation and scale parameters"
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
        help='Number of geobrush smoothing iterations (default: 0, no smoothing)'
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
        '--y-up',
        action='store_true',
        default=True,
        help='Orient normals to point in +Y direction (default: True)'
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
        # Process scene with Gaussian normal estimation
        result = process_gaussian_normal_estimation(
            ply_path=args.input_ply,
            k=args.k,
            prune_factor=args.prune_factor,
            smooth_iterations=args.smooth_iterations,
            smooth_epsilon=args.smooth_epsilon,
            smooth_sparse=args.smooth_sparse,
            output_dir=args.output,
            y_up=args.y_up
        )
        
        print(f"\nSuccess! Processed {len(result['vertices'])} vertices")
        return 0
        
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
