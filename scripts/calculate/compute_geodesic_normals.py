#!/usr/bin/env python3
"""
Wrapper script for geodesic-based normal computation.

This script provides the same command-line interface as the original
compute_geodesic_normals.py but calls functions from utils.research_utils.

For each point:
1. Compute geodesic distances to all other points
2. Select neighbors within geodesic distance radius
3. Use SVD on neighborhood to find normal (direction of minimal variance)
4. Save results to PLY file with normal estimates

Features:
  * Uses geodesic distance radius instead of fixed K-nearest neighbors
  * Automatic radius estimation based on point cloud density
  * Gaussian-weighted confidence scores for each normal estimate
  * Optional subsampling for computational efficiency with interpolation
  * Batch processing for better progress tracking
  * Preserves all Gaussian attributes in output PLY

Performance Tips:
  * Use --subsample for large point clouds (e.g., --subsample 1000 for 100k points)
  * Increase --batch-size for better performance (default: 100)
  * The bottleneck is computing geodesic distances, not the SVD step
  * For very large clouds (>50k points), strongly recommend --subsample

CLI Examples:
  python scripts/compute_geodesic_normals.py input.ply output_with_normals.ply
  python scripts/compute_geodesic_normals.py input.ply output.ply --geodesic-radius 0.05
  python scripts/compute_geodesic_normals.py input.ply output.ply --subsample 500 --limit 2000
  python scripts/compute_geodesic_normals.py input.ply output.ply --min-neighbors 15 --batch-size 200
  python scripts/compute_geodesic_normals.py input.ply output.ply --sigma 0.02
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
except Exception as e:
    print("[ERROR] torch is required for GaussianModel.")
    raise

try:
    from scene.gaussian_model import load_gaussian_model
except Exception as e:
    print("[ERROR] Could not import load_gaussian_model from scene.gaussian_model.")
    raise

try:
    from utils.research_utils import (
        compute_geodesic_normals,
        save_ply_with_normals
    )
except Exception as e:
    print("[ERROR] Could not import from utils.research_utils.")
    raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute geodesic-based normals for Gaussian point cloud.")
    p.add_argument('input_ply', help='Path to input .ply file')
    p.add_argument('output_ply', help='Path to output .ply file with normals')
    p.add_argument('--geodesic-radius', type=float, default=None,
                   help='Geodesic distance radius for neighborhood selection (default: auto-compute from point density)')
    p.add_argument('--min-neighbors', type=int, default=10,
                   help='Minimum number of neighbors required for normal estimation (default: 10)')
    p.add_argument('--subsample', type=int, default=0,
                   help='Subsample point cloud to N points for normal computation (0 = use all points). '
                        'Final output will still contain all original points with interpolated normals.')
    p.add_argument('--limit', type=int, default=0,
                   help='Randomly subsample input to N points before processing (0 = no limit)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for subsampling')
    p.add_argument('--sh-degree', type=int, default=3, help='Spherical harmonics degree (default: 3)')
    p.add_argument('--batch-size', type=int, default=100,
                   help='Batch size for progress tracking (default: 100)')
    p.add_argument('--sigma', type=float, default=None,
                   help='Gaussian kernel sigma for confidence weighting (default: geodesic_radius/3)')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    
    # Validate inputs
    if not os.path.isfile(args.input_ply):
        print(f"[ERROR] Input file not found: {args.input_ply}", file=sys.stderr)
        return 1
    if args.geodesic_radius is not None and args.geodesic_radius <= 0:
        print(f"[ERROR] geodesic_radius must be positive, got {args.geodesic_radius}", file=sys.stderr)
        return 1
    if args.sigma is not None and args.sigma <= 0:
        print(f"[ERROR] sigma must be positive, got {args.sigma}", file=sys.stderr)
        return 1
    if args.min_neighbors < 3:
        print(f"[ERROR] min_neighbors must be at least 3 for SVD, got {args.min_neighbors}", file=sys.stderr)
        return 1
    
    t0 = time.perf_counter()
    
    try:
        # Load Gaussian model
        print(f"[INFO] Loading: {args.input_ply}")
        pts, gaussian_model = load_gaussian_model(args.input_ply, sh_degree=args.sh_degree)
        print(f"[INFO] Loaded {len(pts)} points. Bounds min={pts.min(0)}, max={pts.max(0)}")
        
        # Subsample entire dataset if requested (reduces final output size)
        if args.limit and len(pts) > args.limit:
            print(f"[INFO] Subsampling entire dataset from {len(pts)} -> {args.limit} (seed={args.seed})")
            rng = np.random.default_rng(args.seed)
            sel = rng.choice(len(pts), args.limit, replace=False)
            pts = pts[sel]
            
            # Subsample Gaussian model tensors
            gaussian_model._xyz = torch.nn.Parameter(gaussian_model._xyz[sel].clone())
            gaussian_model._features_dc = torch.nn.Parameter(gaussian_model._features_dc[sel].clone())
            gaussian_model._features_rest = torch.nn.Parameter(gaussian_model._features_rest[sel].clone())
            gaussian_model._opacity = torch.nn.Parameter(gaussian_model._opacity[sel].clone())
            gaussian_model._scaling = torch.nn.Parameter(gaussian_model._scaling[sel].clone())
            gaussian_model._rotation = torch.nn.Parameter(gaussian_model._rotation[sel].clone())
        
        # Determine subsampling for normal computation (for efficiency)
        subsample_indices = None
        if args.subsample and len(pts) > args.subsample:
            print(f"[INFO] Subsampling for normal computation: {args.subsample} points")
            rng = np.random.default_rng(args.seed)
            subsample_indices = np.sort(rng.choice(len(pts), args.subsample, replace=False))
        
        # Compute geodesic-based normals using research_utils
        if subsample_indices is not None:
            # Compute normals only for subsampled points
            normals, confidences = compute_geodesic_normals(pts, 
                                              geodesic_radius=args.geodesic_radius,
                                              min_neighbors=args.min_neighbors,
                                              subsample_indices=subsample_indices,
                                              batch_size=args.batch_size,
                                              sigma=args.sigma,
                                              y_up=True)
        else:
            # Compute normals for all points
            normals, confidences = compute_geodesic_normals(pts,
                                              geodesic_radius=args.geodesic_radius,
                                              min_neighbors=args.min_neighbors,
                                              batch_size=args.batch_size,
                                              sigma=args.sigma,
                                              y_up=True)
        
        print(f"[INFO] Computed normals: mean magnitude = {np.linalg.norm(normals, axis=1).mean():.6f}")
        print(f"[INFO] Computed confidences: mean = {confidences.mean():.6f}, std = {confidences.std():.6f}")

        # Safety: Ensure all normals point upward (positive Y) before saving
        neg_y = normals[:, 1] < 0
        if np.any(neg_y):
            print(f"[INFO] Inverting {neg_y.sum()} normals with negative Y (final pass before save)")
            normals[neg_y] = -normals[neg_y]

        # Save results using research_utils
        save_ply_with_normals(args.output_ply, pts, normals, confidences, gaussian_model)
        
        total_dt = time.perf_counter() - t0
        print(f"[INFO] Done in {total_dt:.2f}s")
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
