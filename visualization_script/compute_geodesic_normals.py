#!/usr/bin/env python3
"""
Compute geodesic-based normal estimates for Gaussian point cloud.

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
  python compute_geodesic_normals.py input.ply output_with_normals.ply
  python compute_geodesic_normals.py input.ply output.ply --geodesic-radius 0.05
  python compute_geodesic_normals.py input.ply output.ply --subsample 500 --limit 2000
  python compute_geodesic_normals.py input.ply output.ply --min-neighbors 15 --batch-size 200
  python compute_geodesic_normals.py input.ply output.ply --sigma 0.02
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import numpy as np
from tqdm import tqdm

# Add the parent directory to the path to import from scene
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
except Exception as e:
    print("[ERROR] torch is required for GaussianModel.")
    raise

try:
    from scene.gaussian_model import GaussianModel, load_gaussian_model
except Exception as e:
    print("[ERROR] Could not import GaussianModel from scene.gaussian_model.")
    raise

try:
    import potpourri3d as pp3d
except Exception as e:
    print("[ERROR] potpourri3d is required.")
    raise

try:
    from plyfile import PlyData, PlyElement
except Exception as e:
    print("[ERROR] plyfile is required.")
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


def estimate_geodesic_radius(points: np.ndarray, percentile: float = 10.0) -> float:
    """
    Estimate a reasonable geodesic radius based on point cloud density.
    Uses nearest neighbor distances in Euclidean space as a proxy.
    """
    from scipy.spatial import KDTree
    
    # Build KDTree for efficient nearest neighbor search
    tree = KDTree(points)
    
    # Sample some points to estimate density
    n_samples = min(1000, len(points))
    sample_indices = np.random.choice(len(points), n_samples, replace=False)
    
    # Find distance to 10th nearest neighbor for each sample
    distances, _ = tree.query(points[sample_indices], k=11)  # k=11 to exclude self
    avg_10nn_dist = distances[:, -1].mean()
    
    # Use a multiple of the average 10-NN distance
    radius = avg_10nn_dist * 3.0
    
    print(f"[INFO] Auto-estimated geodesic radius: {radius:.6f}")
    return radius


def compute_geodesic_normals(points: np.ndarray, 
                            geodesic_radius: float = None,
                            min_neighbors: int = 10,
                            subsample_indices: np.ndarray = None,
                            batch_size: int = 100,
                            sigma: float = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute normal for each point using geodesic distance neighborhoods.
    
    Args:
        points: (N, 3) array of point coordinates
        geodesic_radius: maximum geodesic distance for neighborhood selection
                        If None, will be auto-estimated from point density
        min_neighbors: minimum number of neighbors required for normal estimation
        subsample_indices: if provided, only compute normals for these indices
        batch_size: process points in batches for better progress tracking
        sigma: Gaussian kernel sigma for confidence weighting (default: geodesic_radius/3)
    
    Returns:
        tuple: (normals, confidences) - (N, 3) normal vectors and (N,) confidence values
    """
    n_points = len(points)
    normals = np.zeros((n_points, 3))
    confidences = np.zeros(n_points)
    
    # Auto-estimate radius if not provided
    if geodesic_radius is None:
        geodesic_radius = estimate_geodesic_radius(points)
    
    # Auto-estimate sigma if not provided
    if sigma is None:
        sigma = geodesic_radius / 3.0
    
    print(f"[INFO] Using Gaussian confidence sigma: {sigma:.6f}")
    
    print(f"[INFO] Building PointCloudHeatSolver for {n_points} points...")
    solver = pp3d.PointCloudHeatSolver(points)
    
    # Determine which points to compute normals for
    if subsample_indices is not None:
        compute_indices = subsample_indices
        print(f"[INFO] Computing normals for {len(compute_indices)} subsampled points...")
    else:
        compute_indices = np.arange(n_points)
        print(f"[INFO] Computing normals for all {n_points} points...")
    
    print(f"[INFO] Using geodesic radius: {geodesic_radius:.6f}, min neighbors: {min_neighbors}")
    
    # Optimization: Process in batches to reuse solver
    skipped = 0
    n_compute = len(compute_indices)
    
    # Process in batches for better cache locality
    for batch_start in tqdm(range(0, n_compute, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, n_compute)
        batch_indices = compute_indices[batch_start:batch_end]
        
        for i in batch_indices:
            # Compute geodesic distances from point i to all other points
            geod_dist = solver.compute_distance(i)
            
            # Find all neighbors within geodesic radius
            within_radius = (geod_dist > 1e-10) & (geod_dist <= geodesic_radius)
            neighbor_indices = np.where(within_radius)[0]
            
            # Check if we have enough neighbors
            if len(neighbor_indices) < min_neighbors:
                # Not enough neighbors within radius - assign confidence 0 and skip normal computation
                # (normal will remain zero vector, confidence is 0)
                confidences[i] = 0.0
                skipped += 1
                continue
            
            # Get neighborhood points (centered at origin)
            neighborhood = points[neighbor_indices] - points[i]
            
            # Compute SVD to find principal directions
            U, S, Vt = np.linalg.svd(neighborhood.T @ neighborhood)
            
            # Normal is the direction of minimal variance
            normal = U[:, -1]
            
            # Ensure consistent orientation (Y-up in Polyscope)
            if normal[1] < 0:
                normal = -normal
            
            normals[i] = normal
            
            # Compute Gaussian-weighted confidence
            # Use geodesic distances for Gaussian weighting
            neighbor_geod_dists = geod_dist[neighbor_indices]
            weights = np.exp(-neighbor_geod_dists**2 / (2 * sigma**2))
            confidences[i] = np.sum(weights)
    
    if skipped > 0:
        print(f"[INFO] {skipped} points had fewer than {min_neighbors} neighbors within radius, assigned confidence=0")
    
    # Normalize all normals
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)

    # Ensure all normals point upward (positive Y). If a normal has negative Y,
    # invert it so it points upward. This makes orientation consistent for
    # visualization and downstream processing (Y-up in Polyscope).
    neg_y = normals[:, 1] < 0
    if np.any(neg_y):
        print(f"[INFO] Inverting {neg_y.sum()} normals with negative Y to point upward")
        normals[neg_y] = -normals[neg_y]
    
    # Normalize confidences to [0, 1] range
    if confidences.max() > confidences.min():
        confidences = (confidences - confidences.min()) / (confidences.max() - confidences.min())
        print(f"[INFO] Normalized confidences to [0, 1] range")
    else:
        print(f"[INFO] All confidences equal ({confidences[0]:.6f}), no normalization needed")

    return normals, confidences


def interpolate_normals(points: np.ndarray, 
                       computed_indices: np.ndarray, 
                       computed_normals: np.ndarray,
                       computed_confidences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate normals and confidences for all points based on computed values at subsampled points.
    Uses nearest neighbor interpolation in Euclidean space.
    """
    from scipy.spatial import KDTree
    
    n_points = len(points)
    all_normals = np.zeros((n_points, 3))
    all_confidences = np.zeros(n_points)
    
    # Copy computed normals and confidences
    all_normals[computed_indices] = computed_normals[computed_indices]
    all_confidences[computed_indices] = computed_confidences[computed_indices]
    
    # Find points that need interpolation
    needs_interpolation = np.ones(n_points, dtype=bool)
    needs_interpolation[computed_indices] = False
    interp_indices = np.where(needs_interpolation)[0]
    
    if len(interp_indices) > 0:
        print(f"[INFO] Interpolating normals and confidences for {len(interp_indices)} points...")
        
        # Build KDTree from points with computed normals
        tree = KDTree(points[computed_indices])
        
        # Find nearest computed point for each point needing interpolation
        _, nearest_idx = tree.query(points[interp_indices], k=1)
        
        # Copy normals and confidences from nearest computed point
        all_normals[interp_indices] = computed_normals[computed_indices[nearest_idx]]
        all_confidences[interp_indices] = computed_confidences[computed_indices[nearest_idx]]
    
    return all_normals, all_confidences


def save_ply_with_normals(path: str, points: np.ndarray, normals: np.ndarray, 
                          confidences: np.ndarray = None, gaussian_model: GaussianModel = None):
    """
    Save point cloud with normals and optionally confidences to PLY file.
    Optionally includes Gaussian attributes if model is provided.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    
    # Prepare basic attributes
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')
    ]
    
    # Prepare data arrays
    data_arrays = [
        points[:, 0], points[:, 1], points[:, 2],
        normals[:, 0], normals[:, 1], normals[:, 2]
    ]
    
    # Add confidence if provided
    if confidences is not None:
        dtype_list.append(('confidence', 'f4'))
        data_arrays.append(confidences)
    
    # Add Gaussian attributes if model is provided
    if gaussian_model is not None:
        # Opacities
        opacities = gaussian_model.get_opacity.detach().cpu().numpy()
        dtype_list.append(('opacity', 'f4'))
        data_arrays.append(opacities.squeeze())
        
        # Features DC
        features_dc = gaussian_model.get_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        for i in range(features_dc.shape[1]):
            dtype_list.append((f'f_dc_{i}', 'f4'))
            data_arrays.append(features_dc[:, i])
        
        # Scaling
        scales = gaussian_model.get_scaling.detach().cpu().numpy()
        for i in range(scales.shape[1]):
            dtype_list.append((f'scale_{i}', 'f4'))
            data_arrays.append(scales[:, i])
        
        # Rotation
        rotations = gaussian_model.get_rotation.detach().cpu().numpy()
        for i in range(rotations.shape[1]):
            dtype_list.append((f'rot_{i}', 'f4'))
            data_arrays.append(rotations[:, i])
    
    # Create structured array
    elements = np.empty(len(points), dtype=dtype_list)
    for i, (name, _) in enumerate(dtype_list):
        elements[name] = data_arrays[i]
    
    # Create PLY element and save
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    print(f"[INFO] Saved PLY with normals to: {path}")


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
        
        # Compute geodesic-based normals
        if subsample_indices is not None:
            # Compute normals only for subsampled points
            normals, confidences = compute_geodesic_normals(pts, 
                                              geodesic_radius=args.geodesic_radius,
                                              min_neighbors=args.min_neighbors,
                                              subsample_indices=subsample_indices,
                                              batch_size=args.batch_size,
                                              sigma=args.sigma)
            # Interpolate normals for remaining points
            # normals, confidences = interpolate_normals(pts, subsample_indices, normals, confidences)
        else:
            # Compute normals for all points
            normals, confidences = compute_geodesic_normals(pts,
                                              geodesic_radius=args.geodesic_radius,
                                              min_neighbors=args.min_neighbors,
                                              batch_size=args.batch_size,
                                              sigma=args.sigma)
        
        print(f"[INFO] Computed normals: mean magnitude = {np.linalg.norm(normals, axis=1).mean():.6f}")
        print(f"[INFO] Computed confidences: mean = {confidences.mean():.6f}, std = {confidences.std():.6f}")

        # Safety: Ensure all normals point upward (positive Y) before saving.
        neg_y = normals[:, 1] < 0
        if np.any(neg_y):
            print(f"[INFO] Inverting {neg_y.sum()} normals with negative Y (final pass before save)")
            normals[neg_y] = -normals[neg_y]

        # Save results
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
