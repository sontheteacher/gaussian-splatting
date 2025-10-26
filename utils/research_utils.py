#!/usr/bin/env python3
"""
Research Utilities for Gaussian Splatting Scene Analysis

This module provides core utilities for processing Gaussian splatting scenes:

1. Geodesic-based normal estimation with confidence scores
   - Uses geodesic distance neighborhoods (potpourri3d heat method)
   - Gaussian-weighted confidence scores
   - Optional subsampling for large point clouds
   - BVH-accelerated radius estimation

2. k-NN based normal estimation with connectivity
   - Uses k-nearest neighbor connectivity (BVH-based)
   - SVD-based normal estimation
   - Adaptive radius-based edge pruning
   - Discrete exponential map computation

Note: High-level pipeline functions (like process_gaussian_scene) should be
implemented in wrapper scripts, not in this utility module. This module
provides core building blocks.

Functions:
  Geodesic-based:
    - compute_geodesic_normals() - Main geodesic normal computation
    - interpolate_normals() - Interpolate for non-computed points
    NOTE: estimate_geodesic_radius() commented out (radius computed in build_connectivity)
    
  Gaussian parameter-based:
    - gaussian_normal_estimation() - Estimate normals from rotation/scale
    - quaternion_to_rotation_matrix() - Convert quaternions to matrices
    
  k-NN based:
    - build_connectivity() - Build k-NN graph with adaptive pruning (includes radius estimation)
    - estimate_normals_svd_simple() - SVD-based normals from adjacency
    - geobrush_smoothing() - Distance-weighted normal smoothing
    - compute_exponential_map() - Exponential map computation
    
  Common utilities:
    - save_ply_with_normals() - Save with Gaussian attributes
    - save_to_ply() - Save vertices + normals + edges
    - load_processed_scene() - Load from NPZ
"""

from __future__ import annotations
import numpy as np
import os
import sys
import time
from typing import Tuple, Optional
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
    print("[WARN] potpourri3d not available. Geodesic normal computation will not work.")
    pp3d = None

try:
    from plyfile import PlyData, PlyElement
except Exception as e:
    print("[ERROR] plyfile is required.")
    raise

try:
    from scipy.spatial import KDTree
    from scipy.sparse import csr_matrix
except Exception as e:
    print("[ERROR] scipy is required.")
    raise

try:
    from . import discrete_exp_map as dem
except ImportError:
    try:
        import discrete_exp_map as dem
    except ImportError:
        print("[WARN] discrete_exp_map not available. k-NN based processing will not work.")
        dem = None

# Try to import pybvh for efficient BVH-based operations
try:
    import pybvh
    HAS_PYBVH = True
except ImportError:
    print("[WARN] pybvh not available. Will fall back to scipy KDTree for radius estimation.")
    HAS_PYBVH = False


# ============================================================================
# GEODESIC-BASED NORMAL ESTIMATION (potpourri3d)
# ============================================================================

def estimate_geodesic_radius(points: np.ndarray, percentile: float = 10.0, use_gpu: bool = True, k_neighbors: int = 10) -> float:
    """
    Ultra-fast estimate of geodesic radius using GPU-accelerated nearest neighbors.
    
    Optimizations:
    1. PyTorch GPU KNN for massive speedup (10-50x faster than CPU)
    2. PyTorch tensor operations for vectorized distance computation
    3. Advanced sampling strategies for large point clouds
    4. Vectorized percentile computation
    
    Args:
        points: (N, 3) array of point coordinates
        percentile: Percentile for distance estimation (default: 10.0)
        use_gpu: If True, use GPU acceleration (PyTorch)
        k_neighbors: Number of neighbors to query for density estimation
        
    Returns:
        Estimated geodesic radius
    """
    n_points = len(points)
    
    # Adaptive sampling for very large point clouds
    max_samples = 10000  # Increased for better accuracy
    if n_points > max_samples:
        # Stratified sampling for better coverage
        sample_indices = np.random.choice(n_points, max_samples, replace=False)
        sample_points = points[sample_indices]
        n_samples = max_samples
    else:
        sample_points = points
        sample_indices = np.arange(n_points)
        n_samples = n_points
    
    # Use PyTorch GPU-accelerated k-NN
    try:
        # Get k-NN using our PyTorch implementation
        indices, distances = _torch_knn(sample_points, k_neighbors, use_gpu)
        
        # Vectorized computation of k-th nearest neighbor distances
        kth_distances = distances[:, -1]  # Last column = k-th neighbor
        
        # Use GPU for percentile computation if large dataset
        if n_samples > 1000 and use_gpu:
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            kth_distances_torch = torch.from_numpy(kth_distances).to(device)
            percentile_distance = torch.quantile(kth_distances_torch, percentile / 100.0).cpu().numpy()
        else:
            percentile_distance = np.percentile(kth_distances, percentile)
        
        avg_knn_dist = np.mean(kth_distances)
        
    except Exception as e:
        print(f"[WARN] PyTorch radius estimation failed ({e}), falling back to CPU")
        # Fall back to CPU methods
        avg_knn_dist, percentile_distance = _estimate_radius_cpu(sample_points, k_neighbors, percentile)
    
    # Use adaptive multiple based on point cloud characteristics
    # For dense clouds: smaller multiple, for sparse: larger multiple
    density_factor = 3.0  # Default value
    if percentile_distance > 1e-10:  # Avoid division by zero
        density_factor = max(2.0, min(5.0, 3.0 * (avg_knn_dist / percentile_distance)))
    radius = avg_knn_dist * density_factor
    
    return radius


def _estimate_radius_cpu(points: np.ndarray, k_neighbors: int, percentile: float) -> tuple[float, float]:
    """
    CPU fallback for radius estimation with optimized implementations.
    
    Returns:
        tuple: (avg_knn_dist, percentile_distance)
    """
    n_points = len(points)
    
    # Try BVH first (fastest CPU method)
    if HAS_PYBVH:
        # Build BVH tree
        tree = pybvh.build_bvh_points(points)
        
        # Query k+1 nearest neighbors (excluding self)
        k = k_neighbors + 1
        results = pybvh.knn(points, k, tree)
        
        # Vectorized distance extraction
        kth_distances = []
        for i in range(n_points):
            if len(results[i]) >= k:
                kth_distances.append(results[i][k-1].dist)
            elif len(results[i]) > 0:
                kth_distances.append(results[i][-1].dist)
        
        if len(kth_distances) == 0:
            print("[WARN] No valid distances found, using default values")
            return 0.1, 0.1
        
        kth_distances = np.array(kth_distances)
        
    else:
        # Fall back to scipy KDTree with vectorized operations
        from scipy.spatial import KDTree
        tree = KDTree(points)
        
        # Vectorized KNN query for all points at once
        distances, _ = tree.query(points, k=k_neighbors+1)  # +1 to exclude self
        kth_distances = distances[:, -1]  # Last column = k-th neighbor distance
    
    # Vectorized statistics computation
    avg_knn_dist = np.mean(kth_distances)
    percentile_distance = np.percentile(kth_distances, percentile)
    
    return avg_knn_dist, percentile_distance


def estimate_geodesic_radius_legacy(points: np.ndarray, percentile: float = 10.0, use_bvh: bool = True) -> float:
    """
    Estimate a reasonable geodesic radius based on point cloud density.
    Uses nearest neighbor distances in Euclidean space as a proxy.
    
    Two implementations available:
    1. BVH-based (pybvh) - Fast, recommended for large point clouds
    2. KDTree-based (scipy) - Fallback if pybvh not available
    
    Args:
        points: (N, 3) array of point coordinates
        percentile: Percentile for distance estimation (unused, kept for API compatibility)
        use_bvh: If True and pybvh available, use BVH implementation (default: True)
        
    Returns:
        Estimated geodesic radius
    """
    n_points = len(points)
    
    # Use BVH implementation if available and requested
    if use_bvh and HAS_PYBVH:
        print(f"[INFO] Using BVH-based radius estimation (fast)")
        
        # Build BVH tree
        tree = pybvh.build_bvh_points(points)
        
        # Sample points to estimate density
        n_samples = min(1000, n_points)
        if n_samples < n_points:
            sample_indices = np.random.choice(n_points, n_samples, replace=False)
            sample_points = points[sample_indices]
        else:
            sample_points = points
            n_samples = n_points
        
        # Query k=11 nearest neighbors (excluding self)
        k = 11
        results = pybvh.knn(sample_points, k, tree)
        
        # Extract 10th nearest neighbor distances
        distances_10nn = []
        for i in range(n_samples):
            # results[i] is a list of k nearest neighbors
            # Get the last one (10th neighbor, excluding self)
            if len(results[i]) >= k:
                distances_10nn.append(results[i][k-1].dist)
            elif len(results[i]) > 0:
                distances_10nn.append(results[i][-1].dist)
        
        if len(distances_10nn) == 0:
            print("[WARN] No valid distances found, using default radius 0.1")
            return 0.1
        
        avg_10nn_dist = np.mean(distances_10nn)
        
    else:
        # Fall back to scipy KDTree
        if use_bvh and not HAS_PYBVH:
            print(f"[INFO] pybvh not available, falling back to scipy KDTree")
        else:
            print(f"[INFO] Using KDTree-based radius estimation")
        
        # Build KDTree for efficient nearest neighbor search
        from scipy.spatial import KDTree
        tree = KDTree(points)
        
        # Sample some points to estimate density
        n_samples = min(1000, n_points)
        sample_indices = np.random.choice(n_points, n_samples, replace=False)
        
        # Find distance to 10th nearest neighbor for each sample
        distances, _ = tree.query(points[sample_indices], k=11)  # k=11 to exclude self
        avg_10nn_dist = distances[:, -1].mean()
    
    # Use a multiple of the average 10-NN distance
    radius = avg_10nn_dist * 3.0
    
    print(f"[INFO] Auto-estimated geodesic radius: {radius:.6f} (avg 10-NN dist: {avg_10nn_dist:.6f})")
    return radius


def compute_geodesic_normals(points: np.ndarray, 
                            geodesic_radius: float = None,
                            min_neighbors: int = 10,
                            subsample_indices: np.ndarray = None,
                            batch_size: int = 100,
                            sigma: float = None,
                            y_up: bool = True) -> tuple[np.ndarray, np.ndarray]:
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
        y_up: If True, orient normals to point in +Y direction (default: True)
    
    Returns:
        tuple: (normals, confidences) - (N, 3) normal vectors and (N,) confidence values
    """
    if pp3d is None:
        raise ImportError("potpourri3d is required for geodesic normal computation")
    
    n_points = len(points)
    normals = np.zeros((n_points, 3))
    confidences = np.zeros(n_points)
    
    # Auto-estimate radius if not provided
    if geodesic_radius is None:
        # NOTE: Using simple default since estimate_geodesic_radius is commented out
        # Radius estimation is now handled in build_connectivity
        geodesic_radius = 0.1  # Default conservative radius
        print(f"[INFO] Using default geodesic radius: {geodesic_radius:.6f}")
    
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
    
    # Process in batches
    skipped = 0
    n_compute = len(compute_indices)
    
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
                confidences[i] = 0.0
                skipped += 1
                continue
            
            # Get neighborhood points (centered at origin)
            neighborhood = points[neighbor_indices] - points[i]
            
            # Compute SVD to find principal directions
            U, S, Vt = np.linalg.svd(neighborhood.T @ neighborhood)
            
            # Normal is the direction of minimal variance
            normal = U[:, -1]
            
            # Ensure consistent orientation
            if y_up and normal[1] < 0:
                normal = -normal
            elif not y_up and normal[2] < 0:
                normal = -normal
            
            normals[i] = normal
            
            # Compute Gaussian-weighted confidence
            neighbor_geod_dists = geod_dist[neighbor_indices]
            weights = np.exp(-neighbor_geod_dists**2 / (2 * sigma**2))
            
            # Confidence is sum of Gaussian weights
            confidences[i] = np.sum(weights)
    
    if skipped > 0:
        print(f"[INFO] {skipped} points had fewer than {min_neighbors} neighbors within radius, assigned confidence=0")
    
    # Normalize all normals
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)

    # Final pass: ensure all normals point upward
    if y_up:
        neg_axis = normals[:, 1] < 0
        axis_name = "Y"
    else:
        neg_axis = normals[:, 2] < 0
        axis_name = "Z"
        
    if np.any(neg_axis):
        print(f"[INFO] Inverting {neg_axis.sum()} normals with negative {axis_name} to point upward")
        normals[neg_axis] = -normals[neg_axis]
    
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
                       computed_confidences: np.ndarray,
                       use_gpu: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast interpolate normals and confidences using nearest neighbor search.
    
    Optimizations:
    1. Vectorized operations for all interpolation at once
    2. Memory-efficient processing for large point clouds
    
    Args:
        points: All point positions (N, 3)
        computed_indices: Indices where normals were computed
        computed_normals: Normals array with values at computed_indices
        computed_confidences: Confidences array with values at computed_indices
        use_gpu: If True, use GPU acceleration (parameter kept for compatibility)
        
    Returns:
        tuple: (all_normals, all_confidences) with interpolated values
    """
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
        # Use CPU interpolation (cuML removed for compatibility)
        _interpolate_cpu(points, computed_indices, computed_normals, computed_confidences,
                       interp_indices, all_normals, all_confidences)
    
    return all_normals, all_confidences


def _interpolate_cpu(points: np.ndarray, computed_indices: np.ndarray, 
                    computed_normals: np.ndarray, computed_confidences: np.ndarray,
                    interp_indices: np.ndarray, all_normals: np.ndarray, all_confidences: np.ndarray):
    """CPU fallback for normal interpolation."""
    # Build KDTree from points with computed normals
    tree = KDTree(points[computed_indices])
    
    # Find nearest computed point for each point needing interpolation
    _, nearest_idx = tree.query(points[interp_indices], k=1)
    
    # Vectorized copy of normals and confidences from nearest computed point
    all_normals[interp_indices] = computed_normals[computed_indices[nearest_idx]]
    all_confidences[interp_indices] = computed_confidences[computed_indices[nearest_idx]]


def interpolate_normals_legacy(points: np.ndarray, 
                       computed_indices: np.ndarray, 
                       computed_normals: np.ndarray,
                       computed_confidences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Legacy interpolate normals function (CPU-only).
    Uses nearest neighbor interpolation in Euclidean space.
    
    Args:
        points: All point positions (N, 3)
        computed_indices: Indices where normals were computed
        computed_normals: Normals array with values at computed_indices
        computed_confidences: Confidences array with values at computed_indices
        
    Returns:
        tuple: (all_normals, all_confidences) with interpolated values
    """
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


def compute_pairwise_distances_gpu(points1: np.ndarray, points2: np.ndarray = None, use_gpu: bool = True) -> np.ndarray:
    """
    Ultra-fast pairwise distance computation using PyTorch GPU acceleration.
    
    Optimizations:
    1. PyTorch GPU tensors for vectorized distance computation
    2. Memory-efficient chunking for large datasets
    3. Automatic fallback to CPU if GPU unavailable
    
    Args:
        points1: First set of points (N, 3)
        points2: Second set of points (M, 3). If None, computes distances within points1
        use_gpu: If True, use GPU acceleration
        
    Returns:
        Distance matrix (N, M) or (N, N) if points2 is None
    """
    if points2 is None:
        points2 = points1
    
    n1, n2 = len(points1), len(points2)
    
    # Use GPU acceleration if available
    if use_gpu and torch.cuda.is_available():
        try:
            # Convert to GPU tensors
            device = torch.device('cuda')
            p1_tensor = torch.from_numpy(points1).float().to(device)
            p2_tensor = torch.from_numpy(points2).float().to(device)
            
            # Compute pairwise distances using PyTorch's efficient implementation
            # This uses optimized CUDA kernels
            distances = torch.cdist(p1_tensor, p2_tensor, p=2.0)
            
            # Convert back to numpy
            return distances.cpu().numpy()
            
        except Exception as e:
            print(f"[WARN] GPU distance computation failed ({e}), falling back to CPU")
    
    # CPU fallback using scipy's optimized implementation
    from scipy.spatial.distance import cdist
    return cdist(points1, points2, metric='euclidean')


# ============================================================================
# GAUSSIAN-BASED NORMAL ESTIMATION
# ============================================================================

def quaternion_to_rotation_matrix(quaternions: np.ndarray) -> np.ndarray:
    """
    Convert quaternions to rotation matrices.
    
    Args:
        quaternions: Quaternions (N, 4) in [w, x, y, z] format
        
    Returns:
        Rotation matrices (N, 3, 3)
    """
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Normalize quaternions
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Build rotation matrices (vectorized)
    R = np.zeros((len(quaternions), 3, 3))
    
    # First row
    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    
    # Second row
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - w*x)
    
    # Third row
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)
    
    return R


def gaussian_normal_estimation(vertices: np.ndarray,
                               scales: np.ndarray,
                               rotations: np.ndarray,
                               edges: np.ndarray,
                               y_up: bool = True) -> np.ndarray:
    """
    Estimate normals from Gaussian splat parameters using rotation and scale.
    
    For each Gaussian:
    1. Convert quaternion to rotation matrix R
    2. Find the column of R corresponding to the smallest scale component
    3. Average this direction vector over all k-NN neighbors
    4. Normalize the result
    
    Args:
        vertices: Point positions (N, 3)
        scales: Scale vectors (N, 3) representing (s1, s2, s3)
        rotations: Quaternions (N, 4) in [w, x, y, z] format
        edges: Edge connectivity (M, 2) defining k-NN neighborhoods
        y_up: If True, orient normals to point in +Y direction (default: True)
        
    Returns:
        Normals array (N, 3)
    """
    n_points = len(vertices)
    normals = np.zeros((n_points, 3))
    
    # Convert all quaternions to rotation matrices (vectorized)
    rotation_matrices = quaternion_to_rotation_matrix(rotations)
    
    # Find the smallest scale component for each Gaussian
    min_scale_indices = np.argmin(scales, axis=1)  # (N,) indices in [0, 1, 2]
    
    # Extract the normal vector (column corresponding to min scale) for each Gaussian
    # This is the direction of minimum extent - the "thin" direction of the ellipsoid
    gaussian_normals = np.zeros((n_points, 3))
    for i in range(n_points):
        col_idx = min_scale_indices[i]
        gaussian_normals[i] = rotation_matrices[i, :, col_idx]
    
    # Build adjacency list from edges
    adjacency = [[] for _ in range(n_points)]
    for edge in edges:
        i, j = edge[0], edge[1]
        adjacency[i].append(j)
        adjacency[j].append(i)
    
    # Average normals over k-NN neighborhoods
    for i in range(n_points):
        neighbors = adjacency[i]
        if len(neighbors) == 0:
            # Isolated point - use its own Gaussian normal
            normals[i] = gaussian_normals[i]
        else:
            # Average over neighbors (including self)
            neighbor_normals = gaussian_normals[neighbors]
            avg_normal = np.mean(np.vstack([gaussian_normals[i:i+1], neighbor_normals]), axis=0)
            
            # Normalize
            norm = np.linalg.norm(avg_normal)
            if norm > 1e-8:
                normals[i] = avg_normal / norm
            else:
                normals[i] = gaussian_normals[i]
    
    # Orient normals if requested
    if y_up:
        # Flip normals pointing down
        flip_mask = normals[:, 1] < 0
        normals[flip_mask] *= -1
    
    return normals


# ============================================================================
# k-NN BASED NORMAL ESTIMATION (BVH)
# ============================================================================

def estimate_normals_svd_knn(vertices: np.ndarray, 
                            knn_indices: np.ndarray,
                            knn_distances: np.ndarray,
                            k_neighbors: int = 15,
                            y_up: bool = True,
                            use_gpu: bool = True,
                            return_confidences: bool = False,
                            sigma: float = 0.5,
                            prune_factor: float = 2.0) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    ULTRA-FAST: Direct SVD normal estimation from k-NN indices (no adjacency conversion).
    
    This is the most optimized version - operates directly on k-NN results without
    any intermediate data structure conversions.
    
    Args:
        vertices: Point positions (N, 3)
        knn_indices: k-NN neighbor indices (N, k) from _torch_knn
        knn_distances: k-NN distances (N, k) from _torch_knn  
        k_neighbors: Number of neighbors per point
        y_up: If True, orient normals to point in +Y direction
        use_gpu: If True, use GPU acceleration with torch
        return_confidences: If True, return (normals, confidences) tuple
        sigma: Gaussian kernel sigma for confidence estimation
        prune_factor: Distance-based pruning threshold multiplier
        
    Returns:
        If return_confidences=False: Normals array (N, 3)
        If return_confidences=True: Tuple of (normals, confidences) arrays
    """
    import time
    
    start_total = time.time()
    n_points = len(vertices)
    normals = np.zeros((n_points, 3))
    confidences = np.zeros(n_points) if return_confidences else None
    
    # Determine device
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    
    # No adjacency building needed - direct from k-NN!
    start_adjacency = time.time()
    time_adjacency = time.time() - start_adjacency  # Will be ~0ms
    
    # Convert vertices to torch tensor on device
    start_gpu_transfer = time.time()
    vertices_torch = torch.from_numpy(vertices).float().to(device)
    knn_indices_torch = torch.from_numpy(knn_indices).long().to(device)
    knn_distances_torch = torch.from_numpy(knn_distances).float().to(device)
    time_gpu_transfer = time.time() - start_gpu_transfer
    
    # Apply distance-based pruning
    if prune_factor > 0:
        positive_distances = knn_distances[knn_distances > 0]
        if len(positive_distances) > 0:
            avg_radius = np.mean(positive_distances)
            prune_threshold = avg_radius * prune_factor
            valid_mask = torch.from_numpy(knn_distances <= prune_threshold).to(device)
            knn_indices_torch = knn_indices_torch * valid_mask.long()
            knn_distances_torch = knn_distances_torch * valid_mask.float()
        # If no positive distances, skip pruning
    
    # Ultra-vectorized batch SVD processing
    start_svd_loop = time.time()
    time_svd_computation = 0.0
    time_confidence_computation = 0.0
    
    # Process in batches for GPU memory management
    batch_size = min(1000, n_points)
    
    for batch_start in range(0, n_points, batch_size):
        batch_end = min(batch_start + batch_size, n_points)
        current_batch_size = batch_end - batch_start
        
        # Extract batch data directly from k-NN results
        batch_knn_indices = knn_indices_torch[batch_start:batch_end]  # (B, k)
        batch_knn_distances = knn_distances_torch[batch_start:batch_end]  # (B, k)
        batch_centers = vertices_torch[batch_start:batch_end]  # (B, 3)
        
        # Create valid neighbor mask (exclude invalid indices)
        valid_neighbors = (batch_knn_indices >= 0) & (batch_knn_distances > 0)
        
        start_svd = time.time()
        
        # Vectorized neighbor position gathering
        batch_knn_indices_flat = batch_knn_indices.flatten()
        valid_flat = batch_knn_indices_flat >= 0
        
        # Gather all neighbor positions at once
        neighbor_positions_flat = torch.zeros_like(vertices_torch[batch_knn_indices_flat])
        if valid_flat.any():
            neighbor_positions_flat[valid_flat] = vertices_torch[batch_knn_indices_flat[valid_flat]]
        
        # Reshape to batch format
        neighbor_positions = neighbor_positions_flat.view(current_batch_size, k_neighbors, 3)
        
        # Center the neighborhoods (subtract vertex position)
        batch_centers_expanded = batch_centers.unsqueeze(1).expand(-1, k_neighbors, -1)
        centered_neighborhoods = neighbor_positions - batch_centers_expanded
        
        # Apply valid neighbor mask
        centered_neighborhoods = centered_neighborhoods * valid_neighbors.unsqueeze(-1).float()
        
        try:
            # Batch SVD computation - MASSIVE PARALLELIZATION!
            U, S, Vh = torch.linalg.svd(centered_neighborhoods, full_matrices=False)
            
            # Extract normals (last column of Vh for each batch item)
            batch_normals = Vh[:, -1, :].cpu().numpy()  # (B, 3)
            
            # Apply orientation constraints
            if y_up:
                flip_mask = batch_normals[:, 1] < 0
                batch_normals[flip_mask] = -batch_normals[flip_mask]
            else:
                flip_mask = batch_normals[:, 2] < 0
                batch_normals[flip_mask] = -batch_normals[flip_mask]
            
            # Normalize normals
            norms = np.linalg.norm(batch_normals, axis=1, keepdims=True)
            batch_normals = batch_normals / (norms + 1e-10)
            
            time_svd_computation += time.time() - start_svd
            
            # VECTORIZED confidence computation
            if return_confidences:
                start_confidence = time.time()
                
                # Protect against division by zero
                if sigma <= 0:
                    print(f"[WARN] Invalid sigma value: {sigma}, using default weights")
                    batch_confidence_sums = torch.sum(valid_neighbors.float(), dim=1).cpu().numpy()
                else:
                    # All operations on GPU - no loops!
                    sigma_torch = torch.tensor(sigma, device=device)
                    batch_weights = torch.exp(-batch_knn_distances**2 / (2 * sigma_torch**2))
                    batch_weights = batch_weights * valid_neighbors.float()
                    batch_confidence_sums = torch.sum(batch_weights, dim=1).cpu().numpy()
                
                confidences[batch_start:batch_end] = batch_confidence_sums
                
                time_confidence_computation += time.time() - start_confidence
            
            # Assign computed normals
            normals[batch_start:batch_end] = batch_normals
            
        except Exception as e:
            print(f"[WARN] Batch SVD failed: {e}, falling back to individual processing")
            # Fallback to individual processing for this batch - same as adjacency method
            for i in range(current_batch_size):
                idx = batch_start + i
                try:
                    # Get valid neighbors for this point
                    point_knn_indices = knn_indices[idx]
                    point_knn_distances = knn_distances[idx]
                    valid_mask = (point_knn_indices >= 0) & (point_knn_distances > 0)
                    
                    if not valid_mask.any():
                        # No valid neighbors
                        normals[idx] = np.array([0.0, 0.0, 0.0])
                        if return_confidences:
                            confidences[idx] = 0.0
                        continue
                    
                    # Get valid neighbor positions
                    valid_indices = point_knn_indices[valid_mask]
                    valid_distances = point_knn_distances[valid_mask]
                    
                    # Center the neighborhood
                    neighbors_torch = vertices_torch[valid_indices] - vertices_torch[idx]
                    
                    # Individual SVD
                    U, S, Vh = torch.linalg.svd(neighbors_torch, full_matrices=False)
                    normal = Vh[-1, :].cpu().numpy()
                    
                    # Apply orientation constraints
                    if y_up and normal[1] < 0:
                        normal = -normal
                    elif not y_up and normal[2] < 0:
                        normal = -normal
                    
                    # Normalize
                    normals[idx] = normal / (np.linalg.norm(normal) + 1e-10)
                    
                    # Individual confidence computation
                    if return_confidences:
                        if sigma <= 0:
                            confidences[idx] = len(valid_indices)  # Just count neighbors
                        else:
                            weights = np.exp(-valid_distances**2 / (2 * sigma**2))
                            confidences[idx] = np.sum(weights)
                    
                except Exception as inner_e:
                    print(f"[WARN] Individual SVD failed for point {idx}: {inner_e}")
                    normals[idx] = np.array([0.0, 0.0, 0.0])
                    if return_confidences:
                        confidences[idx] = 0.0
    
    time_svd_loop = time.time() - start_svd_loop
    time_total = time.time() - start_total
    
    # Print timing breakdown
    print(f"\n⚡ ULTRA-FAST NORMAL ESTIMATION TIMING:")
    if time_total > 1e-10:
        # print(f"  Adjacency list build:    {time_adjacency*1000:>8.2f}ms ({time_adjacency/time_total*100:>5.1f}%)")
        print(f"  GPU tensor transfer:     {time_gpu_transfer*1000:>8.2f}ms ({time_gpu_transfer/time_total*100:>5.1f}%)")
        print(f"  SVD computation:         {time_svd_computation*1000:>8.2f}ms ({time_svd_computation/time_total*100:>5.1f}%)")
        if return_confidences and time_confidence_computation > 0:
            print(f"  Confidence computation:  {time_confidence_computation*1000:>8.2f}ms ({time_confidence_computation/time_total*100:>5.1f}%)")
        other_time = max(0, time_total - time_gpu_transfer - time_svd_computation - time_confidence_computation)
        print(f"  Other operations:        {other_time*1000:>8.2f}ms ({other_time/time_total*100:>5.1f}%)")
        print(f"  {'-'*50}")
        print(f"  Total:                   {time_total*1000:>8.2f}ms (100.0%)")
    else:
        print(f"  Total time too small for breakdown: {time_total*1000:.4f}ms")
    
    # Normalize confidences to [0, 1] range if requested
    if return_confidences and confidences is not None:
        if confidences.max() > confidences.min():
            confidences = (confidences - confidences.min()) / (confidences.max() - confidences.min())
            print(f"[INFO] Normalized confidences to [0, 1] range")
        else:
            print(f"[INFO] All confidences equal ({confidences[0]:.6f}), no normalization needed")
    
    if return_confidences:
        return normals, confidences
    else:
        return normals


def estimate_normals_svd_simple(vertices: np.ndarray, 
                               adjacency_lists: list[list[int]], 
                               k_neighbors: int = 15,
                               y_up: bool = True,
                               use_gpu: bool = True,
                               return_confidences: bool = False,
                               sigma: float = 0.5) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Estimate normals using SVD on neighborhoods defined by adjacency lists.
    Direct adjacency input eliminates redundant edge-to-adjacency reconstruction.
    
    Optimized with vectorized batch SVD on GPU for massive speedup.
    
    Args:
        vertices: Point positions (N, 3)
        adjacency_lists: Pre-computed neighbor lists, adjacency_lists[i] = neighbors of vertex i
        k_neighbors: Expected number of neighbors (unused, kept for API compatibility)
        y_up: If True, orient normals to point in +Y direction (default: True)
        use_gpu: If True, use GPU acceleration with torch (default: True)
        return_confidences: If True, return (normals, confidences) tuple (default: False)
        sigma: Gaussian kernel sigma for confidence estimation (default: 0.1)
        
    Returns:
        If return_confidences=False: Normals array (N, 3)
        If return_confidences=True: Tuple of (normals, confidences) arrays
    """
    import time
    
    start_total = time.time()
    n_points = len(vertices)
    normals = np.zeros((n_points, 3))
    confidences = np.zeros(n_points) if return_confidences else None
    # Remove the default_normal initialization - let both methods use [0,0,0] for consistency
    
    # Determine device
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    
    # Use pre-computed adjacency lists directly (no reconstruction needed!)
    start_adjacency = time.time()
    adjacency = adjacency_lists  # Direct assignment - massive time saving!
    time_adjacency = time.time() - start_adjacency
    
    # Convert vertices to torch tensor on device
    start_gpu_transfer = time.time()
    vertices_torch = torch.from_numpy(vertices).float().to(device)
    time_gpu_transfer = time.time() - start_gpu_transfer
    
    # Vectorized batch SVD processing for massive speedup
    start_svd_loop = time.time()
    time_svd_computation = 0.0
    time_confidence_computation = 0.0
    
    # Find maximum neighborhood size for batch tensor allocation
    max_neighbors = max(len(adj) for adj in adjacency) if n_points > 0 else 0
    
    # Process in batches to manage GPU memory
    batch_size = min(1000, n_points)  # Adjust based on GPU memory
    
    for batch_start in range(0, n_points, batch_size):
        batch_end = min(batch_start + batch_size, n_points)
        current_batch_size = batch_end - batch_start
        
        # Prepare batch data
        batch_neighborhoods = []
        batch_valid_mask = []
        batch_neighbor_counts = []
        batch_indices = []
        
        for i in range(batch_start, batch_end):
            neighbor_indices = adjacency[i]
            
            # Remove the < 3 filter - use all available neighbors like k-NN method
            if len(neighbor_indices) == 0:
                # Only skip if NO neighbors at all
                batch_valid_mask.append(False)
                batch_neighborhoods.append(torch.zeros((max_neighbors, 3), device=device))  # Dummy data with correct size
                batch_neighbor_counts.append(0)
                batch_indices.append([])
                continue
            
            # Get neighbor positions relative to current vertex
            neighbors_torch = vertices_torch[neighbor_indices] - vertices_torch[i]
            
            # Ensure all neighborhoods have the same size for batching
            current_size = neighbors_torch.shape[0]
            if current_size < max_neighbors:
                # Pad with zeros
                padding = torch.zeros((max_neighbors - current_size, 3), device=device)
                neighbors_torch = torch.cat([neighbors_torch, padding], dim=0)
            elif current_size > max_neighbors:
                # Truncate to max size
                neighbors_torch = neighbors_torch[:max_neighbors]
            
            batch_neighborhoods.append(neighbors_torch)
            batch_valid_mask.append(True)
            batch_neighbor_counts.append(len(neighbor_indices))
            batch_indices.append(neighbor_indices)
        
        if not any(batch_valid_mask):
            # No valid neighborhoods in this batch
            for i, is_valid in enumerate(batch_valid_mask):
                idx = batch_start + i
                normals[idx] = np.array([0.0, 0.0, 0.0])
                if return_confidences:
                    confidences[idx] = 0.0
            continue
        
        # Stack into batch tensor: (batch_size, max_neighbors, 3)
        start_svd = time.time()
        batch_tensor = torch.stack(batch_neighborhoods, dim=0)  # (B, N, 3)
        
        try:
            # Batch SVD computation - this is the key optimization!
            U, S, Vh = torch.linalg.svd(batch_tensor, full_matrices=False)  # (B, 3, 3), (B, 3), (B, 3, 3)
            
            # Extract normals (last column of Vh for each batch item)
            batch_normals = Vh[:, -1, :].cpu().numpy()  # (B, 3)
            
            # Apply orientation constraints
            if y_up:
                flip_mask = batch_normals[:, 1] < 0
                batch_normals[flip_mask] = -batch_normals[flip_mask]
            else:
                flip_mask = batch_normals[:, 2] < 0
                batch_normals[flip_mask] = -batch_normals[flip_mask]
            
            # Normalize normals
            norms = np.linalg.norm(batch_normals, axis=1, keepdims=True)
            batch_normals = batch_normals / (norms + 1e-10)
            
            time_svd_computation += time.time() - start_svd
            
            # Vectorized batch confidence computation (GPU accelerated!)
            if return_confidences:
                start_confidence = time.time()
                
                # Prepare batch confidence computation on GPU
                batch_confidences = np.zeros(current_batch_size)
                
                # Create padded tensor for all neighborhoods in batch: (B, max_neighbors)
                max_batch_neighbors = max(len(indices) for indices in batch_indices) if batch_indices else 0
                
                if max_batch_neighbors > 0:
                    # Build batch tensor for vectorized distance computation
                    batch_positions = torch.zeros((current_batch_size, max_batch_neighbors, 3), device=device)
                    batch_centers = torch.zeros((current_batch_size, 3), device=device)
                    batch_masks = torch.zeros((current_batch_size, max_batch_neighbors), device=device, dtype=torch.bool)
                    
                    for i, (is_valid, neighbor_indices) in enumerate(zip(batch_valid_mask, batch_indices)):
                        if is_valid and len(neighbor_indices) > 0:
                            idx = batch_start + i
                            # Fill neighbor positions (padded)
                            n_neighbors = len(neighbor_indices)
                            batch_positions[i, :n_neighbors] = vertices_torch[neighbor_indices]
                            batch_centers[i] = vertices_torch[idx]
                            batch_masks[i, :n_neighbors] = True
                    
                    # Vectorized distance computation for entire batch
                    batch_centers_expanded = batch_centers.unsqueeze(1).expand(-1, max_batch_neighbors, -1)
                    batch_distances = torch.norm(batch_positions - batch_centers_expanded, dim=2)  # (B, max_neighbors)
                    
                    # Apply Gaussian kernel vectorized
                    if sigma <= 0:
                        print(f"[WARN] Invalid sigma value: {sigma}, using default weights")
                        batch_weights = batch_masks.float()  # Just count neighbors
                    else:
                        sigma_torch = torch.tensor(sigma, device=device)
                        batch_weights = torch.exp(-batch_distances**2 / (2 * sigma_torch**2))  # (B, max_neighbors)
                        batch_weights = batch_weights * batch_masks.float()
                    batch_confidence_sums = torch.sum(batch_weights, dim=1).cpu().numpy()  # (B,)
                    
                    # Assign to final confidences array
                    for i, is_valid in enumerate(batch_valid_mask):
                        idx = batch_start + i
                        if is_valid:
                            confidences[idx] = batch_confidence_sums[i]
                        else:
                            confidences[idx] = 0.0
                
                time_confidence_computation += time.time() - start_confidence
            
            # Assign computed normals to valid vertices
            for i, is_valid in enumerate(batch_valid_mask):
                idx = batch_start + i
                if is_valid:
                    normals[idx] = batch_normals[i]
                else:
                    normals[idx] = np.array([0.0, 0.0, 0.0])
                    if return_confidences:
                        confidences[idx] = 0.0
                        
        except Exception as e:
            print(f"[WARN] Batch SVD failed: {e}, falling back to individual processing")
            # Fallback to individual processing for this batch
            for i, (is_valid, neighbor_indices) in enumerate(zip(batch_valid_mask, batch_indices)):
                idx = batch_start + i
                if not is_valid:
                    normals[idx] = np.array([0.0, 0.0, 0.0])
                    if return_confidences:
                        confidences[idx] = 0.0
                    continue
                
                try:
                    neighbors_torch = vertices_torch[neighbor_indices] - vertices_torch[idx]
                    U, S, Vh = torch.linalg.svd(neighbors_torch, full_matrices=False)
                    normal = Vh[-1, :].cpu().numpy()
                    
                    if y_up and normal[1] < 0:
                        normal = -normal
                    elif not y_up and normal[2] < 0:
                        normal = -normal
                    
                    normals[idx] = normal / np.linalg.norm(normal)
                    
                    if return_confidences:
                        neighbor_positions = vertices_torch[neighbor_indices]
                        distances = torch.norm(neighbor_positions - vertices_torch[idx], dim=1).cpu().numpy()
                        if sigma <= 0:
                            weights = np.ones_like(distances)  # Default weights
                        else:
                            weights = np.exp(-distances**2 / (2 * sigma**2))
                        confidences[idx] = np.sum(weights)
                        
                except:
                    normals[idx] = np.array([0.0, 0.0, 0.0])
                    if return_confidences:
                        confidences[idx] = 0.0
    
    time_svd_loop = time.time() - start_svd_loop
    time_total = time.time() - start_total
    
    # Print timing breakdown (protect against division by zero)
    print(f"\n⏱️  NORMAL ESTIMATION TIMING BREAKDOWN:")
    if time_total > 0:
        print(f"  Adjacency list build:    {time_adjacency*1000:>8.2f}ms ({time_adjacency/time_total*100:>5.1f}%)")
        print(f"  GPU tensor transfer:     {time_gpu_transfer*1000:>8.2f}ms ({time_gpu_transfer/time_total*100:>5.1f}%)")
        print(f"  SVD computation:         {time_svd_computation*1000:>8.2f}ms ({time_svd_computation/time_total*100:>5.1f}%)")
        if return_confidences:
            print(f"  Confidence computation:  {time_confidence_computation*1000:>8.2f}ms ({time_confidence_computation/time_total*100:>5.1f}%)")
        other_time = time_total - time_adjacency - time_gpu_transfer - time_svd_computation - time_confidence_computation
        print(f"  Other operations:        {other_time*1000:>8.2f}ms ({other_time/time_total*100:>5.1f}%)")
        print(f"  {'-'*50}")
        print(f"  Total:                   {time_total*1000:>8.2f}ms (100.0%)")
    else:
        print(f"  Total time too small to measure accurately: {time_total*1000:.4f}ms")
    
    if return_confidences:
        return normals, confidences
    else:
        return normals


def geobrush_smoothing(vertices: np.ndarray,
                      normals: np.ndarray,
                      edges: np.ndarray,
                      iterations: int = 1,
                      epsilon: float = 1e-6,
                      use_sparse: bool = False) -> np.ndarray:
    """
    Smooth normals using distance-weighted averaging over neighborhoods.
    
    For each vertex v, compute smoothed normal as:
        n(v) = normalize( sum( n(w) / (||w - v||^2 + epsilon) ) for all neighbors w )
    
    This is a bilateral-style smoothing that preserves geometric features while
    reducing noise. The epsilon parameter prevents division by zero for coincident points.
    
    Two implementations available:
    1. Vectorized loop (default) - Good balance of speed and memory
    2. Sparse matrix (use_sparse=True) - Fastest for many iterations, higher memory
    
    Args:
        vertices: Point positions (N, 3)
        normals: Current normal vectors (N, 3)
        edges: Edge connectivity (M, 2) defining neighborhoods
        iterations: Number of smoothing iterations (default: 1)
                   More iterations = more smoothing
        epsilon: Small constant to prevent division by zero (default: 1e-6)
        use_sparse: Use sparse matrix multiplication (faster for iterations >= 3)
        
    Returns:
        Smoothed normals array (N, 3)
    """
    if use_sparse:
        return _geobrush_smoothing_sparse(vertices, normals, edges, iterations, epsilon)
    else:
        return _geobrush_smoothing_vectorized(vertices, normals, edges, iterations, epsilon)


def _geobrush_smoothing_vectorized(vertices: np.ndarray,
                                   normals: np.ndarray,
                                   edges: np.ndarray,
                                   iterations: int,
                                   epsilon: float) -> np.ndarray:
    """Vectorized implementation (good for 1-5 iterations)."""
    
    n_points = len(vertices)
    smoothed_normals = np.zeros_like(normals)
    
    # Pre-compute edge properties (vectorized)
    edge_i = edges[:, 0]
    edge_j = edges[:, 1]
    
    # Compute all edge vectors and squared distances once
    edge_vectors = vertices[edge_j] - vertices[edge_i]
    edge_dist_sq = np.sum(edge_vectors**2, axis=1)
    
    # Compute weights for all edges: 1 / (d^2 + epsilon)
    edge_weights = 1.0 / (edge_dist_sq + epsilon)
    
    # Build adjacency structure for fast access
    # adjacency[i] = list of (neighbor_idx, edge_idx) tuples
    adjacency = [[] for _ in range(n_points)]
    for edge_idx, (i, j) in enumerate(edges):
        adjacency[i].append((j, edge_idx))
        adjacency[j].append((i, edge_idx))
    
    # Perform iterative smoothing
    for iter_idx in range(iterations):
        new_normals = np.zeros_like(smoothed_normals)
        
        # Process all vertices (still need loop, but inner operations are vectorized)
        for i in range(n_points):
            neighbor_data = adjacency[i]
            
            if len(neighbor_data) == 0:
                new_normals[i] = smoothed_normals[i]
                continue
            
            # Extract neighbor indices and edge indices
            neighbor_indices = [n_idx for n_idx, _ in neighbor_data]
            edge_indices = [e_idx for _, e_idx in neighbor_data]
            
            # Vectorized: get all neighbor normals at once
            neighbor_normals = smoothed_normals[neighbor_indices]  # (K, 3)
            
            # Vectorized: get pre-computed weights
            weights = edge_weights[edge_indices]  # (K,)
            
            # Vectorized: weighted sum using broadcasting
            weighted_sum = np.sum(weights[:, np.newaxis] * neighbor_normals, axis=0)
            total_weight = np.sum(weights)
            
            # Normalize the weighted sum
            if total_weight > 0:
                new_normals[i] = weighted_sum / total_weight
            else:
                new_normals[i] = smoothed_normals[i]
        
        # Vectorized normalization of all normals at once
        norms = np.linalg.norm(new_normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Prevent division by zero
        smoothed_normals = new_normals / norms
    
    return smoothed_normals


def _geobrush_smoothing_sparse(vertices: np.ndarray,
                               normals: np.ndarray,
                               edges: np.ndarray,
                               iterations: int,
                               epsilon: float) -> np.ndarray:
    """Sparse matrix implementation (best for many iterations >= 5)."""
    
    n_points = len(vertices)
    
    # Compute all edge weights
    edge_i = edges[:, 0]
    edge_j = edges[:, 1]
    edge_vectors = vertices[edge_j] - vertices[edge_i]
    edge_dist_sq = np.sum(edge_vectors**2, axis=1)
    edge_weights = 1.0 / (edge_dist_sq + epsilon)
    
    # Build sparse weight matrix W where W[i,j] = weight from j to i
    # Need to add both directions for undirected edges
    row_indices = np.concatenate([edge_i, edge_j])
    col_indices = np.concatenate([edge_j, edge_i])
    weights = np.concatenate([edge_weights, edge_weights])
    
    # Create sparse matrix
    W = csr_matrix((weights, (row_indices, col_indices)), shape=(n_points, n_points))
    
    # Row-normalize: each row sums to 1
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    D_inv = csr_matrix((1.0 / row_sums, (np.arange(n_points), np.arange(n_points))), 
                       shape=(n_points, n_points))
    W_normalized = D_inv @ W
    
    # Apply smoothing iterations using matrix multiplication
    smoothed_normals = normals.copy()
    for iter_idx in range(iterations):
        # Matrix-vector multiply for each component (fully vectorized!)
        smoothed_normals = W_normalized @ smoothed_normals
        
        # Normalize to unit length
        norms = np.linalg.norm(smoothed_normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        smoothed_normals = smoothed_normals / norms
    
    return smoothed_normals


def _torch_knn(vertices: np.ndarray, k: int, use_gpu: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Ultra-fast PyTorch k-NN implementation using full GPU vectorization.
    
    Args:
        vertices: Point positions (N, 3)
        k: Number of nearest neighbors
        use_gpu: Use GPU acceleration if available
        
    Returns:
        tuple: (neighbor_indices, distances) both (N, k) arrays
    """
    import torch
    
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    vertices_torch = torch.from_numpy(vertices).float().to(device)
    
    n_points = len(vertices)
    
    # Single GPU operation - compute all pairwise distances at once
    if n_points > 10000:
        # Use chunked approach for large datasets
        return _torch_knn_chunked(vertices_torch, k, device)
    
    # Full vectorized distance matrix computation (N x N)
    distances = torch.cdist(vertices_torch, vertices_torch, p=2)  # (N, N)
    
    # Get k+1 nearest neighbors (including self) then exclude self
    k_actual = min(k + 1, n_points)
    knn_distances, knn_indices = torch.topk(distances, k_actual, largest=False, dim=1)
    
    # Remove self-neighbors (first column is distance 0 to self)
    neighbor_indices = knn_indices[:, 1:k+1].cpu().numpy()  # (N, k)
    neighbor_distances = knn_distances[:, 1:k+1].cpu().numpy()  # (N, k)
    
    return neighbor_indices, neighbor_distances


def _torch_knn_chunked(vertices_torch: torch.Tensor, k: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Chunked k-NN for large datasets to avoid GPU memory issues."""
    n_points = len(vertices_torch)
    chunk_size = min(2000, n_points)  # Adjust based on GPU memory
    
    all_indices = []
    all_distances = []
    
    for start_idx in range(0, n_points, chunk_size):
        end_idx = min(start_idx + chunk_size, n_points)
        chunk_vertices = vertices_torch[start_idx:end_idx]
        
        # Distance from chunk to all points
        chunk_distances = torch.cdist(chunk_vertices, vertices_torch, p=2)
        
        # Get k+1 nearest (including self) then exclude self
        k_actual = min(k + 1, n_points)
        chunk_knn_dist, chunk_knn_idx = torch.topk(chunk_distances, k_actual, largest=False, dim=1)
        
        # Remove self-neighbors
        chunk_indices = chunk_knn_idx[:, 1:k+1].cpu().numpy()
        chunk_dists = chunk_knn_dist[:, 1:k+1].cpu().numpy()
        
        all_indices.append(chunk_indices)
        all_distances.append(chunk_dists)
    
    return np.vstack(all_indices), np.vstack(all_distances)


def build_knn_direct(vertices: np.ndarray, k: int = 15, use_gpu: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """
    ULTRA-FAST: Direct k-NN computation returning raw indices and distances.
    
    This is the fastest approach - no adjacency list conversion, no loops, pure GPU vectorization.
    Use this with estimate_normals_svd_knn() for maximum performance.
    
    Args:
        vertices: Point positions (N, 3)
        k: Number of nearest neighbors
        use_gpu: If True, use GPU acceleration with PyTorch
        
    Returns:
        tuple: (knn_indices, knn_distances, avg_radius)
            - knn_indices: Neighbor indices (N, k)
            - knn_distances: Neighbor distances (N, k)
            - avg_radius: Average neighbor distance
    """
    # Single GPU k-NN call - maximum efficiency
    knn_indices, knn_distances = _torch_knn(vertices, k, use_gpu)
    
    # Compute average radius for potential downstream use
    avg_radius = np.mean(knn_distances[knn_distances > 0])
    
    return knn_indices, knn_distances, avg_radius


def build_connectivity_vectorized(vertices: np.ndarray, k: int = 15, prune_factor: float = 2.0, use_gpu: bool = True) -> tuple[list[list[int]], float]:
    """
    Ultra-optimized vectorized k-NN connectivity with adjacency list output.
    
    Key optimizations:
    1. PyTorch GPU-accelerated k-NN for fast neighbor search
    2. Direct adjacency list construction (no edges intermediate)
    3. Vectorized distance-based pruning
    4. Eliminates adjacency reconstruction in downstream functions
    
    Args:
        vertices: Point positions (N, 3)
        k: Number of nearest neighbors to query
        prune_factor: Multiplier for pruning (keep neighbors < avg_radius * prune_factor)
        use_gpu: If True, use GPU acceleration with PyTorch (default: True)
        
    Returns:
        tuple: (adjacency_lists, avg_radius) where adjacency_lists[i] = list of neighbor indices for vertex i
    """
    n_points = len(vertices)
    
    # Use PyTorch GPU-accelerated k-NN
    try:
        # Get k-NN using PyTorch (GPU accelerated)
        indices, distances = _torch_knn(vertices, k, use_gpu)
        
        # Build adjacency lists directly from k-NN results
        adjacency = [[] for _ in range(n_points)]
        
        # Vectorized distance computation for all neighbors
        all_distances = []
        
        for i in range(n_points):
            valid_neighbors = indices[i][indices[i] >= 0]  # Filter padding
            neighbor_distances = distances[i][:len(valid_neighbors)]
            
            # Store neighbors and their distances
            for j, neighbor_idx in enumerate(valid_neighbors):
                if neighbor_idx != i:  # Skip self-neighbors
                    adjacency[i].append(int(neighbor_idx))
                    all_distances.append(neighbor_distances[j])
        
        # Compute average radius from all neighbor distances
        if len(all_distances) > 0:
            avg_radius = np.mean(all_distances)
        else:
            print("[WARN] No valid neighbors found, using default radius 0.1")
            avg_radius = 0.1
            
    except Exception as e:
        print(f"[WARN] PyTorch KNN failed ({e}), falling back to CPU method")
        # Fallback to edge-based method then convert to adjacency
        edges_initial, edge_dists = _fallback_knn_edges(vertices, k)
        avg_radius = np.mean(edge_dists) if len(edge_dists) > 0 else 0.1
        
        # Convert edges to adjacency lists
        adjacency = [[] for _ in range(n_points)]
        for edge in edges_initial:
            i, j = edge[0], edge[1]
            adjacency[i].append(j)
            adjacency[j].append(i)
    
    # Apply distance-based pruning
    if prune_factor > 0:
        prune_threshold = avg_radius * prune_factor
        
        # Convert to PyTorch for GPU-accelerated pruning
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        vertices_torch = torch.from_numpy(vertices).float().to(device)
        
        # Prune adjacency lists based on distances
        for i in range(n_points):
            if len(adjacency[i]) == 0:
                continue
                
            neighbor_indices = adjacency[i]
            current_pos = vertices_torch[i:i+1]  # (1, 3)
            neighbor_pos = vertices_torch[neighbor_indices]  # (k, 3)
            
            # Compute distances on GPU
            neighbor_distances = torch.norm(neighbor_pos - current_pos, dim=1).cpu().numpy()
            
            # Keep only neighbors within pruning threshold
            keep_mask = neighbor_distances <= prune_threshold
            adjacency[i] = [neighbor_indices[j] for j, keep in enumerate(keep_mask) if keep]
    
    return adjacency, avg_radius


def _fallback_knn_edges(vertices: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Helper function for CPU fallback KNN edge generation."""
    if dem is None:
        raise ImportError("discrete_exp_map module is required for k-NN connectivity")
    if not HAS_PYBVH:
        raise ImportError("pybvh is required for BVH-based connectivity")
    
    edges_initial = dem.generate_knn_edges(vertices, k)
    edge_i = edges_initial[:, 0]
    edge_j = edges_initial[:, 1]
    edge_dists = np.linalg.norm(vertices[edge_i] - vertices[edge_j], axis=1)
    
    return edges_initial, edge_dists


def build_connectivity(vertices: np.ndarray, k: int = 15, prune_factor: float = 2.0, use_gpu: bool = True) -> tuple[list[list[int]], float]:
    """
    Build k-NN connectivity returning adjacency lists directly.
    
    This is the main connectivity function optimized for downstream usage:
    1. GPU-accelerated k-nearest neighbor search  
    2. Direct adjacency list construction (no intermediate edges)
    3. Adaptive pruning based on local density
    4. Eliminates adjacency reconstruction in normal estimation
    
    For each point:
    1. Find k nearest neighbors using GPU-accelerated PyTorch k-NN
    2. Calculate average distance to neighbors 
    3. Compute global average radius across all points
    4. Prune neighbors where distance > avg_radius * prune_factor
    
    Args:
        vertices: Point positions (N, 3)
        k: Number of nearest neighbors to query
        prune_factor: Multiplier for pruning (keep neighbors < avg_radius * prune_factor)
        use_gpu: If True, use GPU acceleration with PyTorch (default: True)
        
    Returns:
        tuple: (adjacency_lists, avg_radius)
            - adjacency_lists: List of neighbor lists, adjacency_lists[i] = neighbors of vertex i
            - avg_radius: Computed average neighborhood radius
    """
    # Use the optimized vectorized implementation
    return build_connectivity_vectorized(vertices, k, prune_factor, use_gpu)


def compute_exponential_map(vertices: np.ndarray,
                           edges: np.ndarray,
                           normals: np.ndarray,
                           root_vertex: Optional[int] = None,
                           local_coordinates: bool = True) -> np.ndarray:
    """
    Compute discrete exponential map on processed scene.
    
    Args:
        vertices: Point positions (N, 3)
        edges: Edge connectivity (M, 2)
        normals: Normal vectors (N, 3)
        root_vertex: Source vertex index (None = centroid)
        local_coordinates: Use local coordinate system
        
    Returns:
        Exponential map coordinates (N, 2)
    """
    if dem is None:
        raise ImportError("discrete_exp_map module is required for exponential map computation")
    
    if root_vertex is None:
        # Find vertex closest to centroid
        centroid = np.mean(vertices, axis=0)
        root_vertex = np.argmin(np.linalg.norm(vertices - centroid, axis=1))
    
    print(f"Computing exponential map from vertex {root_vertex}")
    
    exp_map_start = time.time()
    exp_map = dem.discrete_exp_map(
        vertices, edges, normals, root_vertex, 
        add_locally=local_coordinates
    )
    exp_map_time = time.time() - exp_map_start
    
    print(f"Exponential map computed in {exp_map_time:.2f}s")
    
    return exp_map


def adjacency_to_edges(adjacency_lists: list[list[int]]) -> np.ndarray:
    """
    Convert adjacency lists to edge array for backward compatibility.
    
    Args:
        adjacency_lists: List of neighbor lists, adjacency_lists[i] = neighbors of vertex i
        
    Returns:
        edges: Edge array (M, 2) with undirected edges (i < j)
    """
    edges = []
    for i, neighbors in enumerate(adjacency_lists):
        for j in neighbors:
            # Only add edge once (i < j) to avoid duplicates
            if i < j:
                edges.append([i, j])
    
    return np.array(edges, dtype=np.int32) if edges else np.empty((0, 2), dtype=np.int32)


def compute_discrete_exponential_map(vertices: np.ndarray,
                                   edges: np.ndarray,
                                   normals: np.ndarray,
                                   root_vertex: Optional[int] = None,
                                   local_coordinates: bool = True) -> np.ndarray:
    """
    Compute discrete exponential map using the discrete_exp_map implementation.
    
    This function serves as a wrapper around the discrete_exp_map module, automatically
    selecting the center point of the scene as the root vertex if not specified.
    
    Args:
        vertices: Point positions (N, 3)
        edges: Edge connectivity (M, 2)
        normals: Normal vectors (N, 3)
        root_vertex: Root vertex index. If None, uses scene center (default: None)
        local_coordinates: Use local coordinate system (default: True)
        
    Returns:
        exp_map: 2D exponential map coordinates (N, 2)
    """
    if dem is None:
        raise ImportError("discrete_exp_map module not available. Cannot compute exponential map.")
    
    # Find root vertex as scene center if not specified
    if root_vertex is None:
        # Find vertex closest to scene centroid
        centroid = np.mean(vertices, axis=0)
        distances_to_centroid = np.linalg.norm(vertices - centroid, axis=1)
        root_vertex = np.argmin(distances_to_centroid)
        print(f"[INFO] Auto-selected root vertex {root_vertex} (closest to scene center)")
    
    print(f"[INFO] Computing discrete exponential map from root vertex {root_vertex}...")
    exp_map_start = time.time()
    
    # Call the discrete exponential map function
    exp_map = dem.discrete_exp_map(
        V=vertices,
        E=edges, 
        N=normals,
        root_idx=root_vertex,
        add_locally=local_coordinates
    )
    
    exp_map_time = time.time() - exp_map_start
    print(f"[INFO] Discrete exponential map computed in {exp_map_time:.2f}s")
    
    return exp_map


# ============================================================================
# COMMON UTILITIES
# ============================================================================

def save_ply_with_normals(path: str, points: np.ndarray, normals: np.ndarray, 
                          confidences: np.ndarray = None, gaussian_model: GaussianModel = None):
    """
    Save point cloud with normals and optionally confidences to PLY file.
    Optionally includes Gaussian attributes if model is provided.
    
    Args:
        path: Output PLY file path
        points: Point positions (N, 3)
        normals: Normal vectors (N, 3)
        confidences: Optional confidence values (N,)
        gaussian_model: Optional GaussianModel to include attributes
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


def save_to_ply(vertices: np.ndarray, 
                normals: np.ndarray, 
                edges: np.ndarray,
                output_path: str,
                confidences: Optional[np.ndarray] = None,
                exp_map: Optional[np.ndarray] = None):
    """
    Save processed scene data to PLY format with vertices, normals, and edges.
    
    Args:
        vertices: Vertex positions (N, 3)
        normals: Vertex normals (N, 3)  
        edges: Edge connectivity (M, 2)
        output_path: Output PLY file path
        confidences: Optional confidence values (N,) for each vertex
        exp_map: Optional exponential map coordinates (N, 2) for each vertex
    """
    
    n_vertices = len(vertices)
    n_edges = len(edges)
    
    # Build header dynamically based on optional data
    properties = [
        "property float x",
        "property float y", 
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz"
    ]
    
    if confidences is not None:
        properties.append("property float confidence")
    
    if exp_map is not None:
        properties.extend([
            "property float exp_u",
            "property float exp_v"
        ])
    
    header = f"""ply
format ascii 1.0
element vertex {n_vertices}
{chr(10).join(properties)}
element edge {n_edges}
property int vertex1
property int vertex2
end_header
"""
    
    with open(output_path, 'w') as f:
        f.write(header)
        
        # Write vertices with normals and optional data
        for i in range(n_vertices):
            v = vertices[i]
            n = normals[i]
            line = f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}"
            
            if confidences is not None:
                conf = confidences[i]
                line += f" {conf:.6f}"
            
            if exp_map is not None:
                exp_u, exp_v = exp_map[i]
                line += f" {exp_u:.6f} {exp_v:.6f}"
            
            f.write(line + "\n")
        
        # Write edges
        for edge in edges:
            f.write(f"{edge[0]} {edge[1]}\n")


def load_processed_scene(npz_path: str, model_path: Optional[str] = None) -> dict:
    """
    Load previously processed scene from NPZ file.
    
    Args:
        npz_path: Path to NPZ file
        model_path: Optional path to Gaussian model PLY file
        
    Returns:
        Dictionary with scene data
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Load Gaussian model if path provided
    gaussian_model = None
    if model_path and os.path.exists(model_path):
        try:
            _, gaussian_model = load_gaussian_model(model_path)
        except Exception as e:
            print(f"Warning: Could not load Gaussian model from {model_path}: {e}")
    
    result = {
        'vertices': data['vertices'],
        'gaussian_model': gaussian_model,
        'normals': data['normals'],
        'edges': data['edges'],
        'stats': {k: v.item() if hasattr(v, 'item') else v for k, v in data.items() 
                 if k not in ['vertices', 'normals', 'edges', 'confidences']}
    }
    
    # Include confidence data if available
    if 'confidences' in data:
        result['confidences'] = data['confidences']
    
    return result
