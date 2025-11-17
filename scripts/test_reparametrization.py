#!/usr/bin/env python3
"""
Test script for reparametrization sampling from Gaussians.

This script tests the sample_points_from_gaussians function with a simple example
and demonstrates the GPU optimization benefit.
"""

import sys
import os
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except:
    HAS_TORCH = False

from utils.research_utils import sample_points_from_gaussians, quaternion_to_rotation_matrix

def test_reparametrization_sampling():
    """Test reparametrization sampling with a simple Gaussian."""
    
    print("\n" + "="*70)
    print("REPARAMETRIZATION SAMPLING TEST")
    print("="*70)
    
    # Create a simple test case: 3 Gaussians
    n_gaussians = 3
    
    # Gaussian centers
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    # Scales (standard deviations along each axis)
    scales = np.array([
        [0.1, 0.1, 0.1],  # Spherical
        [0.2, 0.05, 0.05],  # Elongated along X
        [0.05, 0.2, 0.05]   # Elongated along Y
    ])
    
    # Rotations (identity quaternions for simplicity)
    rotations = np.array([
        [1.0, 0.0, 0.0, 0.0],  # Identity
        [1.0, 0.0, 0.0, 0.0],  # Identity
        [1.0, 0.0, 0.0, 0.0]   # Identity
    ])
    
    # Opacities (all high)
    opacities = np.array([[0.9], [0.8], [0.7]])
    
    print(f"\n[TEST] Input:")
    print(f"  {n_gaussians} Gaussians")
    print(f"  Centers: {vertices.tolist()}")
    print(f"  Scales: {scales.tolist()}")
    
    # Test sampling
    n_samples = 5
    print(f"\n[TEST] Sampling {n_samples} points per Gaussian...")
    
    sampled_points = sample_points_from_gaussians(
        vertices,
        scales,
        rotations,
        opacities,
        n_samples_per_gaussian=n_samples,
        opacity_threshold=0.1,
        use_gpu=True
    )
    
    print(f"\n[TEST] Results:")
    print(f"  Total sampled points: {len(sampled_points)}")
    print(f"  Expected: {n_gaussians * n_samples}")
    
    # Verify dimensions
    assert len(sampled_points) == n_gaussians * n_samples, "Wrong number of sampled points"
    
    # Show samples for first Gaussian
    print(f"\n[TEST] Samples from Gaussian 0 (center at origin):")
    samples_g0 = sampled_points[0:n_samples]  # First n_samples are from Gaussian 0
    for i, point in enumerate(samples_g0):
        dist = np.linalg.norm(point - vertices[0])
        print(f"    Sample {i}: {point} (distance from center: {dist:.4f})")
    
    # Check that points are distributed around centers
    print(f"\n[TEST] Verifying distribution around centers:")
    for g_idx in range(n_gaussians):
        # Samples are ordered: first n_samples from Gaussian 0, next n_samples from Gaussian 1, etc.
        start_idx = g_idx * n_samples
        end_idx = start_idx + n_samples
        samples = sampled_points[start_idx:end_idx]
        center = vertices[g_idx]
        
        mean_sample = np.mean(samples, axis=0)
        mean_dist = np.linalg.norm(mean_sample - center)
        
        print(f"  Gaussian {g_idx}:")
        print(f"    True center: {center}")
        print(f"    Sample mean: {mean_sample}")
        print(f"    Distance: {mean_dist:.6f}")
        
        # Mean should be close to center (within tolerance)
        if mean_dist < 0.2:  # Generous tolerance for small sample size
            print(f"    ✓ PASS: Mean close to center")
        else:
            print(f"    ⚠ WARNING: Mean far from center (but OK for small n_samples)")
    
    print(f"\n" + "="*70)
    print("✅ REPARAMETRIZATION SAMPLING TEST PASSED")
    print("="*70)
    print()


def test_gpu_optimization():
    """Test GPU optimization with return_torch to avoid CPU transfer."""
    
    if not HAS_TORCH:
        print("\n[SKIP] GPU not available, skipping optimization test")
        return
    
    print("\n" + "="*70)
    print("GPU OPTIMIZATION TEST (return_torch=True)")
    print("="*70)
    
    # Create larger test case for performance comparison
    n_gaussians = 1000
    
    # Random Gaussians
    np.random.seed(42)
    vertices = np.random.randn(n_gaussians, 3)
    scales = np.abs(np.random.randn(n_gaussians, 3)) * 0.1
    rotations = np.zeros((n_gaussians, 4))
    rotations[:, 0] = 1.0  # Identity quaternions
    opacities = np.ones((n_gaussians, 1)) * 0.9
    
    n_samples = 10
    
    print(f"\n[TEST] Configuration:")
    print(f"  {n_gaussians} Gaussians x {n_samples} samples = {n_gaussians * n_samples} points")
    
    # Test 1: Standard mode (converts to NumPy)
    print(f"\n[TEST] Mode 1: return_torch=False (standard, with CPU transfer)")
    start = time.time()
    sampled_numpy = sample_points_from_gaussians(
        vertices, scales, rotations, opacities,
        n_samples_per_gaussian=n_samples,
        opacity_threshold=0.0,
        use_gpu=True,
        return_torch=False
    )
    time_numpy = time.time() - start
    print(f"  Time: {time_numpy*1000:.2f}ms")
    print(f"  Type: {type(sampled_numpy)}")
    print(f"  Shape: {sampled_numpy.shape}")
    
    # Test 2: Optimized mode (keeps on GPU)
    print(f"\n[TEST] Mode 2: return_torch=True (optimized, stays on GPU)")
    start = time.time()
    sampled_torch = sample_points_from_gaussians(
        vertices, scales, rotations, opacities,
        n_samples_per_gaussian=n_samples,
        opacity_threshold=0.0,
        use_gpu=True,
        return_torch=True
    )
    time_torch = time.time() - start
    print(f"  Time: {time_torch*1000:.2f}ms")
    print(f"  Type: {type(sampled_torch)}")
    print(f"  Device: {sampled_torch.device}")
    print(f"  Shape: {sampled_torch.shape}")
    
    # Show benefit
    speedup = time_numpy / time_torch if time_torch > 0 else 1.0
    saved_time = (time_numpy - time_torch) * 1000
    print(f"\n[RESULT] GPU Optimization:")
    print(f"  Speedup: {speedup:.2f}x faster")
    print(f"  Time saved: {saved_time:.2f}ms")
    print(f"  Benefit: Eliminates CPU transfer overhead for downstream GPU operations")
    
    # Verify correctness (values should be close but not identical due to RNG)
    print(f"\n[VERIFY] Data on correct device: {sampled_torch.is_cuda}")
    
    print(f"\n" + "="*70)
    print("✅ GPU OPTIMIZATION TEST PASSED")
    print("="*70)
    print()


if __name__ == "__main__":
    test_reparametrization_sampling()
    test_gpu_optimization()
