#!/usr/bin/env python3
"""
Visualize point cloud with geodesic-based normals using Polyscope.

Loads a PLY file containing points, normal estimates, and optional confidence scores,
then visualizes them using Polyscope with interactive controls.

Features:
  * Interactive confidence filtering with real-time slider
  * Normal vector visualization with adjustable scaling
  * Color-coded visualization (normals, confidence, opacity, scales)
  * Point cloud subsampling for performance
  * Screenshot capture support

CLI Examples:
  python visualize_normals.py points_with_normals.ply
  python visualize_normals.py points_with_normals.ply --scale 0.005
  python visualize_normals.py points_with_normals.ply --limit 10000 --seed 123
  python visualize_normals.py points_with_normals.ply --offscreen --screenshot output.png
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np

try:
    from plyfile import PlyData
except Exception as e:
    print("[ERROR] plyfile is required.")
    raise

try:
    import polyscope as ps
except Exception as e:
    print("[ERROR] polyscope is required for visualization.")
    raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize point cloud with normals using Polyscope.")
    p.add_argument('ply_file', help='Path to input .ply file with normals')
    p.add_argument('--scale', type=float, default=0.01,
                   help='Scale factor for normal vector length (0.01 = auto-scale, or specify custom value like 0.001-0.1)')
    p.add_argument('--limit', type=int, default=0,
                   help='Limit visualization to N points (0 = show all points, default: 0)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for subsampling (default: 42)')
    p.add_argument('--offscreen', action='store_true', 
                   help='Force offscreen rendering')
    p.add_argument('--screenshot', type=str, default=None, 
                   help='Path to save screenshot (exits after saving)')
    return p.parse_args()


def load_ply_with_normals(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load PLY file with normals, confidence, and optional Gaussian attributes.
    
    Returns:
        points: (N, 3) array of point coordinates
        normals: (N, 3) array of normal vectors
        confidences: (N,) array of confidence values (None if not present)
        attributes: dict of additional attributes (opacities, scales, etc.)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    
    # Load points
    points = np.stack([
        np.asarray(vertex['x']),
        np.asarray(vertex['y']),
        np.asarray(vertex['z'])
    ], axis=1)
    
    # Load normals
    normals = np.stack([
        np.asarray(vertex['nx']),
        np.asarray(vertex['ny']),
        np.asarray(vertex['nz'])
    ], axis=1)
    
    # Load optional attributes
    attributes = {}
    
    # Try to load confidence
    confidences = None
    if 'confidence' in vertex:
        confidences = np.asarray(vertex['confidence'])
        attributes['confidence'] = confidences
    
    # Try to load opacity
    if 'opacity' in vertex:
        attributes['opacity'] = np.asarray(vertex['opacity'])
    
    # Try to load scales
    scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
    if scale_names:
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.stack([np.asarray(vertex[name]) for name in scale_names], axis=1)
        attributes['scales'] = scales
        attributes['scale_max'] = np.max(scales, axis=1)
        attributes['scale_mean'] = np.mean(scales, axis=1)
    
    # Try to load rotations
    rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
    if rot_names:
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rotations = np.stack([np.asarray(vertex[name]) for name in rot_names], axis=1)
        attributes['rotations'] = rotations
    
    return points, normals, confidences, attributes


def main() -> int:
    args = parse_args()
    
    # Validate inputs
    if not os.path.isfile(args.ply_file):
        print(f"[ERROR] Input file not found: {args.ply_file}", file=sys.stderr)
        return 1
    if args.scale < 0:
        print(f"[ERROR] scale must be non-negative, got {args.scale}", file=sys.stderr)
        return 1
    if args.limit < 0:
        print(f"[ERROR] limit must be non-negative, got {args.limit}", file=sys.stderr)
        return 1
    
    try:
        # Load PLY with normals
        print(f"[INFO] Loading: {args.ply_file}")
        points, normals, confidences, attributes = load_ply_with_normals(args.ply_file)
        print(f"[INFO] Loaded {len(points)} points with normals")
        print(f"[INFO] Points bounds: min={points.min(0)}, max={points.max(0)}")
        print(f"[INFO] Normals magnitude: mean={np.linalg.norm(normals, axis=1).mean():.6f}, "
              f"std={np.linalg.norm(normals, axis=1).std():.6f}")
        
        if confidences is not None:
            print(f"[INFO] Confidence range: [{confidences.min():.4f}, {confidences.max():.4f}], "
                  f"mean={confidences.mean():.4f}")
        
        if attributes:
            print(f"[INFO] Additional attributes: {list(attributes.keys())}")
        
        # Subsample if requested
        if args.limit and len(points) > args.limit:
            print(f"[INFO] Subsampling from {len(points)} -> {args.limit} points (seed={args.seed})")
            rng = np.random.default_rng(args.seed)
            indices = rng.choice(len(points), args.limit, replace=False)
            points = points[indices]
            normals = normals[indices]
            if confidences is not None:
                confidences = confidences[indices]
            # Also subsample attributes
            for key in attributes:
                if len(attributes[key]) == len(indices):  # Skip if already subsampled or wrong size
                    continue
                attributes[key] = attributes[key][indices]
            print(f"[INFO] Subsampled to {len(points)} points")
        
        # Initialize Polyscope
        need_offscreen = args.offscreen or ('DISPLAY' not in os.environ)
        print(f"[INFO] Initializing Polyscope (offscreen={need_offscreen})...")
        try:
            ps.init(offscreen=need_offscreen)
        except TypeError:
            ps.init()  # older polyscope version
        
        # Register point cloud
        pc = ps.register_point_cloud("points", points)
        
        # Scale normals for visualization if requested
        if args.scale > 0:
            scaled_normals = normals * args.scale
            print(f"[INFO] Scaling normals by {args.scale} for visualization (magnitude: {np.linalg.norm(scaled_normals, axis=1).mean():.6f})")
        else:
            scaled_normals = normals
            print(f"[INFO] Using normals at original magnitude (1.0)")
        
        # Add basic quantities
        pc.add_vector_quantity("normals", scaled_normals, enabled=True, vectortype='ambient')
        
        # Add normal magnitude as scalar
        normal_mag = np.linalg.norm(normals, axis=1)
        pc.add_scalar_quantity("normal_magnitude", normal_mag, enabled=False, cmap='viridis')
        
        # Add normal components as scalars
        pc.add_scalar_quantity("normal_x", normals[:, 0], enabled=False, cmap='coolwarm')
        pc.add_scalar_quantity("normal_y", normals[:, 1], enabled=False, cmap='coolwarm')
        pc.add_scalar_quantity("normal_z", normals[:, 2], enabled=False, cmap='coolwarm')
        
        # Add color based on normal direction (RGB = xyz normalized to [0,1])
        normal_color = (normals + 1.0) / 2.0  # Map from [-1,1] to [0,1]
        pc.add_color_quantity("normal_color", normal_color, enabled=True)
        
        # Add confidence-based visualization if available
        if confidences is not None:
            pc.add_scalar_quantity("confidence", confidences, enabled=False, cmap='viridis')
            
            # Setup interactive confidence filtering
            confidence_threshold = [confidences.min()]  # Use list for mutable reference
            
            def update_normals():
                """Update displayed normals based on confidence threshold"""
                mask = confidences >= confidence_threshold[0]
                if np.any(mask):
                    # Show normals only for points above threshold
                    filtered_normals = np.zeros_like(scaled_normals)
                    filtered_normals[mask] = scaled_normals[mask]
                    pc.add_vector_quantity("normals", filtered_normals, enabled=True, vectortype='ambient')
                    
                    # Update point colors to show filtered vs unfiltered
                    point_colors = np.zeros((len(points), 3))
                    point_colors[mask] = normal_color[mask]  # Colored by normals
                    point_colors[~mask] = [0.3, 0.3, 0.3]   # Gray for low confidence
                    pc.add_color_quantity("filtered_normals", point_colors, enabled=True)
            
            def gui_callback():
                """GUI callback for interactive controls"""
                ps.imgui.Text(f"Confidence Filtering")
                ps.imgui.Text(f"Range: [{confidences.min():.4f}, {confidences.max():.4f}]")
                
                changed, new_threshold = ps.imgui.SliderFloat(
                    "Min Confidence", confidence_threshold[0], 
                    confidences.min(), confidences.max())
                
                if changed:
                    confidence_threshold[0] = new_threshold
                    update_normals()
                
                visible_count = np.sum(confidences >= confidence_threshold[0])
                ps.imgui.Text(f"Visible normals: {visible_count}/{len(confidences)} ({100*visible_count/len(confidences):.1f}%)")
            
            # Set up GUI callback
            ps.set_user_callback(gui_callback)
            
            # Initial update
            update_normals()
            
        # Add optional Gaussian attributes
        # if 'opacity' in attributes:
        #     pc.add_scalar_quantity("opacity", attributes['opacity'], enabled=False, cmap='plasma')
        
        # if 'scale_max' in attributes:
        #     pc.add_scalar_quantity("scale_max", attributes['scale_max'], enabled=False, cmap='plasma')
        
        # Handle screenshot or interactive mode
        if args.screenshot:
            print(f"[INFO] Capturing screenshot -> {args.screenshot}")
            os.makedirs(os.path.dirname(args.screenshot) or '.', exist_ok=True)
            ps.screenshot(args.screenshot)
            print("[INFO] Screenshot saved. Exiting.")
        else:
            print("[INFO] Launching interactive viewer (close window to finish)...")
            ps.show()
        
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
