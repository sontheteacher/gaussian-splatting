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
    p.add_argument('--scale', type=float, default=0.005,
                   help='Scale factor for normal vector length (default: 0.005, or specify custom value like 0.001-0.1)')
    p.add_argument('--limit', type=int, default=0,
                   help='Limit visualization to N points (0 = show all points, default: 0)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for subsampling (default: 42)')
    p.add_argument('--low-confidence-alpha', type=float, default=0.2,
                   help='Transparency for low confidence points (0.0=invisible, 1.0=fully opaque, default: 0.2)')
    p.add_argument('--offscreen', action='store_true', 
                   help='Force offscreen rendering')
    p.add_argument('--screenshot', type=str, default=None, 
                   help='Path to save screenshot (exits after saving)')
    return p.parse_args()


def load_ply_with_normals(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Load PLY file with normals, confidence, optional Gaussian attributes, and edges.
    
    Returns:
        points: (N, 3) array of point coordinates
        normals: (N, 3) array of normal vectors
        confidences: (N,) array of confidence values (None if not present)
        attributes: dict of additional attributes (opacities, scales, etc.)
        edges: (M, 2) array of edge connectivity (None if not present)
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
    
    # Try to load edges
    edges = None
    if 'edge' in plydata:
        edge_element = plydata['edge']
        edges = np.stack([
            np.asarray(edge_element['vertex1']),
            np.asarray(edge_element['vertex2'])
        ], axis=1)
        print(f"[INFO] Loaded {len(edges)} edges")
    
    return points, normals, confidences, attributes, edges


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
        points, normals, confidences, attributes, edges = load_ply_with_normals(args.ply_file)
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
            # Filter edges to only include edges between subsampled points
            if edges is not None:
                # Create mapping from old indices to new indices
                index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
                # Filter edges to only those between subsampled vertices
                valid_edges = []
                for edge in edges:
                    if edge[0] in index_map and edge[1] in index_map:
                        valid_edges.append([index_map[edge[0]], index_map[edge[1]]])
                edges = np.array(valid_edges) if valid_edges else None
                if edges is not None:
                    print(f"[INFO] Filtered to {len(edges)} edges between subsampled points")
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
        
        # Register edge network if available
        if edges is not None and len(edges) > 0:
            print(f"[INFO] Registering {len(edges)} edges as curve network")
            # Create curve network from edges
            curve_network = ps.register_curve_network("connectivity", points, edges)
            curve_network.set_radius(0.0001, relative=False)  # Set thin radius for clean visualization
            curve_network.set_enabled(False)  # Start disabled to reduce visual clutter
            print(f"[INFO] Edge connectivity registered (disabled by default, thin lines)")
        
        # Scale normals for visualization
        if args.scale > 0:
            scaled_normals = normals * args.scale
            print(f"[INFO] Scaling normals by {args.scale} for visualization (magnitude: {np.linalg.norm(scaled_normals, axis=1).mean():.6f})")
        else:
            scaled_normals = normals * 0.005
            print(f"[INFO] Using default scale 0.005 for visualization (magnitude: {np.linalg.norm(scaled_normals, axis=1).mean():.6f})")
        
        # Add basic quantities
        pc.add_vector_quantity("normals", scaled_normals, enabled=True, vectortype='ambient')
        
        # # Add normal magnitude as scalar
        # normal_mag = np.linalg.norm(normals, axis=1)
        # pc.add_scalar_quantity("normal_magnitude", normal_mag, enabled=False, cmap='viridis')
        
        # # Add normal components as scalars
        # pc.add_scalar_quantity("normal_x", normals[:, 0], enabled=False, cmap='coolwarm')
        # pc.add_scalar_quantity("normal_y", normals[:, 1], enabled=False, cmap='coolwarm')
        # pc.add_scalar_quantity("normal_z", normals[:, 2], enabled=False, cmap='coolwarm')
        
        # Add color based on normal direction (RGB = xyz normalized to [0,1])
        normal_color = (normals + 1.0) / 2.0  # Map from [-1,1] to [0,1]
        pc.add_color_quantity("normal_color", normal_color, enabled=True)
        
        # Add confidence-based visualization if available
        if confidences is not None:
            pc.add_scalar_quantity("confidence", confidences, enabled=False, cmap='viridis')
            
            # Setup interactive confidence filtering with transparency/visibility control
            confidence_threshold = [confidences.min()]  # Use list for mutable reference
            low_confidence_alpha = [args.low_confidence_alpha]  # Visibility for low confidence points (0=invisible, 1=fully visible)
            
            def update_visualization():
                """Update point transparency and normal visibility based on confidence threshold"""
                mask = confidences >= confidence_threshold[0]
                
                # Filter normal vectors - only show for points above threshold
                filtered_normals = np.zeros_like(scaled_normals)
                filtered_normals[mask] = scaled_normals[mask]
                pc.add_vector_quantity("normals", filtered_normals, enabled=True, vectortype='ambient')
                
                # Add normal colors
                pc.add_color_quantity("normal_color", normal_color, enabled=True)
                
                # Create transparency values based on confidence and threshold
                transparency_vals = np.ones(len(points))  # Start with full opacity
                transparency_vals[mask] = 1.0  # High confidence: full opacity  
                transparency_vals[~mask] = low_confidence_alpha[0]  # Low confidence: reduced opacity
                
                # Add transparency quantity
                pc.add_scalar_quantity("transparency_vals", transparency_vals, enabled=False, cmap='viridis')
                
                # Set this quantity to control transparency
                pc.set_transparency_quantity("transparency_vals")
            
            # Apply initial transparency effect
            update_visualization()
            
            def gui_callback():
                """GUI callback for interactive controls"""
                import polyscope.imgui as psim
                
                psim.Text(f"Confidence Filtering & Transparency")
                psim.Text(f"Range: [{confidences.min():.4f}, {confidences.max():.4f}]")
                
                # Confidence threshold slider
                changed_conf, new_threshold = psim.SliderFloat(
                    "Min Confidence", confidence_threshold[0], 
                    confidences.min(), confidences.max())
                
                # Transparency alpha slider for low confidence points
                changed_alpha, new_alpha = psim.SliderFloat(
                    "Low Confidence Alpha", low_confidence_alpha[0], 
                    0.0, 1.0)
                
                if changed_conf or changed_alpha:
                    if changed_conf:
                        confidence_threshold[0] = new_threshold
                    if changed_alpha:
                        low_confidence_alpha[0] = new_alpha
                    update_visualization()
                
                visible_count = np.sum(confidences >= confidence_threshold[0])
                psim.Text(f"High confidence: {visible_count}/{len(confidences)} ({100*visible_count/len(confidences):.1f}%)")
                psim.Text(f"Low confidence transparency: {low_confidence_alpha[0]:.2f} (0=invisible, 1=opaque)")
                
                # Add helpful text
                psim.Separator()
                psim.Text("Tips:")
                psim.Text("• Normal vectors shown only for points above threshold")
                psim.Text("• High confidence: fully opaque, colored by normal direction")
                psim.Text("• Low confidence: transparent based on alpha setting")
                psim.Text("• Adjust threshold to change which points get normals/transparency")
                psim.Text("• Enable 'transparency_vals' quantity to see values")
            
            # Set up GUI callback
            ps.set_user_callback(gui_callback)
            
        else:
            # No confidence data - just use normal colors
            pc.add_color_quantity("normal_color", normal_color, enabled=True)
            
        # Add optional Gaussian attributes
        if 'opacity' in attributes:
            pc.add_scalar_quantity("opacity", attributes['opacity'], enabled=False, cmap='viridis')
        
        if 'scale_max' in attributes:
            pc.add_scalar_quantity("scale_max", attributes['scale_max'], enabled=False, cmap='viridis')
        
        if 'scale_mean' in attributes:
            pc.add_scalar_quantity("scale_mean", attributes['scale_mean'], enabled=False, cmap='viridis')
        
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
