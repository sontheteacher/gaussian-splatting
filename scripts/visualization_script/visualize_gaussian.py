#!/usr/bin/env python3
"""
Minimal Gaussian Splat PLY intrinsic analysis: geodesic distance + log map.
Uses GaussianModel's PLY loading functionality for consistency with the main codebase.

Features:
  * Loads point cloud (.ply) using GaussianModel's load_ply method
  * (Optional) random subsampling via --limit for large clouds
  * Selects a source vertex (user provided or closest to centroid)
  * Computes intrinsic geodesic distances & log map with potpourri3d PointCloudHeatSolver
  * Saves results to an .npz archive regardless of visualization success
  * Attempts Polyscope visualization; if available saves a screenshot automatically
  * Works in headless contexts via --offscreen (best effort). If OpenGL init fails, data still saved.

CLI Examples:
  python ply_geodesic_gaussian.py scene.ply
  python ply_geodesic_gaussian.py scene.ply --limit 200000 --export results_scene.npz
  python ply_geodesic_gaussian.py scene.ply --source-vertex 1234 --screenshot custom.png
  python ply_geodesic_gaussian.py scene.ply --offscreen

Exit Codes:
  0  Success (data computed; screenshot may or may not exist if visualization failed)
  1  Invalid input (file missing, bad source index)
  2  Runtime failure during computation (no outputs written)

Notes:
  * When subsampling, the source vertex index refers to the SUBSAMPLED cloud.
  * For large datasets (>~500k points) you may wish to pre-limit for speed & memory.
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    print("[ERROR] torch is required for GaussianModel.")
    raise

# Add the parent directory to the path to import from scene
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from scene.gaussian_model import GaussianModel, load_gaussian_model
except Exception as e:  # pragma: no cover
    print("[ERROR] Could not import GaussianModel from scene.gaussian_model.")
    raise

try:
    import potpourri3d as pp3d
except Exception as e:  # pragma: no cover
    print("[ERROR] potpourri3d is required.")
    raise

try:
    import polyscope as ps
except Exception:
    ps = None  # We'll handle later


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute geodesic distance & log map on a PLY point cloud using GaussianModel loading.")
    p.add_argument('ply_file', help='Path to input .ply file')
    p.add_argument('--source-vertex', type=int, default=None,
                   help='Index of source vertex (after subsampling if --limit used). Default: closest to centroid')
    p.add_argument('--limit', type=int, default=0,
                   help='Randomly subsample to N points (0 = no limit)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for subsampling')
    p.add_argument('--offscreen', action='store_true', help='Force offscreen Polyscope init (best effort)')
    p.add_argument('--screenshot', type=str, default=None, help='Path to save screenshot (overrides default)')
    p.add_argument('--no-screenshot', action='store_true', help='Disable automatic screenshot saving')
    p.add_argument('--export', type=str, default=None, help='Path to save results .npz (default: derived)')
    p.add_argument('--no-viz', action='store_true', help='Skip all Polyscope visualization (still exports results)')
    p.add_argument('--force-viz', action='store_true', help='Force attempt at visualization even if headless (may crash if no GL context)')
    return p.parse_args()


def choose_source(points: np.ndarray, idx: int | None) -> int:
    if idx is not None:
        if not (0 <= idx < len(points)):
            raise ValueError(f"source vertex {idx} out of range (0..{len(points)-1})")
        return idx
    centroid = points.mean(axis=0)
    return int(np.argmin(np.linalg.norm(points - centroid, axis=1)))


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()

    try:
        # Load using GaussianModel
        print(f"[INFO] Loading: {args.ply_file}")
        pts, gaussian_model = load_gaussian_model(args.ply_file)
        print(f"[INFO] Loaded {len(pts)} points using GaussianModel. Bounds min={pts.min(0)}, max={pts.max(0)}")
        
        # Extract Gaussian properties for visualization and storage
        # print("[INFO] Extracting Gaussian properties...")
        # Only extract what we need for point cloud geodesic analysis
        # opacities = gaussian_model.get_opacity.detach().cpu().numpy()
        # scales = gaussian_model.get_scaling.detach().cpu().numpy()
        # rotations = gaussian_model.get_rotation.detach().cpu().numpy()
        # features_dc = gaussian_model.get_features_dc.detach().cpu().numpy()
        # print(f"[INFO] Extracted properties: opacities {opacities.shape}, "
        #       f"scales {scales.shape}, rotations {rotations.shape}")

        # Subsample
        if args.limit and len(pts) > args.limit:
            print(f"[INFO] Subsampling from {len(pts)} -> {args.limit} (seed={args.seed})")
            rng = np.random.default_rng(args.seed)
            sel = rng.choice(len(pts), args.limit, replace=False)
            pts = pts[sel]
            # Also subsample Gaussian properties (commented out - not needed for point cloud analysis)
            # opacities = opacities[sel]
            # scales = scales[sel]
            # rotations = rotations[sel]
            # features_dc = features_dc[sel]
        else:
            sel = None  # Not used further but kept for potential extension

        # Source vertex
        source_idx = choose_source(pts, args.source_vertex)
        print(f"[INFO] Source vertex: {source_idx}  coord={pts[source_idx]}")

        # Solver
        print("[INFO] Building PointCloudHeatSolver ...")
        t_s = time.perf_counter()
        solver = pp3d.PointCloudHeatSolver(pts)
        build_dt = time.perf_counter() - t_s
        print(f"[INFO] Solver built in {build_dt:.2f}s")

        # Compute geodesic distance
        print("[INFO] Computing geodesic distances (heat method)...")
        t_d = time.perf_counter()
        geod = solver.compute_distance(source_idx)
        geod_dt = time.perf_counter() - t_d
        print(f"[INFO] Geodesic done in {geod_dt:.2f}s  (min={np.nanmin(geod):.6f} max={np.nanmax(geod):.6f})")

        # Compute log map
        print("[INFO] Computing log map ...")
        t_l = time.perf_counter()
        try:
            logmap = solver.compute_log_map(source_idx)
            # Get tangent frames to map log map to 3D space
            basisX, basisY, basisN = solver.get_tangent_frames()
            # Map log map to 3D vectors at each point
            logmap3D = logmap[:, 0, np.newaxis] * basisX + logmap[:, 1, np.newaxis] * basisY
        except Exception as e:
            print(f"[WARN] Log map computation failed: {e}")
            logmap = None
            logmap3D = None
        logmap_dt = time.perf_counter() - t_l
        if logmap is not None:
            print(f"[INFO] Log map done in {logmap_dt:.2f}s  shape={logmap.shape}")
        else:
            print("[INFO] Log map unavailable.")

        # Prepare export path (commented out - not saving results temporarily)
        # if args.export:
        #     export_path = args.export
        # else:
        #     base = os.path.splitext(os.path.basename(args.ply_file))[0]
        #     export_path = f"results_{base}_geodesic_logmap_gaussian.npz"
        # ensure_dir(export_path)

        # Save results BEFORE visualization attempt (commented out - not saving temporarily)
        # np.savez_compressed(
        #     export_path,
        #     points=pts,
        #     source_idx=source_idx,
        #     geodesic=geod,
        #     logmap=logmap if logmap is not None else np.array([]),
        #     logmap3D=logmap3D if logmap3D is not None else np.array([]),
        #     gaussian_opacities=opacities,
        #     gaussian_scales=scales,
        #     gaussian_rotations=rotations,
        #     gaussian_features_dc=features_dc,
        #     gaussian_model=gaussian_model,  # Store the full model for future use
        #     subsample_indices=sel if sel is not None else np.array([])
        # )
        # print(f"[INFO] Results saved to {export_path}")
        print("[INFO] Skipping results saving (temporary mode)")

        # Decide screenshot path
        default_ss = None
        if not args.no_screenshot:
            base = os.path.splitext(os.path.basename(args.ply_file))[0]
            default_ss = os.path.join('screenshots', f"{base}_geodesic_gaussian.png")
        screenshot_path = args.screenshot or default_ss
        if screenshot_path:
            ensure_dir(screenshot_path)

        # Visualization attempt (robust against headless segfaults)
        vis_ok = False
        if args.no_viz:
            print("[INFO] --no-viz specified; skipping Polyscope visualization.")
        elif ps is None:
            print("[WARN] polyscope not available; skipping visualization.")
        else:
            # Heuristic: if offscreen requested OR no DISPLAY, only attempt if user forced and EGL/OSMesa hints present
            need_offscreen = args.offscreen or ('DISPLAY' not in os.environ)
            headless_env = ('DISPLAY' not in os.environ)
            egl_hint = any(k in os.environ for k in [
                'PYOPENGL_PLATFORM', 'EGL_DEVICE_ID', 'MESA_GL_VERSION_OVERRIDE'
            ])
            # Detect WSL without GUI (look for 'microsoft' in kernel release)
            try:
                with open('/proc/sys/kernel/osrelease','r') as f:
                    osrel = f.read().lower()
            except Exception:
                osrel = ''
            wsl_env = 'microsoft' in osrel
            wslg_present = os.path.exists('/mnt/wslg')  # indicates WSLg GUI support
            unsafe_headless = headless_env and not egl_hint and (not wsl_env or (wsl_env and not wslg_present))

            if unsafe_headless and not args.force_viz:
                print("[INFO] Headless environment without EGL/OSMesa (and --force-viz not set); skipping Polyscope to prevent segfault.")
                print("       Set PYOPENGL_PLATFORM=egl or install OSMesa and use --force-viz to attempt offscreen rendering.")
            else:
                print(f"[INFO] Initializing Polyscope (offscreen={need_offscreen}) ...")
                try:
                    try:
                        ps.init(offscreen=need_offscreen)
                    except TypeError:
                        ps.init()  # older polyscope
                    vis_ok = True
                except Exception as e:
                    print(f"[WARN] Polyscope init failed: {e}")
                    vis_ok = False

                if vis_ok:
                    try:
                        pc = ps.register_point_cloud("points", pts)
                        pc.add_scalar_quantity("geodesic_distance", geod, enabled=True, cmap='viridis')
                        
                        if logmap3D is not None:
                            pc.add_vector_quantity("log_map", logmap3D, enabled=True)
                            pc.add_scalar_quantity("logmap_magnitude", np.linalg.norm(logmap3D, axis=1), enabled=True, cmap='plasma')

                        # Add Gaussian-specific visualizations (commented out - not needed for point cloud analysis)
                        # pc.add_scalar_quantity("opacity", opacities.squeeze(), enabled=False, cmap='blues')
                        # pc.add_scalar_quantity("scale_max", np.max(scales, axis=1), enabled=False, cmap='oranges')

                        highlight = np.zeros(len(pts))
                        highlight[source_idx] = 1.0
                        pc.add_scalar_quantity("source", highlight, cmap='reds', enabled=True)

                        if screenshot_path:
                            print(f"[INFO] Capturing screenshot -> {screenshot_path}")
                            ps.screenshot(screenshot_path)
                        else:
                            print("[INFO] Launching interactive viewer (close window to finish)...")
                            ps.show()
                    except Exception as e:
                        print(f"[WARN] Visualization failed after init: {e}")
                        vis_ok = False

        total_dt = time.perf_counter() - t0
        print(f"[INFO] Done in {total_dt:.2f}s (visualization={'ok' if 'vis_ok' in locals() and vis_ok else 'skipped'})")
        return 0

    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except Exception as e:
        # In catastrophic compute failure we have not saved results yet
        print(f"[ERROR] Unexpected failure: {e}", file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())