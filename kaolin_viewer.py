#!/usr/bin/env python3
"""
Interactive Gaussian Splatting Viewer with Kaolin
Restructured from notebook for better performance and organization.
"""

import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from pathlib import Path
from PIL import Image
import argparse

try:
    import kaolin
    import kaolin.render.camera
    import kaolin.visualize
    KAOLIN_AVAILABLE = True
except ImportError:
    print("ERROR: Kaolin not installed. Please install with:")
    print("pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html")
    KAOLIN_AVAILABLE = False

# Gaussian splatting dependencies
from utils.graphics_utils import focal2fov, getWorld2View2, getProjectionMatrix
from utils.system_utils import searchForMaxIteration
from utils.general_utils import PILtoTorch
from gaussian_renderer import render, GaussianModel
from scene.cameras import Camera as GSCamera


class PipelineParamsNoparse:
    """Pipeline parameters without argparse."""
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


class GaussianViewer:
    """Main viewer class for Gaussian Splatting scenes."""
    
    def __init__(self, model_path, sh_degree=3, iteration=-1):
        """
        Initialize the viewer with a trained model.
        
        Args:
            model_path: Path to trained model directory
            sh_degree: Spherical harmonics degree
            iteration: Iteration to load (-1 for latest)
        """
        self.model_path = model_path
        self.sh_degree = sh_degree
        self.iteration = iteration
        
        # Initialize components
        self.gaussians = None
        self.pipeline = PipelineParamsNoparse()
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        
        # Load the model
        self._load_model()
        
        # Camera and visualization
        self.test_camera = None
        self.visualizer = None
        
    def _load_model(self):
        """Load the Gaussian Splatting model."""
        checkpt_dir = os.path.join(self.model_path, "point_cloud")
        
        if not os.path.exists(checkpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpt_dir}")
        
        if self.iteration == -1:
            self.iteration = searchForMaxIteration(checkpt_dir)
            print(f"Auto-detected iteration: {self.iteration}")
        
        checkpt_path = os.path.join(checkpt_dir, f"iteration_{self.iteration}", "point_cloud.ply")
        
        if not os.path.exists(checkpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpt_path}")
        
        print(f"Loading checkpoint: {checkpt_path}")
        self.gaussians = GaussianModel(self.sh_degree)
        self.gaussians.load_ply(checkpt_path)
        
        print(f"✅ Loaded {len(self.gaussians.get_xyz)} Gaussians")
        
    def create_camera(self, width=800, height=800, fovx=0.69, fovy=0.69, 
                     distance=4.0, elevation=0, azimuth=0):
        """
        Create a camera for rendering.
        
        Args:
            width, height: Image dimensions
            fovx, fovy: Field of view in radians
            distance: Distance from origin
            elevation: Elevation angle in degrees
            azimuth: Azimuth angle in degrees
        
        Returns:
            GSCamera object
        """
        # Convert angles to radians
        elev_rad = np.radians(elevation)
        azim_rad = np.radians(azimuth)
        
        # Calculate camera position
        x = distance * np.cos(elev_rad) * np.cos(azim_rad)
        y = distance * np.sin(elev_rad)
        z = distance * np.cos(elev_rad) * np.sin(azim_rad)
        
        # Create rotation matrix (look at origin)
        camera_pos = np.array([x, y, z])
        target = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        R = np.array([right, up, -forward])
        T = -R @ camera_pos
        
        # Create dummy image for Camera class
        dummy_image = Image.new('RGB', (width, height), color=(0, 0, 0))
        
        camera = GSCamera(
            resolution=(width, height),
            colmap_id=0,
            R=R,
            T=T,
            FoVx=fovx,
            FoVy=fovy,
            depth_params=None,
            image=dummy_image,
            invdepthmap=None,
            image_name='viewer_cam',
            uid=0
        )
        
        return camera
    
    def render_image(self, camera=None, width=512, height=512):
        """
        Render an image with the given camera.
        
        Args:
            camera: GSCamera object (creates default if None)
            width, height: Image dimensions if creating default camera
            
        Returns:
            dict: Rendered results with 'image' and 'depth' keys
        """
        if camera is None:
            camera = self.create_camera(width=width, height=height)
        
        with torch.no_grad():
            render_res = render(camera, self.gaussians, self.pipeline, self.background)
            rendering = render_res["render"]
        
        # Convert to numpy
        img_np = (rendering.permute(1, 2, 0) * 255).to(torch.uint8).detach().cpu().numpy()
        
        result = {'image': img_np}
        
        # Add depth if available
        if 'depth' in render_res:
            depth = render_res["depth"]
            depth_np = depth.detach().cpu().numpy()
            if depth_np.ndim == 3 and depth_np.shape[0] == 1:
                depth_np = depth_np.squeeze(0)
            result['depth'] = depth_np
        
        return result
    
    def show_test_render(self, width=512, height=512, save_path=None):
        """
        Show a test rendering with depth visualization.
        
        Args:
            width, height: Image dimensions
            save_path: Optional path to save the images
        """
        print("Rendering test image...")
        
        # Create test camera
        camera = self.create_camera(width=width, height=height)
        
        # Render
        result = self.render_image(camera)
        img_np = result['image']
        
        # Create visualization
        if 'depth' in result:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # RGB image
            axes[0].imshow(img_np)
            axes[0].set_title("Rendered Image")
            axes[0].axis('off')
            
            # Raw depth
            depth_np = result['depth']
            im1 = axes[1].imshow(depth_np, cmap='plasma')
            axes[1].set_title("Raw Depth Map")
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], shrink=0.7)
            
            # Normalized depth
            depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
            im2 = axes[2].imshow(depth_norm, cmap='viridis')
            axes[2].set_title("Normalized Depth")
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], shrink=0.7)
            
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(img_np)
            ax.set_title("Rendered Image")
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved image to {save_path}")
        
        plt.show()
        
        print(f"✅ Rendering complete! Image shape: {img_np.shape}")
        print(f"Number of Gaussians: {len(self.gaussians.get_xyz)}")
    
    def create_kaolin_viewer(self, width=512, height=512):
        """
        Create interactive Kaolin viewer.
        
        Args:
            width, height: Viewer dimensions
            
        Returns:
            Kaolin visualizer object
        """
        if not KAOLIN_AVAILABLE:
            raise ImportError("Kaolin not available for interactive viewing")
        
        # Create test camera and convert to Kaolin format
        test_camera = self.create_camera(width=width, height=height)
        kal_cam = self._convert_gs_to_kaolin_camera(test_camera)
        
        # Calculate scene center for focus
        xyz = self.gaussians.get_xyz.detach().cpu().numpy()
        scene_center = torch.tensor(xyz.mean(axis=0), dtype=torch.float32)
        
        print(f"Scene center: {scene_center.numpy()}")
        print(f"Scene bounds: min={xyz.min(axis=0)}, max={xyz.max(axis=0)}")
        
        # Create render function
        def render_kaolin(kaolin_cam):
            """Render function for Kaolin visualizer."""
            gs_cam = self._convert_kaolin_to_gs_camera(kaolin_cam, width, height)
            
            with torch.no_grad():
                render_res = render(gs_cam, self.gaussians, self.pipeline, self.background)
                rendering = render_res["render"]
            
            return (rendering.permute(1, 2, 0) * 255).to(torch.uint8).detach().cpu()
        
        # Create visualizer
        visualizer = kaolin.visualize.IpyTurntableVisualizer(
            width, height,
            copy.deepcopy(kal_cam),
            render_kaolin,
            focus_at=scene_center,
            world_up_axis=1,  # Y-up
            max_fps=30
        )
        
        self.visualizer = visualizer
        return visualizer
    
    def show_kaolin_viewer(self):
        """Show the Kaolin interactive viewer if it exists."""
        if self.visualizer is not None:
            print("🚀 Launching Kaolin interactive viewer...")
            print("Use mouse to rotate, zoom, and pan the view.")
            self.visualizer.show()
            return True
        else:
            print("❌ No Kaolin viewer created. Call create_kaolin_viewer() first.")
            return False
    
    def create_filtered_viewer(self, width=512, height=512):
        """
        Create viewer with Gaussian filtering controls.
        
        Returns:
            Tuple of (visualizer, scale_range, opacity_range) for external control
        """
        if not KAOLIN_AVAILABLE:
            raise ImportError("Kaolin not available for interactive viewing")
        
        # Analyze Gaussian properties for filter ranges
        scales = self.gaussians.get_scaling.detach().cpu()
        opacities = self.gaussians.get_opacity.detach().cpu()
        max_scale = scales.max(dim=1)[0]
        
        scale_range = (max_scale.min().item(), max_scale.max().item())
        opacity_range = (-10.0, 5.0)  # Typical range for inverse sigmoid
        
        print(f"Scale range: {scale_range}")
        print(f"Opacity range: {opacity_range}")
        
        # Store filter parameters (will be updated externally)
        self.current_scale_threshold = scale_range[1]
        self.current_opacity_threshold = opacity_range[0]
        
        # Create camera
        test_camera = self.create_camera(width=width, height=height)
        kal_cam = self._convert_gs_to_kaolin_camera(test_camera)
        
        # Scene focus
        xyz = self.gaussians.get_xyz.detach().cpu().numpy()
        scene_center = torch.tensor(xyz.mean(axis=0), dtype=torch.float32)
        
        def filtered_render_kaolin(kaolin_cam):
            """Render with filtered Gaussians."""
            # Apply filters
            scaling = self.gaussians._scaling.max(dim=1)[0]
            mask = scaling < self.current_scale_threshold
            
            opacity_mask = self.gaussians._opacity.squeeze() > self.current_opacity_threshold
            mask = mask & opacity_mask
            
            # Create filtered Gaussian model
            tmp_gaussians = GaussianModel(self.gaussians.max_sh_degree)
            tmp_gaussians._xyz = self.gaussians._xyz[mask, :]
            tmp_gaussians._features_dc = self.gaussians._features_dc[mask, ...]
            tmp_gaussians._features_rest = self.gaussians._features_rest[mask, ...]
            tmp_gaussians._opacity = self.gaussians._opacity[mask, ...]
            tmp_gaussians._scaling = self.gaussians._scaling[mask, ...]
            tmp_gaussians._rotation = self.gaussians._rotation[mask, ...]
            tmp_gaussians.active_sh_degree = self.gaussians.max_sh_degree
            
            # Render
            gs_cam = self._convert_kaolin_to_gs_camera(kaolin_cam, width, height)
            
            with torch.no_grad():
                render_res = render(gs_cam, tmp_gaussians, self.pipeline, self.background)
                rendering = render_res["render"]
            
            return (rendering.permute(1, 2, 0) * 255).to(torch.uint8).detach().cpu()
        
        # Create visualizer
        visualizer = kaolin.visualize.IpyTurntableVisualizer(
            width, height,
            copy.deepcopy(kal_cam),
            filtered_render_kaolin,
            focus_at=scene_center,
            world_up_axis=1,
            max_fps=30
        )
        
        return visualizer, scale_range, opacity_range
    
    def update_filters(self, scale_threshold=None, opacity_threshold=None):
        """Update filter thresholds for filtered viewer."""
        if scale_threshold is not None:
            self.current_scale_threshold = scale_threshold
        if opacity_threshold is not None:
            self.current_opacity_threshold = opacity_threshold
    
    def show_statistics(self):
        """Display Gaussian statistics and visualizations."""
        # Analyze properties
        scales = self.gaussians.get_scaling.detach().cpu()
        opacities = self.gaussians.get_opacity.detach().cpu()
        xyz = self.gaussians.get_xyz.detach().cpu().numpy()
        
        max_scale = scales.max(dim=1)[0]
        
        print("Gaussian Statistics")
        print("=" * 50)
        print(f"Total Gaussians: {len(xyz):,}")
        print(f"\nScene bounds:")
        print(f"  X: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
        print(f"  Y: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]")
        print(f"  Z: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")
        print(f"\nScale statistics:")
        print(f"  Min: {max_scale.min().item():.6f}")
        print(f"  Max: {max_scale.max().item():.6f}")
        print(f"  Mean: {max_scale.mean().item():.6f}")
        print(f"  Median: {max_scale.median().item():.6f}")
        print(f"\nOpacity statistics:")
        print(f"  Min: {opacities.min().item():.6f}")
        print(f"  Max: {opacities.max().item():.6f}")
        print(f"  Mean: {opacities.mean().item():.6f}")
        print(f"  Median: {opacities.median().item():.6f}")
        
        # Plot histograms and positions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scale histogram
        axes[0, 0].hist(max_scale.numpy(), bins=100, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Max Scale')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Gaussian Scale Distribution')
        axes[0, 0].set_yscale('log')
        
        # Opacity histogram
        axes[0, 1].hist(opacities.squeeze().numpy(), bins=100, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Opacity')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Opacity Distribution')
        axes[0, 1].set_yscale('log')
        
        # 3D positions - top view
        axes[1, 0].scatter(xyz[:, 0], xyz[:, 1], s=1, alpha=0.5)
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        axes[1, 0].set_title('Gaussian Positions (Top View)')
        axes[1, 0].axis('equal')
        
        # 3D positions - side view
        axes[1, 1].scatter(xyz[:, 0], xyz[:, 2], s=1, alpha=0.5)
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Z')
        axes[1, 1].set_title('Gaussian Positions (Side View)')
        axes[1, 1].axis('equal')
        
        plt.tight_layout()
        plt.show()
    
    def _convert_kaolin_to_gs_camera(self, kal_camera, width, height):
        """Convert Kaolin camera to Gaussian Splatting camera."""
        R = kal_camera.extrinsics.R[0].clone()
        R[1:3] = -R[1:3]
        T = kal_camera.extrinsics.t.squeeze().clone()
        T[1:3] = -T[1:3]
        
        dummy_image = Image.new('RGB', (width, height), color=(0, 0, 0))
        
        camera = GSCamera(
            resolution=(width, height),
            colmap_id=0,
            R=R.transpose(1, 0).cpu().numpy(),
            T=T.cpu().numpy(),
            FoVx=self._compute_cam_fov(kal_camera.intrinsics, 'x'),
            FoVy=self._compute_cam_fov(kal_camera.intrinsics, 'y'),
            depth_params=None,
            image=dummy_image,
            invdepthmap=None,
            image_name='kaolin_cam',
            uid=0
        )
        
        return camera
    
    def _convert_gs_to_kaolin_camera(self, gs_camera):
        """Convert Gaussian Splatting camera to Kaolin camera."""
        view_mat = gs_camera.world_view_transform.transpose(1, 0).clone()
        view_mat[1:3] = -view_mat[1:3]
        
        kal_cam = kaolin.render.camera.Camera.from_args(
            view_matrix=view_mat,
            width=gs_camera.image_width,
            height=gs_camera.image_height,
            fov=gs_camera.FoVx,
            device='cpu'
        )
        
        return kal_cam
    
    def _compute_cam_fov(self, intrinsics, axis='x'):
        """Compute FOV from Kaolin camera intrinsics."""
        aspectScale = intrinsics.width / 2.0
        tanHalfAngle = aspectScale / (intrinsics.focal_x if axis == 'x' else intrinsics.focal_y).item()
        fov = np.arctan(tanHalfAngle) * 2
        return fov


def create_matplotlib_interactive_viewer(viewer, width=512, height=512):
    """Create a matplotlib-based interactive viewer as fallback."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    
    print("🎨 Creating matplotlib-based interactive viewer...")
    
    # Create figure with subplots for controls
    fig = plt.figure(figsize=(12, 8))
    
    # Main image axis
    ax_img = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
    
    # Control axes
    ax_azimuth = plt.subplot2grid((4, 4), (3, 0), colspan=2)
    ax_elevation = plt.subplot2grid((4, 4), (3, 2), colspan=2)
    ax_distance = plt.subplot2grid((4, 4), (0, 3))
    ax_render_btn = plt.subplot2grid((4, 4), (1, 3))
    ax_save_btn = plt.subplot2grid((4, 4), (2, 3))
    
    # Initial render
    camera = viewer.create_camera(width=width, height=height)
    result = viewer.render_image(camera)
    img_display = ax_img.imshow(result['image'])
    ax_img.set_title('Interactive Gaussian Splatting Viewer')
    ax_img.axis('off')
    
    # Sliders
    slider_azimuth = Slider(ax_azimuth, 'Azimuth', 0, 360, valinit=0, valfmt='%d°')
    slider_elevation = Slider(ax_elevation, 'Elevation', -90, 90, valinit=0, valfmt='%d°')  
    slider_distance = Slider(ax_distance, 'Distance', 1, 10, valinit=4, orientation='vertical')
    
    # Buttons
    btn_render = Button(ax_render_btn, 'Render')
    btn_save = Button(ax_save_btn, 'Save')
    
    def update_render(val=None):
        """Update the render with current slider values."""
        azimuth = slider_azimuth.val
        elevation = slider_elevation.val  
        distance = slider_distance.val
        
        # Create camera with new parameters
        camera = viewer.create_camera(
            width=width, height=height,
            distance=distance, 
            elevation=elevation,
            azimuth=azimuth
        )
        
        # Render
        result = viewer.render_image(camera)
        
        # Update image
        img_display.set_array(result['image'])
        ax_img.set_title(f'Render (Az:{azimuth:.0f}°, El:{elevation:.0f}°, D:{distance:.1f})')
        fig.canvas.draw_idle()
    
    def save_image(event):
        """Save current render."""
        azimuth = slider_azimuth.val
        elevation = slider_elevation.val
        distance = slider_distance.val
        
        filename = f"render_az{azimuth:.0f}_el{elevation:.0f}_d{distance:.1f}.png"
        
        camera = viewer.create_camera(
            width=1024, height=1024,  # Higher resolution for saving
            distance=distance,
            elevation=elevation, 
            azimuth=azimuth
        )
        
        result = viewer.render_image(camera)
        plt.imsave(filename, result['image'])
        print(f"💾 Saved: {filename}")
    
    # Connect events
    slider_azimuth.on_changed(update_render)
    slider_elevation.on_changed(update_render)
    slider_distance.on_changed(update_render)
    btn_render.on_clicked(update_render)
    btn_save.on_clicked(save_image)
    
    plt.tight_layout()
    plt.show()
    
    print("🎮 Interactive matplotlib viewer launched!")
    print("Use sliders to control camera, 'Render' to update, 'Save' to export")
    
    return fig


def find_model_paths():
    """Find all available trained models."""
    possible_dirs = ['output', 'data']
    found_models = []
    
    for base_dir in possible_dirs:
        if not os.path.exists(base_dir):
            continue
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                point_cloud_dir = os.path.join(item_path, 'point_cloud')
                if os.path.exists(point_cloud_dir):
                    found_models.append(item_path)
    
    return found_models


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Interactive Gaussian Splatting Viewer')
    parser.add_argument('--model_path', type=str, help='Path to trained model directory')
    parser.add_argument('--width', type=int, default=512, help='Render width')
    parser.add_argument('--height', type=int, default=512, help='Render height')
    parser.add_argument('--iteration', type=int, default=-1, help='Model iteration to load')
    parser.add_argument('--test_only', action='store_true', help='Only run test rendering, no interactive viewer')
    parser.add_argument('--stats', action='store_true', help='Show Gaussian statistics')
    
    args = parser.parse_args()
    
    # Find models if no path provided
    if args.model_path is None:
        available_models = find_model_paths()
        if not available_models:
            print("No trained models found!")
            print("Please train a model first or specify --model_path")
            return
        
        print("Available models:")
        for i, model in enumerate(available_models):
            print(f"  [{i}] {model}")
        
        choice = input("\nSelect a model (number): ")
        try:
            args.model_path = available_models[int(choice)]
        except (ValueError, IndexError):
            args.model_path = available_models[0]
            print(f"Using default: {args.model_path}")
    
    # Create viewer
    print(f"Initializing viewer for: {args.model_path}")
    viewer = GaussianViewer(args.model_path, iteration=args.iteration)
    
    # Show statistics if requested
    if args.stats:
        viewer.show_statistics()
    
    # Run test rendering
    print("\n" + "="*50)
    viewer.show_test_render(width=args.width, height=args.height, 
                           save_path="test_render.png")
    
    if not args.test_only and KAOLIN_AVAILABLE:
        print("\n" + "="*50)
        print("Creating interactive viewer...")
        
        try:
            # Create basic viewer
            visualizer = viewer.create_kaolin_viewer(width=args.width, height=args.height)
            print("✅ Interactive viewer created!")
            
            # Try to display the viewer
            print("🚀 Launching interactive visualizer...")
            try:
                # Check if we're in an interactive environment that supports widgets
                import IPython
                if IPython.get_ipython() is not None:
                    print("Running in IPython/Jupyter - showing interactive viewer")
                    # Use the exact code from the notebook
                    visualizer.show()
                    print("✅ Kaolin interactive viewer launched!")
                    print("Use mouse to rotate, zoom, and pan the view.")
                else:
                    print("Not in Jupyter - attempting to show Kaolin viewer anyway...")
                    # Try to show the Kaolin viewer even outside Jupyter
                    try:
                        visualizer.show()
                        print("✅ Kaolin interactive viewer launched!")
                        print("Use mouse to rotate, zoom, and pan the view.")
                    except Exception as kaolin_error:
                        print(f"Kaolin viewer failed: {kaolin_error}")
                        print("Falling back to matplotlib interactive viewer...")
                        create_matplotlib_interactive_viewer(viewer, args.width, args.height)
            except ImportError:
                print("IPython not available - trying Kaolin viewer directly...")
                try:
                    visualizer.show()
                    print("✅ Kaolin interactive viewer launched!")
                    print("Use mouse to rotate, zoom, and pan the view.")
                except Exception as kaolin_error:
                    print(f"Kaolin viewer failed: {kaolin_error}")
                    print("Falling back to matplotlib interactive viewer...")
                    create_matplotlib_interactive_viewer(viewer, args.width, args.height)
            
            # Create filtered viewer
            filtered_viz, scale_range, opacity_range = viewer.create_filtered_viewer(
                width=args.width, height=args.height)
            print("✅ Filtered viewer created!")
            print(f"Scale range: {scale_range}")
            print(f"Opacity range: {opacity_range}")
            
            # Make objects available globally for interactive use
            globals()['viewer'] = viewer
            globals()['visualizer'] = visualizer
            globals()['filtered_viz'] = filtered_viz
            
            print("\n📖 Interactive viewer launched!")
            print("- Use mouse to rotate, zoom, and pan")
            print("- Objects available: viewer, visualizer, filtered_viz")
            
        except Exception as e:
            print(f"Failed to create interactive viewer: {e}")
            import traceback
            traceback.print_exc()
    
    elif not KAOLIN_AVAILABLE:
        print("Kaolin not available - interactive viewing disabled")
    
    print("\n" + "="*50)
    print("Viewer setup complete!")
    print("Use viewer.show_test_render() for static rendering")
    if KAOLIN_AVAILABLE:
        print("Use viewer.create_kaolin_viewer() for interactive viewing")


if __name__ == "__main__":
    main()