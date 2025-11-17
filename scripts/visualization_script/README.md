# Visualization Scripts Documentation

This directory contains scripts for computing and visualizing geodesic-based normals and geodesic distances on Gaussian splatting point clouds.

## 📋 Files Overview

### 1. **`compute_geodesic_normals.py`** ⭐ Main Script
Computes surface normals for point clouds using geodesic distance neighborhoods with confidence scoring.

**Purpose**: Estimate surface normals that respect the intrinsic geometry of the point cloud surface, not just Euclidean distances.

**Key Features**:
- Geodesic distance-based neighborhood selection (follows surface curvature)
- SVD-based normal estimation (minimal variance direction)
- Gaussian-weighted confidence scores for each normal
- Automatic geodesic radius estimation from point density
- Subsampling support for large point clouds (>50k points)
- Y-up orientation (Polyscope convention)

**Usage**:
```bash
# Basic usage - compute normals for all points
python compute_geodesic_normals.py input.ply output_with_normals.ply

# Fast mode - subsample for computation
python compute_geodesic_normals.py input.ply output.ply --subsample 1000

# Custom parameters
python compute_geodesic_normals.py input.ply output.ply \
    --geodesic-radius 0.05 \
    --sigma 0.02 \
    --min-neighbors 15 \
    --batch-size 200

# Limit input size
python compute_geodesic_normals.py input.ply output.ply --limit 10000 --subsample 500
```

**Algorithm**:
1. Load Gaussian splat PLY using `GaussianModel`
2. Build `PointCloudHeatSolver` for geodesic distance computation
3. For each point:
   - Compute geodesic distances to all other points
   - Select neighbors within geodesic radius
   - Apply SVD on centered neighborhood to find normal (minimal variance)
   - Flip normal to point upward (Y+ direction)
   - Compute Gaussian-weighted confidence: `Σ exp(-d²/(2σ²))`
4. Normalize confidences to [0, 1] range
5. Save with all Gaussian attributes preserved

**Output PLY Structure**:
```
x, y, z              # Point coordinates
nx, ny, nz           # Normal vectors (Y-up)
confidence           # Confidence score [0, 1]
opacity              # Gaussian opacity
f_dc_*               # Spherical harmonics features
scale_*              # Gaussian scale factors
rot_*                # Quaternion rotations
```

**Performance Tips**:
- Use `--subsample 1000` for >50k point clouds (400x speedup)
- Increase `--batch-size` for better cache locality
- Geodesic computation is O(N²) - bottleneck, not SVD
- potpourri3d is CPU-only, no GPU acceleration

---

### 2. **`visualize_normals.py`** ⭐ Main Visualization
Interactive visualization of computed normals with confidence filtering.

**Purpose**: Inspect normal quality and filter by confidence scores interactively.

**Key Features**:
- Interactive confidence slider (real-time filtering)
- Normal vector visualization with adjustable scaling
- Color-coded by normal direction (RGB = XYZ)
- Multiple scalar quantities (confidence, opacity, scales)
- Point subsampling for performance
- Screenshot capture support

**Usage**:
```bash
# Basic visualization
python visualize_normals.py points_with_normals.ply

# With scaling and subsampling
python visualize_normals.py points_with_normals.ply --scale 0.005 --limit 1000

# Save screenshot
python visualize_normals.py points_with_normals.ply \
    --offscreen --screenshot output.png

# Custom random seed for reproducible subsampling
python visualize_normals.py points_with_normals.ply --limit 5000 --seed 123
```

**Interactive Controls**:
- **Confidence Slider**: Drag to filter normals by minimum confidence
- **Point Colors**: High confidence = colored by normal direction, Low = gray
- **Visible Count**: Shows how many normals pass the threshold
- **Scalar Quantities**: Enable/disable different attributes in Polyscope UI

**Visualization Quantities**:
- `normals` - Normal vectors (colored by confidence filtering)
- `normal_color` - Points colored by normal direction (RGB)
- `normal_magnitude` - Length of normal vectors (should be ~1.0)
- `normal_x/y/z` - Individual normal components
- `confidence` - Confidence scores heatmap
- `opacity` - Gaussian opacity values
- `scale_max` - Maximum Gaussian ellipsoid dimension
- `scale_mean` - Average Gaussian ellipsoid size

---

### 3. **`visualize_gaussian.py`** (Legacy)
Geodesic distance and log map visualization using GaussianModel loading.

**Purpose**: Original script for exploring geodesic distances from a source point.

**Key Features**:
- Loads PLY via `GaussianModel` (consistent with main codebase)
- Computes geodesic distances from a source vertex
- Log map computation (tangent space mapping)
- Polyscope visualization with distance coloring

**Usage**:
```bash
# Auto-select source (closest to centroid)
python visualize_gaussian.py scene.ply

# Specify source vertex
python visualize_gaussian.py scene.ply --source-vertex 1234

# With subsampling
python visualize_gaussian.py scene.ply --limit 200000 --export results.npz
```

**Output**:
- `.npz` file with: `points`, `geodesic_distances`, `log_map`, `source_idx`
- Optional Polyscope screenshot

**Use Cases**:
- Explore geodesic distance propagation
- Validate heat method solver
- Visualize intrinsic surface geometry

---

### 4. **`visualize_open3d.py`** (Legacy)
Same as `visualize_gaussian.py` but uses Open3D for PLY loading.

**Purpose**: Alternative loading method for point clouds without Gaussian attributes.

**Differences**:
- Uses `Open3D` PLY reader instead of `GaussianModel`
- Falls back to `plyfile` if Open3D fails
- Simpler, no Gaussian-specific dependencies

**Usage**: Same as `visualize_gaussian.py`

---

## 🔧 Dependencies

### Required Packages
```bash
# Install in gaussian_splatting conda environment
conda activate gaussian_splatting

pip install torch
pip install potpourri3d  # Geodesic distance computation
pip install polyscope    # Interactive 3D visualization
pip install plyfile      # PLY file I/O
pip install open3d       # Point cloud processing (optional)
pip install scipy        # KDTree for interpolation
pip install tqdm         # Progress bars
```

### Conda Environment
```bash
# IMPORTANT: Always activate the correct environment
conda activate gaussian_splatting

# NOT the base environment!
```

---

## 📊 Workflow Example

### End-to-End Normal Estimation

```bash
# Activate correct environment
conda activate gaussian_splatting

# Step 1: Compute normals with confidence
python compute_geodesic_normals.py \
    ./data/stone_gravel_patch/stone_gravel_patch.ply \
    ./data/stone_gravel_patch/with_normals.ply \
    --subsample 1000

# Expected output:
# [INFO] Loaded 429282 points
# [INFO] Auto-estimated geodesic radius: 0.05234
# [INFO] Using Gaussian confidence sigma: 0.01745
# [INFO] Computing normals for 1000 subsampled points...
# [INFO] Inverting 123 normals with negative Y to point upward
# [INFO] Computed normals: mean magnitude = 1.000000
# [INFO] Computed confidences: mean = 0.3275, std = 0.2134
# [INFO] Saved PLY with normals to: ./data/stone_gravel_patch/with_normals.ply

# Step 2: Visualize and inspect quality
python visualize_normals.py \
    ./data/stone_gravel_patch/with_normals.ply \
    --scale 0.005 \
    --limit 2000

# Interactive GUI appears:
# - Use confidence slider to filter low-quality normals
# - Rotate view to inspect normal orientations
# - Check that normals point "upward" (Y+)
```

---

## 🧠 Key Concepts

### Geodesic Distance
The shortest path distance along the surface, not through space. More accurate for curved surfaces than Euclidean distance.

**Why use it?**
- Respects surface topology (doesn't cut through holes)
- Better neighborhood selection on curved surfaces
- More robust to noise and outliers

**Computation**: Heat method via `potpourri3d.PointCloudHeatSolver`

### Normal Estimation via SVD
Find the direction of **minimal variance** in the neighborhood:
1. Center neighborhood: `N = neighbors - point`
2. Compute covariance: `C = N.T @ N`
3. SVD decomposition: `U, S, V = svd(C)`
4. Normal = eigenvector with smallest eigenvalue = `U[:, -1]`

**Why SVD?** Robust to noise, works for arbitrary neighborhood sizes, standard in computer graphics.

### Confidence Scoring
Gaussian-weighted sum of neighbor contributions:
```
confidence = Σ exp(-distance²/(2σ²))
```

**Interpretation**:
- High confidence = dense, consistent neighborhood
- Low confidence = sparse, isolated point or edge/boundary
- Confidence = 0 = insufficient neighbors (< min_neighbors)

### Y-Up Convention
Polyscope uses Y-axis as "up" (not Z like many CAD tools).
- All normals are flipped to have `ny > 0`
- Ensures consistent upward orientation for visualization

---

## 🐛 Troubleshooting

### "Module not found" errors
```bash
# Make sure you're in the correct conda environment!
conda activate gaussian_splatting

# NOT:
conda activate base
```

### Polyscope colormap errors
```
RuntimeError: unrecognized colormap name: 'hot'
```
**Fix**: Updated to use only `'viridis'` colormap (universally supported)

### GLFW/OpenGL errors
```
[polyscope] GLFW emitted error: WGL: Failed to make context current
```
**Fix**: These are warnings, not errors. Script continues. Use `--offscreen` if needed.

### Computation too slow
```
[INFO] Computing normals for all 429282 points...
```
**Fix**: Use `--subsample 1000` for large clouds:
```bash
python compute_geodesic_normals.py input.ply output.ply --subsample 1000
```

### Normals pointing wrong direction
**Check**: Make sure Y-axis is "up" in your coordinate system. If using Z-up, edit:
```python
# In compute_geodesic_normals.py, change:
if normal[1] < 0:  # Y-up
# To:
if normal[2] < 0:  # Z-up
```

---

## 📝 File Structure

```
visualization_script/
├── README.md                      # This file
├── REFACTORING_NOTES.md          # Code cleanup history
├── compute_geodesic_normals.py   # ⭐ Main: compute normals + confidence
├── visualize_normals.py          # ⭐ Main: visualize with confidence filtering
├── visualize_gaussian.py         # Legacy: geodesic distances (GaussianModel)
└── visualize_open3d.py           # Legacy: geodesic distances (Open3D)
```

---

## 🎯 Future Improvements

### Potential Enhancements (not implemented)
1. **GPU Acceleration**: potpourri3d is CPU-only, could port heat method to CUDA
2. **Parallel Processing**: Geodesic solver is not parallelizable (stateful), but SVD could be
3. **Adaptive Radius**: Use local density estimation for per-point radius
4. **Normal Smoothing**: Post-process with bilateral filter
5. **Mesh Reconstruction**: Use normals for Poisson surface reconstruction
6. **Quality Metrics**: Compute normal consistency, smoothness scores

---

## 📚 References

- **Heat Method**: [Crane et al. 2013 - "Geodesics in Heat"](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/)
- **potpourri3d**: [GitHub - nmwsharp/potpourri3d](https://github.com/nmwsharp/potpourri3d)
- **Polyscope**: [GitHub - nmwsharp/polyscope](https://github.com/nmwsharp/polyscope)
- **Gaussian Splatting**: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

---

## 💡 Tips for New Users

1. **Always activate conda environment first**: `conda activate gaussian_splatting`
2. **Start small**: Use `--limit 10000` for quick tests
3. **Use subsampling**: `--subsample 1000` is fast enough for most QA
4. **Check confidence**: Use slider to filter out uncertain normals
5. **Visualize frequently**: Iterate on parameters with quick visualizations
6. **Save intermediate results**: Keep `.ply` files with normals for later analysis

---

## 🔗 Related Files

- `scene/gaussian_model.py` - Core Gaussian model, includes `load_gaussian_model()` utility
- `utils/*.py` - Utility functions for point cloud processing
- `data/` - Input PLY files (Gaussian splat scenes)

---

Last Updated: October 4, 2025
