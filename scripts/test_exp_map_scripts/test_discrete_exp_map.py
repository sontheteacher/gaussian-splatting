import gpytoolbox as gpy
import polyscope as ps
import numpy as np
import cmocean
import sys
import os

# Add the parent directory to the path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import utils.discrete_exp_map as dem
import utils.graphics_utils as utils

# Create a simple sphere mesh for testing
V, F = gpy.icosphere(3)  # Create an icosphere with 3 subdivisions
N = gpy.per_vertex_normals(V, F)
E = gpy.edges(F)
root_idx = 0  # Use the first vertex as root
exp_map = dem.discrete_exp_map(V, E, N, root_idx, add_locally=True)

# V, _, N, _ =  gpy.read_mesh('points.obj', return_N=True)
# # V, _, N, _ =  gpy.read_mesh('bundle_points.obj', return_N=True)
# edge_max_radius = 5e-2
# E = dem.generate_knn_edges(V, k=10, max_radius=edge_max_radius*edge_max_radius)

# root_idx = 0
# freq = 5
# contrast = 0.8
# exp_map = dem.discrete_exp_map(V, E, N, root_idx, add_locally=False)
# map_color = utils.parameterization_dartboard(exp_map, num_cycles=freq, contrast=0.8)
# # exp_map_radius = np.linalg.norm(exp_map, axis=1)
# # exp_map_angle = np.arctan2(exp_map[:,0], exp_map[:,1])
# # max_radius = np.max(exp_map_radius)
# # num_cycles = 2
# # period = max_radius/num_cycles

# # lattice_coords = np.remainder(np.abs(np.floor(exp_map / period)), 2)
# # checkerboard_active = 0.8*np.remainder(np.sum(lattice_coords, axis=1, keepdims=True), 2) + 0.2
# # exp_map_angle_rescaled = (exp_map_angle + np.pi) / (2*np.pi)
# # phase_map = cmocean.cm.phase
# # map_color = checkerboard_active * phase_map(exp_map_angle_rescaled)[:,:3]

ps.init()

ps.register_point_cloud('origin', V[root_idx,:][None,:])

ps_mesh = ps.register_surface_mesh('mesh', V, F)
ps_mesh.add_parameterization_quantity('log map', exp_map)

# ps_mesh = ps.register_point_cloud('point cloud', V)
# ps_mesh.add_color_quantity('log map', map_color)

ps.show()
