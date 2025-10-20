import numpy as np
import heapq
import os
import sys

# Add current directory to Python path to find pybvh.pyd
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import pybvh

from . import graphics_utils as utils


class ExpMapQueueEntry:
    def __init__(self, v, idx, priority):
        self.v = v
        self.idx = idx
        self.priority = priority

    def dist(self):
        return np.linalg.norm(self.v)

    def __lt__(self, other):
        # return self.priority < other.priority
        return self.dist() < other.dist()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'tangent vector {self.v} at index {self.idx}, priority {self.priority}'


def adjacent_vertices(E, i):
    adjacencies0 = set(E[E[:,0] == i, 1])
    adjacencies1 = set(E[E[:,1] == i, 0])
    return adjacencies0 | adjacencies1


def generate_knn_edges(V, k, max_radius=np.inf):
    tree = pybvh.build_bvh_points(V)
    results = pybvh.knn(V, k, tree)
    edges = set()
    for i in range(V.shape[0]):
        for j in range(k):
            i_other = results[i][j].idx
            d = results[i][j].dist
            if i_other == -1 or d > max_radius:
                continue
            e0 = min(i,i_other)
            e1 = max(i,i_other)
            edges.add((e0, e1))
    E = np.array(list(edges), dtype='int32')
    return E


# Compute the exponential map from V[root_idx,:] to every other point in V, propagated along edges E
def discrete_exp_map(V, E, N, root_idx, add_locally=False):
    n = N[root_idx,:]
    random_dir = np.random.rand(3)
    t0 = random_dir - np.dot(random_dir, n)*n
    t0 = t0 / np.linalg.norm(t0) # if this ends up being nan then we somehow ended up with something really close to the normal
    t1 = np.cross(n, t0)

    # make an empty priority queue and visited set
    traversal_queue = [ExpMapQueueEntry(np.zeros((2)), root_idx, 0)]
    exp_map = np.zeros((V.shape[0], 2))
    weights = np.zeros(V.shape[0])
    bases = np.zeros_like(V)
    visited = set()

    bases[root_idx,:] = t0
    
    first_entry = None
    while len(traversal_queue) > 0:
        entry = heapq.heappop(traversal_queue)
        if entry.idx in visited:
            continue
        visited.add(entry.idx)
        if weights[entry.idx] > 0:
            exp_map[entry.idx,:] /= weights[entry.idx]
            if add_locally:
                bases[entry.idx,:] /= weights[entry.idx]

        t0_i = bases[entry.idx,:]
        t0_i = t0_i / (np.linalg.norm(t0_i) + 1e-8)
        n_i = N[entry.idx,:]
        t1_i = np.cross(n_i, t0_i)

        # find adjacent vertices that are unvisited
        traversal_front = adjacent_vertices(E, entry.idx) - visited

        for j in traversal_front:
            # we have the chance to visit a vertex several times before it's actually dequeued, so we can do upwinding through the traversal
            # keep track of total weight and 

            # also need to track which items have been queued already
            if first_entry is None:
                first_entry = j

            # compute vector going from i to each adjacent (unvisited) vertex
            vj_embedded = V[j,:] - V[entry.idx,:]
            dist_ij = np.linalg.norm(vj_embedded)
            weight = 1/(dist_ij+1e-8)
            # then compute vj by projecting onto normal at entry's tangent plane, and rescale to have the same length
            vj_projected = vj_embedded - np.dot(n_i, vj_embedded)*n_i
            vj = vj_projected / (np.linalg.norm(vj_projected) + 1e-8) * dist_ij

            vi = entry.v
            if add_locally:
                # vi is in space i, so we can just directly add it to j
                vj_u = np.dot(vj, t0_i)
                vj_v = np.dot(vj, t1_i)
                vj_uv = vi + np.stack((vj_u, vj_v), axis=0)

                # however, we now need to compute the new basis vector, which we do by rotating t0_i into j space
                n_j = N[j,:]
                rotation_axis_unnormalized = np.cross(n_i, n_j)
                rotation_axis = rotation_axis_unnormalized / (np.linalg.norm(rotation_axis_unnormalized) + 1e-8)
                rotation_angle = np.arccos(np.clip(np.dot(n_i, n_j), -1, 1))
                Rij = utils.rodrigues_numpy(rotation_axis[None,:], rotation_angle)
                t0_j = (t0_i[None,:] @ Rij.T).flatten()
            else:
                # otherwise, vi is in the space of root_idx, so we need to rotate vj into root_idx's space
                rotation_axis_unnormalized = np.cross(n_i, n)
                rotation_axis = rotation_axis_unnormalized / (np.linalg.norm(rotation_axis_unnormalized) + 1e-8)
                rotation_angle = np.arccos(np.clip(np.dot(n_i, n), -1, 1))
                Ri = utils.rodrigues_numpy(rotation_axis[None,:], rotation_angle)
                vj_root = (vj[None,:] @ Ri.T).flatten()
                vj_u = np.dot(vj_root, t0)
                vj_v = np.dot(vj_root, t1)
                vj_uv = vi + np.stack((vj_u, vj_v), axis=0)

            # NOTE: the priority being the scalar sum of distances so far is very important, as the results deteriorate quite quickly otherwise
            # not quite sure why yet...perhaps because we need to defer the upwind average for when we have enough information?
            # I really should try to examine this in more detail, seems worth investigating
            heapq.heappush(traversal_queue, ExpMapQueueEntry(vj_uv, j, entry.priority + dist_ij))
            exp_map[j,:] += weight * vj_uv
            weights[j] += weight
            if add_locally:
                bases[j,:] += weight * t0_j

    return exp_map
