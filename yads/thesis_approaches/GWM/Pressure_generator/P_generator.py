from typing import List

import random
import numpy as np
from pyDOE import lhs
from yads.mesh import Mesh
from yads.thesis_approaches.GWM.Pressure_generator.dists import circle_dist


def P_generator(lhd, coords, cov_mat_fun, cor_d, radius, P_min=10e6, P_max=20e6):
    all_P_scaled = []
    # square mesh
    for i in range(lhd.shape[0]):
        P_unscaled = np.matmul(cov_mat_fun(X=coords, lhd=lhd[i, :], dist_func=circle_dist, cor_d=cor_d, radius=radius,
                                           ), lhd[i, :])
        P_scaled = P_min + P_unscaled * (P_max - P_min)
        all_P_scaled.append(P_scaled)
    return np.array(all_P_scaled)


def P_generator_wrapper(grid: Mesh, nb_boundaries: int, nb_samples: int, cov_mat, cor_ds: List[float], P_min, P_max,
                        seed=42):
    np.random.seed(seed)
    random.seed(seed)

    nb_bd_faces = grid.nb_boundary_faces
    Nx = Ny = nb_bd_faces / 4
    Lx = Ly = grid.measures(item="face")[0] * Nx
    radius = Lx / 2

    # get center of boundary faces
    face_center_coords = []
    for group in grid.face_groups:
        if group != "0":
            cell_idxs = grid.face_groups[group][:, 0]
            for coord in grid.centers(item="face")[cell_idxs]:
                face_center_coords.append(list(coord))

    boundaries = random.sample(range(4), nb_boundaries)
    random_faces = np.random.randint(0, Nx, size=len(boundaries))
    rd_faces_coord = []
    groups = list(grid.face_groups.keys())[1:]
    for i, bound in enumerate(boundaries):
        rd_face_idx = grid.face_groups[groups[bound]][random_faces[i]][0]
        rd_face_coord = grid.centers(item="face")[rd_face_idx]
        rd_faces_coord.append(rd_face_coord)

    denom = np.sqrt((np.array(rd_faces_coord)[:, 0] - Lx / 2) ** 2 + (
                             np.array(rd_faces_coord)[:, 1] - Ly / 2) ** 2)
    circle_coord_x = (np.array(rd_faces_coord)[:, 0] - Lx / 2) * radius * np.sqrt(2) / denom
    circle_coord_y = (np.array(rd_faces_coord)[:, 1] - Ly / 2) * radius * np.sqrt(2) / denom

    circle_coords = np.array([[circle_coord_x[i], circle_coord_y[i]] for i in range(len(circle_coord_x))])

    lhd = lhs(n=nb_boundaries, samples=nb_samples, criterion="maximin")

    P = P_generator(lhd=lhd,
                    coords=circle_coords,
                    cov_mat_fun=cov_mat,
                    radius=radius,
                    P_min=P_min, P_max=P_max, cor_d=cor_ds)

    return P, rd_faces_coord, circle_coords
