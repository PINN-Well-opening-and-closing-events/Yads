import copy
from typing import List

import random
import numpy as np
from pyDOE import lhs
from yads.mesh import Mesh
from yads.thesis_approaches.GWM.Pressure_generator.P_interp_to_P_imp import (
    P_interp_to_P_imp,
)
from yads.thesis_approaches.GWM.Pressure_generator.dists import circle_dist


def P_generator(lhd, coords, cov_mat_fun, cor_d, radius, P_min=10e6, P_max=20e6):
    all_P_scaled = []
    # square mesh

    P_unscaled = np.matmul(
        cov_mat_fun(
            X=coords, lhd=lhd, dist_func=circle_dist, cor_d=cor_d, radius=radius,
        ),
        lhd,
    )
    P_scaled = P_min + P_unscaled * (P_max - P_min)
    all_P_scaled.append(P_scaled)
    return np.array(all_P_scaled)


def P_generator_wrapper(
    grid: Mesh,
    nb_boundaries: int,
    nb_samples: int,
    cov_mat,
    cor_ds: List[float],
    P_min,
    P_max,
    seed=42,
):
    np.random.seed(seed)
    random.seed(seed)

    # get mesh information
    nb_bd_faces = grid.nb_boundary_faces
    Nx = nb_bd_faces / 4
    Lx = Ly = grid.measures(item="face")[0] * Nx
    radius = Lx / 2

    # get center of boundary faces
    face_center_coords = []
    for group in grid.face_groups:
        if group in ["left", "right", "upper", "lower"]:
            cell_idxs = grid.face_groups[group][:, 0]
            for coord in grid.centers(item="face")[cell_idxs]:
                face_center_coords.append(list(coord))

    # generate lhs samples and boundaries
    lhd = lhs(n=nb_boundaries, samples=nb_samples, criterion="maximin")
    rng = np.random.RandomState(seed)

    boundaries = np.empty((nb_samples, nb_boundaries), dtype=int)
    for i in range(nb_samples):
        boundaries[i] = rng.choice(range(4), size=nb_boundaries, replace=False)
    random_faces = np.random.randint(0, Nx, size=(nb_samples, nb_boundaries))

    # care !
    groups = list(grid.face_groups.keys())[1:]
    rd_faces_coord = np.empty((nb_samples, nb_boundaries), dtype=object)
    for i in range(nb_samples):
        for j, bound in enumerate(boundaries[i]):
            rd_face_idx = grid.face_groups[groups[bound]][random_faces[i, j]][0]
            rd_face_coord = grid.centers(item="face")[rd_face_idx]
            rd_faces_coord[i, j] = rd_face_coord

    circle_coords = np.empty((nb_samples, nb_boundaries), dtype=object)
    for i in range(nb_samples):
        for j in range(nb_boundaries):
            denom = np.sqrt(
                (np.array(rd_faces_coord)[i, j][0] - Lx / 2) ** 2
                + (np.array(rd_faces_coord)[i, j][1] - Ly / 2) ** 2
            )

            circle_coord_x = (
                (np.array(rd_faces_coord)[i, j][0] - Lx / 2)
                * radius
                * np.sqrt(2)
                / denom
            )
            circle_coord_y = (
                (np.array(rd_faces_coord)[i, j][1] - Ly / 2)
                * radius
                * np.sqrt(2)
                / denom
            )
            circle_coords[i, j] = np.array([circle_coord_x, circle_coord_y])

    P = np.empty((nb_samples, nb_boundaries), dtype=float)
    for i in range(nb_samples):
        P[i] = P_generator(
            lhd=lhd[i, :],
            coords=circle_coords[i, :],
            cov_mat_fun=cov_mat,
            radius=radius,
            P_min=P_min,
            P_max=P_max,
            cor_d=cor_ds,
        )
    return P, rd_faces_coord, circle_coords


def P_imp_generator(
    grid, nb_samples, nb_boundaries, P_min, P_max, cov_mat, cor_ds, seed, savepath
):
    P, cart_coords, circle_coords = P_generator_wrapper(
        grid, nb_boundaries, nb_samples, cov_mat, cor_ds, P_min, P_max, seed,
    )
    P_interp_to_P_imp(grid, P, circle_coords, savepath)
    return
