import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from yads.numerics.physics import calculate_transmissivity
from yads.numerics.solvers.implicit_pressure_solver import implicit_pressure_solver

from yads.thesis_approaches.GWM.Pressure_generator.P_generator import (
    P_generator_wrapper,
)
from yads.thesis_approaches.GWM.Pressure_generator.cov_mats import cov_matrix_P_dist
from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian

from yads.thesis_approaches.GWM.Pressure_generator.P_interpolation import P_interp


def collinear(p0, p1, p2, eps=1e-6):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < eps


def circle_to_cart_coord(grid, circle_coords):
    face_center_coords = []
    for group in grid.face_groups:
        if group in ["left", "right", "lower", "upper"]:
            cell_idxs = grid.face_groups[group][:, 0]
            for coord in grid.centers(item="face")[cell_idxs]:
                face_center_coords.append(list(coord))
    cart_coords = []
    for cir in circle_coords:
        for car in face_center_coords:
            car_centered = np.array(car) - Lx / 2
            if collinear([0, 0], cir, car_centered):
                if np.sign(car_centered[0]) == np.sign(cir[0]) and np.sign(
                    car_centered[1]
                ) == np.sign(cir[1]):
                    cart_coords.append(car_centered)
    return np.array(cart_coords)


def create_Pb_groups(grid, P_interp, circle_coords):
    cart_coords = circle_to_cart_coord(grid, circle_coords)
    # add by line all groups to grid:
    groups = []
    Pb_dict = {}
    face_length = Lx / (grid.nb_boundary_faces / 4)
    for i, coord in enumerate(cart_coords):
        if np.abs(coord[0]) == Lx / 2:
            line_point_1 = (coord[0] + Lx / 2, coord[1] + Lx / 2 - face_length / 2)
            line_point_2 = (coord[0] + Lx / 2, coord[1] + Lx / 2 + face_length / 2)
        else:
            line_point_1 = (coord[0] + Lx / 2 - face_length / 2, coord[1] + Lx / 2)
            line_point_2 = (coord[0] + Lx / 2 + face_length / 2, coord[1] + Lx / 2)
        Pb_dict[f"boundary_face_{i}"] = P_interp[i]
        group_by_line = (f"boundary_face_{i}", line_point_1, line_point_2)
        groups.append(group_by_line)
    return groups, Pb_dict


if __name__ == "__main__":
    Lx, Ly = 3, 3
    Nx, Ny = 20, 20

    nb_samples = 2
    radius = Lx / 2
    nb_boundaries = 2

    P_min = 10e6
    P_max = 20e6
    grid = create_2d_cartesian(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)

    cor_ds = [3.0 / grid.nb_boundary_faces, 3]
    seed = 7

    P1, cartesian_coords, circle_coords = P_generator_wrapper(
        grid=grid,
        nb_boundaries=nb_boundaries,
        nb_samples=nb_samples,
        cov_mat=cov_matrix_P_dist,
        cor_ds=cor_ds,
        P_min=P_min,
        P_max=P_max,
        seed=seed,
    )

    (
        (P_interp, P),
        (x_interp, x),
        (circle_interp, circle_no_interp),
        (cart_inter, cart_no_inter),
    ) = P_interp(P1[0, :], circle_coords, grid=grid)

    all_P = np.concatenate([P_interp, P[:-1]])
    all_circles_coords = np.concatenate([circle_interp, circle_no_interp[:-1]])
    groups, Pb_dict = create_Pb_groups(grid, all_P, all_circles_coords)
    for group in groups:
        grid.add_face_group_by_line(*group)

    phi = 0.2

    # Porosity
    phi = np.full(grid.nb_cells, phi)
    # Diffusion coefficient (i.e Permeability)
    K = np.full(grid.nb_cells, 100.0e-15)

    T = calculate_transmissivity(grid, K)
    # gaz saturation initialization
    S = np.full(grid.nb_cells, 0.0)
    # Pressure initialization
    P = np.full(grid.nb_cells, 100.0e5)
    mu_w = 0.571e-3
    mu_g = 0.0285e-3

    kr_model = "quadratic"
    # BOUNDARY CONDITIONS #

    # Saturation
    Sb_d = copy.deepcopy(Pb_dict)
    Sb_n = copy.deepcopy(Pb_dict)
    for group in Sb_d.keys():
        Sb_d[group] = 0.0
        Sb_n[group] = None
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}
    P_imp = implicit_pressure_solver(
        grid=grid, K=K, T=T, P=P, S=S, Pb=Pb_dict, Sb_dict=Sb_dict, mu_g=mu_g, mu_w=mu_w
    )

    fig, ax1 = plt.subplots(1, 1)
    cell_centers = grid.centers(item="cell")
    pos = ax1.imshow(P_imp.reshape(Nx, Ny))
    fig.colorbar(pos, ax=ax1)
    plt.show()
