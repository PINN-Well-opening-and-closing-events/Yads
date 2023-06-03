import copy
import os
import numpy as np

from yads.thesis_approaches.GWM.Pressure_generator.P_interpolation import P_interp


def collinear(p0, p1, p2, eps=1e-6):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < eps


def circle_to_cart_coord(grid, circle_coords):
    nb_bd_faces = grid.nb_boundary_faces
    Nx = nb_bd_faces / 4
    Lx = grid.measures(item="face")[0] * Nx

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


def create_Pb_groups(grid, P_inter, circle_coords):
    nb_bd_faces = grid.nb_boundary_faces
    Nx = nb_bd_faces / 4
    Lx = grid.measures(item="face")[0] * Nx
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
        Pb_dict[f"boundary_face_{i}"] = P_inter[i]
        group_by_line = (f"boundary_face_{i}", line_point_1, line_point_2)
        groups.append(group_by_line)
    return groups, Pb_dict


def P_interp_to_P_imp(grid, P, circle_coords, savepath):
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    for i in range(P.shape[0]):
        grid_temp = copy.deepcopy(grid)
        (P_inter, P_no_inter), _, (circle_interp, circle_no_interp) = P_interp(
            P[i, :], circle_coords[i, :], grid=grid_temp
        )
        all_P = np.concatenate([P_inter, P_no_inter[:-1]])
        all_circles_coords = np.concatenate([circle_interp, circle_no_interp[:-1]])
        groups, Pb_dict = create_Pb_groups(grid_temp, all_P, all_circles_coords)
        # TODO: save in appropriate format
    return
