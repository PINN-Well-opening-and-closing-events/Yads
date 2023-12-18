import copy
import os
import numpy as np
import pickle

from yads.mesh import Mesh
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


def add_rotations(grid, groups):
    # TODO: sort groups clockwise
    left, right, up, down = [], [], [], []
    max_x = max(grid.node_coordinates[:, 0])
    max_y = max(grid.node_coordinates[:, 1])

    for group in groups:
        # upper right
        center = (np.array(group[1]) + np.array(group[2]))/2
        if center[0] == max_x:
            right.append([center, group])
        # upper left
        elif center[0] == 0.:
            left.append([center, group])
        # lower right
        elif center[1] == max_y:
            up.append([center, group])
        # lower left
        elif center[1] == 0.:
            down.append([center, group])

    x_up = [center[0] for center, _ in up]
    x_down = [center[0] for center, _ in down]
    y_left = [center[1] for center, _ in left]
    y_right = [center[1] for center, _ in right]
    up_idxs = np.flip(np.argsort(x_up))
    down_idxs = np.argsort(x_down)
    right_idxs = np.argsort(y_right)
    left_idxs = np.flip(np.argsort(y_left))

    # everything sorted
    groups_sorted = [down[i] for i in down_idxs] + [right[i] for i in right_idxs] +\
                    [up[i] for i in up_idxs] + [left[i] for i in left_idxs]

    # add rotations
    def make_rotation(sorted_groups):
        group_names = [name for name, _, _ in sorted_groups]
        group_names = np.concatenate([[group_names[-1]], group_names[:-1]])
        rotated_sorted_groups = []
        for i, (_, n1, n2) in enumerate(sorted_groups):
            rotated_sorted_groups.append((group_names[i], n1, n2))
        return rotated_sorted_groups

    groups_sorted = [group for center, group in groups_sorted]
    all_groups = [groups_sorted]
    num_rota = len(groups_sorted) - 1
    for r in range(num_rota):
        all_groups.append(make_rotation(all_groups[-1]))
    return all_groups


def P_interp_to_P_imp(grid: Mesh, P, circle_coords, savepath):
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
        all_groups = add_rotations(grid=grid_temp, groups=groups)
        folder_name = f"{i}"
        if not os.path.isdir(savepath + "/" + folder_name):
            os.mkdir(savepath + "/" + folder_name)
        for r, group in enumerate(all_groups):
            with open(savepath + "/" + folder_name + f"/{i}_rotation_{r}.pkl", "wb") as f:
                pickle.dump((group, Pb_dict), f)
    return
