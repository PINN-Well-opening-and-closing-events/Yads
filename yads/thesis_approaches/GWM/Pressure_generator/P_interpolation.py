import numpy as np
from yads.thesis_approaches.GWM.Pressure_generator.dists import circle_dist
from itertools import cycle


def P_interp(P, circle_coords, grid):
    face_center_coords = []
    for group in grid.face_groups:
        if group != "0":
            cell_idxs = grid.face_groups[group][:, 0]
            for coord in grid.centers(item="face")[cell_idxs]:
                face_center_coords.append(list(coord))
    nb_bd_faces = grid.nb_boundary_faces
    Nx = Ny = nb_bd_faces / 4
    Lx = Ly = grid.measures(item="face")[0] * Nx
    radius = Lx / 2

    denom = np.sqrt((np.array(face_center_coords)[:, 0] - Lx / 2) ** 2 + (
            np.array(face_center_coords)[:, 1] - Ly / 2) ** 2)
    circle_coord_x = (np.array(face_center_coords)[:, 0] - Lx / 2) * radius * np.sqrt(2) / denom
    circle_coord_y = (np.array(face_center_coords)[:, 1] - Ly / 2) * radius * np.sqrt(2) / denom
    all_circle_coords = np.array([[circle_coord_x[i], circle_coord_y[i]] for i in range(len(circle_coord_x))])

    init_index = None
    for i, coord in enumerate(all_circle_coords):
        if coord[0] == circle_coords[0][0] and coord[1] == circle_coords[0][1]:
            init_index = i
            break

    min_dist = 2 * np.pi * radius * np.sqrt(2) / nb_bd_faces
    sorted_circle_coords = np.concatenate([all_circle_coords[init_index:], all_circle_coords[:init_index + 1]])
    linear_coords = []

    for i, _ in enumerate(sorted_circle_coords):
        lin_coord = i*min_dist
        linear_coords.append(lin_coord)
    linear_coords = np.array(linear_coords)
    idxs_to_interp = []
    idxs_not_to_interp = []

    for i, coord_1 in enumerate(sorted_circle_coords):
        is_not_in_coords = 0
        for coord_2 in circle_coords:
            if coord_1[0] != coord_2[0] or coord_1[1] != coord_2[1]:
                is_not_in_coords += 1
        if is_not_in_coords == len(P):
            idxs_to_interp.append(i)
        else:
            idxs_not_to_interp.append(i)

    coords_to_interp = linear_coords[idxs_to_interp]
    coords_not_to_interp = linear_coords[idxs_not_to_interp]
    P = np.concatenate([P, [P[0]]])
    np.interp(coords_to_interp, coords_not_to_interp, P)

    return np.concatenate([np.interp(coords_to_interp, coords_not_to_interp, P), P]), \
           np.concatenate([coords_to_interp, coords_not_to_interp])
