import numpy as np


def angle_between_points(A, B):
    return np.abs(np.arctan2(B[1], B[0]) - np.arctan2(A[1], A[0]))


def circle_dist(A, B, radius):
    angle = angle_between_points(A, B)
    return angle * radius


def dist(X1, X2):
    return (X1 - X2) ** 2


def compute_circle_coord(grid):
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

    denom = np.sqrt(
        (np.array(face_center_coords)[:, 0] - Lx / 2) ** 2
        + (np.array(face_center_coords)[:, 1] - Ly / 2) ** 2
    )
    circle_coord_x = (
        (np.array(face_center_coords)[:, 0] - Lx / 2) * radius * np.sqrt(2) / denom
    )
    circle_coord_y = (
        (np.array(face_center_coords)[:, 1] - Ly / 2) * radius * np.sqrt(2) / denom
    )
    all_circle_coords = np.array(
        [[circle_coord_x[i], circle_coord_y[i]] for i in range(len(circle_coord_x))]
    )
    return all_circle_coords, radius
