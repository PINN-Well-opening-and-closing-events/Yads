import numpy as np

from yads.mesh import Mesh
from yads.thesis_approaches.GWM.Pressure_generator.dists import (
    circle_dist,
    compute_circle_coord,
)
from itertools import cycle
import math


def P_interp(P, circle_coords, grid: Mesh):
    all_circle_coords, radius = compute_circle_coord(grid=grid)
    sorted_all_circle_coords, _ = sort_clockwise(all_circle_coords)
    linear_coords = []
    dists = []

    for i in range(len(sorted_all_circle_coords) - 1):
        new_dist = circle_dist(
            sorted_all_circle_coords[i],
            sorted_all_circle_coords[i + 1],
            radius=radius * np.sqrt(2),
        )
        if i == 0:
            sum_dist = 0
        else:
            sum_dist = np.sum(dists)
        lin_coord = sum_dist + new_dist
        dists.append(new_dist)
        linear_coords.append(lin_coord)

    last_dist = 2 * np.pi * radius * np.sqrt(2) - circle_dist(
        sorted_all_circle_coords[-1],
        sorted_all_circle_coords[0],
        radius=radius * np.sqrt(2),
    )
    linear_coords.append(np.sum(dists) + last_dist)
    dists.append(last_dist)
    linear_coords.append(0)
    linear_coords = np.array(linear_coords)

    idxs_to_interp = []
    idxs_not_to_interp = []

    def is_element_in_array(element, array):
        for row in array:
            if np.array_equal(row, element):
                return True
        return False

    for i, coord_1 in enumerate(sorted_all_circle_coords):
        if is_element_in_array(coord_1, circle_coords):
            idxs_not_to_interp.append(i)
        else:
            idxs_to_interp.append(i)

    sorted_all_circle_coords = np.concatenate(
        [sorted_all_circle_coords, [sorted_all_circle_coords[0]]]
    )

    coords_to_interp = linear_coords[idxs_to_interp]
    coords_not_to_interp = linear_coords[idxs_not_to_interp]

    circle_coords_to_interp = sorted_all_circle_coords[idxs_to_interp]
    circle_coords_not_to_interp = sorted_all_circle_coords[idxs_not_to_interp]

    _, P_idxs = sort_clockwise(circle_coords)

    circle_coords_not_to_interp = np.concatenate(
        [circle_coords_not_to_interp, [circle_coords_not_to_interp[0]]]
    )
    coords_not_to_interp = np.concatenate(
        [coords_not_to_interp, [coords_not_to_interp[0]]]
    )

    P_no_interp = np.concatenate([P[P_idxs], [P[P_idxs][0]]])
    P_inter = np.interp(
        coords_to_interp,
        coords_not_to_interp,
        P_no_interp,
        period=2 * np.pi * radius * np.sqrt(2),
    )

    return (
        (P_inter, P_no_interp),
        (coords_to_interp, coords_not_to_interp),
        (circle_coords_to_interp, circle_coords_not_to_interp),
        (idxs_to_interp, idxs_not_to_interp),
    )


def sort_clockwise(points):
    polar_points = []
    for point in points:
        x = point[0]
        y = point[1]
        angle = np.arctan2(y, x)
        polar_points.append((point, angle))

    points_tries = sorted(polar_points, key=lambda p: p[1])
    idxs = np.argsort(np.array(polar_points)[:, 1])
    points_result = [p[0] for p in points_tries]

    return np.array(points_result), idxs
