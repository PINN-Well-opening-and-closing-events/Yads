from yads.mesh import Mesh
import numpy as np  # type: ignore

# import jax.numpy as jnp
from yads.numerics.utils import get_dist

from typing import Union


def calculate_transmissivity_1d(grid: Mesh, K: np.ndarray) -> np.ndarray:
    """calculates transmissivity on each face of a given one_D grid

    Args:
        grid: Mesh object
        K: Permeability of each cell in grid np.ndarray size grid.nb_cells

    Returns:
         transmissivity np.ndarray size grid.nb_faces
    """

    if not isinstance(grid, Mesh):
        raise TypeError(f"grid must be a Mesh object (got {type(grid)})")
    if not isinstance(K, np.ndarray):
        raise TypeError(f"K must be a np.ndarray (got {type(K)})")

    face_centers = grid.centers(item="face")
    cell_centers = grid.centers(item="cell")
    face_measures = grid.measures(item="face")

    T = np.full(grid.nb_faces, None)
    for f, i, j in grid.faces(group="0", with_nodes=True):
        d_if = abs(cell_centers[i] - face_centers[f])
        d_jf = abs(cell_centers[j] - face_centers[f])
        T[f] = face_measures[f] * K[i] * K[j] / (d_jf * K[i] + d_if * K[j])

    # Boundaries
    for bound in ["1", "2"]:
        for f, i in grid.faces(group=bound, with_nodes=True):
            d_f = abs(cell_centers[i] - face_centers[f])

            T[f] = face_measures[f] * K[i] / d_f
    return T


def calculate_transmissivity_2d(grid: Mesh, K: np.ndarray) -> np.ndarray:
    """calculates transmissivity on each face of a given two_D grid

    Args:
        grid: Mesh object
        K: Permeability of each cell in grid np.ndarray size grid.nb_cells

    Returns:
         transmissivity np.ndarray size grid.nb_faces
    """

    if not isinstance(grid, Mesh):
        raise TypeError(f"grid must be a Mesh object (got {type(grid)})")
    if not isinstance(K, np.ndarray):
        raise TypeError(f"K must be a np.ndarray (got {type(K)})")

    face_centers = grid.centers(item="face")
    cell_centers = grid.centers(item="cell")
    face_measures = grid.measures(item="face")
    T = np.full(grid.nb_faces, None)

    for bound in grid.face_groups.keys():
        # inner faces
        if bound == "0":
            for f in grid.faces(group=bound, with_nodes=False):
                # find the two adjacent cells
                i, j = grid.face_to_cell(f, face_type="inner")
                d_if = get_dist(cell_centers[i], face_centers[f])
                d_jf = get_dist(cell_centers[j], face_centers[f])
                T[f] = face_measures[f] * K[i] * K[j] / (d_jf * K[i] + d_if * K[j])

        # boundary faces
        else:
            for f, i, j in grid.faces(group=bound, with_nodes=True):
                # find the only adjacent cell
                c = grid.face_to_cell(f, face_type="boundary")
                d_f = get_dist(cell_centers[c], face_centers[f])
                T[f] = face_measures[f] * K[c] / d_f
    return T


def calculate_transmissivity(grid: Mesh, K: np.ndarray) -> np.ndarray:
    if grid.dim == 1:
        return calculate_transmissivity_1d(grid, K)
    elif grid.dim == 2:
        return calculate_transmissivity_2d(grid, K)
    else:
        NotImplementedError("Dimension not supported yet")
