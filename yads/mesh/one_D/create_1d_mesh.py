import numpy as np  # type: ignore
from typing import Tuple

from yads.mesh.mesh import Mesh
from yads.mesh.meshdata import MeshData


def create_1d(nb_cells: int, interval: Tuple[float, float]) -> Mesh:
    """creates a Mesh object representing a 1D grid
    Args:
        nb_cells: number of cells. integer > 0
        interval: cell interval. tuple of float [L, U} such that L < U

    Returns:
        Mesh object
    """
    if nb_cells <= 0:
        raise ValueError(f"nb_cells must be an integer > 0 (got: {nb_cells}")
    if interval[0] >= interval[1]:
        raise ValueError(
            f"interval must be a tuple of float [L, U] such that L < U (got: {interval}"
        )

    nb_nodes = nb_faces = nb_cells + 1

    node_coordinates = np.linspace(*interval, nb_nodes)
    face_centers = node_coordinates
    cell_centers = 0.5 * (node_coordinates[1:] + node_coordinates[:-1])
    face_measures = np.ones((nb_faces,))
    cell_measures = node_coordinates[1:] - node_coordinates[:-1]

    face_groups = {
        "0": np.array(
            list(
                zip(
                    range(1, nb_faces - 1),
                    range(0, nb_cells - 1),
                    range(1, nb_cells),
                )
            )
        ),
        "1": np.array([[0, 0]]),
        "2": np.array([[nb_faces - 1, nb_cells - 1]]),
    }
    cells = np.array(list(zip(range(0, nb_cells), range(1, nb_cells))))
    # inner connectivity
    cell_face_connectivity = [
        [tuple(cell), face] for cell, face in zip(cells, range(1, nb_faces + 1))
    ]
    # boundary connectivity
    cell_face_connectivity.append([(0, None), 0])
    cell_face_connectivity.append([(nb_cells - 1, None), nb_faces - 1])

    return MeshData(
        dim=1,
        nb_cells=nb_cells,
        nb_nodes=nb_nodes,
        nb_faces=nb_faces,
        cell_centers=cell_centers,
        face_centers=face_centers,
        node_coordinates=node_coordinates,
        face_measures=face_measures,
        cell_measures=cell_measures,
        face_groups=face_groups,
        cell_face_connectivity=cell_face_connectivity,
        cell_node_connectivity=cell_face_connectivity,
        cells=cells,
        faces=None,
    )
