import numpy as np  # type: ignore
from typing import Tuple, Mapping, Union, List, Dict

from yads.mesh import Mesh, MeshData


def get_cells_cartesian_2d(Nx: int, Ny: int) -> np.ndarray:
    cells = []
    for i in range(Nx):
        for j in range(Ny):
            # [down_left, down_right, upper_left, upper_right]
            cells.append(
                [
                    i + j * (Nx + 1),
                    i + j * (Nx + 1) + 1,
                    i + (j + 1) * (Nx + 1),
                    i + (j + 1) * (Nx + 1) + 1,
                ]
            )

    return np.array(cells)


def get_faces_cartesian_2d(Nx: int, Ny: int) -> np.ndarray:
    faces = []
    for i in range(Nx):
        for j in range(Ny + 1):
            # horizontal faces
            faces.append([i + j * (Nx + 1), i + j * (Nx + 1) + 1])

    for i in range(Nx + 1):
        for j in range(Ny):
            # vertical faces
            faces.append([i + j * (Nx + 1), i + (j + 1) * (Nx + 1)])
    return np.array(faces)


def get_cell_centers_cartesian_2d(cells, node_coordinates) -> np.ndarray:
    cell_centers = []
    for cell in cells:
        x_center = (node_coordinates[cell[0]][0] + node_coordinates[cell[1]][0]) / 2.0
        y_center = (node_coordinates[cell[0]][1] + node_coordinates[cell[2]][1]) / 2.0
        cell_centers.append([x_center, y_center])
    return np.array(cell_centers)


def get_face_centers_cartesian_2d(faces, node_coordinates) -> np.ndarray:
    face_centers = []
    for face in faces:
        x_center = (node_coordinates[face[0]][0] + node_coordinates[face[1]][0]) / 2.0
        y_center = (node_coordinates[face[0]][1] + node_coordinates[face[1]][1]) / 2.0
        face_centers.append([x_center, y_center])
    return np.array(face_centers)


def get_node_coordinates_cartesian_2d(
    Nx: int, Ny: int, dx: Union[float, int], dy: Union[float, int]
) -> np.ndarray:
    node_coordinates = []
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            node_coordinates.append([i * dx, j * dy])

    return np.array(node_coordinates)


def get_face_groups_cartesian_2d(faces, node_coordinates, Lx, Ly):
    face_groups = {}
    left_group = []
    right_group = []
    inner_group = []
    lower_group = []
    upper_group = []
    for i, face in enumerate(faces):
        # left : group "left"
        if node_coordinates[face[0]][0] == 0.0 and node_coordinates[face[1]][0] == 0.0:
            left_group.append([i, face[0], face[1]])
        # right : group "right"
        elif node_coordinates[face[0]][0] == Lx and node_coordinates[face[1]][0] == Lx:
            right_group.append([i, face[0], face[1]])
        # upper : group "upper"
        elif node_coordinates[face[0]][1] == Ly and node_coordinates[face[1]][1] == Ly:
            upper_group.append([i, face[0], face[1]])
        # lower : group "lower"
        elif (
            node_coordinates[face[0]][1] == 0.0 and node_coordinates[face[1]][1] == 0.0
        ):
            lower_group.append([i, face[0], face[1]])
        # inner : group "0"
        else:
            inner_group.append([i, face[0], face[1]])

    face_groups["0"] = np.array(inner_group)
    face_groups["lower"] = np.array(lower_group)
    face_groups["upper"] = np.array(upper_group)
    face_groups["right"] = np.array(right_group)
    face_groups["left"] = np.array(left_group)

    return face_groups


def get_cell_face_node_connectivity(
    cells: np.ndarray, faces: np.ndarray
) -> Tuple[List, List]:
    """Computes the connectivity between cells.
    There are multiples cases:
        1) There is no connexion
        2) There is 1 connexion or vertex in common -> this is a cell node connectivity
        3) There are 2 connexions or vertices in common -> this is a cell face connectivity
        4) There are 3 connexions or vertices in common -> the compared cells are the same

    Args:
        cells: array of cells, each cell is characterised by three vertex indexes.
             ex: cells = [(1,2,3), (2,3,4)...]
        faces: array of node/vertex indexes corresponding to each face
             ex: [(1,2), (2,3),...]

    Returns:
        cell_face_connectivity: list of face connexions between cells
            ex: [[(c1,c2),f10], ...]
        cell_node_connectivity: list of node connexions between cells
            ex: [[(c1,c2),n3]]
    """
    cell_face_connectivity = []
    cell_node_connectivity = []

    faces_done = []

    # auxiliary function that finds the vertices in common between 2 cells
    def vertex_in_common_cell_cell(c1, c2):
        return [vx for vx in c1 if vx in c2]

    # get all inner face connectivity
    for i, main_cell in enumerate(cells):
        for j, aux_cell in enumerate(cells):
            if i != j:
                vic = vertex_in_common_cell_cell(main_cell, aux_cell)

                # 1 vertex in common -> 1 common node
                if len(vic) == 1:
                    cell_node_connectivity.append([(i, j), vic[0]])

                # 2 vertices in common -> 1 common face
                if len(vic) == 2:
                    for k, f in enumerate(faces):
                        if vic == list(f) or vic == list(reversed(f)):
                            cell_face_connectivity.append([(i, j), k])
                            faces_done.append(k)
                            continue

    def vertex_in_common_cell_face(c, f):
        return [vx for vx in c if vx in f]

    # remove dupes

    for i, elt in enumerate(cell_face_connectivity):
        for sub_elt in cell_face_connectivity[i + 1 :]:
            if elt[1] == sub_elt[1]:
                cell_face_connectivity.remove(sub_elt)

    # border faces
    faces_to_do = [[i, f] for i, f in enumerate(faces) if i not in faces_done]

    for face in faces_to_do:
        for i, cell in enumerate(cells):
            vic = vertex_in_common_cell_face(cell, face[1])

            if len(vic) == 2:
                cell_face_connectivity.append([(i, None), face[0]])

    return cell_face_connectivity, cell_node_connectivity


def create_2d_cartesian(Lx: int, Ly: int, Nx: int, Ny: int) -> Mesh:
    """

    :param Lx: length in x-axis of the domain
    :param Ly: length in y-axis  of the domain
    :param Nx: Number of intervals in the x direction
    :param Ny: Number of intervals in the y direction
    :return:
    """

    dx = Lx / float(Nx)
    dy = Ly / float(Ny)
    nb_cells = Nx * Ny
    nb_nodes = (Nx + 1) * (Ny + 1)
    nb_faces = (Nx + 1) * Ny + Nx * (Ny + 1)
    dim = 2

    node_coordinates = get_node_coordinates_cartesian_2d(Nx, Ny, dx, dy)
    cells = get_cells_cartesian_2d(Nx, Ny)
    faces = get_faces_cartesian_2d(Nx, Ny)
    cell_measures = np.full(nb_cells, dx * dy)

    cell_centers = get_cell_centers_cartesian_2d(cells, node_coordinates)

    face_centers = get_face_centers_cartesian_2d(faces, node_coordinates)

    face_measures = np.concatenate(
        [np.full(Nx * (Ny + 1), dx), np.full(Ny * (Nx + 1), dy)]
    )
    face_groups = get_face_groups_cartesian_2d(faces, node_coordinates, Lx, Ly)

    cell_face_connectivity, cell_node_connectivity = get_cell_face_node_connectivity(
        cells, faces
    )

    return MeshData(
        dim=dim,
        nb_cells=nb_cells,
        nb_nodes=nb_nodes,
        nb_faces=nb_faces,
        node_coordinates=node_coordinates,
        cell_centers=cell_centers,
        face_centers=face_centers,
        cells=cells,
        faces=faces,
        cell_measures=cell_measures,
        face_measures=face_measures,
        face_groups=face_groups,
        cell_face_connectivity=cell_face_connectivity,
        cell_node_connectivity=cell_node_connectivity,
    )


if __name__ == "__main__":
    create_2d_cartesian(4, 2, 2, 2)
