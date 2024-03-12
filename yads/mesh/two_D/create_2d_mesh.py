import itertools
import numpy as np  # type: ignore
from typing import Dict, List, Tuple

from yads.mesh import Mesh
from yads.mesh import MeshData


def get_cells(mesh_dict: Dict) -> np.ndarray:
    """Extracts indexes corresponding to each cell from the mesh_dict

    Args:
        mesh_dict: mesh properties dictionary, the dict must have a "Triangle" property/key

    Returns:
        array of cells, each cell is characterised by three vertex indexes.
            ex: cells = [(1,2,3), (2,3,4)...]
    """
    cells_str = np.array(
        [coord.split(" ")[0:3] for coord in mesh_dict["Triangles"][1:]]
    )
    cells = []
    for coord in cells_str:
        cells.append((int(coord[0]), int(coord[1]), int(coord[2])))
    return np.array(cells)


def get_nb_cells(mesh_dict: Dict) -> int:
    """Extracts the number of cells from the mesh_dict

    Args:
        mesh_dict: mesh properties dictionary, the dict must have a "Triangle" property/key

    Returns:
         number of cells of the mesh
    """
    # number of cells is the number of triangles in case of a mesh triangle
    return int(mesh_dict["Triangles"][0])


def get_nb_nodes(mesh_dict: Dict) -> int:
    """Extracts the number of nodes from the mesh_dict

    Args:
        mesh_dict: mesh properties dictionary, the dict must have a "Vertices" property/key

    Returns:
             number of nodes of the mesh
    """
    # number of nodes corresponds to the number of vertices
    return int(mesh_dict["Vertices"][0])


def get_node_coordinates(mesh_dict: Dict) -> np.ndarray:
    """Extracts the node coordinates from the mesh_dict

    Args:
        mesh_dict: mesh properties dictionary, the dict must have a "Vertices" property/key

    Returns:
         array of node coordinates
            ex: [(x1,y1),..]
    """
    node_coordinates_str = np.array(
        [coord.split(" ")[0:2] for coord in mesh_dict["Vertices"][1:]]
    )
    node_coordinates = []
    for coord in node_coordinates_str:
        node_coordinates.append([np.float64(coord[0]), np.float64(coord[1])])
    return np.array(node_coordinates)


def get_face_centers(faces: np.ndarray, node_coordinates: np.ndarray) -> np.ndarray:
    """Computes the center of each face of the mesh

    Args
        faces: array of node indexes corresponding to each face
             ex: [(1,2), (2,3),...]
        node_coordinates: array of node coordinates.
                        ex: [(x1,y1),..]

    Returns:
         array of face centers
    """
    face_center = []

    # auxiliary function that computes the center of 2 points
    def get_center(a, b):
        center = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        return center

    for face in faces:
        face_center.append(
            get_center(node_coordinates[face[1] - 1], node_coordinates[face[2] - 1])
        )
    return np.array(face_center)


def get_cell_centers(cells: np.ndarray, node_coordinates: np.ndarray) -> np.ndarray:
    """Computes the center of each cell of the mesh, the mesh must be triangle

    Args:
        cells: array of cells, each cell is characterised by three vertex indexes.
             ex: cells = [(1,2,3), (2,3,4)...]
        node_coordinates: array of node coordinates.
                        ex: [(x1,y1),..]

    Returns:
         array of cell centers:
            ex: [center1, center2,...]
    """

    # auxiliary function that computes the center of a triangle described by 3 points
    def tri_center(a, b, c):
        center = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3]
        return center

    cell_centers = []
    for idx in cells:
        cell_centers.append(
            tri_center(
                node_coordinates[int(idx[0]) - 1],
                node_coordinates[int(idx[1]) - 1],
                node_coordinates[int(idx[2]) - 1],
            )
        )
    return np.array(cell_centers)


def get_face_measures(faces: np.ndarray, node_coordinates: np.ndarray) -> np.ndarray:
    """Computes the length of each face

    Args:
        faces: array of node indexes corresponding to each face
             ex: [(1,2), (2,3),...]
        node_coordinates: array of node coordinates.
                        ex: [(x1,y1),..]

    Returns:
        array of face measures (lengths)
    """
    face_measures = []

    # auxiliary function that computes the distance between 2 points
    def get_measure(a, b):
        measure = np.linalg.norm(a - b)
        return measure

    for face in faces:
        face_measures.append(
            get_measure(node_coordinates[face[1] - 1], node_coordinates[face[2] - 1])
        )
    return np.array(face_measures)


def get_cell_measures(cells: np.ndarray, node_coordinates: np.ndarray):
    """Computes the area of each cell

    Args:
        cells: array of cells, each cell is characterised by three vertex indexes.
             ex: cells = [(1,2,3), (2,3,4)...]
        node_coordinates: array of node coordinates.
                        ex: [(x1,y1),..]

    Returns:
        array of cell measures (areas)
    """
    cell_measures = []

    # auxiliary function the computes the area of a triangle described by 3 points
    def get_measure(a, b, c):
        measure = a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])
        return abs(measure / 2)

    for cell in cells:
        cell_measures.append(
            get_measure(
                node_coordinates[cell[0] - 1],
                node_coordinates[cell[1] - 1],
                node_coordinates[cell[2] - 1],
            )
        )
    return np.array(cell_measures)


def get_faces(cells: np.ndarray) -> List:
    """Find each unique face from an ensemble of cells

    Args:
        cells: array of cells, each cell is characterised by three vertex indexes.
             ex: cells = [(1,2,3), (2,3,4)...]

    Returns:
        List of node indexes corresponding to each face
             ex: [(0,2,5), (1,3,6),...]
    """
    faces = []
    # iter over all cells
    for idx in cells:
        # find all combinations of 2 vertices that correspond to a face
        faces_combin = list(itertools.combinations(idx, 2))
        # iter over all the face combination
        for face in faces_combin:
            # if the face is new, add it to [faces]
            if face not in faces and tuple(reversed(face)) not in faces:
                faces.append(face)
    for f, face in enumerate(faces):
        faces[f] = (f, *face)
    return faces


def get_groups(faces: list, edges: list) -> Dict[str, np.ndarray]:
    """Computes a dictionary of face groups.
    For an inner face, the key is 0
    For all other faces, the key respects the original meshfile loaded.

    Args:
        faces: list of node indexes corresponding to each face
             ex: [(1,2), (2,3),...]
        edges: list of border faces and their corresponding groups
            ex: [[(1,2),4]...] -> face corresponding to vertex 1 and 2 belongs to group 4

    Returns:
         dictionary of face groups
    """
    import copy

    faces_copy_s = copy.deepcopy(faces)
    faces_copy = []
    for i in range(len(faces_copy_s)):
        faces_copy.append(faces_copy_s[i][1:])
    face_groups = {}  # type: ignore
    # first we add all faces that are on the edges and remove them from the face list

    for face, group in edges:
        idx = None
        if str(group) not in face_groups:
            face_groups[str(group)] = []
        if face in faces_copy:
            faces_copy.remove(face)
            for f, i, j in faces_copy_s:
                if (i, j) == face:
                    faces_copy_s.remove((f, i, j))
                    idx = f
        elif tuple(reversed(face)) in faces_copy:
            faces_copy.remove(tuple(reversed(face)))
            for f, i, j in faces_copy_s:
                if (i, j) == tuple(reversed(face)):
                    idx = f
                    faces_copy_s.remove((f, i, j))
        else:
            print(f"{face} not in face list, this is not possible. O_o")
        face_groups[str(group)].append((idx, *face))

    # all remaining faces are inner faces
    face_groups["0"] = faces_copy_s
    for key in face_groups.keys():
        face_groups[key] = np.array(face_groups[key])
    return face_groups


def get_face_groups(
    mesh_dict: Dict, cells: np.ndarray
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Computes faces and face groups.
    faces is a list of unique face contained in the mesh. A face is described by 2 vertices.
    face_groups is a dict where keys are groups and items are faces belonging to a group

    Args:
        mesh_dict: mesh properties dictionary, the dict must have a "Edges" property/key
        cells: array of cells, each cell is characterised by three vertex indexes.
             ex: cells = [(1,2,3), (2,3,4)...]

    Returns:
         face_groups: dict where key are groups and items are (face_index, vi, vj)
                    ex: {0:[(f0,v1,v2), (f1,v2,3)], 1:[(f5,v3,v4)],...}
        faces:  array of node/vertex indexes corresponding to each face
             ex: [(1,2), (2,3),...]
    """
    # extract unique faces from all triangles faces (cells)
    faces = get_faces(cells)
    # now we need to find to which group belongs each face (inner, border, border_left...)
    edges_str = np.array([coord.split(" ")[0:3] for coord in mesh_dict["Edges"][1:]])
    edges = []
    for i, idx in enumerate(edges_str):
        edges.append([(int(idx[0]), int(idx[1])), int(idx[2])])
    face_groups = get_groups(faces, edges)
    return face_groups, np.array(faces)


def get_cell_face_node_connectivity(
    cells: np.ndarray, faces: np.ndarray, face_groups: Dict[str, np.ndarray]
) -> Tuple[List, List]:
    """Computes the connectivity between cells.
    The are multiples cases:
        1) There is no connexion
        2) There is 1 connexion or vertex in common -> this is a cell node connectivity
        3) There are 2 connexions or vertices in common -> this is a cell face connectivity
        4) There are 3 connexions or vertices in common -> the compared cells are the same

    Args:
        cells: array of cells, each cell is characterised by three vertex indexes.
             ex: cells = [(1,2,3), (2,3,4)...]
        faces: array of node/vertex indexes corresponding to each face
             ex: [(1,2), (2,3),...]
        face_groups: dict where key are groups and items are (face_index, vi, vj)
                    ex: {0:[(f0,v1,v2), (f1,v2,3)], 1:[(f5,v3,v4)],...}

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
                # 2 vertices in common -> 1 common face
                if len(vic) == 2:
                    for k, f in enumerate(faces):
                        if vic == list(f[1:]) or vic == list(reversed(f[1:])):
                            cell_face_connectivity.append([(i, j), k])
                            faces_done.append(k)
                # 1 vertex in common -> 1 common node
                if len(vic) == 1:
                    cell_node_connectivity.append([(i, j), vic[0]])

    def vertex_in_common_cell_face(c, f):
        return [vx for vx in c if vx in f]

    # remove dupes

    for i, elt in enumerate(cell_face_connectivity):
        for sub_elt in cell_face_connectivity[i + 1 :]:
            if elt[1] == sub_elt[1]:
                cell_face_connectivity.remove(sub_elt)

    # border faces
    faces_to_do = [f for f in faces if f[0] not in faces_done]
    for face in faces_to_do:
        for i, cell in enumerate(cells):
            vic = vertex_in_common_cell_face(cell, face[1:])
            if len(vic) == 2:
                cell_face_connectivity.append([(i, None), face[0]])

    return cell_face_connectivity, cell_node_connectivity


######################################################
# GEOM DICT AUXILIARY FUNCTIONS
######################################################
def get_nb_cells_from_geom_dict(geom_dict: Dict) -> int:
    return int(geom_dict["nb_cells"][0])


def get_nb_nodes_from_geom_dict(geom_dict: Dict) -> int:
    return int(geom_dict["nb_nodes"][0])


def get_node_coord_from_geom_dict(geom_dict: Dict) -> np.ndarray:
    """Extracts the node coordinates from the mesh_dict

    Args:
        geom_dict: mesh properties dictionary, the dict must have a "Vertices" property/key

    Returns:
         array of node coordinates
            ex: [(x1,y1),..]
    """
    node_coordinates_str = np.array(
        [coord.split(" ")[0:2] for coord in geom_dict["node_coordinates"][0:]]
    )
    node_coordinates = []
    for coord in node_coordinates_str:
        node_coordinates.append([np.float64(coord[0]), np.float64(coord[1])])
    return np.array(node_coordinates)


def get_cells_from_geom_dict(geom_dict: Dict) -> np.ndarray:
    """Extracts indexes corresponding to each cell from the geom_dict

    Args:
        geom_dict: mesh properties dictionary, the dict must have a "cells" property/key

    Returns:
        array of cells, each cell is characterised by three vertex indexes.
            ex: cells = [(1,2,3), (2,3,4)...]
    """
    cells_str = np.array([coord.split(" ")[0:3] for coord in geom_dict["cells"][0:]])
    cells = []
    for coord in cells_str:
        cells.append((int(coord[0]) + 1, int(coord[1]) + 1, int(coord[2]) + 1))
    return np.array(cells)


def get_cell_measures_from_geom_dict(geom_dict: Dict):
    cell_measures = []
    for measure_str in geom_dict["cell_measures"]:
        cell_measures.append(np.float64(measure_str))
    return np.array(cell_measures)


def get_cell_face_conn_from_geom_dict(geom_dict: Dict):
    return


def get_mesh_area_from_geom_dict(geom_dict: Dict):
    return


def get_border_area_from_geom_dict(geom_dict: Dict):
    return


def create_2d(mesh_dict: Dict, geom_dict: Dict = None) -> Mesh:
    if geom_dict:
        nb_cells = get_nb_cells_from_geom_dict(geom_dict)
        nb_nodes = get_nb_nodes_from_geom_dict(geom_dict)
        node_coordinates = get_node_coord_from_geom_dict(geom_dict)
        cells = get_cells_from_geom_dict(geom_dict)
        cell_measures = get_cell_measures_from_geom_dict(geom_dict)

    else:
        cells = get_cells(mesh_dict)
        nb_cells = get_nb_cells(mesh_dict)
        nb_nodes = get_nb_nodes(mesh_dict)
        node_coordinates = get_node_coordinates(mesh_dict)
        cell_measures = get_cell_measures(cells, node_coordinates)

    assert len(cells) == nb_cells
    assert len(node_coordinates) == nb_nodes
    assert len(cell_measures) == nb_cells

    face_groups, faces = get_face_groups(mesh_dict, cells)
    nb_faces = len(faces)
    face_count = 0
    for group in face_groups.keys():
        face_count += len(face_groups[group])
    assert face_count == nb_faces
    cell_centers = get_cell_centers(cells, node_coordinates)
    assert len(cell_centers) == nb_cells
    face_centers = get_face_centers(faces, node_coordinates)
    assert len(face_centers) == nb_faces

    face_measures = get_face_measures(faces, node_coordinates)
    assert len(face_measures) == nb_faces
    cell_face_connectivity, cell_node_connectivity = get_cell_face_node_connectivity(
        cells, faces, face_groups
    )

    return MeshData(
        dim=2,
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
