from yads.mesh import Mesh
from yads.mesh.two_D.create_2d_mesh import create_2d
from typing import Dict


def parse_mesh_file(data: list, prop_list: list) -> Dict:
    """Extracts the properties contained in data using a property list

    Args:
        data: raw data from a meshfile, list of str
        prop_list: properties to extract from data, list of str

    Returns:
         dictionary of properties (in prop_list) extracted from raw data
    """
    mesh_dict = {}
    prop_values = []
    prop = None
    for line in data:
        if line in prop_list:
            prop = line
        if line != "" and line != prop:
            prop_values.append(line)
        else:
            if prop:
                mesh_dict[prop] = prop_values
                prop_values = []

    return mesh_dict


def parse_geom_file(data: list, geom_prop_list: list) -> Dict:
    geom_dict = {}
    prop_values = []
    prop = None
    for line in data:
        if line in geom_prop_list:
            prop = line
        if line != "" and line != prop:
            prop_values.append(line)
        else:
            if prop:
                geom_dict[prop] = prop_values
                prop_values = []

    return geom_dict


def load_mesh_2d(meshfile: str, geom_file: str = None) -> Mesh:
    """Loads a 2D mesh from path

    Args:
        meshfile: path to a 2D meshfile, supported extensions are .mesh
        geom_file: path to a file containing geometric properties of the 2D meshfile.
                   This can avoid the computation of geometric properties by yads

    Returns:
        Mesh object corresponding to the meshfile
    """
    if not isinstance(meshfile, str):
        raise TypeError(f"meshfile must be a string (got {type(meshfile)}")

    extension = meshfile.split(".")[-1]
    if extension != "mesh":
        raise NotImplementedError(
            f"file extension not supported yet (got {'.' + str(extension)}) "
        )
    try:
        with open(meshfile) as f:
            data = [line.replace("\n", "") for line in f.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError(f"invalid path to meshfile (got {meshfile})")

    prop_list = [
        "Dimension",
        "Identifier",
        "Geometry",
        "Vertices",
        "Edges",
        "Triangles",
        "SubDomainFromMesh",
        "SubDomainFromGeom",
        "VertexOnGeometricVertex",
        "VertexOnGeometricEdge",
        "EdgeOnGeometricEdge",
    ]

    geom_prop_list = [
        "nb_cells",
        "nb_nodes",
        "nb_faces",
        "node_coordinates",
        "cell_centers",
        "face_centers",
        "cells",
        "faces",
        "cell_measures",
        "face_measures",
        "face_groups",
        "cell_face_connectivity",
        "cell_node_connectivity",
        "Mesh area",
        "Border measure",
    ]
    mesh_dict = parse_mesh_file(data, prop_list)

    if mesh_dict["Dimension"] != ["2"]:
        raise ValueError(
            f"expected a 2 dimensional mesh from meshfile (got {mesh_dict['Dimension']})"
        )

    if geom_file:
        if not isinstance(geom_file, str):
            raise TypeError(f"geometric file must be a string (got {type(geom_file)}")

        extension = geom_file.split(".")[-1]
        if extension != "txt":
            raise NotImplementedError(
                f"file extension not supported yet (got {'.' + str(extension)}) "
            )
        try:
            with open(geom_file) as f:
                geom_data = [line.replace("\n", "") for line in f.readlines()]
        except FileNotFoundError:
            raise FileNotFoundError(f"invalid path to geometric file (got {geom_file})")

        geom_dict = parse_geom_file(geom_data, geom_prop_list)
        grid = create_2d(mesh_dict, geom_dict)
        del geom_dict

    else:
        grid = create_2d(mesh_dict)

    del mesh_dict

    return grid


if __name__ == "__main__":
    import time

    mesh_f = "../../../meshes/2D/Tests_with_geom/permea_twistz/permea_twistz.mesh"
    # geom_f = "../../../meshes/2D/Tests_with_geom/permea_twistz/permea_twistz_geom_info.txt"
    # mesh_f = "../../../meshes/2D/Tests_with_geom/square_1_1/square_1_1.mesh"
    # geom_f = "../../../meshes/2D/Tests_with_geom/square_1_1/square_1_1_geom_info.txt"
    load_mesh_2d(mesh_f)
