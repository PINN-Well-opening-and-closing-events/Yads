from yads.mesh import Mesh
from yads.mesh import MeshData
from yads.mesh.one_D import load_meshfile, export_vtk_1d
from yads.mesh.two_D import (
    load_mesh_2d,
    export_vtk_2d_triangular,
    export_vtk_2d_cartesian,
)
import json
import numpy as np


def export_vtk(
    path: str, grid: Mesh, cell_data=None, point_data=None, field_data=None
) -> None:
    if grid.dim == 1:
        export_vtk_1d(
            path=path,
            grid=grid,
            cell_data=cell_data,
            point_data=point_data,
            field_data=field_data,
        )
    elif grid.dim == 2:
        if grid.type == "triangular":
            export_vtk_2d_triangular(
                path=path,
                grid=grid,
                cell_data=cell_data,
                point_data=point_data,
                field_data=field_data,
            )
        elif grid.type == "cartesian":
            export_vtk_2d_cartesian(
                path=path,
                grid=grid,
                cell_data=cell_data,
                point_data=point_data,
                field_data=field_data,
            )
    else:
        raise NotImplementedError("Dimension not supported yet")
    return


def find_dim(meshfile: str) -> int:
    extension = meshfile.split(".")[-1]
    dim = 0

    if extension not in ["mesh", "msh"]:
        raise NotImplementedError(
            f"file extension not supported yet (got {'.' + str(extension)}) "
        )

    if not isinstance(meshfile, str):
        raise TypeError(f"meshfile must be a string (got {type(meshfile)}")

    try:
        with open(meshfile) as f:
            data = [line.replace("\n", "") for line in f.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError(f"invalid path to meshfile (got {meshfile})")

    # msh files are only 1D atm
    if extension == "msh":
        return 1
    try:
        dim_line = data.index("Dimension")
        dim = int(data[dim_line + 1])
    except ValueError:
        for i, line in enumerate(data):
            if "Dimension" in line:
                dim = int(line[9:])

    # 1D files are noted as 3D and real 3D not supported yet
    if dim == 3:
        dim = 1
    assert dim != 0
    return dim


def load_mesh(meshfile: str) -> Mesh:
    dim = find_dim(meshfile)
    if dim == 1:
        return load_meshfile(meshfile)
    elif dim == 2:
        return load_mesh_2d(meshfile)
    else:
        raise NotImplementedError("Dimension not supported yet")


def load_json(jsonfile: str) -> Mesh:
    with open(jsonfile, "r") as f:
        f = json.load(f)
        return MeshData(
            dim=f["dimension"],
            nb_cells=f["nb_cells"],
            nb_nodes=f["nb_nodes"],
            nb_faces=f["nb_faces"],
            node_coordinates=np.array(f["node_coordinates"]),
            cell_centers=np.array(f["cell_centers"]),
            face_centers=f["face_centers"],
            cells=np.array(f["cells"]),
            faces=np.array(f["faces"]),
            cell_measures=np.array(f["cell_measures"]),
            face_measures=np.array(f["face_measures"]),
            face_groups={
                key: np.array(value) for (key, value) in f["face_groups"].items()
            },
            cell_face_connectivity=f["cell_face_connectivity"],
            cell_node_connectivity=f["cell_node_connectivity"],
        )
