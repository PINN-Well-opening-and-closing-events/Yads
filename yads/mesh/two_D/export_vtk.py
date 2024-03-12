from pyevtk.hl import unstructuredGridToVTK, gridToVTK  # type: ignore
from pyevtk.vtk import VtkTriangle  # type: ignore
import numpy as np  # type: ignore
from typing import Dict
from yads.mesh import Mesh
from yads.mesh.two_D.load_2d import load_mesh_2d


def grid_to_xyz_2d_triangular(grid: Mesh):
    """extract information from a triangular 2d grid
    Args:
        grid: Mesh object corresponding to a 2D grid

    Returns:
        x,y,z: lists of node coordinates, must be C or F contiguous
        conn: list of nodes index corresponding to the connectivity of triangular cells
        offset: enable the reading of 'conn' (size grid.nb_cells)
        ctype: list of vtk objects, size grid.nb_cells (VtkTriangle only in this case)
            ex: conn = [1, 2, 3, 1, 3, 4]
                offset = [3,6]
                ctype: [VtkTriangle.tid, VtkTriangle.tid]
            We can read this example as:
                offset[0] = 3, ctype[0] = VtkTriangle.tid --> first object is a Triangle composed of 3 nodes
                the 3 nodes of this first object are given by conn --> conn[0: offset[0] ] = [1, 2, 3]
                then:
                offset[1] = 6, ctype[1] = VtkTriangle.tid --> second object is a Triangle composed of 3 nodes
                the 3 nodes of this second object are given by conn --> conn[offset[0]: offset[1]-1] = [1, 3 4]
    """
    coord = grid.node_coordinates
    x, y = np.ascontiguousarray(coord[:, 0]), np.ascontiguousarray(coord[:, 1])
    z = np.zeros(len(x))

    assert x.flags["C_CONTIGUOUS"] or x.flags["F_CONTIGUOUS"]
    assert y.flags["C_CONTIGUOUS"] or y.flags["F_CONTIGUOUS"]
    assert z.flags["C_CONTIGUOUS"] or z.flags["F_CONTIGUOUS"]

    # for triangles only
    conn = grid.cells.reshape(3 * grid.nb_cells) - 1
    offset = np.zeros(grid.nb_cells)
    ctype = np.zeros(grid.nb_cells)
    for i in range(len(offset)):
        offset[i] = (i + 1) * 3
        ctype[i] = VtkTriangle.tid

    assert ctype.shape == offset.shape
    return x, y, z, conn, offset, ctype


def export_vtk_2d_triangular(
    path: str, grid: Mesh, cell_data=None, point_data=None, field_data=None
) -> None:
    """converts a grid and its properties in a vtk extension file.

     Args:
        path: name of the file without extension where data should be saved.
        grid: Mesh object corresponding to a two_D grid
        cell_data: optional dictionary containing arrays with cell centered data.
                    Keys should be the names of the data arrays.
                        ex: cell_data={"Pressure": p, "Temperature": temp} with size(p) = size(temp) = grid.nb_cells
        point_data: optional dictionary containing arrays with node centered data.
                    Keys should be the names of the data arrays.
                        ex: point_data={"Transmissivity": p} with size(p)=grid.nb_nodes
        field_data: optional dictionary with variables associated with the field.
                    Keys should be the names of the variable stored in each array.
    Returns:
        None
    """
    if not isinstance(grid, Mesh):
        raise TypeError(f"grid must be a Mesh object (got {type(grid)})")
    if grid.type != "triangular":
        raise TypeError(f"grid must be triangular (got {grid.type})")

    if not isinstance(path, str):
        raise TypeError(f"path must be a string (got {type(path)})")
    if not isinstance(cell_data, Dict) and cell_data:
        raise TypeError(f"cell_data must be a dict (got {type(cell_data)}")
    if not isinstance(point_data, Dict) and point_data:
        raise TypeError(f"point_data must be a dict (got {type(point_data)}")
    if not isinstance(field_data, Dict) and field_data:
        raise TypeError(f"field_data must be a dict (got {type(field_data)}")

    if cell_data:
        for key, value in [*cell_data.items()]:
            if not value.size == grid.nb_cells:
                raise AssertionError(
                    f"cell_data key has wrong size (key: '{key}' expected {grid.nb_cells}, got {value.size})"
                )
    if point_data:
        for key, value in [*point_data.items()]:
            if not value.size == grid.nb_nodes:
                raise AssertionError(
                    f"point_data key has wrong size (key: '{key}' expected {grid.nb_nodes}, got {value.size})"
                )

    x, y, z, conn, offset, ctype = grid_to_xyz_2d_triangular(grid)
    print("exporting 2d grid at: " + path)
    unstructuredGridToVTK(
        path,
        x,
        y,
        z,
        connectivity=conn,
        offsets=offset,
        cell_types=ctype,
        cellData=cell_data,
        pointData=point_data,
        fieldData=field_data,
    )
    print("2d grid successfully exported")
    return


def grid_to_xyz_2d_cartesian(grid: Mesh):
    coord = grid.node_coordinates
    X, Y = np.ascontiguousarray(coord[:, 0]), np.ascontiguousarray(coord[:, 1])
    X, Y = list(dict.fromkeys(X)), list(dict.fromkeys(Y))
    Z = 0.0

    nx_points, ny_points = len(list(dict.fromkeys(X))), len(list(dict.fromkeys(Y)))
    x = np.zeros((nx_points, ny_points, 1))
    y = np.zeros((nx_points, ny_points, 1))
    z = np.zeros((nx_points, ny_points, 1))

    for j in range(ny_points):
        for i in range(nx_points):
            x[i, j, 0] = X[i]
            y[i, j, 0] = Y[j]
            z[i, j, 0] = Z
    x, y, z = np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(z)
    return x, y, z


def transform_cell_data_2d_cartesian(grid: Mesh, cell_data):
    coord = grid.node_coordinates
    nx_cells, ny_cells = (
        len(list(dict.fromkeys(coord[:, 0]))) - 1,
        len(list(dict.fromkeys(coord[:, 1]))) - 1,
    )
    for key in cell_data.keys():
        cell_data[key] = (
            cell_data[key].reshape(nx_cells, ny_cells).T.reshape(nx_cells * ny_cells)
        )
    for key in cell_data.keys():
        cell_data[key] = np.ascontiguousarray(cell_data[key])
    return cell_data


def export_vtk_2d_cartesian(
    path: str, grid: Mesh, cell_data=None, point_data=None, field_data=None
) -> None:
    if grid.type != "cartesian":
        raise TypeError(f"grid must be cartesian (got {grid.type})")
    if not isinstance(grid, Mesh):
        raise TypeError(f"grid must be a Mesh object (got {type(grid)})")
    if not isinstance(path, str):
        raise TypeError(f"path must be a string (got {type(path)})")
    if not isinstance(cell_data, Dict) and cell_data:
        raise TypeError(f"cell_data must be a dict (got {type(cell_data)}")
    if not isinstance(point_data, Dict) and point_data:
        raise TypeError(f"point_data must be a dict (got {type(point_data)}")
    if not isinstance(field_data, Dict) and field_data:
        raise TypeError(f"field_data must be a dict (got {type(field_data)}")

    if cell_data:
        for key, value in [*cell_data.items()]:
            if not value.size == grid.nb_cells:
                raise AssertionError(
                    f"cell_data key has wrong size (key: '{key}' expected {grid.nb_cells}, got {value.size})"
                )
    if point_data:
        for key, value in [*point_data.items()]:
            if not value.size == grid.nb_nodes:
                raise AssertionError(
                    f"point_data key has wrong size (key: '{key}' expected {grid.nb_nodes}, got {value.size})"
                )

    x, y, z = grid_to_xyz_2d_cartesian(grid)

    assert x.flags["C_CONTIGUOUS"] or x.flags["F_CONTIGUOUS"]
    assert y.flags["C_CONTIGUOUS"] or y.flags["F_CONTIGUOUS"]
    assert z.flags["C_CONTIGUOUS"] or z.flags["F_CONTIGUOUS"]

    cell_data = transform_cell_data_2d_cartesian(grid, cell_data)
    print("exporting 2d grid at: " + path)
    gridToVTK(
        path, x, y, z, cellData=cell_data, pointData=point_data, fieldData=field_data
    )
    return


if __name__ == "__main__":
    from yads.mesh.two_D import create_2d_cartesian

    cartesian = create_2d_cartesian(10, 2, 10, 2)
    export_vtk_2d_cartesian("./cartesian_test", cartesian)

    """
    square = load_mesh_2d("../../../meshes/2D/Square/square.mesh")

    export_vtk_2d_triangular("./square", square)

    disk = load_mesh_2d("../../../meshes/2D/Disk/disk.mesh")
    export_vtk_2d_triangular("./disk", disk)

    disk_hole = load_mesh_2d(
        "../../../meshes/2D/Tests/disk_8_hole_4/disk_8_hole_4.mesh"
    )
    export_vtk_2d_triangular("./disk_hole", disk_hole)

    el = load_mesh_2d("../../../meshes/2D/L/L.mesh")
    export_vtk_2d_triangular("./l", el)

    twistz = load_mesh_2d(
        "../../../meshes/2D/Tests_with_geom/permea_twistz/permea_twistz.mesh"
    )
    export_vtk_2d_triangular("./twistz", twistz)

    twistz_geom = load_mesh_2d(
        "../../../meshes/2D/Tests_with_geom/permea_twistz/permea_twistz.mesh",
        "../../../meshes/2D/Tests_with_geom/permea_twistz/permea_twistz_geom_info.txt",
    )
    export_vtk_2d_triangular("./twistz_geom", twistz_geom)
    """
