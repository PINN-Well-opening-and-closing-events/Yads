from pyevtk.hl import gridToVTK  # type: ignore
import numpy as np  # type: ignore
from typing import Tuple, Dict
from yads.mesh import Mesh

# issues: for a grid with cells = nb_cells and points = nb_points
# polyLinesToVTK: nb_cells = 1, points = nb_points
# gridToVTK: cells = nb_cells, points = nb_points
# linesToVTK : cells = nb_cells/2, points = nb_points or cells = nb_cells, points = 2 * nb_points


def grid_to_xyz_1d(grid: Mesh) -> Tuple:
    """extracts x,y,z node coordinates of a Mesh

    Args:
        grid: one_D Mesh object

    Returns:
         x,y,z size(grid.node_coordinates,1,1)
    """
    x = grid.node_coordinates
    y = z = np.zeros(1)
    return x, y, z


def export_vtk_1d(
    path: str, grid: Mesh, cell_data=None, point_data=None, field_data=None
) -> None:
    """converts a grid and its properties in a vtk extension file.

     Args:
        path: name of the file without extension where data should be saved.
        grid: Mesh object corresponding to a one_D grid
        cell_data: optional dictionary containing arrays with cell centered data.
                    Keys should be the names of the data arrays.
                        ex: cell_data={"Pressure": p, "Temperature": temp} with size(p) = size(temp) = grid.nb_cells
        point_data: optional dictionary containing arrays with node centered data.
                    Keys should be the names of the data arrays.
                        ex: point_data={"Transmissivity": p} with size(p)=grid.nb_cells
        field_data: optional dictionary with variables associated with the field.
                    Keys should be the names of the variable stored in each array.
    Returns:

    """
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

    x, y, z = grid_to_xyz_1d(grid)

    if cell_data:
        for (key, value) in [*cell_data.items()]:
            if not value.size == grid.nb_cells:
                raise AssertionError(
                    f"cell_data key has wrong size (key: '{key}' expected {grid.nb_cells}, got {value.size})"
                )
    if point_data:
        for (key, value) in [*point_data.items()]:
            if not value.size == grid.nb_nodes:
                raise AssertionError(
                    f"point_data key has wrong size (key: '{key}' expected {grid.nb_nodes}, got {value.size})"
                )

    print("exporting 1d grid at: " + path)
    gridToVTK(
        path, x, y, z, cellData=cell_data, pointData=point_data, fieldData=field_data
    )
    print("1d grid successfully exported")
    return
