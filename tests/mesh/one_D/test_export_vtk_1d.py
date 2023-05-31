from yads.mesh.one_D.export_vtk_1d import export_vtk_1d  # type: ignore
from yads.mesh.one_D.create_1d_mesh import create_1d  # type: ignore
import numpy as np  # type: ignore
import pytest


def test_wrong_inputs():
    grid = create_1d(nb_cells=101, interval=(0, 1))
    cell_prop = np.ones(grid.nb_cells)
    node_prop = np.ones(grid.nb_nodes)

    with pytest.raises(TypeError, match="grid must be a Mesh object"):
        export_vtk_1d("./test", "error")
    with pytest.raises(TypeError, match="path must be a string"):
        export_vtk_1d(0.0, grid)
    with pytest.raises(TypeError, match="cell_data must be a dict"):
        export_vtk_1d("./test", grid, cell_data="error")
    with pytest.raises(TypeError, match="point_data must be a dict"):
        export_vtk_1d("./test", grid, point_data="error")
    with pytest.raises(TypeError, match="field_data must be a dict"):
        export_vtk_1d("./test", grid, field_data="error")
    with pytest.raises(AssertionError, match="cell_data key has wrong size"):
        export_vtk_1d(
            "./test", grid, cell_data={"valid": cell_prop, "error": node_prop}
        )
    with pytest.raises(AssertionError, match="point_data key has wrong size"):
        export_vtk_1d(
            "./test", grid, point_data={"valid": node_prop, "error": cell_prop}
        )
