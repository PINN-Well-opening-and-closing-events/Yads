from yads.mesh.two_D.export_vtk import export_vtk_2d_triangular, export_vtk_2d_cartesian
from yads.mesh.two_D.load_2d import load_mesh_2d
from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian
import numpy as np  # type: ignore
import pytest
import os


def test_wrong_inputs():
    square = create_2d_cartesian(10, 10, 5, 5)

    cell_prop = np.ones(square.nb_cells)
    node_prop = np.ones(square.nb_nodes)
    # export_vtk_2d_cartesian("./cartesian_test", square)
    # if os.path.exists("./cartesian_test.vtk"):
    #     os.remove("./cartesian_test.vtk")

    
    # with pytest.raises(TypeError, match="grid must be a Mesh object"):
    #     export_vtk_2d_triangular("./test", "error")
    # with pytest.raises(TypeError, match="path must be a string"):
    #     export_vtk_2d_triangular(0.0, square)
    # with pytest.raises(TypeError, match="cell_data must be a dict"):
    #     export_vtk_2d_triangular("./test", square, cell_data="error")
    # with pytest.raises(TypeError, match="point_data must be a dict"):
    #     export_vtk_2d_triangular("./test", square, point_data="error")
    # with pytest.raises(TypeError, match="field_data must be a dict"):
    #     export_vtk_2d_triangular("./test", square, field_data="error")
    # with pytest.raises(AssertionError, match="cell_data key has wrong size"):
    #     export_vtk_2d_triangular(
    #         "./test", square, cell_data={"valid": cell_prop, "error": node_prop}
    #     )
    # with pytest.raises(AssertionError, match="point_data key has wrong size"):
    #     export_vtk_2d_triangular(
    #         "./test", square, point_data={"valid": node_prop, "error": cell_prop}
    #     )
    pass


def test_exports():
    # disk = load_mesh_2d("./meshes/2D/Disk/disk.mesh")
    # export_vtk_2d_triangular("./test", disk)

    # disk_hole = load_mesh_2d("./meshes/2D/Tests/disk_8_hole_4/disk_8_hole_4.mesh")
    # export_vtk_2d_triangular("./test", disk_hole)

    # l = load_mesh_2d("./meshes/2D/L/L.mesh")
    # export_vtk_2d_triangular("./test", l)

    if os.path.exists("test.vtu"):
        os.remove("test.vtu")
