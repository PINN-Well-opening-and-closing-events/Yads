import numpy as np
import pytest


from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian
from yads.wells import Well


def test_MeshData_properties():
    square_3 = create_2d_cartesian(Lx=3, Ly=3, Nx=3, Ny=3)
    assert square_3.type == "cartesian"
    assert square_3.dim == 2
    assert square_3.nb_cells == 9
    assert square_3.nb_nodes == 16
    assert square_3.nb_faces == 24

    square_1 = create_2d_cartesian(Lx=2, Ly=2, Nx=2, Ny=2)
    assert len(square_1.cell_node_connectivity) == 4
    assert len(square_1.cell_face_connectivity) == 12

    assert square_1.nb_groups == 5
    assert square_1.cell_groups == {}


def test_MeshData_methods_wrong_inputs():
    square_3 = create_2d_cartesian(Lx=3, Ly=3, Nx=3, Ny=3)

    with pytest.raises(ValueError, match="unknown face : face index out of range"):
        square_3.face_to_cell(face=-1, face_type="inner")
        square_3.face_to_cell(face=square_3.nb_faces + 1, face_type="inner")

    with pytest.raises(ValueError, match="invalid face_type, valid face_type are 'inner' and 'boundary'."):
        square_3.face_to_cell(face=0, face_type="error")

    with pytest.raises(ValueError, match="unknown cell : cell index out of range"):
        square_3.cell_to_face(cell=-1)
        square_3.cell_to_face(cell=square_3.nb_cells + 1)

    with pytest.raises(ValueError, match="unknown face : face index out of range"):
        square_3.group(face=-1)
        square_3.group(face=square_3.nb_cells + 1)

    with pytest.raises(AssertionError):
        square_3.add_cell_group_by_index(name="error", cells_idx=[-1])
        square_3.add_cell_group_by_index(name="error", cells_idx=[square_3.nb_cells + 1])


def test_MeshData_methods():
    square_3 = create_2d_cartesian(Lx=3, Ly=3, Nx=3, Ny=3)
    assert square_3.group(face=0) == ['lower']
    # group creation tests
    assert square_3.cell_groups == {}
    square_3.add_cell_group_by_coord(name="coord_group", coord=np.array([[0.5, 0.5]]))
    assert len(list(square_3.cell_groups.keys())) == 1
    square_3.add_cell_group_by_index(name="index_group", cells_idx=[0, 1])
    assert len(list(square_3.cell_groups.keys())) == 2
    square_3.add_cell_group_by_square(name="square_group", up_left=(0., 1.), down_right=(1., 0))
    assert len(list(square_3.cell_groups.keys())) == 3

    #
    assert len(square_3.cell_to_face(cell=1)) == 4

    # well tests
    well_test = Well(
        name="well test",
        cell_group=np.array([[0.5, 0.5]]),
        radius=0.1,
        control={"Neumann": 0.},
        s_inj=1.0,
        schedule=[[0, 0]],
        mode="injector",
    )

    square_3.connect_well(well=well_test)
