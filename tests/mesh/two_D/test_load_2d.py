import pytest
from yads.mesh.two_D.load_2d import load_mesh_2d  # type: ignore


def test_wrong_inputs():

    with pytest.raises(TypeError, match=r"meshfile must be a string"):
        load_mesh_2d(404)
    with pytest.raises(NotImplementedError, match=r"file extension not supported yet"):
        load_mesh_2d("./meshes/2D/Disk/disk.err")
    with pytest.raises(FileNotFoundError, match=r"invalid path to meshfile"):
        load_mesh_2d("error.mesh")
    with pytest.raises(TypeError, match=r"geometric file must be a string"):
        load_mesh_2d(
            "./meshes/2D/Tests_with_geom/disk_8_hole_4/disk_8_hole_4.mesh", 404
        )
    with pytest.raises(NotImplementedError, match=r"file extension not supported yet"):
        load_mesh_2d(
            "./meshes/2D/Tests_with_geom/disk_8_hole_4/disk_8_hole_4.mesh",
            "./meshes/2D/Tests_with_geom/disk_8_hole_4/disk_8_hole_4_geom_info.err",
        )
    with pytest.raises(FileNotFoundError, match=r"invalid path to geometric file"):
        load_mesh_2d(
            "./meshes/2D/Tests_with_geom/disk_8_hole_4/disk_8_hole_4.mesh", "error.txt",
        )


def test_square_1_1():
    square_1_1 = load_mesh_2d("./meshes/2D/Tests/square_1_1/square_1_1.mesh")

    assert square_1_1.nb_cells == 2
    assert square_1_1.nb_nodes == 4
    assert square_1_1.nb_faces == 5
    assert len(square_1_1.cells) == square_1_1.nb_cells
    assert len(square_1_1.measures(item="cell")) == square_1_1.nb_cells
    assert len(square_1_1.measures(item="face")) == square_1_1.nb_faces
    assert len(list(square_1_1.face_groups.keys())) == 5


def test_disk_10():
    disk_10 = load_mesh_2d("./meshes/2D/Tests/disk_10/disk_10.mesh")

    assert disk_10.nb_cells == 16
    assert disk_10.nb_nodes == 14
    assert disk_10.nb_faces == 29
    assert len(disk_10.cells) == disk_10.nb_cells
    assert len(disk_10.measures(item="cell")) == disk_10.nb_cells
    assert len(disk_10.measures(item="face")) == disk_10.nb_faces
    assert len(list(disk_10.face_groups.keys())) == 2


def test_disk_8_hole_4():
    disk_8_hole_4 = load_mesh_2d("./meshes/2D/Tests/disk_8_hole_4/disk_8_hole_4.mesh")

    assert disk_8_hole_4.nb_cells == 12
    assert disk_8_hole_4.nb_nodes == 12
    assert disk_8_hole_4.nb_faces == 24
    assert len(disk_8_hole_4.cells) == disk_8_hole_4.nb_cells
    assert len(disk_8_hole_4.measures(item="cell")) == disk_8_hole_4.nb_cells
    assert len(disk_8_hole_4.measures(item="face")) == disk_8_hole_4.nb_faces
    assert len(list(disk_8_hole_4.face_groups.keys())) == 3


def test_disk_8_hole_4_with_geom():
    disk_8_hole_4 = load_mesh_2d(
        "./meshes/2D/Tests_with_geom/disk_8_hole_4/disk_8_hole_4.mesh",
        "./meshes/2D/Tests_with_geom/disk_8_hole_4/disk_8_hole_4_geom_info.txt",
    )

    assert disk_8_hole_4.nb_cells == 12
    assert disk_8_hole_4.nb_nodes == 12
    assert disk_8_hole_4.nb_faces == 24
    assert len(disk_8_hole_4.cells) == disk_8_hole_4.nb_cells
    assert len(disk_8_hole_4.measures(item="cell")) == disk_8_hole_4.nb_cells
    assert len(disk_8_hole_4.measures(item="face")) == disk_8_hole_4.nb_faces
    assert len(list(disk_8_hole_4.face_groups.keys())) == 3


"""
def test_permea_twistz_with_geom():
    permea_twistz = load_mesh_2d(
        "./meshes/2D/Tests_with_geom/permea_twistz/permea_twistz.mesh",
        "./meshes/2D/Tests_with_geom/permea_twistz/permea_twistz_geom_info.txt",
    )

    assert len(permea_twistz.measures(item="cell")) == permea_twistz.nb_cells
    assert len(permea_twistz.measures(item="face")) == permea_twistz.nb_faces
"""


def test_disk_10_with_geom():
    disk_10 = load_mesh_2d(
        ".//meshes/2D/Tests_with_geom/disk_10/disk_10.mesh",
        "./meshes/2D/Tests_with_geom/disk_10/disk_10_geom_info.txt",
    )

    assert disk_10.nb_cells == 16
    assert disk_10.nb_nodes == 14
    assert disk_10.nb_faces == 29
    assert len(disk_10.cells) == disk_10.nb_cells
    assert len(disk_10.measures(item="cell")) == disk_10.nb_cells
    assert len(disk_10.measures(item="face")) == disk_10.nb_faces
    assert len(list(disk_10.face_groups.keys())) == 2


def test_square_1_1_with_geom():
    square_1_1 = load_mesh_2d(
        "./meshes/2D/Tests_with_geom/square_1_1/square_1_1.mesh",
        "./meshes/2D/Tests_with_geom/square_1_1/square_1_1_geom_info.txt",
    )

    assert square_1_1.nb_cells == 2
    assert square_1_1.nb_nodes == 4
    assert square_1_1.nb_faces == 5
    assert len(square_1_1.cells) == square_1_1.nb_cells
    assert len(square_1_1.measures(item="cell")) == square_1_1.nb_cells
    assert len(square_1_1.measures(item="face")) == square_1_1.nb_faces
    assert len(list(square_1_1.face_groups.keys())) == 5
