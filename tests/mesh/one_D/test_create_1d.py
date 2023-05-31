import pytest

from yads.mesh.one_D.create_1d_mesh import create_1d  # type: ignore


def test_wrong_inputs():

    with pytest.raises(ValueError, match=r"nb_cells must be an integer > 0"):
        create_1d(-1, (0, 1))

    with pytest.raises(
        ValueError, match=r"interval must be a tuple of float \[L, U\] such that L < U"
    ):
        create_1d(1, (1, 0))

    with pytest.raises(ValueError, match=r"centers is valid for 'cell' or 'face'"):
        create_1d(2, (0, 1)).centers(item="error")

    with pytest.raises(ValueError, match=r"measures is valid for 'cell' or 'face'"):
        create_1d(1, (0, 1)).measures(item="error")

    with pytest.raises(ValueError, match=r"unknown group"):
        create_1d(1, (0, 1)).faces(group="error")


def test_cartesian_1d_one_cell():
    m = create_1d(1, (0, 2))

    assert m.nb_cells == 1
    assert m.nb_faces == 2
    assert m.nb_nodes == 2

    assert m.centers(item="cell") == [1]
    assert all([a == b for a, b in zip(m.centers(item="face"), [0, 2])])
    assert m.measures(item="cell") == [2]
    assert all([a == b for a, b in zip(m.measures(item="face"), [1, 1])])


def test_cartesian_1d_two_cell():
    m = create_1d(2, (0, 2))

    assert m.nb_cells == 2
    assert m.nb_faces == 3
    assert m.nb_nodes == 3

    assert all([a == b for a, b in zip(m.centers(item="cell"), [0.5, 1.5])])
    assert all([a == b for a, b in zip(m.centers(item="face"), [0, 1, 2])])
    assert all([a == b for a, b in zip(m.measures(item="cell"), [1, 1])])
    assert all([a == b for a, b in zip(m.measures(item="face"), [1, 1])])
