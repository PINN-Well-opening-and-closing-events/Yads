import pytest
import numpy as np  # type: ignore
from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian

from yads.numerics.physics.calculate_transmissivity import calculate_transmissivity_1d  # type: ignore
from yads.mesh.one_D.create_1d_mesh import create_1d  # type: ignore
from yads.mesh.two_D.load_2d import load_mesh_2d
from yads.numerics.physics.calculate_transmissivity import calculate_transmissivity_2d


def test_wrong_inputs():
    with pytest.raises(TypeError, match=r"grid must be a Mesh object"):
        grid_err = "error"
        K = np.ones(10)
        calculate_transmissivity_1d(grid_err, K)
    with pytest.raises(TypeError, match=r"K must be a np.ndarray"):
        grid = create_1d(nb_cells=10, interval=(0, 1))
        K_err = "error"
        calculate_transmissivity_1d(grid, K_err)


def test_calculate_transmissivity_1d():
    m = create_1d(nb_cells=4, interval=(0, 1))
    p = np.ones(m.nb_cells)
    T = calculate_transmissivity_1d(m, p)

    assert len(T) == m.nb_faces
    assert all([a == b for a, b in zip(T, [8.0, 4.0, 4.0, 4.0, 8.0])])


def test_calculate_transmissivity_2d():
    grid = create_2d_cartesian(50 * 200, 1000, 10, 1)
    K = np.random.random_sample((grid.nb_cells,)) + 1.0e-3
    T = calculate_transmissivity_2d(grid, K)
    assert len(T) == grid.nb_faces

    """
    grid = load_mesh_2d("meshes/2D/Tests_with_geom/permea_twistz/permea_twistz.mesh")
    K = np.random.random_sample((grid.nb_cells,)) + 1.0e-3
    T = calculate_transmissivity_2d(grid, K)
    assert len(T) == grid.nb_faces
    """
