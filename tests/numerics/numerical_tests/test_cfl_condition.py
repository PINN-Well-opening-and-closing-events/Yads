import pytest
import numpy as np  # type: ignore

from yads.mesh import load_mesh
from yads.numerics.numerical_tests.cfl_condition import cfl_condition
from yads.physics import dfw_dsw


def test_wrong_inputs():
    grid = load_mesh("./meshes/2D/Tests/rod_10_1/rod_10_1.mesh")
    F = np.ones(grid.nb_faces)
    F_well = {}
    phi = np.ones(grid.nb_cells)
    mu_wrong, mu_ok = -1, 1
    Pb = {"1": 2.0, "2": 1.0}

    with pytest.raises(ValueError, match=r"viscosity must be positive"):
        cfl_condition(grid, phi, F, F_well, dfw_dsw, Pb, mu_ok, mu_wrong)


def test_output():
    grid = load_mesh("./meshes/2D/Tests/rod_10_1/rod_10_1.mesh")
    F = np.ones(grid.nb_faces)
    F_well = {}
    phi = np.ones(grid.nb_cells)
    Pb = {"1": 2.0, "2": 1.0}
    dt, _ = cfl_condition(grid, phi, F, F_well, dfw_dsw, Pb, 1.0, 1.0)
    assert dt > 0.0
