import pytest

from yads.mesh.one_D.load_1d import load_meshfile, load_mesh, load_msh  # type: ignore


def test_wrong_inputs():
    with pytest.raises(TypeError, match=r"meshfile must be a string"):
        load_meshfile(0)
    with pytest.raises(FileNotFoundError, match=r"invalid path to meshfile"):
        load_meshfile("error/path.msh")
    with pytest.raises(NotImplementedError, match=r"file extension not supported yet"):
        load_meshfile("error/path.err")
    with pytest.raises(FileNotFoundError, match=r"invalid path to meshfile"):
        load_msh("error/path.msh")
    with pytest.raises(FileNotFoundError, match=r"invalid path to meshfile"):
        load_mesh("error/path.mesh")
