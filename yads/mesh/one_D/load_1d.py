from yads.mesh.one_D.create_1d_mesh import create_1d
from yads.mesh import Mesh


def load_mesh(meshfile: str) -> Mesh:
    """loads a mesh .mesh from a meshfile

    Args:
        meshfile: path to a meshfile

    Returns:
        Mesh object corresponding to the given meshfile
    """
    try:
        with open(meshfile) as f:
            data = [line.replace("\n", "") for line in f.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError(f"invalid path to meshfile (got {meshfile})")

    # one_D case
    vx_line_nb = data.index("Vertices") + 1
    vx = int(data[vx_line_nb])
    nb_cells = vx - 1  # Only true for one_D

    lower_bound = float(data[vx_line_nb + 1].split(" ")[0])
    upper_bound = float(data[vx_line_nb + vx].split(" ")[0])
    interval = (lower_bound, upper_bound)
    return create_1d(nb_cells, interval)


def load_msh(meshfile: str) -> Mesh:
    """loads a mesh .msh from a meshfile

    Args:
        meshfile: path to a meshfile

    Returns:
        Mesh object corresponding to the given meshfile
    """
    try:
        with open(meshfile) as f:
            data = [line.replace("\n", "") for line in f.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError(f"invalid path to meshfile (got {meshfile})")

    # one_D case
    vx_line_nb = data.index("$Nodes") + 1
    vx = int(data[vx_line_nb])
    nb_cells = vx - 1  # Only true for one_D

    lower_bound = float(data[vx_line_nb + 1].split(" ")[1])
    upper_bound = float(data[vx_line_nb + vx].split(" ")[1])
    interval = (lower_bound, upper_bound)
    return create_1d(nb_cells, interval)


def load_meshfile(meshfile: str) -> Mesh:
    """loads a mesh from a meshfile, supported extensions are [.msh, .mesh]

    Args:
        meshfile: path to a meshfile

    Returns:
        Mesh object corresponding tot the given meshfile
    """

    if not isinstance(meshfile, str):
        raise TypeError(f"meshfile must be a string (got {type(meshfile)}")

    extension = meshfile.split(".")[-1]
    if extension == "mesh":
        return load_mesh(meshfile)
    elif extension == "msh":
        return load_msh(meshfile)
    else:
        raise NotImplementedError(
            f"file extension not supported yet (got {'.' + str(extension)}) "
        )
