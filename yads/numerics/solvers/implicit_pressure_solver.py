import numpy as np  # type: ignore
from typing import Union, List

from yads.mesh import Mesh
from yads.wells import Well

from yads.numerics.physics.peaceman_formula import peaceman_radius
from yads.physics.relative_permeability import kr


def implicit_pressure_solver(
    grid: Mesh,
    K: np.ndarray,
    T: np.ndarray,
    P: np.ndarray,
    S: np.ndarray,
    Pb: Union[dict, None],
    Sb_dict: Union[dict, None],
    mu_g: Union[float, int],
    mu_w: Union[float, int],
    wells: Union[List[Well], None] = None,
    kr_model: str = "cross",
) -> np.ndarray:
    """implicitly calculates pressure

    Args:
        grid: yads.mesh.cartesian.Cartesian object
        K: diffusion coefficient (i.e. permeability), np.ndarray size(grid.nb_cells)
        T: transmissivity , np.ndarray size(grid.nb_faces)
        S:
        P:
        Pb: pressure boundary conditions dict
            example: Pb = {"left":1.0, "right": 2.0}
        Sb_dict:
        wells: Well object describing a well
        mu_g: water viscosity
        mu_w: oil viscosity
        kr_model:
    Returns:
        pressure, size(grid.nb_cells)
    """

    # input checks
    if len(K) != grid.nb_cells:
        raise ValueError(
            f"K length must match grid.nb_cells: {grid.nb_cells} (got: {len(K)})"
        )
    if np.any((K < 0)):
        raise ValueError(f"Permeability K must contain only positive values")

    #### Pressure solver ####
    if wells:
        for well in wells:
            grid.connect_well(well)

    A = np.zeros((grid.nb_cells, grid.nb_cells))
    b = np.zeros(grid.nb_cells)

    # inner faces
    for f in grid.faces(group="0", with_nodes=False):

        i, j = grid.face_to_cell(f, face_type="inner")
        # upwinding
        if P[i] >= P[j]:
            m = kr(S[i], model=kr_model) / mu_g + kr(1.0 - S[i], model=kr_model) / mu_w
        else:
            m = kr(S[j], model=kr_model) / mu_g + kr(1.0 - S[j], model=kr_model) / mu_w

        value = T[f] * m
        A[i, i] += value
        A[i, j] -= value
        A[j, j] += value
        A[j, i] -= value

    # boundary faces
    if Pb:
        for bound in Pb.keys():
            for f in grid.faces(group=bound, with_nodes=False):

                c = grid.face_to_cell(f, face_type="boundary")
                # upwinding
                m = None
                if P[c] >= Pb[bound]:
                    m = (
                        kr(S[c], model=kr_model) / mu_g
                        + kr(1.0 - S[c], model=kr_model) / mu_w
                    )

                else:
                    if Sb_dict["Dirichlet"][bound] is not None:
                        m = (
                            kr(Sb_dict["Dirichlet"][bound], model=kr_model) / mu_g
                            + kr(1.0 - Sb_dict["Dirichlet"][bound], model=kr_model)
                            / mu_w
                        )
                    if Sb_dict["Neumann"][bound] is not None:
                        m = Sb_dict["Neumann"][bound]

                    if (
                        Sb_dict["Dirichlet"][bound] is None
                        and Sb_dict["Neumann"][bound] is None
                    ):
                        m = 0.0
                A[c, c] += T[f] * m
                b[c] += T[f] * m * Pb[bound]

    # standard well model
    if wells:
        for well in wells:
            if "Neumann" in well.control:
                for c in grid.cell_groups[well.name]:
                    value = well.control["Neumann"]
                    b[c] -= value

            elif "Dirichlet" in well.control:
                for c in grid.cell_groups[well.name]:
                    # upwinding
                    m = None
                    if P[c] >= well.control["Dirichlet"]:
                        m = (
                            kr(S[c], model=kr_model) / mu_g
                            + kr(1.0 - S[c], model=kr_model) / mu_w
                        )

                    else:
                        m = (
                            kr(well.injected_saturation, model=kr_model) / mu_g
                            + kr(1.0 - well.injected_saturation, model=kr_model) / mu_w
                        )
                    ip = well.ip
                    if ip < 0:
                        dx, dy = well.dx, well.dy
                        re = peaceman_radius(dx, dy)
                        ip = 2.0 * np.pi * K[c] / np.log(re / well.radius)
                        well.set_ip(ip)
                    value = ip * m
                    A[c, c] += value
                    b[c] += value * well.control["Dirichlet"]
    return np.linalg.solve(A, b)
