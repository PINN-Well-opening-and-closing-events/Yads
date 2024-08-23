import copy

import numpy as np  # type: ignore
from typing import Union, List

from yads.mesh import Mesh
from yads.wells import Well

from yads.numerics.utils import clipping_S
from yads.physics import kr, d_kr_ds


def implicit_saturation_solver(
    grid: Mesh,
    P: np.ndarray,
    S: np.ndarray,
    T: np.ndarray,
    phi: np.ndarray,
    Pb: dict,
    Sb_dict: dict,
    dt: Union[float, int],
    mu_g: Union[float, int],
    wells: Union[List[Well], None] = None,
    eps: Union[float, int] = 1e-8,
    max_newton_iter=20,
    kr_model: str = "cross",
):
    """implicitly Updates saturation through a Newton

    Args:
        grid: yads.mesh.Mesh object
        P: pressure, np.ndarray size(grid.nb_cells)
        S: water saturation, np.ndarray size(grid.nb_cells)
        T: transmissivity , np.ndarray size(grid.nb_faces)
        phi: porosity in each cell, np.ndarray size(grid.nb_cells)
        Pb: pressure boundary conditions dict
            example: Pb = {"left":1.0, "right": 2.0}
        Sb_dict: water saturation boundary conditions dict
            example: Sb_dict = {"Neumann":{"left":1.0, "right": 0.2}, "Dirichlet": "left":None, "right":None}
        dt: time step
        mu_g: gaz/injection phase viscosity
        wells:
        eps: newton tolerance
        max_newton_iter:
        kr_model:
    Returns:
        updated water saturation
    """
    # inputs checks
    if len(phi) != grid.nb_cells:
        raise ValueError(
            f"phi length must match grid.nb_cells: {grid.nb_cells} (got: {len(phi)})"
        )
    if not all([1.0 >= sw >= 0.0 for sw in S]):
        print("S:", S)
        print("P:", P)
        raise ValueError(f"Saturation S must have all its values between 0 and 1")
    if np.any((phi < 0)):
        raise ValueError(f"Porosity phi must contain only positive values")
    if dt < 0:
        raise ValueError(f"Time step dt must be positive: (got {dt})")
    if eps <= 0.0:
        raise ValueError(f"newton stop criterion eps must be positive: (got {dt})")

    #### Newton constants ####
    cell_measures = grid.measures(item="cell")
    dt_value = dt / np.multiply(cell_measures, phi)
    S_i_plus_1 = None
    #### Newton construction ####
    stop = False
    step = 0
    S_save = copy.deepcopy(S)

    while not stop:
        # jacobian and flux init
        jacobian = np.zeros((grid.nb_cells, grid.nb_cells))
        N_i = np.zeros(grid.nb_cells)
        if step == 0:
            S_i = S
        else:
            S_i = S_i_plus_1

        # inner faces
        for f in grid.faces(group="0", with_nodes=False):
            # an inner face has always 2 adjacent cells
            # cells[0] -> front, cells[1] -> back
            front, back = grid.face_to_cell(f, face_type="inner")
            # upwinding
            value = T[f] * (P[front] - P[back])
            if P[front] >= P[back]:
                F = kr(S_i[front], model=kr_model) / mu_g * value
                dF_ds = (
                    d_kr_ds(S_i[front], model=kr_model, negative=False) / mu_g * value
                )
                # front cell
                jacobian[front, front] += dF_ds
                # back cell
                jacobian[back, front] -= dF_ds

            else:
                F = kr(S_i[back], model=kr_model) / mu_g * value
                dF_ds = d_kr_ds(S_i[back], model=kr_model) / mu_g * value
                # front cell
                jacobian[front, back] += dF_ds
                # back cell
                jacobian[back, back] -= dF_ds
            # front cell
            N_i[front] += F
            # back cell
            N_i[back] -= F

        # boundary faces
        for group in Pb.keys():
            for f in grid.faces(group=group, with_nodes=False):
                cell = grid.face_to_cell(f, face_type="boundary")
                value = T[f] * (P[cell] - Pb[group])
                if P[cell] >= Pb[group]:
                    F = kr(S_i[cell], model=kr_model) / mu_g * value
                    dF_ds = (
                        d_kr_ds(S_i[cell], model=kr_model, negative=False)
                        / mu_g
                        * value
                    )
                    jacobian[cell, cell] += dF_ds
                else:
                    F = kr(Sb_dict["Dirichlet"][group], model=kr_model) / mu_g * value
                N_i[cell] += F

        if wells:
            for well in wells:
                if "Neumann" in well.control:
                    for c in grid.cell_groups[well.name]:
                        N_i[c] += well.control["Neumann"]

                elif "Dirichlet" in well.control:
                    for c in grid.cell_groups[well.name]:
                        # ip already set in IMP part
                        ip = well.ip
                        # upwinding
                        value = ip * (P[c] - well.control["Dirichlet"])
                        if P[c] >= well.control["Dirichlet"]:
                            F = kr(S_i[c], model=kr_model) / mu_g * value
                            dF_ds = (
                                d_kr_ds(S_i[c], model=kr_model, negative=False)
                                / mu_g
                                * value
                            )
                            jacobian[c, c] += dF_ds
                        else:
                            F = (
                                kr(well.injected_saturation, model=kr_model)
                                / mu_g
                                * value
                            )
                        N_i[c] += F
        jacobian = np.multiply(jacobian, dt_value)
        N_i = np.multiply(N_i, dt_value)
        # accumulation term:
        jacobian = np.eye(grid.nb_cells) + jacobian
        N_i = S_i - S + N_i
        delta_S = np.linalg.solve(jacobian, -N_i)
        S_i_plus_1 = S_i + delta_S
        S_i_plus_1 = clipping_S(S_i_plus_1)
        step += 1
        # stop criterion
        norm = np.linalg.norm(N_i, ord=2)
        # print(f"norm: {norm:0.2E}")
        if norm <= eps:
            stop = True

        if step == max_newton_iter:
            print("Newton has not converged in {} steps".format(max_newton_iter))
            return implicit_saturation_solver(
                grid,
                P,
                S_save,
                T,
                phi,
                Pb,
                Sb_dict,
                dt / 2.0,
                mu_g,
                wells=wells,
                max_newton_iter=max_newton_iter,
                kr_model=kr_model,
                eps=eps,
            )

    return S_i_plus_1, dt, step
