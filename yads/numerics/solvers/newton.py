import numpy as np

from yads.numerics.physics import peaceman_radius
from yads.wells import Well
from yads.physics.relative_permeability import kr, d_kr_ds
from yads.mesh import Mesh
from typing import Union, List


def res(
    grid: Mesh,
    S_i,
    Pb,
    Sb_dict,
    phi,
    K,
    T,
    mu_g,
    mu_w,
    dt_init,
    wells: Union[List[Well], None] = None,
    kr_model="cross",
    P_guess=None,
    S_guess=None,
):
    V = grid.measures(item="cell")
    P = P_guess
    S = S_guess

    if wells:
        for well in wells:
            grid.connect_well(well)

    # system construction for the Newton
    B = np.zeros(2 * grid.nb_cells)

    for cell in range(grid.nb_cells):
        # accumulation term
        value = V[cell] * phi[cell] / dt_init
        accu_g = value * (S[cell] - S_i[cell])
        accu_w = value * ((1.0 - S[cell]) - (1.0 - S_i[cell]))

        B[cell] = accu_g
        B[grid.nb_cells + cell] = accu_w

    # inner faces
    for face in grid.faces(group="0", with_nodes=False):
        # an inner face has always 2 adjacent cells
        # cells[0] -> front, cells[1] -> back
        front, back = grid.face_to_cell(face, face_type="inner")
        # upwinding
        value = T[face] * (P[front] - P[back])
        if P[front] >= P[back]:
            # Flux
            F_g = kr(S[front], model=kr_model) / mu_g * value
            F_w = kr(1.0 - S[front], model=kr_model) / mu_w * value

        else:
            # Flux
            F_g = kr(S[back], model=kr_model) / mu_g * value
            F_w = kr(1.0 - S[back], model=kr_model) / mu_w * value

        B[front] += F_g
        B[grid.nb_cells + front] += F_w
        B[back] -= F_g
        B[grid.nb_cells + back] -= F_w

    # boundary faces
    for group in Pb.keys():
        if Pb[group]:
            for face in grid.faces(group=group, with_nodes=False):
                cell = grid.face_to_cell(face, face_type="boundary")
                value = T[face] * (P[cell] - Pb[group])
                if P[cell] >= Pb[group]:
                    # Flux
                    F_g = kr(S[cell], model=kr_model) / mu_g * value
                    F_w = kr(1.0 - S[cell], model=kr_model) / mu_w * value

                else:
                    if Sb_dict["Dirichlet"][group] is not None:
                        F_g = (
                            kr(Sb_dict["Dirichlet"][group], model=kr_model)
                            / mu_g
                            * value
                        )
                        F_w = (
                            kr(1.0 - Sb_dict["Dirichlet"][group], model=kr_model)
                            / mu_w
                            * value
                        )

                    else:
                        ######### RISKY NULL FLUX #####
                        F_g = 0.0
                        F_w = 0.0

                B[cell] += F_g
                B[grid.nb_cells + cell] += F_w

    if wells:
        for well in wells:
            ip = well.ip
            if "Neumann" in well.control:
                for c in grid.cell_groups[well.name]:
                    B[c] += well.control["Neumann"]

            elif "Dirichlet" in well.control:
                for c in grid.cell_groups[well.name]:
                    if ip < 0:
                        dx, dy = well.dx, well.dy
                        re = peaceman_radius(dx, dy)
                        ip = 2.0 * np.pi * K[c] / np.log(re / well.radius)
                        well.set_ip(ip)

                    value = ip * (P[c] - well.control["Dirichlet"])
                    # upwinding
                    if P[c] >= well.control["Dirichlet"]:
                        # Productor
                        if well.is_injector:
                            continue
                        F_g = kr(S[c], model=kr_model) / mu_g * value
                        F_w = kr((1.0 - S[c]), model=kr_model) / mu_w * value

                    else:
                        # Injector
                        if well.is_productor:
                            # print(f"{well.name} is in injector mode when it should be injector")
                            continue
                        F_g = (
                            kr(well.injected_saturation, model=kr_model) / mu_g * value
                        )

                        F_w = (
                            kr(1.0 - well.injected_saturation, model=kr_model)
                            / mu_w
                            * value
                        )

                    B[c] += F_g
                    B[grid.nb_cells + c] += F_w

    # Algebric correction of jacobian in case of injection phase disappears
    for i in range(grid.nb_cells):
        B[i] += B[grid.nb_cells + i]
    return B


def j(
    grid: Mesh,
    Pb,
    Sb_dict,
    phi,
    K,
    T,
    mu_g,
    mu_w,
    dt_init,
    wells: Union[List[Well], None] = None,
    kr_model="cross",
    P_guess=None,
    S_guess=None,
):

    V = grid.measures(item="cell")
    P = P_guess
    S = S_guess

    # system construction for the Newton
    jacobian = np.zeros((2 * grid.nb_cells, 2 * grid.nb_cells))

    for cell in range(grid.nb_cells):
        # accumulation term
        value = V[cell] * phi[cell] / dt_init

        # d_accu_g_ds
        jacobian[cell, grid.nb_cells + cell] += value
        # d_accu_w_ds
        jacobian[grid.nb_cells + cell, grid.nb_cells + cell] -= value

    # inner faces
    for f in grid.faces(group="0", with_nodes=False):
        # an inner face has always 2 adjacent cells
        # cells[0] -> front, cells[1] -> back
        front, back = grid.face_to_cell(f, face_type="inner")
        # upwinding
        value = T[f] * (P[front] - P[back])
        if P[front] >= P[back]:
            dF_g_dS = (
                    d_kr_ds(S[front], model=kr_model, negative=False) * value / mu_g
            )
            dF_w_dS = (
                    d_kr_ds(1.0 - S[front], model=kr_model, negative=True)
                    * value
                    / mu_w
            )
            # dF_g_dP
            dF_g_dP = kr(S[front], model=kr_model) * T[f] / mu_g

            # dF_w_dP
            dF_w_dP = kr(1.0 - S[front], model=kr_model) * T[f] / mu_w

            # derivatives w.r.t the Saturation
            jacobian[front, grid.nb_cells + front] += dF_g_dS
            jacobian[grid.nb_cells + front, grid.nb_cells + front] += dF_w_dS

            jacobian[back, grid.nb_cells + front] -= dF_g_dS
            jacobian[grid.nb_cells + back, grid.nb_cells + front] -= dF_w_dS

        else:
            dF_g_dS = (
                    d_kr_ds(S[back], model=kr_model, negative=False) * value / mu_g
            )
            dF_w_dS = (
                    d_kr_ds(1.0 - S[back], model=kr_model, negative=True) * value / mu_w
            )
            dF_g_dP = kr(S[back], model=kr_model) * T[f] / mu_g
            dF_w_dP = kr(1.0 - S[back], model=kr_model) * T[f] / mu_w

            # derivatives w.r.t the Saturation
            jacobian[front, grid.nb_cells + back] += dF_g_dS
            jacobian[grid.nb_cells + front, grid.nb_cells + back] += dF_w_dS

            jacobian[back, grid.nb_cells + back] -= dF_g_dS
            jacobian[grid.nb_cells + back, grid.nb_cells + back] -= dF_w_dS

        # derivatives w.r.t the Pressure
        jacobian[front, front] += dF_g_dP
        jacobian[grid.nb_cells + front, front] += dF_w_dP

        jacobian[front, back] += -dF_g_dP
        jacobian[grid.nb_cells + front, back] += -dF_w_dP

        jacobian[back, front] -= dF_g_dP
        jacobian[grid.nb_cells + back, front] -= dF_w_dP

        jacobian[back, back] -= -dF_g_dP
        jacobian[grid.nb_cells + back, back] -= -dF_w_dP

    # boundary faces
    for group in Pb.keys():
        if Pb[group]:
            for f in grid.faces(group=group, with_nodes=False):
                cell = grid.face_to_cell(f, face_type="boundary")
                value = T[f] * (P[cell] - Pb[group])
                if P[cell] >= Pb[group]:
                    dF_g_dS = (
                            d_kr_ds(S[cell], model=kr_model, negative=False)
                            / mu_g
                            * value
                    )
                    dF_w_dS = (
                            d_kr_ds(1.0 - S[cell], model=kr_model, negative=True)
                            / mu_w
                            * value
                    )
                    dF_g_dP = kr(S[cell], model=kr_model) * T[f] / mu_g
                    dF_w_dP = kr(1.0 - S[cell], model=kr_model) * T[f] / mu_w

                    # derivatives w.r.t the Saturation
                    jacobian[cell, grid.nb_cells + cell] += dF_g_dS
                    jacobian[grid.nb_cells + cell, grid.nb_cells + cell] += dF_w_dS
                else:
                    if Sb_dict["Dirichlet"][group] is not None:
                        dF_g_dP = (
                                kr(Sb_dict["Dirichlet"][group], model=kr_model)
                                * T[f]
                                / mu_g
                        )
                        dF_w_dP = (
                                kr(1.0 - Sb_dict["Dirichlet"][group], model=kr_model)
                                * T[f]
                                / mu_w
                        )
                    else:
                        ######### RISKY NULL FLUX #####
                        dF_w_dP = 0.0
                        dF_g_dP = 0.0

                # derivatives w.r.t the Pressure
                jacobian[cell, cell] += dF_g_dP
                jacobian[grid.nb_cells + cell, cell] += dF_w_dP

    if wells:
        for well in wells:
            ip = well.ip
            if "Dirichlet" in well.control:
                for c in grid.cell_groups[well.name]:
                    if ip < 0:
                        dx, dy = well.dx, well.dy
                        re = peaceman_radius(dx, dy)
                        ip = 2.0 * np.pi * K[c] / np.log(re / well.radius)
                        well.set_ip(ip)

                    value = ip * (P[c] - well.control["Dirichlet"])
                    # upwinding
                    if P[c] >= well.control["Dirichlet"]:
                        # Productor
                        if well.is_injector:
                            print(
                                f"{well.name} is in productor mode when it should be injector"
                            )
                            continue
                        dF_g_dS = (
                                d_kr_ds(S[c], model=kr_model, negative=False)
                                * value
                                / mu_g
                        )
                        dF_w_dS = (
                                d_kr_ds(1.0 - S[c], model=kr_model, negative=True)
                                * value
                                / mu_w
                        )
                        dF_g_dP = ip / mu_g * kr(S[c], model=kr_model)
                        dF_w_dP = ip / mu_w * kr(1.0 - S[c], model=kr_model)

                        jacobian[c, grid.nb_cells + c] += dF_g_dS
                        jacobian[grid.nb_cells + c, grid.nb_cells + c] += dF_w_dS
                    else:
                        # Injector
                        if well.is_productor:
                            print(
                                f"{well.name} is in injector mode when it should be productor"
                            )
                            continue

                        dF_g_dP = ip * well.injected_saturation / mu_g
                        dF_w_dP = ip * (1.0 - well.injected_saturation) / mu_w

                    # derivatives w.r.t the Pressure
                    jacobian[c, c] += dF_g_dP
                    jacobian[grid.nb_cells + c, c] += dF_w_dP

    # Algebric correction of jacobian in case of injection phase disappears
    for i in range(grid.nb_cells):
        # injection phase lines
        jacobian[i] += jacobian[grid.nb_cells + i]
    return jacobian
