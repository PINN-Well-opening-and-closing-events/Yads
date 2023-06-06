import numpy as np

from yads.numerics.physics import peaceman_radius
from yads.wells import Well
from yads.physics.relative_permeability import kr, d_kr_ds
from yads.mesh import Mesh
from typing import Union, List


def compute_speed(
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

    F = np.zeros(2 * grid.nb_faces + 2)

    # for cell in range(grid.nb_cells):
    #     # accumulation term
    #     value = V[cell] * phi[cell] / dt_init
    #     accu_g = value * (S[cell] - S_i[cell])
    #     accu_w = value * ((1.0 - S[cell]) - (1.0 - S_i[cell]))
    #
    #     B[cell] = accu_g
    #     B[grid.nb_cells + cell] = accu_w

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

        F[face] += F_g
        F[grid.nb_faces + face] += F_w

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

                F[face] += F_g
                F[grid.nb_faces + face] += F_w

    if wells:
        for well in wells:
            ip = well.ip
            if "Neumann" in well.control:
                for c in grid.cell_groups[well.name]:
                    F[2 * grid.nb_faces] += well.control["Neumann"]
                    F[2 * grid.nb_faces + 1] += 0.0
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

                    F[2 * grid.nb_faces + 1] += F_g
                    F[2 * grid.nb_faces + 2] += F_w
    return F


def compute_grad_P(
    grid: Mesh, Pb, T, P_guess=None,
):
    P = P_guess
    grad_P = np.zeros(grid.nb_faces)

    # inner faces
    for face in grid.faces(group="0", with_nodes=False):
        # an inner face has always 2 adjacent cells
        # cells[0] -> front, cells[1] -> back
        front, back = grid.face_to_cell(face, face_type="inner")
        # upwinding
        value = T[face] * (P[front] - P[back])
        grad_P[face] += value

    # boundary faces
    for group in Pb.keys():
        if Pb[group]:
            for face in grid.faces(group=group, with_nodes=False):
                cell = grid.face_to_cell(face, face_type="boundary")
                value = T[face] * (P[cell] - Pb[group])
                grad_P[face] += value
    return grad_P
