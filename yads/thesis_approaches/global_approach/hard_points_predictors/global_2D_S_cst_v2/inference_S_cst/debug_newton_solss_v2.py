import copy
from typing import Union, List
import numpy as np

from yads.mesh import Mesh
import yads.mesh as ym
from yads.numerics import peaceman_radius
from yads.numerics.solvers import newton_relaxation
from yads.wells import Well
from yads.numerics.utils import clipping_P, clipping_S
from yads.physics.relative_permeability import kr, d_kr_ds
from yads.numerics.solvers.newton import res


def solss_newton_step_v2(
    grid: Mesh,
    P_i,
    S_i,
    Pb,
    Sb_dict,
    phi,
    K,
    T,
    mu_g,
    mu_w,
    dt_init,
    dt_min,
    wells: Union[List[Well], None] = None,
    max_newton_iter=20,
    eps=1e-6,
    kr_model="cross",
    P_guess=None,
    S_guess=None,
    debug_newton_mode=False,
    debug_newton_path="./debug_newton_autopath",
):
    V = grid.measures(item="cell")
    stop = False
    step = 0
    S_save = copy.deepcopy(S_i)
    P_save = copy.deepcopy(P_i)
    nb_newton = 0.0
    P = P_guess
    S = S_guess
    B_list = []

    norm_dict = {"L_inf": [], "L2": []}
    while not stop:
        ############## CREATE FUNCTION FOR IT ##################
        # system construction for the Newton
        jacobian = np.zeros((2 * grid.nb_cells, 2 * grid.nb_cells))
        B = np.zeros(2 * grid.nb_cells)

        for cell in range(grid.nb_cells):
            # accumulation term
            value = V[cell] * phi[cell] / dt_init
            # print(value, S[cell], S_i[cell])
            accu_g = value * (S[cell] - S_i[cell])
            accu_w = value * ((1.0 - S[cell]) - (1.0 - S_i[cell]))

            B[cell] = accu_g
            B[grid.nb_cells + cell] = accu_w
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
                # Flux
                F_g = kr(S[front], model=kr_model) / mu_g * value
                F_w = kr(1.0 - S[front], model=kr_model) / mu_w * value

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
                # Flux
                F_g = kr(S[back], model=kr_model) / mu_g * value
                F_w = kr(1.0 - S[back], model=kr_model) / mu_w * value

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

            B[front] += F_g
            B[grid.nb_cells + front] += F_w
            B[back] -= F_g
            B[grid.nb_cells + back] -= F_w

        # boundary faces
        for group in Pb.keys():
            for f in grid.faces(group=group, with_nodes=False):
                cell = grid.face_to_cell(f, face_type="boundary")
                value = T[f] * (P[cell] - Pb[group])
                if P[cell] >= Pb[group]:
                    # Flux
                    F_g = kr(S[cell], model=kr_model) / mu_g * value
                    F_w = kr(1.0 - S[cell], model=kr_model) / mu_w * value

                    dF_g_dS = (
                        d_kr_ds(S[cell], model=kr_model, negative=False) / mu_g * value
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
                        F_g = 0.0
                        F_w = 0.0
                        dF_w_dP = 0.0
                        dF_g_dP = 0.0
                B[cell] += F_g
                B[grid.nb_cells + cell] += F_w

                # derivatives w.r.t the Pressure
                jacobian[cell, cell] += dF_g_dP
                jacobian[grid.nb_cells + cell, cell] += dF_w_dP

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
                                print(
                                    f"{well.name} is in productor mode when it should be injector"
                                )
                                continue
                            F_g = kr(S[c], model=kr_model) / mu_g * value
                            F_w = kr((1.0 - S[c]), model=kr_model) / mu_w * value
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
                            F_g = (
                                kr(well.injected_saturation, model=kr_model)
                                / mu_g
                                * value
                            )

                            F_w = (
                                kr(1.0 - well.injected_saturation, model=kr_model)
                                / mu_w
                                * value
                            )
                            # print(F_g, F_w)
                            dF_g_dP = ip * well.injected_saturation / mu_g
                            dF_w_dP = ip * (1.0 - well.injected_saturation) / mu_w

                        B[c] += F_g
                        B[grid.nb_cells + c] += F_w
                        # derivatives w.r.t the Pressure
                        jacobian[c, c] += dF_g_dP
                        jacobian[grid.nb_cells + c, c] += dF_w_dP

        # Algebric correction of jacobian in case of injection phase disappears
        for i in range(grid.nb_cells):
            # injection phase lines
            jacobian[i] += jacobian[grid.nb_cells + i]
            B[i] += B[grid.nb_cells + i]
        #####################################################################

        if debug_newton_mode:
            debug_newton_path = debug_newton_path.replace(".json", "")
            filename = debug_newton_path + "_" + str(step)
            print(f"exporting debug vtk at {filename}")
            ym.export_vtk(
                filename,
                grid=grid,
                cell_data={"P": P, "S gas": S, "S water": 1.0 - S, "K": K, "phi": phi,},
            )
        # ITERCRIT criterion (test on the norm of the normalised residual Res)
        norm = np.max(np.abs(B) * dt_init / np.concatenate([V, V]))
        if len(norm_dict["L_inf"]) != 0:
            print(norm / norm_dict["L_inf"][-1] ** 2)
        norm_dict["L_inf"].append(norm)
        norm_dict["L2"].append(np.linalg.norm(B, ord=2))
        print(norm_dict)
        B_list.append(B.tolist())
        if norm <= eps:
            stop = True

        if not stop:
            # Newton solving
            PS = np.concatenate([P, S])
            try:
                delta_PS = np.linalg.solve(jacobian, -B)
                # compute relaxation factor
                relax = newton_relaxation.compute_relaxation(
                    S_obj=0.1, delta_S=delta_PS[grid.nb_cells :]
                )
                if relax != 1.0:
                    print(f"relaxed with relax {relax}")

                PS += relax * delta_PS
                P, S = PS[: grid.nb_cells], PS[grid.nb_cells :]
                S = clipping_S(S)
                P = clipping_P(P, P_min=0.0, P_max=500.0e5)
                # stop criterion
                nb_newton += 1.0
                step += 1
            except np.linalg.LinAlgError:
                print("Linalg error")
                if dt_init / 2 >= dt_min:
                    print(f"Singular matrix, trying with new dt: {dt_init / 2}")
                    return solss_newton_step_v2(
                        grid=grid,
                        P_i=P_save,
                        S_i=S_save,
                        Pb=Pb,
                        Sb_dict=Sb_dict,
                        phi=phi,
                        K=K,
                        T=T,
                        mu_g=mu_g,
                        mu_w=mu_w,
                        dt_init=dt_init / 2,
                        dt_min=dt_min,
                        wells=wells,
                        max_newton_iter=max_newton_iter,
                        eps=eps,
                        kr_model=kr_model,
                        P_guess=P_guess,
                        S_guess=S_guess,
                        debug_newton_mode=debug_newton_mode,
                        debug_newton_path=debug_newton_path,
                    )
                else:
                    print("timestep has reached minimal timestep allowed")
                    nb_newton = 123456789
                    dt_init = -1
                    stop = True

        if step == max_newton_iter and not stop:
            # check if last step has converged before changing dt
            B = res(
                grid=grid,
                S_i=S_i,
                Pb=Pb,
                Sb_dict=Sb_dict,
                phi=phi,
                K=K,
                T=T,
                mu_g=mu_g,
                mu_w=mu_w,
                dt_init=dt_init,
                wells=wells,
                kr_model=kr_model,
                P_guess=P,
                S_guess=S,
            )
            norm = np.max(np.abs(B) * dt_init / np.concatenate([V, V]))
            norm_dict["L2"].append(np.linalg.norm(B, ord=2))
            norm_dict["L_inf"].append(norm)
            if norm <= eps:
                return P, S, dt_init, nb_newton, norm_dict, B_list

            if dt_init / 2 >= dt_min:
                print(
                    "Newton has not converged in {} steps, trying with new dt: {}".format(
                        max_newton_iter, dt_init / 2
                    )
                )
                return solss_newton_step_v2(
                    grid=grid,
                    P_i=P_save,
                    S_i=S_save,
                    Pb=Pb,
                    Sb_dict=Sb_dict,
                    phi=phi,
                    K=K,
                    T=T,
                    mu_g=mu_g,
                    mu_w=mu_w,
                    dt_init=dt_init / 2,
                    dt_min=dt_min,
                    wells=wells,
                    max_newton_iter=max_newton_iter,
                    eps=eps,
                    kr_model=kr_model,
                    P_guess=P_guess,
                    S_guess=S_guess,
                    debug_newton_mode=debug_newton_mode,
                    debug_newton_path=debug_newton_path,
                )
            else:
                print("timestep has reached minimal timestep allowed")
                nb_newton = 123456789
                dt_init = -1
                stop = True
    return P, S, dt_init, nb_newton, norm_dict, B_list
