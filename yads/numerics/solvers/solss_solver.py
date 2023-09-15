import copy
from typing import Union, List

import numpy as np

from yads.mesh import Mesh
from yads.numerics.utils import clipping_S, clipping_P
from yads.wells import Well
from yads.numerics.solvers import newton_relaxation
from yads.numerics.solvers.newton import res, j


def solss_newton_step(
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
    debug_newton_path="./debug_newton_autopath.json",
):

    step = 0
    S_save = copy.deepcopy(S_i)
    P_save = copy.deepcopy(P_i)
    nb_newton = 0.0

    P = P_guess
    S = S_guess

    norm_dict = {"L_inf": [], "L2": [], "B": []}

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
    V = grid.measures(item="cell")
    # ITERCRIT criterion (test on the norm of the normalised residual Res)
    norm = np.max(np.abs(B) * dt_init / np.concatenate([V, V]))
    norm_dict["B"].append(B)
    norm_dict["L_inf"].append(norm)
    norm_dict["L2"].append(np.linalg.norm(B, ord=2))
    # print(f"norm: {np.linalg.norm(B, ord=2):0.2E}")
    while not norm <= eps:
        jacobian = j(
            grid=grid,
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
        # Newton solving
        PS = np.concatenate([P, S])
        delta_PS = np.linalg.solve(jacobian, -B)

        relax = newton_relaxation.compute_relaxation(
            S_obj=0.1, delta_S=delta_PS[grid.nb_cells :]
        )
        PS += relax * delta_PS

        P, S = PS[: grid.nb_cells], PS[grid.nb_cells :]

        S = clipping_S(S)
        P = clipping_P(P, P_min=0.0, P_max=500.0e6)

        # stop criterion
        nb_newton += 1
        step += 1

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
        norm_dict["B"].append(B)
        norm_dict["L_inf"].append(norm)
        norm_dict["L2"].append(np.linalg.norm(B, ord=2))
        # print(f"norm: {np.linalg.norm(B, ord=2):0.2E}")
        if debug_newton_mode:
            import json
            if step - 1 == 0:
                debug_dict = {'newton_step_data': {}}
            else:
                with open(debug_newton_path, "r") as f:
                    debug_dict = json.load(f)
            debug_dict["newton_step_data"][step] = {
                "P_i_plus_1": P.tolist(),
                "S_i_plus_1": S.tolist(),
                "Residual": B.tolist(),
            }
            with open(debug_newton_path, "w") as f:
                json.dump(debug_dict, f)

        if step == max_newton_iter:
            if norm <= eps:
                return P, S, dt_init, nb_newton, norm_dict

            if dt_init / 2 >= dt_min:
                print(
                    "Newton has not converged in {} steps, trying with new dt: {}".format(
                        max_newton_iter, dt_init / 2
                    )
                )
                return solss_newton_step(
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
                dt_init = -1
                break
    # print(nb_newton, norms)
    return P, S, dt_init, nb_newton, norm_dict
