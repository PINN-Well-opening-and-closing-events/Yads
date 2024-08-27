from typing import Union, List
import copy
from yads.mesh import Mesh

from yads.numerics.solvers.solss_solver import solss_newton_step
from yads.numerics.solvers.implicit_pressure_solver import implicit_pressure_solver
from yads.wells import Well
from yads.numerics.physics import (
    calculate_transmissivity,
    compute_speed,
    compute_grad_P,
)


def raw_solss(
    grid: Mesh,
    P,
    S,
    Pb,
    Sb_dict,
    phi,
    K,
    mu_g,
    mu_w,
    dt_init,
    total_sim_time,
    kr_model: str = "cross",
    max_newton_iter=10,
    eps=1e-6,
    wells: Union[List[Well], None] = None,
):
    """

    grid:
    P:
    S: initial gas saturation or injected phase
    Pb:
    Sb_dict:
    phi:
    K:
    mu_g: gas viscosity or injected phase viscosity
    mu_w:
    kr_model:
    dt_init:
    total_sim_time:
    auto_dt:
    eps:

    Returns

    """
    T = calculate_transmissivity(grid, K)
    stop = False
    step = 0
    dt = dt_init
    S_i_plus_1 = None
    P_i_plus_1 = None
    dt_list = []
    total_time = 0.0
    newton_list = []
    dt_min = 1

    simulation_state = {
        "metadata": {
            "kr_model": kr_model,
            "P0": P.tolist(),
            "S0": S.tolist(),
            "Pb": Pb,
            "Sb_dict": Sb_dict,
            "T": T.tolist(),
            "K": K.tolist(),
            "Phi": phi.tolist(),
            "dt": dt_init,
            "wells": [well.well_to_dict() for well in wells],
            "eps": eps,
            "max_newton_iter": max_newton_iter,
            "mu_w": mu_w,
            "mu_g": mu_g,
            "total_sim_time": total_sim_time,
        },
        "data": {},
    }

    simulation_state["metadata"]["grid data"] = {
        "Lx": max(grid.node_coordinates[:, 0]),
        "Ly": max(grid.node_coordinates[:, 1]),
        "nb_cells": grid.nb_cells,
        "nb_faces": grid.nb_faces,
        "nb_nodes": grid.nb_nodes,
        "cell_centers": grid.centers(item="cell").tolist(),
        "type": grid.type,
        "dimension": grid.dim,
    }
    if wells:
        for well in wells:
            grid.connect_well(well)

    while not stop:
        dt_save = copy.deepcopy(dt)
        S_save = copy.deepcopy(S)
        effective_wells = []

        if wells:
            for well in wells:
                for schedule in well.schedule:
                    if schedule[0] <= total_time < schedule[1]:
                        effective_wells.append(well)

        if step == 0:
            S_i = S
            P_i = P

        else:
            S_i = S_i_plus_1
            P_i = P_i_plus_1

        P_guess, S_guess = P_i, S_i

        P_i_plus_1, S_i_plus_1, dt, nb_newton = solss_newton_step(
            grid=grid,
            P_i=P_i,
            S_i=S_i,
            Pb=Pb,
            Sb_dict=Sb_dict,
            phi=phi,
            K=K,
            T=T,
            mu_g=mu_g,
            mu_w=mu_w,
            dt_init=dt,
            dt_min=dt_min,
            wells=effective_wells,
            max_newton_iter=max_newton_iter,
            eps=eps,
            kr_model=kr_model,
            P_guess=P_guess,
            S_guess=S_guess,
        )
        # check if newton has failed
        if dt < 0:
            stop = True
            continue

        # number of newton iterations
        newton_list.append(nb_newton)
        # number of newton fails
        i = 0
        if dt != dt_save:
            while dt_save / 2**i != dt:
                newton_list.append(-1)
                i += 1
        dt_list.append(dt)

        total_time += dt

        step += 1

        # update simulation state
        simulation_state["data"][str(total_time)] = {
            "step": step,
            "P": P_i_plus_1.tolist(),
            "S": S_i_plus_1.tolist(),
            "dt": dt,
            "nb_newton": max_newton_iter * i + nb_newton,
            "total_time": total_time,
        }
        if total_time >= total_sim_time:
            stop = True
            continue

        # schedule check and total time check
        for well in wells:
            # 1st case: well is already effective, we check for the closing
            if well in effective_wells:
                for schedule in well.schedule:
                    # check in which opening/closing we are
                    if schedule[0] < total_time < schedule[1]:
                        # check if the next timestep brings the simulation after the closing
                        if total_time + dt > schedule[1]:
                            # update timestep to fit with the schedule
                            dt = schedule[1] - total_time
            # 2nd case: well is not yet effective
            else:
                for schedule in well.schedule:
                    # find the first next schedule for well opening
                    if total_time < schedule[0]:
                        if total_time + dt > schedule[0]:
                            # update timestep to fit with the schedule
                            dt = schedule[0] - total_time
                            break
        assert dt != 0

        if total_time + dt > total_sim_time:
            dt = total_sim_time - total_time

    return simulation_state


def raw_solss_1_iter(
    grid: Mesh,
    P,
    S,
    Pb,
    Sb_dict,
    phi,
    K,
    mu_g,
    mu_w,
    dt_init,
    total_sim_time,
    kr_model: str = "cross",
    max_newton_iter=10,
    eps=1e-6,
    wells: Union[List[Well], None] = None,
    debug_newton_mode=False,
):
    """

    grid:
    P:
    S: initial gas saturation or injected phase
    Pb:
    Sb_dict:
    phi:
    K:
    mu_g: gas viscosity or injected phase viscosity
    mu_w:
    kr_model:
    dt_init:
    total_sim_time:
    auto_dt:
    eps:

    Returns

    """
    P_init = copy.deepcopy(P)
    T = calculate_transmissivity(grid, K)
    P = implicit_pressure_solver(
        grid=grid,
        K=K,
        T=T,
        P=P,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        mu_g=mu_g,
        mu_w=mu_w,
        kr_model=kr_model,
        wells=wells,
    )
    # print(grid.face_groups.items())
    grad_P = compute_grad_P(grid=grid, Pb=Pb, T=T, P_guess=P)

    F = compute_speed(
        grid=grid,
        S_i=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        T=T,
        dt_init=dt_init,
        mu_g=mu_g,
        mu_w=mu_w,
        kr_model=kr_model,
        P_guess=P,
        S_guess=S,
        wells=wells,
    )
    step = 0
    total_time = 0.0

    simulation_state = {
        "metadata": {
            "kr_model": kr_model,
            "P_init": P_init.tolist(),
            "P_imp": P.tolist(),
            "F": F.tolist(),
            "grad_P": grad_P.tolist(),
            "Pb": Pb,
            "Sb_dict": Sb_dict,
            "T": T.tolist(),
            "K": K.tolist(),
            "Phi": phi.tolist(),
            "dt_init": dt_init,
            "wells": [well.well_to_dict() for well in wells],
            "eps": eps,
            "max_newton_iter": max_newton_iter,
            "mu_w": mu_w,
            "mu_g": mu_g,
            "total_sim_time": total_sim_time,
        },
        "data": {},
    }

    simulation_state["metadata"]["grid data"] = {
        "Lx": max(grid.node_coordinates[:, 0]),
        "Ly": max(grid.node_coordinates[:, 1]),
        "nb_cells": grid.nb_cells,
        "nb_faces": grid.nb_faces,
        "nb_nodes": grid.nb_nodes,
        "cell_centers": grid.centers(item="cell").tolist(),
        "type": grid.type,
        "dimension": grid.dim,
    }

    if wells:
        for well in wells:
            grid.connect_well(well)

    effective_wells = []

    if wells:
        for well in wells:
            for schedule in well.schedule:
                if schedule[0] <= total_time < schedule[1]:
                    effective_wells.append(well)

    S_i = S
    P_i = P

    P_guess, S_guess = P_i, S_i
    P_i_plus_1, S_i_plus_1, dt, nb_newton, norm_dict = solss_newton_step(
        grid=grid,
        P_i=P_i,
        S_i=S_i,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        T=T,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=dt_init,
        dt_min=dt_init,
        wells=effective_wells,
        max_newton_iter=max_newton_iter,
        eps=eps,
        kr_model=kr_model,
        P_guess=P_guess,
        S_guess=S_guess,
        debug_newton_mode=debug_newton_mode,
        debug_newton_path="test.json",
    )

    F_final = compute_speed(
        grid=grid,
        S_i=S_i,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        T=T,
        dt_init=dt_init,
        mu_g=mu_g,
        mu_w=mu_w,
        kr_model=kr_model,
        P_guess=P_i_plus_1,
        S_guess=S_i_plus_1,
        wells=wells,
    )

    total_time += dt
    # update simulation state
    simulation_state["data"][str(total_time)] = {
        "step": step,
        "S0": S.tolist(),
        "P": P_i_plus_1.tolist(),
        "S": S_i_plus_1.tolist(),
        "dt": dt,
        "dt_init": dt_init,
        "nb_newton": nb_newton,
        "total_time": total_time,
        "Res": norm_dict["B"][-1].tolist(),
        "F_final": F_final.tolist(),
    }
    assert (
        simulation_state["data"][str(total_time)]["dt_init"]
        == simulation_state["metadata"]["dt_init"]
    )
    return simulation_state


def raw_solss_n_iter(
    grid: Mesh,
    P,
    S,
    Pb,
    Sb_dict,
    phi,
    K,
    mu_g,
    mu_w,
    dt_init,
    total_sim_time,
    kr_model: str = "cross",
    max_newton_iter=10,
    eps=1e-6,
    wells: Union[List[Well], None] = None,
    n: int = 1,
):
    """

    grid:
    P:
    S: initial gas saturation or injected phase
    Pb:
    Sb_dict:
    phi:
    K:
    mu_g: gas viscosity or injected phase viscosity
    mu_w:
    kr_model:
    dt_init:
    total_sim_time:
    auto_dt:
    eps:

    Returns

    """

    # correct simulation total time and timestep to avoid Newton > max iter
    total_sim_time = dt_init
    dt = dt_init / 2**n
    # end of corrections
    T = calculate_transmissivity(grid, K)
    step = 0
    dt_list = []
    total_time = 0.0
    newton_list = []
    dt_min = dt_init / 2**5
    S_i_plus_1 = None
    P_i_plus_1 = None
    i = 0

    simulation_state = {
        "metadata": {
            "kr_model": kr_model,
            "P0": P.tolist(),
            "Pb": Pb,
            "Sb_dict": Sb_dict,
            "T": T.tolist(),
            "K": K.tolist(),
            "Phi": phi.tolist(),
            "dt_init": dt_init,
            "wells": [well.well_to_dict() for well in wells],
            "eps": eps,
            "max_newton_iter": max_newton_iter,
            "mu_w": mu_w,
            "mu_g": mu_g,
            "total_sim_time": total_sim_time,
        },
        "data": {},
    }

    simulation_state["metadata"]["grid data"] = {
        "Lx": max(grid.node_coordinates[:, 0]),
        "Ly": max(grid.node_coordinates[:, 1]),
        "nb_cells": grid.nb_cells,
        "nb_faces": grid.nb_faces,
        "nb_nodes": grid.nb_nodes,
        "cell_centers": grid.centers(item="cell").tolist(),
        "type": grid.type,
        "dimension": grid.dim,
    }
    if wells:
        for well in wells:
            grid.connect_well(well)

    dt_save = copy.deepcopy(dt)
    effective_wells = wells

    S_i = S
    P_i = P

    P_guess, S_guess = P_i, S_i

    while total_time < total_sim_time and dt != -1:
        P_i_plus_1, S_i_plus_1, dt, nb_newton = solss_newton_step(
            grid=grid,
            P_i=P_i,
            S_i=S_i,
            Pb=Pb,
            Sb_dict=Sb_dict,
            phi=phi,
            K=K,
            T=T,
            mu_g=mu_g,
            mu_w=mu_w,
            dt_init=dt,
            dt_min=dt_min,
            wells=effective_wells,
            max_newton_iter=max_newton_iter,
            eps=eps,
            kr_model=kr_model,
            P_guess=P_guess,
            S_guess=S_guess,
        )

        # number of newton iterations
        newton_list.append(nb_newton)
        dt_list.append(dt)

        total_time += dt
        # print(f"Simulation progress: {total_time/total_sim_time*100}%")
        # number of newton fails
        if dt != dt_save and total_time < total_sim_time and dt != -1:
            while dt_save / 2**i != dt:
                i += 1

        # check if we need to synch dt with total_sim_time
        if dt + total_time > total_sim_time:
            dt = total_sim_time - total_time
    simulation_state["data"][str(total_time)] = {
        "step": step,
        "S0": S.tolist(),
        "P": P_i_plus_1.tolist(),
        "S": S_i_plus_1.tolist(),
        "dt": dt,
        "nb_newton": max_newton_iter * i + sum(newton_list),
        "total_time": total_time,
        "dt_init": dt_init,
        "n": n,
        "newtons": newton_list,
    }
    # print(max_newton_iter, i, sum(newton_list), n, total_sim_time, dt_init, total_time, newton_list)
    assert (
        simulation_state["data"][str(total_time)]["dt_init"]
        == simulation_state["metadata"]["dt_init"]
    )
    return simulation_state


def raw_solss_1_hard_iter(
    grid: Mesh,
    P,
    S,
    Pb,
    Sb_dict,
    phi,
    K,
    mu_g,
    mu_w,
    dt_init,
    total_sim_time,
    kr_model: str = "cross",
    max_newton_iter=100,
    eps=1e-6,
    wells: Union[List[Well], None] = None,
):
    """

    grid:
    P:
    S: initial gas saturation or injected phase
    Pb:
    Sb_dict:
    phi:
    K:
    mu_g: gas viscosity or injected phase viscosity
    mu_w:
    kr_model:
    dt_init:
    total_sim_time:
    auto_dt:
    eps:

    Returns

    """
    T = calculate_transmissivity(grid, K)
    P = implicit_pressure_solver(
        grid=grid,
        K=K,
        T=T,
        P=P,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        mu_g=mu_g,
        mu_w=mu_w,
        kr_model=kr_model,
        wells=wells,
    )
    step = 0
    total_time = 0.0

    #### ONLY MODIF FROM RAW SOLSS 1 ITER
    # ensure there are no more that maxiter iterations
    dt_min = 0.99 * dt_init
    #### End of modification

    simulation_state = {
        "metadata": {
            "kr_model": kr_model,
            "P0": P.tolist(),
            "Pb": Pb,
            "Sb_dict": Sb_dict,
            "T": T.tolist(),
            "K": K.tolist(),
            "Phi": phi.tolist(),
            "dt_init": dt_init,
            "wells": [well.well_to_dict() for well in wells],
            "eps": eps,
            "max_newton_iter": max_newton_iter,
            "mu_w": mu_w,
            "mu_g": mu_g,
            "total_sim_time": total_sim_time,
        },
        "data": {},
    }

    simulation_state["metadata"]["grid data"] = {
        "Lx": max(grid.node_coordinates[:, 0]),
        "Ly": max(grid.node_coordinates[:, 1]),
        "nb_cells": grid.nb_cells,
        "nb_faces": grid.nb_faces,
        "nb_nodes": grid.nb_nodes,
        "cell_centers": grid.centers(item="cell").tolist(),
        "type": grid.type,
        "dimension": grid.dim,
    }
    if wells:
        for well in wells:
            grid.connect_well(well)

    effective_wells = []

    if wells:
        for well in wells:
            for schedule in well.schedule:
                if schedule[0] <= total_time < schedule[1]:
                    effective_wells.append(well)

    S_i = S
    P_i = P

    P_guess, S_guess = P_i, S_i

    P_i_plus_1, S_i_plus_1, dt, nb_newton = solss_newton_step(
        grid=grid,
        P_i=P_i,
        S_i=S_i,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        T=T,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=dt_init,
        dt_min=dt_min,
        wells=effective_wells,
        max_newton_iter=max_newton_iter,
        eps=eps,
        kr_model=kr_model,
        P_guess=P_guess,
        S_guess=S_guess,
    )
    i = 0
    if dt != dt_init and dt != -1:
        while dt_init / 2**i != dt:
            i += 1

    # update simulation state
    simulation_state["data"][str(total_time)] = {
        "step": step,
        "S0": S.tolist(),
        "P": P_i_plus_1.tolist(),
        "S": S_i_plus_1.tolist(),
        "dt": dt,
        "dt_init": dt_init,
        "nb_newton": min(max_newton_iter, max_newton_iter * i + nb_newton),
        "total_time": total_time,
    }
    assert (
        simulation_state["data"][str(total_time)]["dt_init"]
        == simulation_state["metadata"]["dt_init"]
    )
    return simulation_state
