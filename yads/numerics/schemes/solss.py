from typing import Union, List
import json
import copy
import time
import yads.mesh as ym
from yads.mesh import Mesh

from yads.numerics.solvers.solss_solver import solss_newton_step
from yads.wells import Well
from yads.numerics.physics import calculate_transmissivity
from yads.numerics.timestep_variation_control import update_dt
from yads.numerics.utils import newton_list_format

from yads.numerics.utils import P_closest_to_dt_in_json, S_closest_to_dt_in_json


def solss(
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
    auto_dt=None,
    max_newton_iter=10,
    eps=1e-6,
    wells: Union[List[Well], None] = None,
    save: bool = False,
    save_step: int = 1,
    save_path: str = "./auto_save_path_",
    save_states_to_json=False,
    json_savepath="./json_auto_save_path.json",
    cheating_path=None,
    cheating_P_bool=False,
    cheating_S_bool=False,
    debug_newton_mode=False,
    debug_newton_path="./debug_newton_autopath",
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
    init_time = time.time()
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
    if auto_dt:
        dt_min = auto_dt[-2]

    ##### REPLACE WITH A FUNCTION ########
    state_dict = {"simulation data": {}, "metadata": {}}
    # record simulation metadata for json export
    if save_states_to_json:

        state_dict["metadata"]["other"] = {
            "total_sim_time": total_sim_time,
            "max_newton_iter": max_newton_iter,
        }
        state_dict["metadata"]["grid data"] = {
            "Lx": max(grid.node_coordinates[:, 0]),
            "Ly": max(grid.node_coordinates[:, 1]),
            "nb_cells": grid.nb_cells,
            "nb_faces": grid.nb_faces,
            "nb_nodes": grid.nb_nodes,
            "cell_centers": grid.centers(item="cell").tolist(),
            "type": grid.type,
            "dimension": grid.dim,
        }
        state_dict["metadata"]["well data"] = {}
        if wells is not None:
            for well in wells:
                state_dict["metadata"]["well data"][well.name] = {
                    "cell_group": well.cell_group.tolist(),
                    "control": well.control,
                    "sat_inj": well.injected_saturation,
                    "radius": well.radius,
                    "schedule": well.schedule,
                }
        # record initial state
        state_dict["simulation data"][str(total_time)] = {}
        state_dict["simulation data"][str(total_time)]["dt"] = 0.0
        state_dict["simulation data"][str(total_time)]["P"] = P.tolist()
        state_dict["simulation data"][str(total_time)]["S"] = S.tolist()
        state_dict["simulation data"][str(total_time)]["total_time"] = str(total_time)
        state_dict["simulation data"][str(total_time)]["step"] = step
        state_dict["simulation data"][str(total_time)]["nb_newton"] = 0.0
        state_dict["simulation data"][str(total_time)]["effective wells"] = []
    ##########################################################################

    if save:
        if step % save_step == 0:
            filename = save_path + str(step)
            ym.export_vtk(
                filename,
                grid=grid,
                cell_data={"P": P, "S gas": S, "S water": 1.0 - S, "K": K, "phi": phi,},
            )
    ### connect wells to grid
    if wells:
        for well in wells:
            grid.connect_well(well)

    # beginning of Newton iter

    while not stop:
        start_time = time.time()
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

        # cheat for initial newton guess
        if cheating_P_bool or cheating_S_bool:
            if None != cheating_path:
                if cheating_P_bool:
                    print("CHEATING P")
                    P_guess = P_closest_to_dt_in_json(cheating_path, total_time + dt)
                if cheating_S_bool:
                    print("CHEATING S")
                    S_guess = S_closest_to_dt_in_json(cheating_path, total_time + dt)

        P_i_plus_1, S_i_plus_1, dt, nb_newton, _ = solss_newton_step(
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
            debug_newton_mode=debug_newton_mode,
            debug_newton_path=debug_newton_path[:-5] + "_" + str(step + 1) + ".json",
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
            while dt_save / 2 ** i != dt:
                newton_list.append(-1)
                i += 1
        dt_list.append(dt)

        total_time += dt

        step += 1
        mid_time = time.time()
        #### REPLACE WITH FUNC #####
        # save state to json
        if save_states_to_json:
            state_dict["simulation data"][str(total_time)] = {}
            state_dict["simulation data"][str(total_time)]["dt"] = dt
            state_dict["simulation data"][str(total_time)]["P"] = P_i_plus_1.tolist()
            state_dict["simulation data"][str(total_time)]["S"] = S_i_plus_1.tolist()
            state_dict["simulation data"][str(total_time)]["total_time"] = total_time
            state_dict["simulation data"][str(total_time)]["step"] = step
            state_dict["simulation data"][str(total_time)]["nb_newton"] = (
                max_newton_iter * i + nb_newton
            )
            state_dict["simulation data"][str(total_time)]["effective wells"] = [
                well.name for well in effective_wells if effective_wells
            ]
        ###############################
        print(
            f"step: {step} done in {(mid_time - start_time):0.4f} seconds with dt={dt:0.3E}s"
        )

        if save:
            if step % save_step == 0:
                filename = save_path + str(step)
                ym.export_vtk(
                    filename,
                    grid=grid,
                    cell_data={
                        "P": P_i_plus_1,
                        "S gas": S_i_plus_1,
                        "S water": 1.0 - S_i_plus_1,
                        "K": K,
                        "phi": phi,
                    },
                )

        ############### REPLACE WITH FUNC ##############
        if total_time >= total_sim_time:
            stop = True
            continue

        if auto_dt is not None:
            dt = update_dt(S_save, S, auto_dt, dt)

        # schedule check and total time check
        if wells is not None:
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

    if save_states_to_json:
        print("saving result to json")
        state_dict["metadata"]["newton_list"] = newton_list_format(
            newton_list, max_newton_iter
        )
        state_dict["metadata"]["dt_list"] = dt_list
        with open(json_savepath, "w") as f:
            json.dump(state_dict, f)
    print("total simulation time: ", total_time)
    print("--- %s seconds ---" % (time.time() - init_time))
    return newton_list_format(newton_list, max_newton_iter), dt_list
