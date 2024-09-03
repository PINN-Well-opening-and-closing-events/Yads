import json

import numpy as np  # type: ignore
import copy

from typing import Union, List
from yads.wells import Well

from yads.numerics.physics import calculate_transmissivity

from yads.numerics.solvers.implicit_pressure_solver import implicit_pressure_solver
from yads.numerics.solvers.implicit_saturation_solver import implicit_saturation_solver

from yads.numerics.timestep_variation_control import update_dt
from yads.numerics.utils import newton_list_format

import yads.mesh as ym
import time


def impims(
    grid: ym.Mesh,
    P: np.ndarray,
    S: np.ndarray,
    Pb: Union[dict, None],
    Sb_dict: Union[dict, None],
    phi: np.ndarray,
    K: np.ndarray,
    mu_g: Union[float, int],
    mu_w: Union[float, int],
    total_sim_time: Union[float, int],
    max_newton_iter: int,
    dt_init: Union[float, int],
    kr_model: str = "cross",
    auto_dt: Union[List[Union[float, int]], None] = None,
    wells: Union[List[Well], None] = None,
    save: bool = False,
    save_step: int = 1,
    save_path: str = "./auto_save_path_",
    save_states_to_json=False,
    json_savepath="./json_auto_save_path.json",
):
    """solver for incompressible two-phase flow using Darcy's equation.
    Uses a 2 points finite volume method.
    Solves implicitly the pressure (imp)  and implicitly the saturation (ims)

    Args:
        grid: yads.mesh.Mesh object
        P: pressure, np.ndarray size(grid.nb_cells)
        S: water saturation, np.ndarray size(grid.nb_cells)
        Pb: pressure boundary conditions dict
            example: Pb = {"left":1.0, "right": 2.0}
        Sb_dict: water saturation boundary conditions dict
            example: Sb_dict = {"Neumann":{"left":1.0, "right": 0.2}, "Dirichlet": {"left":None, "right":None}
        phi: porosity in each cell, np.ndarray size(grid.nb_cells)
        K: diffusion coefficient (i.e. permeability), np.ndarray size(grid.nb_cells)
        mu_g: water viscosity
        mu_w: oil viscosity
        total_sim_time: total simulation time
        max_newton_iter: maximum number of newton iteration
        dt_init: initial time step
        kr_model:
        auto_dt
        wells: list of Well objects
        save: if true: save at save_path
        save_step: if save: save every save step the simulation state
        save_path: path where to save simulation
        json_savepath:
        save_states_to_json:
    """

    init_time = time.time()
    T = calculate_transmissivity(grid, K)

    total_time = 0.0
    stop = False
    nb_iter = 0
    dt = dt_init
    newton_list = []
    dt_list = []

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
        if wells:
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
        state_dict["simulation data"][str(total_time)]["step"] = nb_iter
        state_dict["simulation data"][str(total_time)]["nb_newton"] = 0.0
        state_dict["simulation data"][str(total_time)]["effective wells"] = []

    if wells:
        for well in wells:
            grid.connect_well(well)

    while not stop:
        start_time = time.time()
        S_save = copy.deepcopy(S)
        dt_save = copy.deepcopy(dt)
        effective_wells = []
        if wells:
            for well in wells:
                for schedule in well.schedule:
                    if schedule[0] <= total_time < schedule[1]:
                        effective_wells.append(well)

        P = implicit_pressure_solver(
            grid,
            K,
            T,
            P,
            S,
            Pb,
            Sb_dict,
            mu_g,
            mu_w,
            wells=effective_wells,
            kr_model=kr_model,
        )

        S, dt, nb_newton = implicit_saturation_solver(
            grid,
            P,
            S,
            T,
            phi,
            Pb,
            Sb_dict,
            dt,
            mu_g,
            wells=effective_wells,
            kr_model=kr_model,
            max_newton_iter=max_newton_iter,
        )

        total_time += dt
        nb_iter += 1
        dt_list.append(dt)
        # number of newton iterations
        newton_list.append(nb_newton)

        # save state to json
        if save_states_to_json:
            state_dict["simulation data"][str(total_time)] = {}
            state_dict["simulation data"][str(total_time)]["dt"] = dt
            state_dict["simulation data"][str(total_time)]["P"] = P.tolist()
            state_dict["simulation data"][str(total_time)]["S"] = S.tolist()
            state_dict["simulation data"][str(total_time)]["total_time"] = total_time
            state_dict["simulation data"][str(total_time)]["step"] = nb_iter
            state_dict["simulation data"][str(total_time)]["nb_newton"] = 0.0
            state_dict["simulation data"][str(total_time)]["effective wells"] = [
                well.name for well in effective_wells if effective_wells
            ]

        # number of newton fails
        if dt != dt_save:
            i = 0
            while dt_save / 2**i != dt:
                newton_list.append(-1)
                i += 1
        if save:
            if nb_iter % save_step == 0:
                filename = save_path + str(nb_iter)
                ym.export_vtk(
                    filename,
                    grid=grid,
                    cell_data={
                        "P": P,
                        "S gaz": S,
                        "S water": 1.0 - S,
                        "K": K,
                        "phi": phi,
                    },
                )

        if auto_dt is not None:
            dt = update_dt(S_save, S, auto_dt, dt)
        mid_time = time.time()
        print(f"step: {nb_iter} done in {(mid_time - start_time)} seconds")
        print(f"total sim time: {100*total_time/total_sim_time:0.2}%")
        # stop criterion
        if total_time >= total_sim_time:
            stop = True

        # schedule check and total time check
        if wells:
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
            newton_list, max_newton_iter=max_newton_iter
        )
        state_dict["metadata"]["dt_list"] = dt_list
        with open(json_savepath, "w") as f:
            json.dump(state_dict, f)
    print("total simulation time: ", total_time)
    print("--- %s seconds ---" % (time.time() - init_time))
    return newton_list_format(newton_list, max_newton_iter), dt_list
