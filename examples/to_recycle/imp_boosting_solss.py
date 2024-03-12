import copy
import json
import time
from typing import Union, List
from yads.mesh import Mesh
from yads.wells import Well

import numpy as np
import yads.mesh as ym

from yads.numerics.solvers import implicit_pressure_solver
from yads.numerics.solvers.solss_solver import solss_newton_step
from yads.numerics.physics import calculate_transmissivity
from yads.numerics.timestep_variation_control import update_dt
from yads.numerics.utils import newton_list_format

from yads.numerics.utils import P_closest_to_dt_in_json, S_closest_to_dt_in_json


def imp_boosting_solss(
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
    save: bool = True,
    save_step: int = 1,
    save_path: str = "./auto_save_path_",
    save_states_to_json=False,
    json_savepath="./json_auto_save_path.json",
    cheating_path=None,
    cheating_P_bool=False,
    cheating_S_bool=False,
    debug_newton_mode=False,
    debug_newton_path="./debug_newton_autopath.json",
):
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
    dt_min = auto_dt[-2]

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

    if wells:
        for well in wells:
            grid.connect_well(well)

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
        if cheating_P_bool:
            print("CHEATING P")
            P_guess = implicit_pressure_solver(
                grid=grid,
                K=K,
                T=T,
                P=P,
                S=S,
                Pb=Pb,
                Sb_dict=Sb_dict,
                mu_g=mu_g,
                mu_w=mu_w,
                wells=effective_wells,
                kr_model=kr_model,
            )
        if cheating_S_bool and cheating_path is not None:
            print("CHEATING S")
            S_guess = S_closest_to_dt_in_json(cheating_path, total_time + dt)

        # save for debug mode
        if debug_newton_mode:
            x_centers = np.array([elt[0] for elt in grid.centers(item="cell")])
            debug_dict = {
                "metadata": {
                    "step": step + 1,
                    "cell_centers": x_centers.tolist(),
                    "P_i": P_i.tolist(),
                    "S_i": S_i.tolist(),
                },
                "newton_step_data": {},
            }
            with open(debug_newton_path[:-5] + "_" + str(step + 1) + ".json", "w") as f:
                json.dump(debug_dict, f)

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
            while dt_save / 2**i != dt:
                newton_list.append(-1)
                i += 1
        dt_list.append(dt)

        total_time += dt

        step += 1
        mid_time = time.time()
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

        print(
            f"step: {step} done in {(mid_time - start_time):0.4f} seconds with dt={dt:0.3E}s"
        )

        if total_time >= total_sim_time:
            stop = True
            continue

        if auto_dt is not None:
            dt = update_dt(S_save, S, auto_dt, dt)

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

    print("saving result to json")
    if save_states_to_json:
        state_dict["metadata"]["newton_list"] = newton_list_format(
            newton_list, max_newton_iter
        )
        state_dict["metadata"]["dt_list"] = dt_list
        with open(json_savepath, "w") as f:
            json.dump(state_dict, f)
    print("total simulation time: ", total_time)
    print("--- %s seconds ---" % (time.time() - init_time))
    return newton_list_format(newton_list, max_newton_iter), dt_list


if __name__ == "__main__":
    #### SHP CO 1D CASE ####
    # This is the S 1D mesh parameters
    # pseudo 1d: 1000 m y / 10 000 m x
    grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 200, 1)

    dz = 100  # 100 meters
    real_volume = np.multiply(grid.measures(item="cell"), dz)
    grid.change_measures("cell", real_volume)

    phi = 0.2

    # Porosity
    phi = np.full(grid.nb_cells, phi)
    # Diffusion coefficient (i.e Permeability)
    K = np.full(grid.nb_cells, 100.0e-15)

    # gaz saturation initialization
    S = np.full(grid.nb_cells, 0.0)
    # Pressure initialization
    P = np.full(grid.nb_cells, 100.0e5)

    mu_w = 0.571e-3
    mu_g = 0.0285e-3

    kr_model = "quadratic"
    # BOUNDARY CONDITIONS #
    # Pressure
    Pb = {"left": 110.0e5, "right": 100.0e5}

    # Saturation
    # only water left and right
    Sb_d = {"left": 0.0, "right": 0.0}
    Sb_n = {"left": None, "right": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

    dt = 100 * (60 * 60 * 24 * 365.25)  # in years
    total_sim_time = 10000 * (60 * 60 * 24 * 365.25)
    # auto_dt = [0.25, 0.5, 1.0, 1.1, 1.1, 1., total_sim_time/100]
    auto_dt = [0.25, 0.5, 1.0, 1.1, 1.1, 1.0, total_sim_time]
    max_newton_iter = 20
    save_step = 1

    save = True

    injector_two = Well(
        name="injector 2",
        cell_group=np.array([[7500.0, 500.0]]),
        radius=0.1,
        control={"Dirichlet": 105.0e5},
        s_inj=0.0,
        schedule=[[0.0, total_sim_time]],
        mode="injector",
    )

    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[3000.0, 500.0]]),
        radius=0.1,
        control={"Dirichlet": 115.0e5},
        s_inj=1.0,
        schedule=[
            [0.4 * total_sim_time, 0.6 * total_sim_time],
        ],
        mode="injector",
    )

    imp_boosting_solss(
        grid,
        P,
        S,
        Pb,
        Sb_dict,
        phi,
        K,
        mu_g,
        mu_w,
        kr_model=kr_model,
        total_sim_time=total_sim_time,
        dt_init=dt,
        max_newton_iter=max_newton_iter,
        save=False,
        save_step=save_step,
        wells=[well_co2, injector_two],
        auto_dt=auto_dt,
        save_states_to_json=True,
        cheating_path="../../examples/saves/newton_1d_rekterino/solss/vanilla/vanilla.json",
        cheating_P_bool=True,
        cheating_S_bool=True,
        debug_newton_mode=False,
        json_savepath="./imp_S_boost_solss.json",
    )
