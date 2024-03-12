from typing import Union, List
import json
import copy
import time

from matplotlib import pyplot as plt

import yads.mesh as ym
from yads.mesh import Mesh
import sys

sys.path.append("/work/lechevaa/PycharmProjects/IMPES/Yads")
sys.path.append("/home/irsrvhome1/R16/lechevaa/YADS/Yads")

from yads.numerics.solvers.solss_solver_depreciated import solss_newton_step
from yads.wells import Well
from yads.numerics import calculate_transmissivity, implicit_pressure_solver
from yads.numerics.timestep_variation_control import update_dt
from yads.numerics.utils import newton_list_format

from yads.thesis_approaches.global_approach.hard_points_predictors import (
    FNO2d,
    UnitGaussianNormalizer,
)
import torch
import pickle
import numpy as np


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

    if save:
        if step % save_step == 0:
            filename = save_path + str(step)
            ym.export_vtk(
                filename,
                grid=grid,
                cell_data={
                    "P": P,
                    "S gas": S,
                    "S water": 1.0 - S,
                    "K": K,
                    "phi": phi,
                },
            )
    ### connect wells to grid
    if wells:
        for well in wells:
            grid.connect_well(well)

    # load hybrid model
    model_path = "../../yads/thesis_approaches/global_approach/hard_points_predictors/global_2D_S_var/inference/model"
    S_model = FNO2d(modes1=12, modes2=12, width=64, n_features=4)
    S_model.load_state_dict(
        torch.load(
            model_path + "/best_model_global_2d_S_var_v2_FNO_7300.pt",
            map_location=torch.device("cpu"),
        )
    )
    q_normalizer = pickle.load(open(model_path + "/q_normalizer.pkl", "rb"))
    S_normalizer = pickle.load(open(model_path + "/S_normalizer.pkl", "rb"))
    dt_normalizer = pickle.load(open(model_path + "/dt_normalizer.pkl", "rb"))
    P_normalizer = pickle.load(open(model_path + "/P_imp_normalizer.pkl", "rb"))

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

        P_imp = implicit_pressure_solver(
            grid=grid,
            K=K,
            T=T,
            P=P_i,
            S=S_i,
            Pb=Pb,
            Sb_dict=Sb_dict,
            mu_g=mu_g,
            mu_w=mu_w,
            kr_model=kr_model,
            wells=effective_wells,
        )

        if effective_wells:
            q_flat_zeros = np.zeros((95 * 60))
            q_flat_zeros[1784] = -np.log10(-effective_wells[0].control["Neumann"])
            log_q = torch.from_numpy(np.reshape(q_flat_zeros, (95, 60, 1)))
            log_dt = torch.from_numpy(np.full((95, 60, 1), np.log(dt)))
            S_n = torch.from_numpy(np.array(np.reshape(S_i, (95, 60, 1))))
            P_imp_torch = torch.from_numpy(np.array(np.reshape(P_imp, (95, 60, 1))))
            # normalizer prep
            log_q_n = q_normalizer.encode(log_q)
            log_dt_n = dt_normalizer.encode(log_dt)
            S_n = S_normalizer.encode(S_n)

            P_imp_torch = P_normalizer.encode(P_imp_torch)

            x = torch.cat([log_q_n, log_dt_n, S_n, P_imp_torch], 2).float()
            x = x.reshape(1, 95, 60, 4)
            S_pred = S_model(x)
            S_pred = S_pred.detach().numpy()
            S_pred = np.reshape(S_pred, (95 * 60))
            S_guess, P_guess = S_pred, P_imp
            print("Using hybrid model")
            # # Debug plot
            # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 12))
            # q_plot = q_normalizer.decode(log_q_n)
            # S_plot = S_normalizer.decode(S_n)
            # P_imp_plot = P_normalizer.decode(P_imp_torch)
            #
            # ax1.imshow(q_plot.reshape(95, 60).T)
            # ax2.imshow(S_plot.reshape(95, 60).T, vmin=0, vmax=1)
            # ax3.imshow(np.power(P_imp_plot.reshape(95, 60).T, 10))
            # ax4.imshow(S_pred.reshape(95, 60).T, vmin=0, vmax=1)
            #
            # fig.axes[1].invert_yaxis()
            # fig.axes[2].invert_yaxis()
            # fig.axes[0].invert_yaxis()
            # fig.axes[3].invert_yaxis()
            # plt.show()

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
                        "P imp": P_imp,
                    },
                )

        ############### REPLACE WITH FUNC ##############
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

    print("total simulation time: ", total_time)
    print("--- %s seconds ---" % (time.time() - init_time))
    return newton_list_format(newton_list, max_newton_iter), dt_list
