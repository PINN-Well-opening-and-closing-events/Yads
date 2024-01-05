import copy

import pandas as pd

from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian
import numpy as np
import subprocess as sp
from yads.numerics.physics import calculate_transmissivity
from yads.numerics.solvers import solss_newton_step, implicit_pressure_solver
import os
import pickle
from pyDOE import lhs

from yads.thesis_approaches.data_generation import raw_solss_1_iter
from yads.wells import Well

from models.FNO import FNO2d, UnitGaussianNormalizer
from json import load
import pickle
import torch
from yads.mesh import Mesh

from typing import Union, List


def hybrid_newton_inference(
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
    kr_model: str = "quadratic",
    max_newton_iter=200,
    eps=1e-6,
    wells: Union[List[Well], None] = None,
    P_guess=None,
    S_guess=None,
):
    dt = dt_init
    i = 0

    if wells:
        for well in wells:
            grid.connect_well(well)

    effective_wells = wells

    S_i = S
    P_i = P

    if P_guess is None:
        P_guess = P_i
    if S_guess is None:
        S_guess = S_i

    P_i_plus_1, S_i_plus_1, dt_sim, nb_newton, norms = solss_newton_step(
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
        dt_min=dt,
        wells=effective_wells,
        max_newton_iter=max_newton_iter,
        eps=eps,
        kr_model=kr_model,
        P_guess=P_guess,
        S_guess=S_guess,
    )

    return P_i_plus_1, S_i_plus_1, dt_sim, nb_newton, norms


def main():

    for seed, sample_num in enumerate(sample_dirs):
        rota_files = os.listdir("../" + folder_path + sample_num)
        if not os.path.isdir(save_dir + sample_num):
            sp.call(f"mkdir {save_dir + sample_num}", shell=True)
        # set new seed for each batch
        np.random.seed(seed)
        # scaling lhs data
        nb_data = len(rota_files)
        dt = 1 * (60 * 60 * 24 * 365.25)  # in years
        lhd = lhs(3, samples=nb_data, criterion="maximin")
        q_pow = -np.power(10, -5.0 + lhd[:, 0] * (-3.0 - (-5.0)))
        dt_init = 1 * dt + lhd[:, 1] * (10 * dt - 1 * dt)
        S0 = lhd[:, 2] * 0.6
        # rounding to avoid saving issues
        q_pow = np.round(q_pow, 6)
        dt_init = np.round(dt_init, 1)
        for i, rota in enumerate(rota_files):
            with open("../" + folder_path + sample_num + "/" + rota, 'rb') as f:
                (groups, Pb_dict) = pickle.load(f)

            data_dict = {"q": q_pow[i], "total_sim_time": dt_init[i], "S0": S0}

            # prepare for save
            well_co2 = Well(
                name="well co2",
                cell_group=np.array([[Lx/2, Ly/2]]),
                radius=0.1,
                control={"Neumann": q_pow[i]},
                s_inj=1.0,
                schedule=[[0.0,  dt_init[i]],],
                mode="injector",
            )
            grid_temp = copy.deepcopy(grid)
            for group in groups:
                grid_temp.add_face_group_by_line(*group)
            # Saturation
            Sb_d = copy.deepcopy(Pb_dict)
            Sb_n = copy.deepcopy(Pb_dict)
            for group in Sb_d.keys():
                Sb_d[group] = S0[i]
                Sb_n[group] = None
            Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}
            S = np.full(grid_temp.nb_cells, S0[i])
            P_imp = implicit_pressure_solver(
                grid=grid_temp,
                K=K,
                T=T,
                P=P0,
                S=S,
                Pb=Pb_dict,
                Sb_dict=Sb_dict,
                mu_g=mu_g,
                mu_w=mu_w,
                wells=[well_co2]
            )
            # classic sim
            P_i_plus_1, S_i_plus_1, dt_sim, nb_newton, norms = hybrid_newton_inference(
                grid=grid_temp,
                P=P_imp,
                S=S,
                Pb=Pb_dict,
                Sb_dict=Sb_dict,
                phi=phi,
                K=K,
                mu_g=mu_g,
                mu_w=mu_w,
                dt_init=dt_init[i],
                kr_model=kr_model,
                max_newton_iter=max_newton_iter,
                eps=1e-6,
                wells=[well_co2],
                P_guess=P_imp,
                S_guess=S,
            )
            data_dict["P_i_plus_1_classic"] = P_i_plus_1.tolist()
            data_dict["S_i_plus_1_classic"] = S_i_plus_1.tolist()
            data_dict["nb_newton_classic"] = nb_newton
            data_dict["dt_sim_classic"] = dt_sim
            data_dict["norms_classic"] = norms

            # Hybrid sim
            well_loc_idx = 40

            # shape prep
            q_flat_zeros = np.zeros((Nx * Ny))
            q_flat_zeros[well_loc_idx] = -np.log10(-q_pow[i])
            log_q = torch.from_numpy(np.reshape(q_flat_zeros, (Nx, Ny, 1)))
            log_dt = torch.from_numpy(np.full((Nx, Ny, 1), np.log(dt_init[i])))
            S_n = torch.from_numpy(np.array(np.reshape(S, (Nx, Ny, 1))))
            P_imp_local_n = torch.from_numpy(
                np.array(np.reshape(P_imp, (Nx, Ny, 1)))
            )

            # normalizer prep
            log_q_n = q_normalizer.encode(log_q)
            log_dt_n = dt_normalizer.encode(log_dt)

            P_imp_local_n = P_imp_normalizer.encode(np.log10(P_imp_local_n))
            #
            x = torch.cat([log_q_n, log_dt_n, S_n, P_imp_local_n], 2).float()
            x = x.reshape(1, Nx, Ny, 4)
            S_pred = model(x)
            S_pred = S_pred.detach().numpy()

            S_pred = np.reshape(S_pred, (Nx * Ny))

            P_i_plus_1, S_i_plus_1, dt_sim, nb_newton, norms = hybrid_newton_inference(
                grid=grid_temp,
                P=P_imp,
                S=S,
                Pb=Pb_dict,
                Sb_dict=Sb_dict,
                phi=phi,
                K=K,
                mu_g=mu_g,
                mu_w=mu_w,
                dt_init=dt_init[i],
                kr_model=kr_model,
                max_newton_iter=max_newton_iter,
                eps=1e-6,
                wells=[well_co2],
                P_guess=P_imp,
                S_guess=S_pred,
            )

            data_dict["S_i_plus_1_hybrid"] = S_i_plus_1.tolist()
            data_dict["P_i_plus_1_hybrid"] = P_i_plus_1.tolist()
            data_dict["nb_newton_hybrid"] = nb_newton
            data_dict["dt_sim_hybrid"] = dt_sim
            data_dict["norms_hybrid"] = norms
            print(
                f"step {i}: Newton classic {data_dict['nb_newton_classic']}, hybrid {data_dict['nb_newton_hybrid']}"
            )
            print(f"step number {i} finished")
            df = pd.DataFrame([data_dict])
            save_path = save_dir + sample_num + "/" + rota
            # save to csv
            df.to_csv(save_path + ".csv", sep="\t", index=False)
            if data_dict['nb_newton_classic'] < data_dict['nb_newton_hybrid']:
                import matplotlib.pyplot as plt
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
                ax1.imshow(np.array(data_dict["S_i_plus_1_classic"]).reshape(Nx, Ny))
                ax2.imshow(S_pred. reshape(Nx, Ny))
                ax3.imshow(S.reshape(Nx, Ny))
                ax4.imshow(np.abs(np.array(data_dict["S_i_plus_1_classic"]).reshape(Nx, Ny) - S_pred. reshape(Nx, Ny)))
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.xaxis.set_ticks([])
                    ax.yaxis.set_ticks([])
                print("Classic: ", data_dict["norms_classic"]['L_inf'])
                print("Hybrid: ", data_dict["norms_hybrid"]['L_inf'])
                plt.show()


if __name__ == "__main__":
    Nx, Ny = 9, 9
    nb_bound_faces = Nx * 2 + Ny * 2
    dxy = 50
    Lx, Ly = Nx * dxy, Ny * dxy

    P_min = 10e6
    P_max = 20e6

    grid = create_2d_cartesian(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)

    assert grid.nb_cells == Nx * Ny

    # Porosity
    phi = 0.2
    phi = np.full(grid.nb_cells, phi)
    # Diffusion coefficient (i.e Permeability)
    K = np.full(grid.nb_cells, 100.0e-15)

    T = calculate_transmissivity(grid, K)
    # Pressure initialization
    P0 = np.full(grid.nb_cells, 100.0e5)
    mu_w = 0.571e-3
    mu_g = 0.0285e-3
    max_newton_iter = 200
    kr_model = "quadratic"

    folder_path = "nb_samples_3000_nb_boundaries_3/"
    sample_dirs = os.listdir("../" + folder_path)
    # create dirs
    save_dir = "results_" + folder_path
    if not os.path.isdir(save_dir):
        sp.call(f"mkdir {save_dir}", shell=True)

    S_model = model = FNO2d(modes1=12, modes2=12, width=64, n_features=4)
    S_model.load_state_dict(
        torch.load(
            "models/GWM_3000_3_checkpoint_2000.pt",
            map_location=torch.device("cpu"),
        )["model"]
    )

    q_normalizer = pickle.load(open("models/q_normalizer.pkl", "rb"))
    P_imp_normalizer = pickle.load(open("models/P_imp_normalizer.pkl", "rb"))
    dt_normalizer = pickle.load(open("models/dt_normalizer.pkl", "rb"))

    main()
