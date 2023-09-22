import copy
from typing import Union, List
from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt
from mpi4py import MPI
import os
import subprocess as sp
import torch
import numpy as np
import pickle
from models.FNO import FNO2d, UnitGaussianNormalizer

import sys

sys.path.append("/")
sys.path.append("/home/irsrvhome1/R16/lechevaa/yads")
sys.path.append("/")

from yads.mesh.utils import load_json
from yads.numerics.physics import calculate_transmissivity
from yads.numerics.solvers import solss_newton_step, implicit_pressure_solver
from yads.wells import Well
from yads.mesh import Mesh
from yads.numerics.solvers.newton import res
import yads.mesh as ym


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
    total_sim_time,
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


def launch_inference(qt, log_qt, i, test_P, test_S):
    print(f"launching step number {i}, with {qt[0], qt[1], qt[2][0]}")
    dict_save = {"q": qt[0], "total_sim_time": qt[1], "S0": qt[2][0]}
    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[1675.0, 1725]]),
        radius=0.1,
        control={"Neumann": qt[0]},
        s_inj=1.0,
        schedule=[[0.0, qt[1]],],
        mode="injector",
    )
    Sb_dict["Dirichlet"] = {
        "injector_one": qt[2][0],
        "injector_two": qt[2][0],
        "right": 0.0,
    }
    S = np.full(grid.nb_cells, qt[2][0])
    ################################
    P_imp = implicit_pressure_solver(
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
        wells=[well_co2],
    )

    dict_save["P_imp"] = P_imp.tolist()

    # Data prep for model
    well_x, well_y = 1475, 2225
    grid_dxy = 50
    d = 4
    cells_d = grid.find_cells_inside_square(
        (grid_dxy * (well_x / grid_dxy - d), grid_dxy * (well_y / grid_dxy + d)),
        (grid_dxy * (well_x / grid_dxy + d), grid_dxy * (well_y / grid_dxy - d)),
    )

    well_loc_idx = 40
    P_imp_local = P_imp[cells_d]
    S_n = S[cells_d]

    # Hybrid model
    # shape prep
    local_shape = 2 * ext + 1
    q_flat_zeros = np.zeros((local_shape * local_shape))
    q_flat_zeros[well_loc_idx] = log_qt[0]
    log_q = torch.from_numpy(np.reshape(q_flat_zeros, (local_shape, local_shape, 1)))
    log_dt = torch.from_numpy(np.full((local_shape, local_shape, 1), log_qt[1]))
    S_n = torch.from_numpy(np.array(np.reshape(S_n, (local_shape, local_shape, 1))))
    P_imp_local_n = torch.from_numpy(
        np.array(np.reshape(P_imp_local, (local_shape, local_shape, 1)))
    )

    # normalizer prep
    log_q_n = q_normalizer.encode(log_q)
    log_dt_n = dt_normalizer.encode(log_dt)

    P_imp_local_n = P_imp_normalizer.encode(np.log10(P_imp_local_n))
    #
    x = torch.cat([log_q_n, log_dt_n, S_n, P_imp_local_n], 2).float()
    x = x.reshape(1, local_shape, local_shape, 4)
    S_pred = model(x)
    S_pred = S_pred.detach().numpy()

    S_pred = np.reshape(S_pred, (local_shape * local_shape))

    dict_save["S_predict_local"] = S_pred.tolist()

    # Domain Decomposition

    DD_grid = ym.two_D.create_2d_cartesian(grid_dxy * (2 * ext + 1), grid_dxy * (2 * ext + 1),
                                           (2 * ext + 1), (2 * ext + 1))
    S_DD = S[cells_d]
    K_DD = K[cells_d]
    phi_DD = phi[cells_d]
    P_imp_DD = P_imp[cells_d]

    well_co2_DD = Well(
        name="well co2 DD",
        cell_group=np.array([[grid_dxy * (2 * ext + 1)/2, grid_dxy * (2 * ext + 1)/2]]),
        radius=0.1,
        control={"Neumann": qt[0]},
        s_inj=1.0,
        schedule=[[0.0, qt[1]], ],
        mode="injector",
    )

    # draw wider rectangle of extension (d+1)
    cells_d_plus_1 = grid.find_cells_inside_square(
        (grid_dxy * (well_x / grid_dxy - (d+1)), grid_dxy * (well_y / grid_dxy + (d+1))),
        (grid_dxy * (well_x / grid_dxy + (d+1)), grid_dxy * (well_y / grid_dxy - (d+1))),
    )

    # intersect with local domain of extension d
    cells_DD = [c for c in cells_d_plus_1 if c not in cells_d]
    # remove edges
    cells_DD_centers = grid.centers(item='cell')[cells_DD]

    ur = (well_x + (d+1) * grid_dxy, well_y + (d+1) * grid_dxy)
    ul = (well_x + (d+1) * grid_dxy, well_y - (d+1) * grid_dxy)
    lr = (well_x - (d+1) * grid_dxy, well_y + (d+1) * grid_dxy)
    ll = (well_x - (d+1) * grid_dxy, well_y - (d+1) * grid_dxy)
    edges_idx = []
    cells_DD_wo_edges = []
    for k, c in enumerate(cells_DD_centers):
        if c[0] == ur[0] and c[1] == ur[1]:
            edges_idx.append(k)
        elif c[0] == ul[0] and c[1] == ul[1]:
            edges_idx.append(k)
        elif c[0] == lr[0] and c[1] == lr[1]:
            edges_idx.append(k)
        elif c[0] == ll[0] and c[1] == ll[1]:
            edges_idx.append(k)
        else:
            cells_DD_wo_edges.append(cells_DD[k])

    cells_DD = cells_DD_wo_edges

    # create group for each cell
    # add by line all groups to grid:
    groups = []
    Pb_dict_DD = {}
    Sb_d_DD = {}
    Sb_n_DD = {}

    # get center of boundary faces of DD grid
    face_center_coords = []
    for group in DD_grid.face_groups:
        if group in ["left", "right", "upper", "lower"]:
            cell_idxs = DD_grid.face_groups[group][:, 0]
            for coord in DD_grid.centers(item="face")[cell_idxs]:
                face_center_coords.append(list(coord))

    translation = (well_x - d * grid_dxy - grid_dxy/2, well_y - d * grid_dxy - grid_dxy/2)
    for k, coord_DD in enumerate(face_center_coords):
        DD_index = None
        for j, coord in enumerate(grid.centers(item='cell')[cells_DD]):
            if coord[0] == coord_DD[0] + translation[0] - grid_dxy/2 and coord[1] == coord_DD[1] + translation[1]:
                DD_index = j
            elif coord[0] == coord_DD[0] + translation[0] + grid_dxy / 2 and coord[1] == coord_DD[1] + translation[1]:
                DD_index = j
            elif coord[0] == coord_DD[0] + translation[0] and coord[1] == coord_DD[1] + translation[1] + grid_dxy / 2:
                DD_index = j
            elif coord[0] == coord_DD[0] + translation[0] and coord[1] == coord_DD[1] + translation[1] - grid_dxy / 2:
                DD_index = j

        if np.abs(coord_DD[0]) == 0 or np.abs(coord_DD[0]) == grid_dxy * (2 * ext + 1):
            line_point_1 = (coord_DD[0], coord_DD[1] - grid_dxy/2)
            line_point_2 = (coord_DD[0], coord_DD[1] + grid_dxy/2)
        else:
            line_point_1 = (coord_DD[0] - grid_dxy/2, coord_DD[1])
            line_point_2 = (coord_DD[0] + grid_dxy/2, coord_DD[1])

        Pb_dict_DD[f"boundary_face_{k}"] = P_imp[cells_DD[DD_index]]
        Sb_d_DD[f"boundary_face_{k}"] = S[cells_DD[DD_index]]
        Sb_n_DD[f"boundary_face_{k}"] = None

        group_by_line = (f"boundary_face_{k}", line_point_1, line_point_2)
        groups.append(group_by_line)
        DD_grid.add_face_group_by_line(*group_by_line)

    Sb_dict_DD = {"Dirichlet": Sb_d_DD, "Neumann": Sb_n_DD}
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # ax1.imshow(P_imp.reshape(95, 60).T)
    # ax2.imshow(S.reshape(95, 60).T)
    # ax1.set_title('P imp')
    # ax2.set_title('S')
    # ax1.invert_yaxis()
    # ax2.invert_yaxis()
    # fig.suptitle('Reference')
    # plt.show()
    # #############   INFERENCE ##############
    # Standard
    P_i_plus_1, S_i_plus_1, dt_sim, nb_newton, norms = hybrid_newton_inference(
        grid=grid,
        P=P_imp,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=qt[1],
        total_sim_time=qt[1],
        kr_model=kr_model,
        max_newton_iter=max_newton_iter,
        eps=1e-6,
        wells=[well_co2],
        P_guess=P_imp,
        S_guess=S,
    )

    dict_save["P_i_plus_1_classic"] = P_i_plus_1.tolist()
    dict_save["S_i_plus_1_classic"] = S_i_plus_1.tolist()
    dict_save["nb_newton_classic"] = nb_newton
    dict_save["dt_sim_classic"] = dt_sim
    dict_save["norms_classic"] = norms

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4))
    # ax1.imshow(P_imp.reshape(95, 60).T)
    # ax2.imshow(S.reshape(95, 60).T)
    # ax3.imshow(P_i_plus_1.reshape(95, 60).T)
    # ax4.imshow(S_i_plus_1.reshape(95, 60).T)
    # ax1.set_title('P imp')
    # ax2.set_title('S')
    # ax3.set_title('P sol')
    # ax4.set_title('S sol')
    # ax1.invert_yaxis()
    # ax2.invert_yaxis()
    # ax3.invert_yaxis()
    # ax4.invert_yaxis()
    # fig.suptitle(f'Reference: {nb_newton}')
    # plt.show()

    # Domain Decomposition
    P_DD_plus_1, S_DD_plus_1, _, nb_newton, _ = hybrid_newton_inference(
        grid=DD_grid,
        P=P_imp_DD,
        S=S_DD,
        Pb=Pb_dict_DD,
        Sb_dict=Sb_dict_DD,
        phi=phi_DD,
        K=K_DD,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=qt[1],
        total_sim_time=qt[1],
        kr_model=kr_model,
        max_newton_iter=max_newton_iter,
        eps=1e-6,
        wells=[well_co2_DD],
        P_guess=P_imp_DD,
        S_guess=S_DD,
    )
    dict_save["S_DD_local"] = S_DD_plus_1.tolist()
    dict_save["P_DD_local"] = P_DD_plus_1.tolist()
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(30, 4))
    # ax1.imshow(P_imp_DD.reshape(9, 9).T)
    # ax2.imshow(S_DD.reshape(9, 9).T)
    # ax3.imshow(P_DD_plus_1.reshape(9, 9).T)
    # ax4.imshow(S_DD_plus_1.reshape(9, 9).T)
    # ax5.imshow(S_pred.reshape(9, 9).T)
    #
    # ax1.invert_yaxis()
    # ax2.invert_yaxis()
    # ax3.invert_yaxis()
    # ax4.invert_yaxis()
    # ax5.invert_yaxis()
    #
    # ax1.set_title('P_imp_DD')
    # ax2.set_title('S_DD')
    # ax3.set_title('P_DD_plus_1')
    # ax4.set_title('S_DD_plus_1')
    # ax5.set_title('S_pred')
    # fig.suptitle(f'DD local: {nb_newton}')
    # plt.show()
    #
    S_DD_global = copy.deepcopy(S)
    S_DD_global[cells_d] = S_DD_plus_1

    P_DD_global = copy.deepcopy(P_imp)
    P_DD_global[cells_d] = P_DD_plus_1

    P_i_plus_1, S_i_plus_1, dt_sim, nb_newton, norms = hybrid_newton_inference(
        grid=grid,
        P=P_DD_global,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=qt[1],
        total_sim_time=qt[1],
        kr_model=kr_model,
        max_newton_iter=max_newton_iter,
        eps=1e-6,
        wells=[well_co2],
        P_guess=P_DD_global,
        S_guess=S_DD_global,
    )

    dict_save["S_i_plus_1_DD"] = S_i_plus_1.tolist()
    dict_save["P_i_plus_1_DD"] = P_i_plus_1.tolist()
    dict_save["nb_newton_DD"] = nb_newton
    dict_save["dt_sim_DD"] = dt_sim
    dict_save["norms_DD"] = norms

    # Hybrid
    S_pred_global = copy.deepcopy(S)
    S_pred_global[cells_d] = S_pred

    P_i_plus_1, S_i_plus_1, dt_sim, nb_newton, norms = hybrid_newton_inference(
        grid=grid,
        P=P_imp,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=qt[1],
        total_sim_time=qt[1],
        kr_model=kr_model,
        max_newton_iter=max_newton_iter,
        eps=1e-6,
        wells=[well_co2],
        P_guess=P_imp,
        S_guess=S_pred_global,
    )

    dict_save["S_i_plus_1_hybrid"] = S_i_plus_1.tolist()
    dict_save["P_i_plus_1_hybrid"] = P_i_plus_1.tolist()
    dict_save["nb_newton_hybrid"] = nb_newton
    dict_save["dt_sim_hybrid"] = dt_sim
    dict_save["norms_hybrid"] = norms
    print(
        f"step {i}: Newton classic {dict_save['nb_newton_classic']}, hybrid {dict_save['nb_newton_hybrid']}, "
        f"DD {dict_save['nb_newton_DD']}"
    )
    print(f"step number {i} finished")
    return dict_save


def main():
    test["log_q"] = -np.log10(-test["q"])
    test["log_dt"] = np.log(test["dt"])
    qts = test[["q", "dt", "S0_local"]].to_numpy()
    log_qts = test[["log_q", "log_dt"]].to_numpy()
    P_imps = test['P_imp_local'].to_numpy()
    for i in range(len(test)):
        result = launch_inference(qt=qts[i], log_qt=log_qts[i], i=i, test_P=P_imps[i], test_S=None)
        df = pd.DataFrame([result])
        df.to_csv(
            f"./results/quantification_{ext}_test_{rank}_{len(test)}_{i}.csv",
            sep="\t",
            index=False,
        )
        if rank == 0:
            print(f"saving simulation number {i}")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nb_proc = comm.Get_size()
    ext = 4
    if rank == 0:
        test_full = test = pd.read_csv("data/case_1_q_5_5_dt_1_10_S_0_06.csv",
                                       converters={"P_imp_local": literal_eval, "S0_local": literal_eval},
                                       sep="\t")

        save_dir = "results"
        test_split = np.array_split(test_full, nb_proc)

        if not os.path.isdir(save_dir):
            sp.call(f"mkdir {save_dir}", shell=True)
        # define reservoir setup
        grid = load_json("../../../../../../meshes/SHP_CO2/2D/SHP_CO2_2D_S.json")
        # Boundary groups creation
        grid.add_face_group_by_line("injector_one", (0.0, 0.0), (0.0, 1000.0))
        grid.add_face_group_by_line("injector_two", (2500.0, 3000.0), (3500.0, 3000.0))

    else:
        grid = None
        test_split = None

    test = comm.scatter(test_split, root=0)

    grid = comm.bcast(grid, root=0)
    # define reservoir setup

    # Permeability barrier zone creation
    barrier_1 = grid.find_cells_inside_square((1000.0, 2000.0), (1250.0, 0))
    barrier_2 = grid.find_cells_inside_square((2250.0, 3000.0), (2500.0, 1000.0))
    barrier_3 = grid.find_cells_inside_square((3500.0, 3000.0), (3750.0, 500.0))

    phi = 0.2
    # Porosity
    phi = np.full(grid.nb_cells, phi)
    # Diffusion coefficient (i.e Permeability)
    K = np.full(grid.nb_cells, 100.0e-15)
    permeability_barrier = 1.0e-15
    K[barrier_1] = permeability_barrier
    K[barrier_2] = permeability_barrier
    K[barrier_3] = permeability_barrier
    # Pressure initialization
    P = np.full(grid.nb_cells, 100.0e5)
    T = calculate_transmissivity(grid, K)

    # viscosity
    mu_w = 0.571e-3
    mu_g = 0.0285e-3

    kr_model = "quadratic"
    # BOUNDARY CONDITIONS #
    # Pressure
    Pb = {"injector_one": 110.0e5, "injector_two": 105.0e5, "right": 100.0e5}
    # Saturation
    Sb_d = {"injector_one": 0.0, "injector_two": 0.0, "right": 0.0}
    Sb_n = {"injector_one": None, "injector_two": None, "right": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

    max_newton_iter = 200
    eps = 1e-6

    S_model = model = FNO2d(modes1=12, modes2=12, width=64, n_features=4)
    S_model.load_state_dict(
        torch.load(
            "models/checkpoint_best_model_4_local_2d_1500.pt",
            map_location=torch.device("cpu"),
        )
    )
    q_normalizer = pickle.load(open("models/q_normalizer.pkl", "rb"))
    P_imp_normalizer = pickle.load(open("models/P_imp_normalizer.pkl", "rb"))
    dt_normalizer = pickle.load(open("models/dt_normalizer.pkl", "rb"))
    main()
