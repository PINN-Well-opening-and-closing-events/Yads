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
from pyDOE import lhs

import sys

sys.path.append("/home/AD.NORCERESEARCH.NO/anle/Yads")
from yads.mesh.utils import load_json
from yads.numerics.physics import calculate_transmissivity
from yads.numerics.solvers import solss_newton_step, implicit_pressure_solver
from yads.wells import Well
from yads.mesh import Mesh
import yads.mesh as ym
from yads.numerics.solvers.newton import res


def is_well_loc_ok(well_loc, grid: Mesh, K, i):
    well_x, well_y = well_loc
    cells_d = grid.find_cells_inside_square(
        (grid_dxy * (well_x / grid_dxy - d), grid_dxy * (well_y / grid_dxy + d)),
        (grid_dxy * (well_x / grid_dxy + d), grid_dxy * (well_y / grid_dxy - d)),
    )

    # Ensure the local domain has the correct size
    if len(cells_d) != (2 * d + 1) * (2 * d + 1):
        return False
    # Ensure local domain does not touch boundary
    for c in cells_d:
        if c in fbd_cells:
            return False
    cells_d_plus_i = grid.find_cells_inside_square(
        (
            grid_dxy * (well_x / grid_dxy - (d + i)),
            grid_dxy * (well_y / grid_dxy + (d + i)),
        ),
        (
            grid_dxy * (well_x / grid_dxy + (d + i)),
            grid_dxy * (well_y / grid_dxy - (d + i)),
        ),
    )
    # Ensure no permeability barrier are reached
    if not np.all(K[cells_d_plus_i] == 100.0e-15):
        return False
    return True


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


def launch_inference(qt, log_qt, i, well_loc, grid):
    tmp_well_location = grid.centers(item="cell")[well_loc]
    print(f"launching step number {i}, with {qt[0], qt[1], qt[2], tmp_well_location}")
    dict_save = {
        "q": qt[0],
        "total_sim_time": qt[1],
        "S0": qt[2],
        "well_loc_idx": well_loc,
        "well_loc": tmp_well_location.tolist(),
    }

    well_co2 = Well(
        name="well co2",
        cell_group=np.array([tmp_well_location]),
        radius=0.1,
        control={"Neumann": qt[0]},
        s_inj=1.0,
        schedule=[
            [0.0, qt[1]],
        ],
        mode="injector",
    )
    Sb_dict["Dirichlet"] = {
        "injector_one": qt[2],
        "injector_two": qt[2],
        "right": 0.0,
    }
    S = np.full(grid.nb_cells, qt[2])
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
    well_x, well_y = tmp_well_location
    grid_dxy = 50
    d = 4
    cells_d = grid.find_cells_inside_square(
        (grid_dxy * (well_x / grid_dxy - d), grid_dxy * (well_y / grid_dxy + d)),
        (grid_dxy * (well_x / grid_dxy + d), grid_dxy * (well_y / grid_dxy - d)),
    )

    well_loc_idx = 40
    P_imp_local = P_imp[cells_d]
    S_n = S[cells_d]
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

    #### GWM prediction
    # normalizer prep
    GWM_log_q_n = GWM_q_normalizer.encode(log_q)
    GWM_log_dt_n = GWM_dt_normalizer.encode(log_dt)
    GWM_P_imp_local_n = GWM_P_imp_normalizer.encode(np.log10(P_imp_local_n))
    #
    x = torch.cat([GWM_log_q_n, GWM_log_dt_n, S_n, GWM_P_imp_local_n], 2).float()
    x = x.reshape(1, local_shape, local_shape, 4)
    GWM_S_pred = GWM_S_model(x)
    GWM_S_pred = GWM_S_pred.detach().numpy()

    GWM_S_pred = np.reshape(GWM_S_pred, (local_shape * local_shape))

    dict_save["GWM_S_predict_local"] = GWM_S_pred.tolist()

    #### Local prediction
    # normalizer prep
    local_log_q_n = local_q_normalizer.encode(log_q)
    local_log_dt_n = local_dt_normalizer.encode(log_dt)
    local_P_imp_local_n = local_P_imp_normalizer.encode(np.log10(P_imp_local_n))
    #
    x = torch.cat([local_log_q_n, local_log_dt_n, S_n, local_P_imp_local_n], 2).float()
    x = x.reshape(1, local_shape, local_shape, 4)
    local_S_pred = local_S_model(x)
    local_S_pred = local_S_pred.detach().numpy()

    local_S_pred = np.reshape(local_S_pred, (local_shape * local_shape))

    dict_save["Local_S_predict_local"] = local_S_pred.tolist()

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

    ### GWM Hybrid Newton
    GWM_S_pred_global = copy.deepcopy(S)
    GWM_S_pred_global[cells_d] = GWM_S_pred

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
        S_guess=GWM_S_pred_global,
    )
    dict_save["GWM_S_i_plus_1_hybrid"] = S_i_plus_1.tolist()
    dict_save["GWM_P_i_plus_1_hybrid"] = P_i_plus_1.tolist()
    dict_save["GWM_nb_newton_hybrid"] = nb_newton
    dict_save["GWM_dt_sim_hybrid"] = dt_sim
    dict_save["GWM_norms_hybrid"] = norms

    ### Local Hybrid Newton
    Local_S_pred_global = copy.deepcopy(S)
    Local_S_pred_global[cells_d] = local_S_pred

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
        S_guess=Local_S_pred_global,
    )

    dict_save["Local_S_i_plus_1_hybrid"] = S_i_plus_1.tolist()
    dict_save["Local_P_i_plus_1_hybrid"] = P_i_plus_1.tolist()
    dict_save["Local_nb_newton_hybrid"] = nb_newton
    dict_save["Local_dt_sim_hybrid"] = dt_sim
    dict_save["Local_norms_hybrid"] = norms

    ### Domain decomposition
    DD_grid = ym.two_D.create_2d_cartesian(
        grid_dxy * (2 * ext + 1), grid_dxy * (2 * ext + 1), (2 * ext + 1), (2 * ext + 1)
    )
    S_DD = S[cells_d]
    K_DD = K[cells_d]
    phi_DD = phi[cells_d]
    P_imp_DD = P_imp[cells_d]

    well_co2_DD = Well(
        name="well co2 DD",
        cell_group=np.array(
            [[grid_dxy * (2 * ext + 1) / 2, grid_dxy * (2 * ext + 1) / 2]]
        ),
        radius=0.1,
        control={"Neumann": qt[0]},
        s_inj=1.0,
        schedule=[
            [0.0, qt[1]],
        ],
        mode="injector",
    )

    # draw wider rectangle of extension (d+1)
    cells_d_plus_1 = grid.find_cells_inside_square(
        (
            grid_dxy * (well_x / grid_dxy - (d + 1)),
            grid_dxy * (well_y / grid_dxy + (d + 1)),
        ),
        (
            grid_dxy * (well_x / grid_dxy + (d + 1)),
            grid_dxy * (well_y / grid_dxy - (d + 1)),
        ),
    )

    # intersect with local domain of extension d
    cells_DD = [c for c in cells_d_plus_1 if c not in cells_d]
    # remove edges
    cells_DD_centers = grid.centers(item="cell")[cells_DD]

    ur = (well_x + (d + 1) * grid_dxy, well_y + (d + 1) * grid_dxy)
    ul = (well_x + (d + 1) * grid_dxy, well_y - (d + 1) * grid_dxy)
    lr = (well_x - (d + 1) * grid_dxy, well_y + (d + 1) * grid_dxy)
    ll = (well_x - (d + 1) * grid_dxy, well_y - (d + 1) * grid_dxy)
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

    translation = (
        well_x - d * grid_dxy - grid_dxy / 2,
        well_y - d * grid_dxy - grid_dxy / 2,
    )
    for k, coord_DD in enumerate(face_center_coords):
        DD_index = None
        for j, coord in enumerate(grid.centers(item="cell")[cells_DD]):
            if (
                coord[0] == coord_DD[0] + translation[0] - grid_dxy / 2
                and coord[1] == coord_DD[1] + translation[1]
            ):
                DD_index = j
            elif (
                coord[0] == coord_DD[0] + translation[0] + grid_dxy / 2
                and coord[1] == coord_DD[1] + translation[1]
            ):
                DD_index = j
            elif (
                coord[0] == coord_DD[0] + translation[0]
                and coord[1] == coord_DD[1] + translation[1] + grid_dxy / 2
            ):
                DD_index = j
            elif (
                coord[0] == coord_DD[0] + translation[0]
                and coord[1] == coord_DD[1] + translation[1] - grid_dxy / 2
            ):
                DD_index = j

        if np.abs(coord_DD[0]) == 0 or np.abs(coord_DD[0]) == grid_dxy * (2 * ext + 1):
            line_point_1 = (coord_DD[0], coord_DD[1] - grid_dxy / 2)
            line_point_2 = (coord_DD[0], coord_DD[1] + grid_dxy / 2)
        else:
            line_point_1 = (coord_DD[0] - grid_dxy / 2, coord_DD[1])
            line_point_2 = (coord_DD[0] + grid_dxy / 2, coord_DD[1])

        Pb_dict_DD[f"boundary_face_{k}"] = P_imp[cells_DD[DD_index]]
        Sb_d_DD[f"boundary_face_{k}"] = S[cells_DD[DD_index]]
        Sb_n_DD[f"boundary_face_{k}"] = None

        group_by_line = (f"boundary_face_{k}", line_point_1, line_point_2)
        groups.append(group_by_line)
        DD_grid.add_face_group_by_line(*group_by_line)

    Sb_dict_DD = {"Dirichlet": Sb_d_DD, "Neumann": Sb_n_DD}

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

    # fig, axs = plt.subplots(2, 3)
    # axs[0][0].imshow(np.array(S_DD_global.reshape(95, 60)).T, vmin=0, vmax=1)
    # axs[0][1].imshow(np.array(Local_S_pred_global.reshape(95, 60)).T, vmin=0, vmax=1)
    # axs[0][2].imshow(np.array(GWM_S_pred_global.reshape(95, 60)).T, vmin=0, vmax=1)

    # axs[1][0].imshow(np.array(dict_save["S_i_plus_1_DD"]).reshape(95, 60).T, vmin=0, vmax=1)
    # axs[1][1].imshow(np.array(dict_save["Local_S_i_plus_1_hybrid"]).reshape(95, 60).T, vmin=0, vmax=1)
    # axs[1][2].imshow(np.array(dict_save["GWM_S_i_plus_1_hybrid"]).reshape(95, 60).T, vmin=0, vmax=1)

    # plt.savefig(f'debug/{i}_{well_loc}.png')

    print(
        f"""step {i}: Newton classic {dict_save['nb_newton_classic']}, GWM hybrid {dict_save['GWM_nb_newton_hybrid']}, local Hybrid {dict_save['Local_nb_newton_hybrid']}, Local DD {dict_save['nb_newton_DD']}"""
    )
    print(f"step number {i} finished")
    return dict_save


def main():
    for i in range(len(well_locs)):
        # set seed to avoid similar lhs
        np.random.seed(well_locs[i])
        lhs_sample = 10
        lhd = lhs(3, samples=lhs_sample, criterion="maximin")
        # scaling lhs data
        q_pow = -np.power(10, -5.0 + lhd[:, 0] * (-3.0 - (-5.0)))
        dt_init = 1e-1 * dt + lhd[:, 1] * (10 * dt - 1e-1 * dt)
        S0 = lhd[:, 2] * 0.6
        # rounding to avoid saving issues
        q_pow = np.round(q_pow, 6)
        dt_init = np.round(dt_init, 1)
        for sample in range(lhs_sample):
            grid_tmp = copy.deepcopy(grid)
            qts = np.array([q_pow[sample], dt_init[sample], S0[sample]])
            log_qts = np.array([-np.log10(-q_pow[sample]), np.log(dt_init[sample])])
            result = launch_inference(
                qt=qts, log_qt=log_qts, i=sample, well_loc=well_locs[i], grid=grid_tmp
            )
            df = pd.DataFrame([result])
            df.to_csv(
                f"./results_variable/quantification_{ext}_test_{rank}_{well_locs[i]}_{sample}.csv",
                sep="\t",
                index=False,
            )
            if rank == 0:
                print(f"saving simulation number {i + 1}")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nb_proc = comm.Get_size()
    ext = d = 4
    grid_dxy = 50
    np.random.seed(42)
    dt = 1 * (60 * 60 * 24 * 365.25)  # in years

    if rank == 0:
        save_dir = "results_variable"
        if not os.path.isdir(save_dir):
            sp.call(f"mkdir {save_dir}", shell=True)
        # define reservoir setup
        grid = load_json("../../../../../meshes/SHP_CO2/2D/SHP_CO2_2D_S.json")
        # Boundary groups creation
        grid.add_face_group_by_line("injector_one", (0.0, 0.0), (0.0, 1000.0))
        grid.add_face_group_by_line("injector_two", (2500.0, 3000.0), (3500.0, 3000.0))

    else:
        grid = None
        test_split = None

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
    #### Loading GWM model
    if rank == 0:
        print("Loading GWM model")
    GWM_S_model = FNO2d(modes1=12, modes2=12, width=64, n_features=4)
    GWM_S_model.load_state_dict(
        torch.load(
            "models/GWM_3100_checkpoint_2500.pt",
            map_location=torch.device("cpu"),
        )["model"]
    )

    GWM_q_normalizer = pickle.load(open("models/GWM_q_normalizer.pkl", "rb"))
    GWM_P_imp_normalizer = pickle.load(open("models/GWM_P_imp_normalizer.pkl", "rb"))
    GWM_dt_normalizer = pickle.load(open("models/GWM_dt_normalizer.pkl", "rb"))

    ##### Loading local model
    if rank == 0:
        print("Loading local model")
    local_S_model = FNO2d(modes1=12, modes2=12, width=64, n_features=4)
    local_S_model.load_state_dict(
        torch.load(
            "../../../local_approach/SHPCO2/data/case_0/models/checkpoint_best_model_4_local_2d_1500.pt",
            map_location=torch.device("cpu"),
        )
    )

    local_q_normalizer = pickle.load(
        open("../../../local_approach/SHPCO2/data/case_0/models/q_normalizer.pkl", "rb")
    )
    local_P_imp_normalizer = pickle.load(
        open(
            "../../../local_approach/SHPCO2/data/case_0/models/P_imp_normalizer.pkl",
            "rb",
        )
    )
    local_dt_normalizer = pickle.load(
        open(
            "../../../local_approach/SHPCO2/data/case_0/models/dt_normalizer.pkl", "rb"
        )
    )
    ###### Compute well locs
    if rank == 0:
        print("Computing well locs")
        # pre-compute cells the are on boundary (forbidden)
        fbd_cells_1 = []
        for group in grid.face_groups.keys():
            if group != "0":
                for f in grid.faces(group=group, with_nodes=False):
                    c = grid.face_to_cell(f, face_type="boundary")
                    fbd_cells_1.append(c)
        # pre-compute cells the are two cells away from boundary (forbidden)
        fbd_cells_2 = []
        for group in grid.face_groups.keys():
            if group == "0":
                for f in grid.faces(group=group, with_nodes=False):
                    c1, c2 = grid.face_to_cell(f, face_type="inner")
                    # check if one of the cells is on the boundary
                    if c1 in fbd_cells_1:
                        fbd_cells_2.append(c2)
                    elif c2 in fbd_cells_1:
                        fbd_cells_2.append(c1)
        # Stay far from boundary conditions
        fbd_cells_3 = list(grid.find_cells_inside_square((0.0, 1000.0), (1250.0, 0)))
        fbd_cells_4 = list(
            grid.find_cells_inside_square((2500.0, 3000.0), (3500.0, 2500.0))
        )
        fbd_cells_5 = list(
            grid.find_cells_inside_square((3750.0, 3000.0), (4500.0, 0.0))
        )

        fbd_cells = fbd_cells_1 + fbd_cells_2 + fbd_cells_3 + fbd_cells_4 + fbd_cells_5

        fbd_cells = list(dict.fromkeys(fbd_cells))

        print("Forbidden cells computed")
        well_locs_cell_idxs = [
            i
            for i, _ in enumerate(grid.cells)
            if is_well_loc_ok(grid.centers(item="cell")[i], grid, K, 3)
        ]
        print(f"Found {len(well_locs_cell_idxs)} possible well locations")
        np.random.seed(42)
        nb_well_samples = 200
        print(f"Sampling {nb_well_samples} well locations in the location pool")
        well_loc_samples = np.random.randint(
            low=0, high=len(well_locs_cell_idxs) - 1, size=nb_well_samples
        )
        well_loc_samples_cell_idxs = [well_locs_cell_idxs[i] for i in well_loc_samples]
        well_locs_split = np.array_split(well_loc_samples_cell_idxs, nb_proc)

    else:
        well_locs = None
        well_locs_split = None

    well_locs = comm.scatter(well_locs_split, root=0)

    if rank == 0:
        print("Launching main script")
    main()
