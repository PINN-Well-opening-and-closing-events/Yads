import os

import pandas as pd
from mpi4py import MPI
import numpy as np
from pyDOE import lhs
import time
import sys
import subprocess as sp

sys.path.append("/")
sys.path.append("/home/irsrvhome1/R16/lechevaa/YADS/Yads")

from yads.wells import Well
import yads.mesh as ym
from yads.numerics import calculate_transmissivity, implicit_pressure_solver


def main():
    save_dir = "P_imp_naive_generation"
    nb_data = 500
    np.random.seed(42)
    lhd = lhs(5, samples=nb_data, criterion="maximin")

    # grid = ym.two_D.create_2d_cartesian(51*20, 51*20, 51, 51)
    grid = ym.utils.load_json("meshes/51x51_20.json")
    # create dirs
    if not os.path.isdir(save_dir):
        sp.call(f"mkdir {save_dir}", shell=True)

    # Diffusion coefficient (i.e Permeability)
    K = np.full(grid.nb_cells, 100.0e-15)
    T = calculate_transmissivity(grid, K)

    # gaz saturation initialization
    S = np.full(grid.nb_cells, 0.0)
    # Pressure initialization
    P = np.full(grid.nb_cells, 100.0e5)

    # viscosity
    mu_w = 0.571e-3
    mu_g = 0.0285e-3

    kr_model = "quadratic"
    # BOUNDARY CONDITIONS #
    # Saturation
    Sb_d = {"left": 0.0, "upper": 0.0, "right": 0.0, "lower": 0.0}
    Sb_n = {"left": None, "upper": None, "right": None, "lower": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

    dt = 1 * (60 * 60 * 24 * 365.25)  # in years

    # scaling lhs data
    Pl = np.round(100e5 + lhd[:, 0] * (200e5 - 100e5), 1)
    Pr = np.round(100e5 + lhd[:, 1] * (200e5 - 100e5), 1)
    Pd = np.round(100e5 + lhd[:, 2] * (200e5 - 100e5), 1)
    Pu = np.round(100e5 + lhd[:, 3] * (200e5 - 100e5), 1)
    q = np.round(100e5 + lhd[:, 4] * (200e5 - 100e5), 1)

    for i in range(nb_data):
        if q[i] == max([Pr[i], Pd[i], Pu[i], Pl[i], q[i]]):
            print(f"launching Pressure generation number {i}")
            well_co2 = Well(
                name="well co2",
                cell_group=np.array([[20 * 51 / 2.0, 20 * 51 / 2]]),
                radius=0.1,
                control={"Dirichlet": q[i]},
                s_inj=1.0,
                schedule=[[0.0, dt],],
                mode="injector",
            )
            Pb = {"left": Pl[i], "upper": Pu[i], "right": Pr[i], "lower": Pd[i]}
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
                wells=[well_co2],
            )

            df_dict = {"P": P.tolist(), "Pb": Pb, "q": q[i]}
            df = pd.DataFrame([df_dict])
            df.to_csv(save_dir + f"/P_{i}.csv", sep="\t", index=False)


if __name__ == "__main__":
    print("launching generate data")
    start_time = time.time()
    main()
    print(f"realised in {time.time() - start_time:3e} seconds")
