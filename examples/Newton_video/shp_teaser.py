import pickle
import os

import numpy as np
import time

from yads.wells import Well
from yads.mesh.utils import load_json
from yads.numerics.schemes.solss import solss


def main():
    grid = load_json("../../meshes/SHP_CO2/2D/SHP_CO2_2D_S.json")

    # Boundary groups creation
    grid.add_face_group_by_line("injector_one", (0.0, 0.0), (0.0, 1000.0))
    grid.add_face_group_by_line("injector_two", (2500.0, 3000.0), (3500.0, 3000.0))

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
    # gaz saturation initialization
    S = np.full(grid.nb_cells, 0.0)
    # Pressure initialization
    P = np.full(grid.nb_cells, 100.0e5)

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

    dt = 2 * (60 * 60 * 24 * 365.25)  # in years

    max_newton_iter = 200
    eps = 1e-6
    os.makedirs("physical_video", exist_ok=True)
    os.makedirs("physical_video/shp_teaser", exist_ok=True)
    os.makedirs("physical_video/newton_lists", exist_ok=True)
    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[1475.0, 2225]]),
        radius=0.1,
        control={"Neumann": -(10**-3.1)},
        s_inj=1.0,
        schedule=[
            [2 * dt, 12 * dt],
        ],
        mode="injector",
    )

    newton_list, dt_list = solss(
        grid=grid,
        P=P,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=dt,
        total_sim_time=40 * dt,
        kr_model=kr_model,
        wells=[well_co2],
        max_newton_iter=max_newton_iter,
        eps=eps,
        save=True,
        save_step=1,
        save_path="physical_video/shp_teaser/shp_teaser_",
    )
    with open(
        "physical_video/newton_lists/shp_teaser_newton_list_classic.pkl", "wb"
    ) as fp:
        pickle.dump(newton_list, fp)


if __name__ == "__main__":
    print("launching video teasing")
    start_time = time.time()
    main()
    print(f"realised in {time.time() - start_time:3e} seconds")
