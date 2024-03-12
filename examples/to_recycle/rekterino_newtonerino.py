import numpy as np  # type: ignore
import yads.mesh as ym
import yads.numerics as yn
from yads.wells import Well

import time

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

"""
yn.impims_solver(
    grid,
    P,
    S,
    Pb,
    Sb_dict,
    phi,
    K,
    mu_g,
    mu_w,
    total_sim_time=total_sim_time,
    dt_init=dt,
    max_newton_iter=max_newton_iter,
    save=save,
    save_path="./saves/newton_1d_rekterino/impims/impims_",
    save_step=save_step,
    wells=[injector_two, well_co2],
    auto_dt=auto_dt,
    save_states_to_json=True,
    json_savepath="./saves/newton_1d_rekterino/impims/vanilla/vanilla.json",
    kr_model=kr_model
)


# gaz saturation initialization
S = np.full(grid.nb_cells, 0.0)
# Pressure initialization
P = np.full(grid.nb_cells, 100.0e5)
"""

yn.solss(
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
    save=save,
    save_path="./saves/newton_1d_rekterino/solss/solss_",
    save_step=save_step,
    wells=[well_co2, injector_two],
    auto_dt=auto_dt,
    save_states_to_json=False,
    # json_savepath="./saves/newton_1d_rekterino/solss/vanilla/vanilla.json",
    # cheating_path="./saves/newton_1d_rekterino/solss/vanilla/vanilla.json",
    cheating_S_bool=False,
    cheating_P_bool=False,
    debug_newton_mode=False,
    debug_newton_path="./saves/debug/vanilla/vanilla.json",
)
