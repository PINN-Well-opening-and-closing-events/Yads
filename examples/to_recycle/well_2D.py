import numpy as np  # type: ignore
import yads.mesh as ym
import yads.numerics as yn
from yads.wells import Well
import time

#### IMPES: IMPLICIT PRESSURE EXPLICIT SATURATION (SOLVER) ####

#### System initialization ####
load_time = time.time()

# 2D example
grid = ym.two_D.create_2d_cartesian(1089, 1089, 33, 33)


# dz = 100  # 100 meters
# real_volume = np.multiply(grid.measures(item="cell"), dz)
# grid.change_measures("cell", real_volume)

# PHYSICS

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
Pb = {"left": 100.0e5, "right": 100.0e5}

# Saturation

Sb_d = {"left": 0.0, "right": 0.0}
Sb_n = {"left": None, "right": None}
Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}


dt = 0.1 * (60 * 60 * 24 * 365.25)  # in years
total_sim_time = 1.0 * (60 * 60 * 24 * 365.25)
eps = 1e-6

well_co2 = Well(
    name="well co2",
    cell_group=np.array([[545.0, 545.0]]),
    radius=0.1,
    control={"Neumann": -0.002},
    s_inj=1.0,
    schedule=[
        [0.0, total_sim_time],
    ],
    mode="injector",
)

max_iter = 20
save_step = 1
save = True


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
    max_newton_iter=max_iter,
    save=save,
    save_path="./saves/cartesian_test/well_2D/solss_",
    save_step=save_step,
    wells=[well_co2],
    auto_dt=None,
    save_states_to_json=False,
    # json_savepath="./saves/newton_1d_rekterino/solss/vanilla/vanilla.json",
    # cheating_path="./saves/newton_1d_rekterino/solss/vanilla/vanilla.json",
    cheating_S_bool=False,
    cheating_P_bool=False,
    debug_newton_mode=False,
    debug_newton_path="./saves/debug/vanilla/vanilla.json",
)
