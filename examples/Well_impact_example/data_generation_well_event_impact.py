import numpy as np
import yads.mesh as ym
from yads.wells import Well
from yads.numerics.schemes.solss import solss
import pickle
import os

# create 1D cartesian mesh 10 000m x 1000m with 200 cells in x direction and 1 in y direction
grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 201, 1)

# define initial time step and total simulation time in seconds
dt = 1 * (60 * 60 * 24 * 365.25)  # in years

#### PHYSICS ####
# Porosity
phi = np.full(grid.nb_cells, 0.2)
# Diffusion coefficient (i.e Permeability)
K = np.full(grid.nb_cells, 100.0e-15)
# gaz saturation initialization
S = np.full(grid.nb_cells, 0.0)
# Pressure initialization
P = np.full(grid.nb_cells, 100.0e5)
# viscosity
mu_w = 0.571e-3
mu_g = 0.0285e-3
# relative permeability model
kr_model = "quadratic"

# BOUNDARY CONDITIONS #
# Pressure
Pb = {"left": 110.0e5, "right": 100.0e5}
# Saturation
# only water left and right
Sb_d = {"left": 0.0, "right": 0.0}
Sb_n = {"left": None, "right": None}
Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

well_co2 = Well(
    name="well co2",
    cell_group=np.array([[5000.0, 500.0]]),
    radius=0.1,
    control={"Dirichlet": 200.0e5},
    s_inj=1.0,
    schedule=[
        [4 * dt, 24 * dt],
    ],
    mode="injector",
)

eps = 1e-6
max_newton_iter = 200

os.makedirs("well_impact_video", exist_ok=True)
os.makedirs("newton_list", exist_ok=True)
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
    dt_init=2 * dt,
    total_sim_time=60 * dt,
    kr_model=kr_model,
    wells=[well_co2],
    max_newton_iter=max_newton_iter,
    eps=eps,
    save=True,
    save_step=1,
    save_path="well_impact_video/well_impact",
    save_states_to_json=True,
    json_savepath="./well_impact.json",
)

with open("newton_list/well_event_newton_list.pkl", "wb") as fp:
    pickle.dump(newton_list, fp)
