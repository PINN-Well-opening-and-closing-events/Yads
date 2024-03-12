import numpy as np
import yads.mesh as ym
import yads.numerics as yn
from yads.wells import Well

# create 1D cartesian mesh 10 000m x 1000m with 200 cells in x direction and 1 in y direction
grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 200, 1)

# define initial time step and total simulation time in seconds
dt = 1 * (60 * 60 * 24 * 365.25)  # in years
total_sim_time = 3 * (60 * 60 * 24 * 365.25)  # in years

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
    cell_group=np.array([[3000.0, 500.0]]),
    radius=1.0,
    control={"Dirichlet": 200.0e5},
    s_inj=1.0,
    schedule=[
        [0.4 * total_sim_time, 0.6 * total_sim_time],
    ],
    mode="injector",
)

productor = Well(
    name="productor",
    cell_group=np.array([[7500.0, 500.0]]),
    radius=1.0,
    control={"Dirichlet": 100.0e5},
    s_inj=0.0,
    schedule=[
        [0.0, total_sim_time],
    ],
    mode="productor",
)

# Fully Implicit scheme
yn.solss(
    grid=grid,
    P=P,
    S=S,
    Pb=Pb,
    Sb_dict=Sb_dict,
    phi=phi,
    K=K,
    mu_g=mu_g,
    mu_w=mu_w,
    kr_model=kr_model,
    total_sim_time=total_sim_time,
    dt_init=dt,
    max_newton_iter=10,
    wells=[well_co2, productor],
    save_states_to_json=True,
    json_savepath="./discontinuity_figure_grad_P.json",
)
