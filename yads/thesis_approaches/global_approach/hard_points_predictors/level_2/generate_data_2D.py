import yads.mesh as ym
import numpy as np
from yads.wells import Well
from yads.predictors.hard_points_predictors.level_2.utils import args_to_dict
from yads.predictors.hard_points_predictors.level_2.find_hard_points import (
    find_hard_points,
)
from pyDOE import lhs

# S SHP CO2
grid = ym.two_D.create_2d_cartesian(50 * 95, 50 * 60, 95, 60)
dz = 100  # 100 meters
real_volume = np.multiply(grid.measures(item="cell"), dz)
grid.change_measures("cell", real_volume)


# There are 3 barriers inside the shp_c02 case:


grid.add_face_group_by_line("injector_one", (0.0, 0.0), (0.0, 1000.0))
grid.add_face_group_by_line("injector_two", (2500.0, 3000.0), (3500.0, 3000.0))


barrier_1 = grid.find_cells_inside_square((1000.0, 2000.0), (1250.0, 0))
barrier_2 = grid.find_cells_inside_square((2250.0, 3000.0), (2500.0, 1000.0))
barrier_3 = grid.find_cells_inside_square((3500.0, 3000.0), (3750.0, 500.0))

phi = 0.2
permeability_barrier = 1.0e-15

# Porosity
phi = np.full(grid.nb_cells, phi)
# Diffusion coefficient (i.e Permeability)
K = np.full(grid.nb_cells, 100.0e-15)
K[barrier_1] = permeability_barrier
K[barrier_2] = permeability_barrier
K[barrier_3] = permeability_barrier


# gaz saturation initialization
S = np.full(grid.nb_cells, 0.0)
# Pressure initialization
P = np.full(grid.nb_cells, 100.0e5)

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


dt = 1 * (60 * 60 * 24 * 365.25)  # in years
total_sim_time = 30 * (60 * 60 * 24 * 365.25)
max_newton_iter = 10

productor = Well(
    name="productor",
    cell_group=np.array([[7500.0, 500.0]]),
    radius=0.1,
    control={"Dirichlet": 100.0e5},
    s_inj=0.0,
    schedule=[[0.0, total_sim_time]],
    mode="productor",
)

well_co2 = Well(
    name="well co2",
    cell_group=np.array([[3000.0, 500.0]]),
    radius=0.1,
    control={"Neumann": -0.02},
    s_inj=1.0,
    schedule=[[0.0, total_sim_time],],
    mode="injector",
)

data_dict = args_to_dict(
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
    total_sim_time=total_sim_time,
    kr_model=kr_model,
    max_newton_iter=max_newton_iter,
    eps=1e-6,
    wells=[well_co2, productor],
)

lhd = lhs(3, samples=100, criterion="maximin")

q_pow = -np.power(10, -4.5 + lhd[:, 0] * (-3 - (-4.5)))
dt_init = 1e-3 * dt + lhd[:, 1] * (2 * dt - 1e-3 * dt)

q_pow = np.round(q_pow, 6)
dt_init = np.round(dt_init, 1)

S0 = lhd[:, 2]

find_hard_points(
    var_dict={"dt_init": dt_init, "q": q_pow, "S": S0},
    data_dict=data_dict,
    savepath="data/5000_level_2_2D_S_10_100_newtons",
)
