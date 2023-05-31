from matplotlib import pyplot as plt
import yads.numerics as yn
import yads.mesh as ym
import numpy as np

from yads.wells import Well
from yads.predictors.hard_points_predictors.global_2D.utils import args_to_dict
from yads.predictors.hard_points_predictors.global_2D.find_hard_points import (
    find_hard_points,
)
from yads.mesh.utils import load_json
from pyDOE import lhs

print("loading mesh")
grid = load_json("../../../../meshes/SHP_CO2/2D/SHP_CO2_2D_S.json")
# grid = ym.two_D.create_2d_cartesian(50*95, 50*60, 95, 60)
print("mesh loaded")
print("generating reservoir configuration")
# dz = 100  # 100 meters
# real_volume = np.multiply(grid.measures(item="cell"), dz)
# grid.change_measures("cell", real_volume)

grid.add_face_group_by_line("injector_one", (0.0, 0.0), (0.0, 1000.0))
grid.add_face_group_by_line("injector_two", (2500.0, 3000.0), (3500.0, 3000.0))

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

dt = 10 * (60 * 60 * 24 * 365.25)  # in years
total_sim_time = 10 * (60 * 60 * 24 * 365.25)
max_newton_iter = 100
eps = 1e-6

# usual well pos [1500., 2250]
# New well pos
well_co2 = Well(
    name="well co2",
    cell_group=np.array([[1500.0, 2250]]),
    radius=0.1,
    control={"Neumann": -(10 ** -5)},
    s_inj=1.0,
    schedule=[[0.0, total_sim_time],],
    mode="injector",
)
print("reservoir configuration generated")

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
    eps=eps,
    wells=[well_co2],
)

print("generating samples")
lhd = lhs(2, samples=4, criterion="maximin")
print("samples generated")

# q_pow = -np.power(10, -5.0 + lhd[:, 0] * (-3 - (-5.0)))
# dt_init = 1e-3 * dt + lhd[:, 1] * (2 * dt - 1e-3 * dt)
#
# q_pow = np.round(q_pow, 6)
# dt_init = np.round(dt_init, 1)

# find_hard_points(data_dict=data_dict,
#                  var_dict={"dt_init": [4*dt], "q": [-4.5e-5]}, savepath="data/test")

max_iter = 100
save_step = 1
save = True

print("launching visualization")
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
    max_newton_iter=max_iter,
    eps=eps,
    save=save,
    save_path="./visualization/shp_co2_S_",
    save_step=save_step,
    wells=[well_co2],
    auto_dt=None,
    save_states_to_json=False,
    json_savepath="./visualization/debug/shp_co2_S_json.",
    # cheating_path="./saves/newton_1d_rekterino/solss/vanilla/vanilla.json",
    cheating_S_bool=False,
    cheating_P_bool=False,
    debug_newton_mode=False,
    debug_newton_path="./visualization/debug/8_criterion/shp_co2_S_",
)
print("visualization save to vtk")
