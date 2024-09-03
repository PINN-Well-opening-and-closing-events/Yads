import numpy as np  # type: ignore
import yads.mesh as ym
import yads.numerics as yn
from yads.wells import Well
import time

load_time = time.time()
grid = ym.two_D.create_2d_cartesian(250 * 19, 250 * 12, 19, 12)
print("mesh loaded in %s seconds" % (time.time() - load_time))


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

# BOUNDARY CONDITIONS #
# Pressure
Pb = {"injector_one": 200.0e5, "injector_two": 200.0e5, "right": 100.0e5}

# Saturation
Sb_d = {"injector_one": 1.0, "injector_two": 1.0, "right": 0.0}
Sb_n = {"injector_one": None, "injector_two": None, "right": None}
Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}


dt = 60 * 60 * 24 * 365.25
total_sim_time = 10000 * 60 * 60 * 24 * 365.25
auto_dt = [0.1, 0.2, 1.0, 2.0, 2.0, 60 * 60 * 24 * 30, 60 * 60 * 24 * 365.25]
max_iter = -1
save_step = 10
save = False

well_1 = Well(
    name="well_1",
    cell_group=np.array([[0.0, 0.0]]),
    radius=0.1,
    control={"Dirichlet": 110.0e6},
    sw=1.0,
    schedule=[[0.0, total_sim_time / 2.0]],
)

print("IMPLICIT")
yn.impims(
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
    max_iter=max_iter,
    auto_dt=auto_dt,
    save=save,
    save_path="./saves/shp_c02/test_uno_",
    save_step=save_step,
    wells=[],
)
