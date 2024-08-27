import numpy as np  # type: ignore
import yads.mesh as ym
import yads.numerics as yn
import yads.physics as yp
from yads.wells import Well

# grid = ym.load_mesh("../meshes/2D/Tests/rod_10_1/rod_10_1.mesh")
grid = ym.two_D.create_2d_cartesian(4, 2, 2, 2)

# Porosity
phi = np.ones(grid.nb_cells)
# Diffusion coefficient (i.e Permeability)
K = np.ones(grid.nb_cells)
# Water saturation initialization
S = np.full(grid.nb_cells, 0.1)
# Pressure initialization
P = np.full(grid.nb_cells, 1.5)

T = yn.calculate_transmissivity(grid, K)

mu_w = 1.0
mu_g = 1.0

M = yp.total_mobility(S, mu_w, mu_g)

# BOUNDARY CONDITIONS #
# Pressure
Pb = {"left": 1.2, "right": 1.0}

# Saturation
Sb_d = {"left": 0.1, "right": 0.1}
Sb_n = {"left": None, "right": None}
Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

dt = 0.1
save = False
total_sim_time = 10.0
max_iter = 200
save_step = 1

well_1 = Well(
    name="well_1",
    cell_group=np.array([[4.1, 0.1]]),
    radius=0.1,
    control={"Dirichlet": 3.0},
    sw=1.0,
    schedule=[[0.0, 1.0]],
)

print("IMPLICIT")
yn.impims_solver(
    grid,
    P,
    S,
    Pb,
    Sb_dict,
    phi,
    K,
    mu_w,
    mu_g,
    total_sim_time=total_sim_time,
    dt_init=dt,
    max_iter=max_iter,
    save=save,
    save_path="./saves/cartesian_test/impims_01_/impims_01_",
    save_step=save_step,
    wells=[well_1],
)
