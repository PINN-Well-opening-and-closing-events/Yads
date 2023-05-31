import numpy as np  # type: ignore
import yads.mesh as ym
import yads.numerics as yn
from yads.wells import Well
import matplotlib.pyplot as plt

import time

#### SHP CO 2D CASE ####
# This is the S 2D mesh parameters
# pseudo 1d: 1000 m y / 10 000 m x
grid = ym.two_D.create_2d_cartesian(100 * 47, 100 * 30, 47, 30)
# check that there are 200 cells
print(grid.nb_cells)

dz = 100  # 100 meters
real_volume = np.multiply(grid.measures(item="cell"), dz)
grid.change_measures("cell", real_volume)


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
Pb = {"injector_one": 110.0e5, "injector_two": 105.0e5, "right": 100.0e5}

# Saturation
# only water left and right
Sb_d = {"injector_one": 0.0, "injector_two": 0.0, "right": 0.0}
Sb_n = {"injector_one": None, "injector_two": None, "right": None}
Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}


dt = 60 * 60 * 24 * 365.25
total_sim_time = 10000 * 60 * 60 * 24 * 365.25
auto_dt = [0.25, 0.5, 1.0, 1.1, 1.1, 1.0, total_sim_time]
max_iter = -1
save_step = 1
save = False


well_co2 = Well(
    name="well co2",
    cell_group=np.array([[1500.0, 2250.0]]),
    radius=0.1,
    control={"Dirichlet": 115.0e5},
    s_inj=1.0,
    schedule=[[100 / 10000 * total_sim_time, 8000 / 10000 * total_sim_time],],
)

"""
newton_list, dt_list = yn.impims_solver(
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
    save=save,
    save_path="./saves/newton_1d_rekterino/impims/impims_",
    save_step=save_step,
    wells=[well_co2],
    auto_dt=auto_dt,
)


"""
newton_list, dt_list = yn.solss(
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
    save=save,
    save_path="./saves/newton_1d_rekterino/solss/solss_",
    save_step=save_step,
    wells=[well_co2],
    auto_dt=auto_dt,
    max_newton_iter=50,
)


fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
ax1.plot(np.cumsum(dt_list), np.cumsum(newton_list))
ax2.plot(np.cumsum(dt_list), newton_list)
ax3.plot(np.cumsum(dt_list), dt_list)

for i, schedule in enumerate(well_co2.schedule):

    if i == 0:
        ax1.axvline(x=schedule[0], color="red", ls="--", lw=2, label="well opening")
        ax1.axvline(x=schedule[1], color="green", ls="--", lw=2, label="well closing")
        ax2.axvline(x=schedule[0], color="red", ls="--", lw=2, label="well opening")
        ax2.axvline(x=schedule[1], color="green", ls="--", lw=2, label="well closing")
        ax3.axvline(x=schedule[0], color="red", ls="--", lw=2, label="well opening")
        ax3.axvline(x=schedule[1], color="green", ls="--", lw=2, label="well closing")
    else:
        ax1.axvline(x=schedule[0], color="red", ls="--", lw=2)
        ax1.axvline(x=schedule[1], color="green", ls="--", lw=2)
        ax2.axvline(x=schedule[0], color="red", ls="--", lw=2)
        ax2.axvline(x=schedule[1], color="green", ls="--", lw=2)
        ax3.axvline(x=schedule[0], color="red", ls="--", lw=2)
        ax3.axvline(x=schedule[1], color="green", ls="--", lw=2)

plt.xlabel("time")

ax1.set_title("cumulative sum of newton iterations")
ax1.set(ylabel="newton iterations cumsum")
ax2.set_title("number of newton iterations")
ax2.set(ylabel="newton iterations")
ax3.set_title("dt value evolution through simulation time")
ax3.set(ylabel="dt value")

plt.legend()
fig.savefig("./saves/newton_1d_rekterino/newton_rekterino.png")
plt.show()
