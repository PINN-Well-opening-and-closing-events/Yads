import numpy as np  # type: ignore
import yads.mesh as ym
import yads.numerics as yn
from yads.wells import Well
import time
import matplotlib.pyplot as plt

#### IMPES: IMPLICIT PRESSURE EXPLICIT SATURATION (SOLVER) ####

#### System initialization ####
load_time = time.time()

# load a mesh from meshes folder
# 2D example
grid = ym.load_mesh("../meshes/2D/Tests/rod_10_1/rod_10_1.mesh")
print("mesh loaded in %s seconds" % (time.time() - load_time))

# PHYSICS

# Porosity
phi = np.full(grid.nb_cells, 0.1)
# Diffusion coefficient (i.e Permeability)
K = np.full(grid.nb_cells, 1.0e-12)
# Water saturation initialization
S = np.full(grid.nb_cells, 0.01)
# Pressure initialization
P = np.full(grid.nb_cells, 100.0e5)

mu_w = 0.571e-3
mu_g = 0.0285e-3

# BOUNDARY CONDITIONS #
# Pressure
Pb = {"1": 110.0e5, "2": 95.0e5}

# Saturation
Sb_d = {"1": 1.0, "2": 0.1}
Sb_n = {"1": None, "2": None}
Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

dt = 0.01
save = True
total_sim_time = 10.0
max_iter = 10000

well_1 = Well(
    name="well_1",
    cell_group=np.array([[4.1, 0.1]]),
    radius=0.1,
    control={"Dirichlet": 110.0e5},
    sw=1.0,
    schedule=[[1.0, 2.5], [5.0, 7.5]],
)


# IMPES SOLVER

dt_list_with_well = yn.impes_solver(
    grid,
    P,
    S,
    Pb,
    Sb_dict,
    phi,
    K,
    mu_g,
    mu_w,
    wells=[well_1],
    total_sim_time=total_sim_time,
    dt_init=dt,
    max_iter=max_iter,
    auto_dt=False,
    save=save,
    save_path="./saves/well_opening_showcase/with_well/rod_10_1_with_well_",
    save_step=1,
    return_dt=True,
)

# Water saturation initialization
S = np.full(grid.nb_cells, 0.01)
# Pressure initialization
P = np.full(grid.nb_cells, 100e5)

dt_list_no_well = yn.impes_solver(
    grid,
    P,
    S,
    Pb,
    Sb_dict,
    phi,
    K,
    mu_g,
    mu_w,
    wells=[],
    total_sim_time=total_sim_time,
    dt_init=dt,
    max_iter=max_iter,
    auto_dt=False,
    save=save,
    save_path="./saves/well_opening_showcase/no_well/rod_10_1_no_well_",
    save_step=1,
    return_dt=True,
)

fig = plt.figure()
plt.plot(dt_list_with_well, label="with well")
plt.plot(dt_list_no_well, label="no well")


for i, schedule in enumerate(well_1.schedule):
    if i == 0:
        plt.axvline(
            x=schedule[0] / dt, color="red", ls="--", lw=2, label="well opening"
        )
        plt.axvline(
            x=schedule[1] / dt, color="green", ls="--", lw=2, label="well closing"
        )
    else:
        plt.axvline(x=schedule[0] / dt, color="red", ls="--", lw=2)
        plt.axvline(x=schedule[1] / dt, color="green", ls="--", lw=2)

plt.xlabel("step")
plt.ylabel("limit timestep value")
plt.title(
    "Comparison of limit timestep evolutions with and without a well opening with dt = {}".format(
        dt
    )
)
plt.legend()
fig.savefig("./saves/well_opening_showcase/well_opening_showcase.png")
plt.show()
