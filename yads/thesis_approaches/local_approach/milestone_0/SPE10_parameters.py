import numpy as np  # type: ignore
from matplotlib import pyplot as plt

import yads.mesh as ym
from yads.numerics import calculate_transmissivity
from yads.numerics.solvers.solss_solver_depreciated import solss_newton_step
from yads.wells import Well

# ft to m
ft = 0.3
factor_x, factor_y = 1.0, 1.0
grid = ym.two_D.create_2d_cartesian(
    int(1200 * ft * factor_x),
    int(2200 * ft * factor_y),
    int(60 * factor_x),
    int(220 * factor_y),
)
# grid.to_json("../SPE_mesh/SPE_10")
# grid = ym.utils.load_json("../SPE_mesh/SPE_10.json")

dz = 2  # 2 meters
real_volume = np.multiply(grid.measures(item="cell"), dz)
grid.change_measures("cell", real_volume)

phi = 0.4

# Porosity
phi = np.full(grid.nb_cells, phi)
# Diffusion coefficient (i.e Permeability)
K = np.full(grid.nb_cells, 2.0e-7)
T = calculate_transmissivity(grid, K)

# gaz saturation initialization
S = np.full(grid.nb_cells, 0.0)
# Pressure initialization
P = np.full(grid.nb_cells, 100.0e5)

mu_w = 0.571e-3
mu_g = 0.0285e-3

# BOUNDARY CONDITIONS #
# Pressure
Pb = {"left": None, "upper": None, "right": None, "lower": None}

# Saturation
Sb_d = {"left": None, "upper": None, "right": None, "lower": None}
Sb_n = {"left": None, "upper": None, "right": None, "lower": None}
Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}


dt = 60 * 60 * 24 * 1
max_newton_iter = 20

well_CO2 = Well(
    name="well_CO2",
    cell_group=np.array([[1200 * ft * factor_x * 0.5, 2200 * ft * factor_y * 0.5]]),
    radius=0.1,
    control={"Dirichlet": 0.6 * 700.0e5},
    s_inj=1.0,
    schedule=[[0.0, dt]],
    mode="injector",
)

well_1 = Well(
    name="well_1",
    cell_group=np.array([[1200 * ft * factor_x, 2200 * ft * factor_y]]),
    radius=0.1,
    control={"Dirichlet": 0.4 * 700.0e5},
    s_inj=1.0,
    schedule=[[0.0, dt]],
    mode="productor",
)

well_2 = Well(
    name="well_2",
    cell_group=np.array([[0, 2200 * ft * factor_y]]),
    radius=0.1,
    control={"Dirichlet": 0.4 * 700.0e5},
    s_inj=1.0,
    schedule=[[0.0, dt]],
    mode="productor",
)

well_3 = Well(
    name="well_3",
    cell_group=np.array([[1200 * ft * factor_x, 0]]),
    radius=0.1,
    control={"Dirichlet": 0.4 * 700.0e5},
    s_inj=1.0,
    schedule=[[0.0, dt]],
    mode="productor",
)

well_4 = Well(
    name="well_4",
    cell_group=np.array([[0, 0]]),
    radius=0.1,
    control={"Dirichlet": 0.4 * 700.0e5},
    s_inj=1.0,
    schedule=[[0.0, dt]],
    mode="productor",
)

P_i_plus_1, S_i_plus_1, dt_sim, nb_newton, norms = solss_newton_step(
    grid=grid,
    P_i=P,
    S_i=S,
    Pb=Pb,
    Sb_dict=Sb_dict,
    phi=phi,
    K=K,
    T=T,
    mu_g=mu_g,
    mu_w=mu_w,
    dt_init=dt,
    dt_min=dt,
    wells=[well_CO2, well_1, well_2, well_3, well_4],
    max_newton_iter=max_newton_iter,
    eps=1e-6,
    kr_model="quadratic",
    P_guess=P,
    S_guess=S,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

print(grid.cells.shape, dt_sim)
ax1.imshow(P_i_plus_1.reshape(int(60 * factor_x), int(220 * factor_y)).T)
ax2.imshow(S_i_plus_1.reshape(int(60 * factor_x), int(220 * factor_y)).T)

fig.axes[1].invert_yaxis()
fig.axes[0].invert_yaxis()
# ax1.title.set_text(f"Well flow {10**q:.2E}, timestep {np.exp(dt)/(60*60*24*365.25):.1E} years")
plt.show()
