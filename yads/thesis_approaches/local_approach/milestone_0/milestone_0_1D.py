import numpy as np  # type: ignore
import yads.mesh as ym
import yads.numerics as yn
from yads.wells import Well
from yads.thesis_approaches.data_generation import raw_solss_1_iter
import matplotlib.pyplot as plt

grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 200, 1)

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
Pb = {"left": 110.0e5, "right": 100.0e5}

# Saturation
# only water left and right
Sb_d = {"left": 0.0, "right": 0.0}
Sb_n = {"left": None, "right": None}
Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

dt = 155103597.1  # in years
total_sim_time = 155103597.1
max_newton_iter = -1

well_co2 = Well(
    name="well co2",
    cell_group=np.array([[5000.0, 500.0]]),
    radius=0.1,
    control={"Neumann": -9.5e-4},
    s_inj=1.0,
    schedule=[[0, total_sim_time]],
    mode="injector",
)

sim_state = raw_solss_1_iter(
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
    total_sim_time=dt,
    kr_model=kr_model,
    max_newton_iter=max_newton_iter,
    eps=1e-6,
    wells=[well_co2],
    debug_newton_mode=True,
)

print(sim_state["data"][str(dt)].keys())
P_plot = np.array(sim_state["data"][str(dt)]["P"]) / 10**6
S_plot = sim_state["data"][str(dt)]["S"]

first_non_zero_S_x = (np.round(S_plot, 6) != 0).argmax(axis=0)
last_non_zero_S_x = len(S_plot) - (np.flip(np.round(S_plot, 6)) != 0).argmax(axis=0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

ax2.axvline(x=first_non_zero_S_x, ls="--", c="r")
ax2.axvline(x=last_non_zero_S_x, ls="--", c="r")
ax1.scatter(range(len(P_plot)), P_plot, s=6)
ax2.scatter(range(len(S_plot)), S_plot, s=6)

ax1.set(xlabel="x", ylabel="P (MPa)")
ax2.set(
    xlabel="x",
    ylabel="S",
    title=f"well extension: {last_non_zero_S_x - first_non_zero_S_x} cells",
)
# plt.show()
