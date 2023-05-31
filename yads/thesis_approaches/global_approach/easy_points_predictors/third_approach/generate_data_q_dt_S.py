import yads.mesh as ym
import numpy as np
from yads.wells import Well
from yads.predictors.easy_points_predictors.third_approach.utils import args_to_dict
from yads.predictors.easy_points_predictors.third_approach.make_pipeline import (
    make_pipeline_lhs,
)
from pyDOE import lhs
import pandas as pd
from ast import literal_eval

grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 256, 1)

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
Pb = {"left": 100.0e5, "right": 100.0e5}

# Saturation

Sb_d = {"left": None, "right": None}
Sb_n = {"left": None, "right": None}
Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

dt = 1 * (60 * 60 * 24 * 365.25)  # in years
total_sim_time = 30 * (60 * 60 * 24 * 365.25)
max_newton_iter = 20

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

nb_samples = 5000
lhd = lhs(2, samples=nb_samples, criterion="maximin")

q = -np.power(10, -6 + lhd[:, 0] * 3)
dt_init = 1e-3 * dt + lhd[:, 1] * (dt - 1e-3 * dt)
df = pd.read_csv(
    "../second_approach/data/second_approach_5000.csv",
    converters={"S": literal_eval, "P": literal_eval, "S0": literal_eval},
)

S0 = np.array(list(df["S"].sample(nb_samples)))
del df

make_pipeline_lhs(
    var_dict={"dt_init": dt_init, "q": q, "S": S0},
    data_dict=data_dict,
    savepath="data/third_approach_5000",
)
