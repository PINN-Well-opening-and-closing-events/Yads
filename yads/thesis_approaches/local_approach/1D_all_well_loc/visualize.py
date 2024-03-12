import pandas as pd
from ast import literal_eval
from yads.wells import Well
import numpy as np
import yads.mesh as ym

df_1 = pd.read_csv(
    "data/manuscrit/all_sim_5000_4_1_2500_1447.csv",
    sep="\t",
    converters={"well_loc": literal_eval},
)

print(df_1.columns)

print(df_1["well_loc"])
well_1 = Well(
    name="well 2",
    cell_group=np.array(np.array(df_1["well_loc"])),
    radius=0.1,
    control={"Neumann": -1},
    s_inj=1.0,
    schedule=[
        [0.0, 1],
    ],
    mode="injector",
)

df_2 = pd.read_csv(
    "data/manuscrit/all_sim_5000_4_1_2500_1259.csv",
    sep="\t",
    converters={"well_loc": literal_eval},
)

print(df_1["well_loc"])
well_2 = Well(
    name="well 2",
    cell_group=np.array(np.array(df_2["well_loc"])),
    radius=0.1,
    control={"Neumann": -1},
    s_inj=1.0,
    schedule=[
        [0.0, 1],
    ],
    mode="injector",
)

grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 200, 1)
grid.connect_well(well_1)
grid.connect_well(well_2)
for c in grid.cell_groups[well_1.name]:
    print(c)
for c in grid.cell_groups[well_2.name]:
    print(c)
