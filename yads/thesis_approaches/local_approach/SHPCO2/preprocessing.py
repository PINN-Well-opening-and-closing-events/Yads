import pandas as pd
from ast import literal_eval
import numpy as np
import os
import yads.mesh as ym
import matplotlib.pyplot as plt
path_dir = "data/case_0/data/train_q_5_3_dt_1_10_S_0_06_P_imp.csv"
dists = [4]

grid = ym.utils.load_json("../../../../meshes/SHP_CO2/2D/SHP_CO2_2D_S.json")
print(grid.centers(item="cell")[1784])
i = 0

# df = pd.read_csv(
#     path_dir,
#     sep="\t",
#     nrows=1800,
#     skiprows=list(range(1, 1800 * i + 1)),
#     converters={"S": literal_eval, "P_imp": literal_eval, "S0": literal_eval},
# )

df = pd.read_csv(path_dir, sep='\t',
                 converters={'S': literal_eval, 'S0': literal_eval, 'P_imp': literal_eval})


# print(len(df), (i + 1) * 1800)
# create all light_df for training
light_local_df = []
well_x, well_y = 1475, 2225
grid_dxy = 50

P_imp_local_list = []
S_local_list = []
S0_local_list = []


for idx in range(len(df)):
    P_imp_global = np.array(df["P_imp"].loc[idx])
    S_global = np.array(df["S"].loc[idx])
    S0_global = np.array(df["S0"].loc[idx])
    for d in dists:
        cells_d = grid.find_cells_inside_square(
            (grid_dxy * (well_x / grid_dxy - d), grid_dxy * (well_y / grid_dxy + d)),
            (grid_dxy * (well_x / grid_dxy + d), grid_dxy * (well_y / grid_dxy - d)),
        )
        if idx == 0:
            print(len(cells_d))
        P_imp_local = P_imp_global[cells_d]
        S_local = S_global[cells_d]
        S0_local = S0_global[cells_d]
        P_imp_local_list.append(P_imp_local.tolist())
        S_local_list.append(S_local.tolist())
        S0_local_list.append(S0_local.tolist())

df["P_imp_local"] = P_imp_local_list
df["S_local"] = S_local_list
df["S0_local"] = S0_local_list


print(len(S0_local_list))
# df[["q", "dt", "P_imp_local", "S_local", "S0_local"]].to_csv(
#     path_dir[:-4] + f"_extension_4_{i}.csv", sep="\t", index=False
# )
df[['q', 'dt', 'P_imp_local', 'S_local', 'S0_local']].to_csv(path_dir[:-4] + f"_extension_4.csv", sep='\t', index=False)





