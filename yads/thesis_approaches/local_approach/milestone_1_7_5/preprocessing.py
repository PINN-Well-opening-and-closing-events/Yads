import pandas as pd
from ast import literal_eval
import numpy as np
import os
import yads.mesh as ym
import matplotlib.pyplot as plt
path_dir = "data/light_train_q_5_3_dt_1_10_v2.csv"
dists = [10]

grid = ym.utils.load_json("../../../../meshes/SHP_CO2/2D/SHP_CO2_2D_S.json")
print(grid.centers(item="cell")[1784])
i = 7
# 1800
# df = pd.read_csv(
#     path_dir,
#     sep="\t",
#     nrows=1800,
#     skiprows=list(range(1, 1800 * i + 1)),
#     converters={"S": literal_eval, "P_imp": literal_eval, "S0": literal_eval},
# )

# df = pd.read_csv(path_dir, sep='\t',
#                  converters={'S': literal_eval, 'P_imp': literal_eval, 'S0': literal_eval})

train_df = pd.read_csv("data/train_q_5_3_dt_1_10_v2.csv", sep='\t')
test_df = pd.read_csv("data/test_q_5_3_dt_1_10_v2.csv", sep='\t')
big_df = pd.concat([train_df, test_df])
big_df.to_csv('data/all_sim_q_5_3_dt_1_10_v2.csv', sep='\t', index=False)
# print(len(df), (i + 1) * 1800)
# create all light_df for training
light_local_df = []
well_x, well_y = 1475, 2225
grid_dxy = 50

P_imp_local_list = []
S_local_list = []
S0_local_list = []


# Permeability barrier zone creation
barrier_1 = grid.find_cells_inside_square((1000.0, 2000.0), (1250.0, 0))
barrier_2 = grid.find_cells_inside_square((2250.0, 3000.0), (2500.0, 1000.0))
barrier_3 = grid.find_cells_inside_square((3500.0, 3000.0), (3750.0, 500.0))

# Diffusion coefficient (i.e Permeability)
K = np.full(grid.nb_cells, 100.0e-15)
permeability_barrier = 1.0e-15
K[barrier_1] = permeability_barrier
K[barrier_2] = permeability_barrier
K[barrier_3] = permeability_barrier


# for idx in range(len(df)):
#     P_imp_global = np.array(df["P_imp"].loc[idx])
#     S_global = np.array(df["S"].loc[idx])
#     S0_global = np.array(df["S0"].loc[idx])
#     for d in dists:
#         cells_d = grid.find_cells_inside_square(
#             (grid_dxy * (well_x / grid_dxy - d), grid_dxy * (well_y / grid_dxy + d)),
#             (grid_dxy * (well_x / grid_dxy + d), grid_dxy * (well_y / grid_dxy - d)),
#         )
#         if idx == 0:
#             print(len(cells_d))
#         P_imp_local = P_imp_global[cells_d]
#         S_local = S_global[cells_d]
#         S0_local = S0_global[cells_d]
#         P_imp_local_list.append(P_imp_local.tolist())
#         S_local_list.append(S_local.tolist())
#         S0_local_list.append(S0_local.tolist())


d = 4

## Usual well
cells_d = grid.find_cells_inside_square(
            (grid_dxy * (well_x / grid_dxy - d), grid_dxy * (well_y / grid_dxy + d)),
            (grid_dxy * (well_x / grid_dxy + d), grid_dxy * (well_y / grid_dxy - d)),
        )

K_local = K[cells_d]

## Short distance well
well_x_2, well_y_2 = 1675, 1725


cells_d_2 = grid.find_cells_inside_square(
            (grid_dxy * (well_x_2 / grid_dxy - d), grid_dxy * (well_y_2 / grid_dxy + d)),
            (grid_dxy * (well_x_2 / grid_dxy + d), grid_dxy * (well_y_2 / grid_dxy - d)),
        )

## Long distance well
well_x_3, well_y_3 = 2975, 425


cells_d_3 = grid.find_cells_inside_square(
            (grid_dxy * (well_x_3 / grid_dxy - d), grid_dxy * (well_y_3 / grid_dxy + d)),
            (grid_dxy * (well_x_3 / grid_dxy + d), grid_dxy * (well_y_3 / grid_dxy - d)),
        )

# df["P_imp_local"] = P_imp_local_list
# df["S_local"] = S_local_list
# df["S0_local"] = S0_local_list

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(K_local.reshape(2*d + 1, 2*d+1).T)
ax1.invert_yaxis()

K_global = K
K_global[cells_d] = 1000e-15
K_global[cells_d_2] = 500e-15
K_global[cells_d_3] = 250e-15
ax2.imshow(K_global.reshape(95, 60).T)
ax2.invert_yaxis()
plt.show()
# df[["q", "dt", "P_imp_local", "S_local", "S0_local"]].to_csv(
#     path_dir[:-4] + f"_extension_10_{i}.csv", sep="\t", index=False
# )

# df[['q', 'dt', 'P_imp_local', 'S_local', 'S0_local']].to_csv(path_dir[:-4] + f"_extension_10.csv", sep='\t', index=False)
