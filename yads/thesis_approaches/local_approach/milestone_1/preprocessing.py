import pandas as pd
from ast import literal_eval
import numpy as np
import os
import yads.mesh as ym

path_dir = "more_data"
filenames = os.listdir(path_dir)
dists = [5, 10]

Pb = {"left": 100.0e5, "upper": 100.0e5, "right": 100.0e5, "lower": 100.0e5}
grid = ym.utils.load_json("meshes/51x51_20.json")

for d in dists:
    if not os.path.isdir(path_dir + "_extension_" + str(d)):
        os.mkdir(path_dir + "_extension_" + str(d))

# create all light_df for training
for file in filenames:
    file_df = pd.read_csv(
        path_dir + "/" + file,
        sep="\t",
        converters={
            "S": literal_eval,
            "P": literal_eval,
            "P_imp": literal_eval,
            "Res": literal_eval,
        },
    )

    P_imp_global = np.array(file_df["P_imp"].loc[0])
    Res_global = np.array(file_df["Res"].loc[0])
    S_global = np.array(file_df["S"].loc[0])
    well_loc = S_global.argmax()

    for d in dists:
        cells_d = grid.find_cells_inside_square(
            (20 * (25 - d), 20 * (26 + d)), (20 * (26 + d), 20 * (25 - d))
        )
        P_imp_local = P_imp_global[cells_d]
        S_local = S_global[cells_d]
        Res_local_g = Res_global[cells_d]
        Res_local_w = Res_global[[grid.nb_cells + cell for cell in cells_d]]
        file_df["P_imp_local"] = [P_imp_local.tolist()]
        file_df["S_local"] = [S_local.tolist()]
        file_df["Res_local_g"] = [Res_local_g.tolist()]
        file_df["Res_local_w"] = [Res_local_w.tolist()]
        local_light_df = file_df[
            ["q", "dt", "P_imp_local", "S_local", "Res_local_w", "Res_local_g"]
        ]
        savepath = path_dir + "_extension_" + str(d)
        local_light_df.to_csv(savepath + "/" + file, sep="\t", index=False)

# fuse all df
for d in dists:
    list_of_df = []
    filenames = os.listdir(path_dir + "_extension_" + str(d))
    for file in filenames:
        file_df = pd.read_csv(path_dir + "_extension_" + str(d) + "/" + file, sep="\t")
        list_of_df.append(file_df)
    big_df = pd.concat(list_of_df)
    big_df.to_csv("all_sim_well_extension_" + str(d) + ".csv", sep="\t", index=False)
