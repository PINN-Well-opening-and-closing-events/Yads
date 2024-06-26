import pandas as pd
from ast import literal_eval
import numpy as np
import os

path_dir = "data/manuscrit"
filenames = os.listdir(path_dir)
print(len(filenames))
dists = [10]
Pb = {"left": 110.0e5, "right": 100.0e5}

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

    if P_imp_global.argmax() == well_loc:
        P_imp_global_with_bc = np.concatenate(
            [np.array([Pb["left"]]), P_imp_global, np.array([Pb["right"]])]
        )

        first_non_zero_S_x = (np.round(S_global, 6) != 0).argmax(axis=0)
        last_non_zero_S_x = len(S_global) - (
            np.flip(np.round(S_global, 6)) != 0
        ).argmax(axis=0)
        file_df["well_extension"] = last_non_zero_S_x - first_non_zero_S_x

        for d in dists:
            P_imp_local = P_imp_global[well_loc - d : well_loc + d + 1]
            S_local = S_global[well_loc - d : well_loc + d + 1]
            Res_local = Res_global[well_loc - d : well_loc + d + 1]
            file_df["P_imp_local"] = [P_imp_local.tolist()]
            file_df["S_local"] = [S_local.tolist()]
            file_df["Res_local"] = [Res_local.tolist()]
            local_light_df = file_df[
                [
                    "well_loc",
                    "q",
                    "dt",
                    "P_imp_local",
                    "S_local",
                    "well_extension",
                    "Res_local",
                ]
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
    print(len(big_df))
    big_df.to_csv("all_sim_well_extension_" + str(d) + ".csv", sep="\t", index=False)
