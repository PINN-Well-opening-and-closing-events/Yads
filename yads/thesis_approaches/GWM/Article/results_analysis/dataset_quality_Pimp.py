from scipy.stats import wasserstein_distance
import pandas as pd
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append("/home/AD.NORCERESEARCH.NO/anle/Yads")
import yads.mesh as ym
from yads.mesh.utils import load_json


def compute_distance(sample_1, sample_2):
    # normalize
    sample_1 = (sample_1 - sample_1.min()) / (sample_1.max() - sample_1.min())
    sample_2 = (sample_2 - sample_2.min()) / (sample_2.max() - sample_2.min())
    return wasserstein_distance(sample_1, sample_2)


def compute_dataset_quality(physical_df, synthetic_df):
    physical_data = np.array(
        [np.array(row["P_imp_local"]) for _, row in physical_df.iterrows()]
    )
    synthetic_data = np.array(
        [np.array(row["P_imp"]) for _, row in synthetic_df.iterrows()]
    )

    score = np.full(len(synthetic_data), np.inf)

    for i, synthetic_row in enumerate(synthetic_data):
        distances = np.apply_along_axis(
            compute_distance, 1, physical_data, synthetic_row
        )
        score[i] = np.min(distances)
    return np.max(score)


if __name__ == "__main__":
    synthetic_df = pd.read_csv(
        "../data/train_gwm_dataset_3100.csv",
        sep="\t",
        converters={"P_imp": literal_eval},
        nrows=10,
    )

    physical_df = pd.read_csv(
        "../../../local_approach/SHPCO2/data/case_0/data/train_q_5_3_dt_1_10_S_0_06_P_imp_extension_4.csv",
        converters={"P_imp_local": literal_eval},
        sep="\t",
        nrows=10,
    )

    physical_variable_df = pd.read_csv(
        "data/results_variable_2.csv",
        sep="\t",
        converters={"P_imp": literal_eval, "well_loc": literal_eval},
        nrows=10,
    )

    grid = load_json("../../../../../meshes/SHP_CO2/2D/SHP_CO2_2D_S.json")

    d = 4
    grid_dxy = 50
    well_locs = physical_variable_df["well_loc"]
    Pimp_local_df = []
    for i, well_loc in enumerate(well_locs):
        well_x, well_y = well_loc[0], well_loc[1]
        cells_d = grid.find_cells_inside_square(
            (grid_dxy * (well_x / grid_dxy - d), grid_dxy * (well_y / grid_dxy + d)),
            (grid_dxy * (well_x / grid_dxy + d), grid_dxy * (well_y / grid_dxy - d)),
        )
        Pimp = np.array(physical_variable_df["P_imp"].loc[i])
        Pimp_local = Pimp[cells_d]
        Pimp_local_df.append(Pimp_local)

    physical_variable_df["P_imp_local"] = Pimp_local_df

    print(compute_dataset_quality(physical_df, synthetic_df))
    print(compute_dataset_quality(physical_variable_df, synthetic_df))
    print(compute_dataset_quality(physical_variable_df, synthetic_df))