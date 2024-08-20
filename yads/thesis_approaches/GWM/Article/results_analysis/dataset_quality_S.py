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
    sample_1 = (sample_1 - np.min(sample_1)) / (max(sample_1) - min(sample_1))
    sample_2 = (sample_2 - np.min(sample_2)) / (max(sample_2) - min(sample_2))
    return wasserstein_distance(sample_1, sample_2)


def compute_dataset_quality(df, col1, col2):
    score = df.apply(lambda x: compute_distance(x[col1], x[col2]), axis=1)
    return np.max(score), np.mean(score)




if __name__ == '__main__':
    physical_df = pd.read_csv(
                'data/results_train.csv',
                converters={"GWM_S_predict_local": literal_eval, "Local_S_predict_local": literal_eval, "S_i_plus_1_classic": literal_eval, 'S_DD_local':literal_eval},
                sep="\t", nrows=100
            )

    physical_variable_df = pd.read_csv('data/results_variable_2.csv', sep='\t', 
                      converters= {'GWM_S_predict_local': literal_eval, 'Local_S_predict_local': literal_eval, "S_i_plus_1_classic": literal_eval, 'S_DD_local':literal_eval,
                                   'well_loc':literal_eval},
                       nrows=100)
    
    print(physical_variable_df.columns)
    grid = load_json("../../../../../meshes/SHP_CO2/2D/SHP_CO2_2D_S.json")


    d = 4
    grid_dxy = 50
    well_locs = physical_variable_df['well_loc']
    S_true_local_df = []
    for i, well_loc in enumerate(well_locs):
        well_x, well_y = well_loc[0], well_loc[1]
        cells_d = grid.find_cells_inside_square(
        (grid_dxy * (well_x / grid_dxy - d), grid_dxy * (well_y / grid_dxy + d)),
        (grid_dxy * (well_x / grid_dxy + d), grid_dxy * (well_y / grid_dxy - d)),
        )
        S_true = np.array(physical_variable_df['S_i_plus_1_classic'].loc[i])
        S_true_local = S_true[cells_d]
        S_true_local_df.append(S_true_local)

    physical_variable_df['S_true_local'] = S_true_local_df

    print(compute_dataset_quality(physical_variable_df, 'GWM_S_predict_local', 'S_true_local'))
    print(compute_dataset_quality(physical_variable_df, 'Local_S_predict_local', 'S_true_local'))
    print(compute_dataset_quality(physical_variable_df, 'S_DD_local', 'S_true_local'))
    

    well_x, well_y = 1475, 2225
    S_true_local_df = []

    cells_d = grid.find_cells_inside_square(
    (grid_dxy * (well_x / grid_dxy - d), grid_dxy * (well_y / grid_dxy + d)),
    (grid_dxy * (well_x / grid_dxy + d), grid_dxy * (well_y / grid_dxy - d)),
    )
    for i in range(len(physical_df)):
        S_true = np.array(physical_df['S_i_plus_1_classic'].loc[i])
        S_true_local = S_true[cells_d]
        S_true_local_df.append(S_true_local)

    physical_df['S_true_local'] = S_true_local_df
    
    print(compute_dataset_quality(physical_df, 'GWM_S_predict_local', 'S_true_local'))
    print(compute_dataset_quality(physical_df, 'Local_S_predict_local', 'S_true_local'))
    print(compute_dataset_quality(physical_df, 'S_DD_local', 'S_true_local'))
