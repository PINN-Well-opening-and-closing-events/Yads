from scipy.stats import wasserstein_distance
import pandas as pd
from ast import literal_eval
import numpy as np
from mpi4py import MPI


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


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nb_proc = comm.Get_size()


if rank == 0:
    # Setup code
    synthetic_df = pd.read_csv(
        "../data/train_gwm_dataset_3100.csv",
        sep="\t",
        converters={"P_imp": literal_eval},
        nrows=10,
    )

    synthetic_split = np.array_split(synthetic_df, nb_proc)

elif rank == 1:
    physical_df = pd.read_csv(
        "../../../local_approach/SHPCO2/data/case_0/data/train_q_5_3_dt_1_10_S_0_06_P_imp_extension_4.csv",
        converters={"P_imp_local": literal_eval},
        sep="\t",
        nrows=10,
    )
    physical_split = np.array_split(physical_df, nb_proc)
else:
    physical_split = None
    synthetic_split = None

physical_df = comm.scatter(physical_split, root=1)
synthetic_df = comm.scatter(synthetic_split, root=0)


results_split = compute_dataset_quality(physical_df, synthetic_df)

results = comm.gather(results_split, root=0)

print(results)
