import pandas as pd
import os


all_df = []

folder_path = [
    "test_nb_samples_100_nb_boundaries_1_size_9_9",
    "test_nb_samples_1000_nb_boundaries_2_size_9_9",
    "test_nb_samples_1000_nb_boundaries_3_size_9_9",
    "test_nb_samples_1000_nb_boundaries_4_size_9_9",
]

for d in folder_path:
    df_list = []
    rota_dirs = os.listdir(d)
    for r in rota_dirs:
        data_file = os.listdir(d + "/" + r)
        for data in data_file:
            df = pd.read_csv(d + "/" + r + "/" + data, sep="\t")
            df_list.append(df)
    big_single_df = pd.concat(df_list)
    print(len(big_single_df))
    big_single_df.to_csv(d + ".csv", sep="\t", index=False)
    all_df.append(big_single_df)

big_df = pd.concat(all_df)
big_df.to_csv("gwm_dataset_3100.csv", sep="\t", index=False)
print(len(big_df))
