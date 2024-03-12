import pandas as pd
import os

df_list = []
folder_path = [
    "test_nb_samples_100_nb_boundaries_1_size_9_9.csv",
    "test_nb_samples_1000_nb_boundaries_2_size_9_9.csv",
    "test_nb_samples_1000_nb_boundaries_3_size_9_9.csv",
    "test_nb_samples_1000_nb_boundaries_4_size_9_9.csv",
]
# sample_dirs = os.listdir(folder_path)
#
# for d in sample_dirs:
#     rota_dirs = os.listdir(folder_path + "/" + d)
#     for r in rota_dirs:
#         df = pd.read_csv(folder_path + "/" + d + "/" + r, sep="\t")
#         df_list.append(df)

for df in folder_path:
    df_list.append(pd.read_csv(df, sep="\t"))
big_df = pd.concat(df_list)

# big_df.to_csv(folder_path + ".csv", sep="\t", index=False)
big_df.to_csv("gwm_dataset_3100.csv", sep="\t", index=False)
