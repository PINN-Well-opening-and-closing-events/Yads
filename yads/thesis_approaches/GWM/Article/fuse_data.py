import pandas as pd
import os

df_list = []
folder_path = "test_nb_samples_3000_nb_boundaries_3"
sample_dirs = os.listdir(folder_path)

for d in sample_dirs:
    rota_dirs = os.listdir(folder_path + "/" + d)
    for r in rota_dirs:
        df = pd.read_csv(folder_path + "/" + d + "/" + r, sep="\t")
        df_list.append(df)

big_df = pd.concat(df_list)

big_df.to_csv(folder_path + ".csv", sep="\t", index=False)

