import pandas as pd
import os

path_dir = "data/data_split/"
filenames = os.listdir(path_dir)
list_of_df = []
for file in filenames:
    file_df = pd.read_csv(path_dir + "/" + file, sep="\t")
    print(len(file_df))
    list_of_df.append(file_df)

big_df = pd.concat(list_of_df)

big_df.to_csv("data/train_q_5_3_dt_1_10_v2_extension_10.csv", sep="\t", index=False)
