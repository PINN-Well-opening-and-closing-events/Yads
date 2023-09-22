import pandas as pd
import os

path_dir = "data/case_1_q_5_5_dt_1_10_S_0_06"
filenames = os.listdir(path_dir)
list_of_df = []
for file in filenames:
    file_df = pd.read_csv(path_dir + "/" + file, sep="\t")
    list_of_df.append(file_df)

print(len(list_of_df))
big_df = pd.concat(list_of_df)

big_df.to_csv(path_dir + ".csv", sep="\t", index=False)
