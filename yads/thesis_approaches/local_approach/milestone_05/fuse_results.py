import pandas as pd
import os

path_dir = "sci_pres/results_full_linear_interp/train_interp_mid_well_extension_10"
filenames = os.listdir(path_dir)
list_of_df = []
for file in filenames:
    file_df = pd.read_csv(path_dir + "/" + file, sep="\t")
    list_of_df.append(file_df)

big_df = pd.concat(list_of_df)

big_df.to_csv(path_dir + ".csv", sep="\t", index=False)
