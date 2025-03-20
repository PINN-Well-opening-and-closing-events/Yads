import pandas as pd
import os
import shutil
import re

for dir in ["ood", "train", "test", "validation"]:
    path_dir = f"data/{dir}"
    if not os.path.exists(path_dir):
        continue
    filenames = sorted(os.listdir(path_dir))
    filenames.sort(key=lambda x: tuple(map(int, re.findall(r'\d+', x)[1:])))

    list_of_df = []
    for file in filenames:
        file_df = pd.read_csv(path_dir + "/" + file, sep="\t")
        list_of_df.append(file_df)
    big_df = pd.concat(list_of_df)

    big_df.to_csv(path_dir + ".csv", sep="\t", index=False)
    # clean dir after fusing
    if os.path.isdir(path_dir):
        shutil.rmtree(path_dir)
