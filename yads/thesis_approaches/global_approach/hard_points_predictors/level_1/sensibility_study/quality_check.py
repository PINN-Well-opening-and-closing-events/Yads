import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt

# df_02 = pd.read_csv("debug_path/sensibility_study_train_S_02.csv")
# df_03 = pd.read_csv("debug_path/sensibility_study_train_S_03.csv")
df_04 = pd.read_csv("debug_path/sensibility_study_train_S_04.csv")
# iters = df_04['i'].value_counts()
# del df_04

# df_true = pd.read_csv("debug_path/sensibility_study_train_S_true.csv")
# iters_true = df_true['i'].value_counts()
# del df_true
# df_05 = pd.read_csv("debug_path/sensibility_study_train_S_05.csv")
# iters = df_05['i'].value_counts()
# del df_05


def create_sample(i, list_of_df):
    sample = pd.DataFrame()
    for j, df in enumerate(list_of_df):
        i_sample = df[df["i"] == i]
        i_sample["name"] = j
        sample = pd.concat([sample, i_sample])
    return sample


# list_of_df = [df_05]

# improved cases
# improved_count = 0
# same_count = 0
# worth_count = 0
# newton_inf_100_count = 0
# newt_inf_100_sol_sup_100 = 0
# for i in iters.index:
#     if iters[i] < iters_true[i]:
#         improved_count += 1
#     elif iters[i] == iters_true[i]:
#         same_count += 1
#     elif iters[i] > iters_true[i]:
#         worth_count += 1
#     if iters[i] <= 100:
#         newton_inf_100_count += 1
#     if iters[i] <= 100 < iters_true[i]:
#         newt_inf_100_sol_sup_100 += 1
#         print(i)
# print(f"improved count {improved_count}")
# print(f"same count {same_count}")
# print(f"worth count {worth_count}")
# print(f"newton_inf_100_count {newton_inf_100_count}")
# print(f"newton_inf_100_sol_sup_100 {newt_inf_100_sol_sup_100}")

sample_0 = create_sample(250, [df_04])

for i, idx in enumerate(sample_0.index):
    if i % 10 == 0 and i <= 100:
        plt.plot(literal_eval(df_04["S_i_plus_1"][idx]), label=f"Iteration number {i}")
    # plt.plot(literal_eval(df_true['S_i_plus_1'][idx]), label="True saturation")
plt.ylabel("Saturation")
plt.title("True Saturation")
# plt.ylim(ymin=-0.2, ymax=1.1)
plt.legend()
plt.show()
