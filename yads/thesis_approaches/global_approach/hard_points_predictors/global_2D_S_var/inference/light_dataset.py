import pandas as pd
from ast import literal_eval

test_full = pd.read_csv("data/train_q_5_3_dt_1_10_v2.csv", sep="\t")

light_test = test_full[["q", "S0", "dt_init", "S_boundary", "P_imp"]]

print(light_test.columns)
light_test.to_csv("data/light_train_q_5_3_dt_1_10_v2.csv", sep="\t", index=False)
