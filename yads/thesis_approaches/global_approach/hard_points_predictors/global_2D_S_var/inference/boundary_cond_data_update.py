import pandas as pd
from ast import literal_eval

test_full = pd.read_csv(
    "data/test_q_5_3_dt_1_10_v2.csv", converters={"S0": literal_eval}, sep="\t", nrows=2
)

print(test_full.columns)

# find all first well events

init_events = test_full[test_full["well_event"] == 0]

S_boundary = [None] * len(test_full)

for i in range(len(test_full)):
    df_line = test_full.iloc[i]
    for j in range(len(init_events)):
        df_init_line = init_events.iloc[j]
        if int(df_line["simulation_number"]) == int(df_init_line["simulation_number"]):
            S_boundary[i] = [df_init_line["S0"][0]]

test_full["S_boundary"] = S_boundary

print(test_full.columns)
test_full.to_csv("data/better_test_q_5_3_dt_1_10_v2.csv", sep="\t", index=False)
