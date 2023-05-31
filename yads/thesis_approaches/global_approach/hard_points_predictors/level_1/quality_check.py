import pandas as pd
from ast import literal_eval
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv(
    "data/5000_data_10_100_newton.csv",
    converters={"S": literal_eval, "P": literal_eval},
)

hard_df = pd.read_csv(
    "data/5000_data_10_100_newton_hard.csv",
    converters={"S": literal_eval, "P": literal_eval},
)

giga_df = pd.read_csv(
    "data/5000_data_10_100_newton_giga_hard.csv",
    converters={"S": literal_eval, "P": literal_eval},
)

print(df.columns)
print(hard_df.columns)
print(giga_df.columns)
#
# print(f"sizes: all {len(df)}, hard {len(hard_df)}, giga_hard {len(giga_df)}")
# df_sup_10 = df[df['nb_newton'] > 10]
# print(f"all newton > 10: {len(df_sup_10)}")
#
# all_hard_df = pd.concat([hard_df, giga_df])
# print(len(all_hard_df))
# print(len(np.intersect1d(df['q'], all_hard_df['q'])))
# print(f"number of q in common between all newton > 10 and hard + giga_hard {len(np.intersect1d(df_sup_10['q'], all_hard_df['q']))}")
# print(f"number of dt_init in common between all newton > 10 and hard + giga_hard {len(np.intersect1d(df_sup_10['dt_init'], all_hard_df['dt_init']))}")
# # print(len(np.intersect1d(df_sup_10['dt_init'], all_hard_df['dt_init'])))
# print(f"number of total_time in common between all newton > 10 and hard + giga_hard {len(np.intersect1d(df_sup_10['dt_init'], all_hard_df['total_time']))}")
#
# # print(len(np.intersect1d(df_sup_10['dt_init'], all_hard_df['total_time'])))
# # print(hard_df['dt'], hard_df['dt_init'], hard_df['total_time'])
# print(hard_df['dt_init'].equals(hard_df['dt']))

print(len(df.index), len(hard_df.index), len(giga_df.index))
# all_hard_df = pd.concat([giga_df, hard_df])
df["log_q"] = np.log10(-df["q"])
df_sup_10 = df[df["nb_newton"] > 10]
print(len(df_sup_10))

# print(df_sup_10['dt_init'], all_hard_df['dt_init'])
# print(df_sup_10['dt_init'].values.sort() == all_hard_df['dt_init'].values.sort())
# print(df_sup_10['q'].values.sort() == all_hard_df['q'].values.sort())
# i = 0
# j = 0
# k = 0
# myst_qt = []
# myst_qt_rm = hard_df[['q', 'dt_init']].values.tolist()
# print(len(df), len(hard_df))
# for qt in df[['q', 'dt_init']].values:
#     if myst_qt_rm:
#         if qt[0] in list(list(zip(*myst_qt_rm))[0]):
#             j = list(list(zip(*myst_qt_rm))[0]).index(qt[0])
#             if qt[1] in list(list(zip(*myst_qt_rm))[1]):
#                 i = list(list(zip(*myst_qt_rm))[1]).index(qt[1])
#                 if i == j:
#                     del myst_qt_rm[j]
#                     myst_qt.append(qt)
#
#
# print(len(myst_qt), len(myst_qt_rm))
# myst_qt = np.array(myst_qt)
# myst_qt_rm = np.array(myst_qt_rm)

# miss_qt = []
# for qt in df_sup_10[['q', 'dt_init']].values:
#     if qt in all_hard_df[['q', 'dt_init']].values:
#         miss_qt.append(qt)
# print(len(miss_qt))

# for qt in df_sup_10[['q', 'dt_init']].values:


# all_hard_df['log_q'] = np.log10(-all_hard_df['q'])
hard_df["log_q"] = np.log10(-hard_df["q"])
giga_df["log_q"] = np.log10(-giga_df["q"])

fig = plt.figure(figsize=(8, 8))

# plt.scatter(df['log_q'], df['dt_init'], label=f"initial dt({len(df)})")
plt.scatter(
    df_sup_10["log_q"],
    df_sup_10["dt_init"],
    label=f"Newton > 10 initial dt({len(df_sup_10)})",
)
plt.scatter(hard_df["log_q"], hard_df["dt_init"], label=f"hard ({len(hard_df)})")
plt.scatter(giga_df["log_q"], giga_df["dt_init"], label=f"giga ({len(giga_df)})")
# plt.scatter(np.log10(-myst_qt[:, 0]), myst_qt[:, 1], label=f"Newton > 10 initial dt({len(df_sup_10)})")
# plt.scatter(np.log10(-myst_qt_rm[:, 0]), myst_qt_rm[:, 1], label=f"mystery points ({len(df_sup_10)})")

plt.xlabel("$log_{10}(q)$")
plt.ylabel("t")
plt.legend()
plt.show()

# print(all_hard_df['log_q'].values.sort() == df_sup_10)
# df.hist('nb_newton')
# plt.show()
