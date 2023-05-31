import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval

test = pd.read_csv(
    "data/q_4_25_dt_01_1/all_sim_1000_1_0_1000_4.csv",
    sep="\t",
    converters={"S": literal_eval, "P": literal_eval},
)

P_plot = np.array(test["P"].loc[0]) / 10 ** 6
S_plot = np.array(test["S"].loc[0])

first_non_zero_S_x = (np.round(S_plot, 6) != 0).argmax(axis=0)
last_non_zero_S_x = len(S_plot) - (np.flip(np.round(S_plot, 6)) != 0).argmax(axis=0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

ax2.axvline(x=first_non_zero_S_x, ls="--", c="r")
ax2.axvline(x=last_non_zero_S_x, ls="--", c="r")
ax1.scatter(range(len(P_plot)), P_plot, s=6)
ax2.scatter(range(len(S_plot)), S_plot, s=6)

ax1.set(
    xlabel="x",
    ylabel="P (MPa)",
    title=f'q: {test["q"].values[0]}, '
    f'dt {test["dt"].values[0]}, '
    f'nb_newton: {test["nb_newton"].values[0]}',
)
ax2.set(
    xlabel="x",
    ylabel="S",
    title=f"well extension: {last_non_zero_S_x - first_non_zero_S_x} cells",
)
plt.show()
