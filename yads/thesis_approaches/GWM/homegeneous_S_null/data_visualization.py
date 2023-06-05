import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(
    "idc/idc_20.csv",
    sep="\t",
    converters={"Pb": literal_eval, "P_imp": literal_eval, "S": literal_eval},
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 16))

P_imp = np.array(df["P_imp"].loc[0])
S = np.array(df["S"].loc[0])
Nx = Ny = int(np.sqrt(len(P_imp)))

pos1 = ax1.imshow(P_imp.reshape(Nx, Ny).T, vmin=10e6, vmax=20e6)
plt.colorbar(pos1, ax=ax1)
pos2 = ax2.imshow(S.reshape(Nx, Ny).T, vmin=0.0, vmax=1.0)
plt.colorbar(pos2, ax=ax2)
ax1.invert_yaxis()
ax1.set_xticks([])
ax1.set_yticks([])

ax2.invert_yaxis()
ax2.set_xticks([])
ax2.set_yticks([])

plt.show()
