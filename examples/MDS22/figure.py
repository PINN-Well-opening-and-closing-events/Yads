import matplotlib.pyplot as plt
import json
from matplotlib import rc
import numpy as np

data_dict = json.load(open("discontinuity_figure_grad_P.json", "rb"))
print(data_dict.keys())
x = np.array(data_dict["metadata"]["grid data"]["cell_centers"])[:, 0]

dts = list(data_dict["simulation data"].keys())
P_cont = data_dict["simulation data"][dts[1]]["P"]
S_cont = data_dict["simulation data"][dts[1]]["S"]

P_disc = data_dict["simulation data"][dts[3]]["P"]
S_disc = data_dict["simulation data"][dts[3]]["S"]

fig = plt.figure(figsize=(12, 12))

ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

ax1.scatter(x, P_cont, s=10)
ax2.scatter(x, P_disc, s=10)
ax3.scatter(x, S_cont, s=10)
ax4.scatter(x, S_disc, s=10)

ax1.set_ylim([90e5, 200e5])
ax2.set_ylim([90e5, 200e5])
ax3.set_ylim([-0.1, 0.3])
ax4.set_ylim([-0.1, 0.3])

ax1.set_xlim([0.0, 10000])
ax2.set_xlim([0.0, 10000])
ax3.set_xlim([0.0, 10000])
ax4.set_xlim([0.0, 10000])

ax1.set_ylabel("P (MPa)", fontsize=16)
ax2.set_ylabel("P (MPa)", fontsize=16)
ax3.set_ylabel("S", fontsize=16)
ax4.set_ylabel("S", fontsize=16)

ax1.set_xlabel("x (meters)", fontsize=16)
ax2.set_xlabel("x (meters)", fontsize=16)
ax3.set_xlabel("x (meters)", fontsize=16)
ax4.set_xlabel("x (meters)", fontsize=16)

ax1.title.set_text(r"$\bf{Pressure}$ $\bf{before}$ $\bf{well}$ $\bf{opening}$")
ax1.title.set_size(16)
ax2.title.set_text(r"$\bf{Pressure}$ $\bf{after}$ $\bf{well}$ $\bf{opening}$")
ax2.title.set_size(16)
ax3.title.set_text(r"$\bf{Saturation}$ $\bf{before}$ $\bf{well}$ $\bf{opening}$")
ax3.title.set_size(16)
ax4.title.set_text(r"$\bf{Saturation}$ $\bf{after}$ $\bf{well}$ $\bf{opening}$")
ax4.title.set_size(16)

ax1.ticklabel_format(axis="y", scilimits=(5, 5))
ax2.ticklabel_format(axis="y", scilimits=(5, 5))

ax2.vlines(
    x=2975,
    ymin=0,
    ymax=200e5,
    colors="red",
    ls="--",
    lw=1,
    label="Well C02 injector",
    alpha=0.5,
)
ax4.vlines(
    x=2975,
    ymin=-0.1,
    ymax=1.1,
    colors="red",
    ls="--",
    lw=1,
    label="Well C02 injector",
    alpha=0.5,
)
# ax1.grid(True)
# ax2.grid(True)
# ax3.grid(True)
# ax4.grid(True)
# fig.suptitle(r"Example of $\bf{Pressure}$ and $\bf{Saturation}$ $\bf{discontinuities}$ induced by a $\bf{well}$ $\bf{event}$ in a 1D reservoir\n", fontsize=18)
fig.tight_layout()
ax4.legend(prop={"size": 17})
ax2.legend(prop={"size": 17})
plt.savefig("MDS_PS_discontinuity.png", dpi=2400)
plt.show()
