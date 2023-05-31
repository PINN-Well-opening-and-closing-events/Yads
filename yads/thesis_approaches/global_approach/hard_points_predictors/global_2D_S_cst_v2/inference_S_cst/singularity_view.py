import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

test_df = pd.read_csv(
    "results/quantification_test_0_1_0.csv",
    sep="\t",
    converters={
        "norms_hybrid": literal_eval,
        "norms_classic": literal_eval,
        "norms_P_sol_S0": literal_eval,
        "norms_P_sol_S_pred": literal_eval,
    },
)


# 'Residuals_hybrid': literal_eval,
#                                             'Residuals_classic': literal_eval,
# i = 6
# fig, axs = plt.subplots(2, 2, figsize=(12, 12))
#
#
# # Res_g = np.log10(np.abs(np.array(test_df['Residuals_hybrid'][0][0][:95*60])))
# # Res_w = np.log10(np.abs(np.array(test_df['Residuals_hybrid'][0][0][95*60:])))
# print(np.argmax(np.abs(test_df['Residuals_hybrid'][0][i])))
# print(np.argmax(np.abs(test_df['Residuals_classic'][0][i])))
# print(test_df['Residuals_hybrid'][0][i][np.argmax(np.abs(test_df['Residuals_hybrid'][0][i]))])
# print(test_df['Residuals_classic'][0][i][np.argmax(np.abs(test_df['Residuals_classic'][0][i]))])
#
# print(test_df['norms_hybrid'][0]['L_inf'][i])
# print(test_df['norms_hybrid'][0]['L2'][i])
#
# print(test_df['norms_classic'][0]['L_inf'][i])
# print(test_df['norms_classic'][0]['L2'][i])

# Res_g = np.array(test_df['Residuals_hybrid'][0][i][:95*60])
# Res_w = np.array(test_df['Residuals_hybrid'][0][i][95*60:])
#
# im = axs[0][0].imshow(Res_g.reshape(95, 60).T)
# axs[0][1].imshow(Res_w.reshape(95, 60).T)
# axs[0][0].invert_yaxis()
# axs[0][1].invert_yaxis()
# axs[0][0].set_title(f'Hybrid gas residual at iteration {i}')
# axs[0][1].set_title(f'Hybrid water residual at iteration {i}')
#
# # Res_g = np.log10(np.abs(np.array(test_df['Residuals_classic'][0][0][:95*60])))
# # Res_w = np.log10(np.abs(np.array(test_df['Residuals_classic'][0][0][95*60:])))
#
# Res_g = np.array(test_df['Residuals_classic'][0][i][:95*60])
# Res_w = np.array(test_df['Residuals_classic'][0][i][95*60:])
#
# axs[1][0].imshow(Res_g.reshape(95, 60).T)
# axs[1][1].imshow(Res_w.reshape(95, 60).T)
# axs[1][0].invert_yaxis()
# axs[1][1].invert_yaxis()
# axs[1][0].set_title(f'Classic gas residual at iteration {i}')
# axs[1][1].set_title(f'Classic water residual at iteration {i}')
#
#
# axs[0][0].axis('off')
# axs[0][1].axis('off')
# axs[1][0].axis('off')
# axs[1][1].axis('off')
#
# fig.colorbar(im, ax=axs.ravel().tolist())
#
# plt.show()

# i, j = 0, 0
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#
# axs[0].scatter(range(0, len(test_df['norms_hybrid'][i]['L_inf'])), test_df['norms_hybrid'][i]['L_inf'], label='Hybrid norm')
# axs[0].scatter(range(0, len(test_df['norms_classic'][i]['L_inf'])), test_df['norms_classic'][i]['L_inf'], label='Classic norm')
# axs[0].axhline(1e-6, linewidth=1.0, linestyle='dashed', color='black', label='stop criterion')
# # axs[0].set_xticklabels(range(0, len(test_df['norms_classic'][i]['L_inf'])), fontsize=12)
# axs[0].set_title(rf'$L_\infty$ residual evolution through Newton iterations')
# axs[0].set_ylabel(r'$L_\infty$')
# axs[0].set_yscale('log')
#
# axs[1].scatter(range(0, len(test_df['norms_hybrid'][i]['L2'])), test_df['norms_hybrid'][i]['L2'], label='Hybrid norm')
# axs[1].scatter(range(0, len(test_df['norms_classic'][i]['L2'])), test_df['norms_classic'][i]['L2'], label='Classic norm')
# axs[1].set_title(rf'$L_{2}$ residual evolution through Newton iterations')
# axs[1].set_ylabel(r'$L_{2}$')
# axs[1].set_yscale('log')
#
# axs[0].legend()
# axs[1].legend()
#
#
# plt.show()
i = 0

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(
    range(0, len(test_df["norms_hybrid"][i]["L_inf"])),
    test_df["norms_hybrid"][i]["L_inf"],
    label=r"$\bf{R(P_{imp}, S_{pred})}$",
    marker="x",
    s=200,
)
ax.scatter(
    range(0, len(test_df["norms_classic"][i]["L_inf"])),
    test_df["norms_classic"][i]["L_inf"],
    label=r"$\bf{R(P_{imp}, S_{0})}$",
    marker="o",
    s=200,
)

# plt.scatter(range(0, len(test_df['norms_P_sol_S0'][i]['L_inf'])), test_df['norms_P_sol_S0'][i]['L_inf'],
#             label='P sol S0 norm',
#             marker='^')

ax.scatter(
    range(0, len(test_df["norms_P_sol_S_pred"][i]["L_inf"])),
    test_df["norms_P_sol_S_pred"][i]["L_inf"],
    label=r"$\bf{R(P_{sol}, S_{pred})}$",
    marker="s",
    s=200,
)
ax.patch.set_edgecolor("black")
ax.patch.set_linewidth("2")
ax.tick_params(
    axis="both", which="major", labelsize=20, width=2, length=10, labelcolor="black"
)

plt.axhline(
    1e-6, linewidth=1.0, linestyle="dashed", color="black", label=r"\bf{stop criterion}"
)
plt.title(rf"$L_\infty$ residual evolution through Newton iterations", fontsize=20)
plt.ylabel(r"$\mathbf{\Vert R(X) \Vert_{\infty}}$", fontsize=20)
plt.xlabel(r"$\mathbf{Newton}$" + "  " + "$\mathbf{iterations}$", fontsize=20)
plt.yscale("log")
plt.legend(fontsize=13)
plt.show()
