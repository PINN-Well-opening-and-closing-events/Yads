import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

test = pd.read_csv("result_quantification_test_16_samples.csv", sep="\t")
test["improvement"] = test["nb_newton_classic"] - test["nb_newton_hybrid"]
print(test["improvement"].describe())

fig = plt.figure(figsize=(10, 10))
plt.scatter(
    x=test["nb_newton_classic"],
    y=test["nb_newton_hybrid"],
    color=(68 / 255, 114 / 255, 196 / 255),
)

lims = [
    np.min([plt.xlim(), plt.ylim()]),
    np.max([plt.xlim(), plt.ylim()]),
]
plt.plot(lims, lims, "k-", alpha=0.75, zorder=0)
plt.xlabel(
    r"Number of Newton iterations using" "\n" r"$\bf{Classic}$ $\bf{methodology}$"
)
plt.ylabel(
    r"Number of Newton iterations using" "\n" r"$\bf{Hybrid}$ $\bf{methodology}$"
)
# plt.legend(loc='upper right')
# plt.text(0.4, 0.90, f"Hybrid iterations > Classic iterations\n ({len(qt_index_df[qt_index_df['nb_newton_classic'] < qt_index_df['nb_newton_boosted']])} cases)",
#              horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes, fontsize=16)
# plt.text(0.87, 0.15, f"Hybrid iterations < Classic iterations\n ({len(qt_index_df[qt_index_df['nb_newton_classic'] >= qt_index_df['nb_newton_boosted']])} cases)",
#              horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes, fontsize=16)

plt.title(
    "Comparison of total number of Newton iterations on the test set using \nclassic and hybrid methodologies",
    fontsize=18,
)
fig.tight_layout()
plt.savefig("newton_comparison.png", dpi=500)
plt.show()
