import pickle
import matplotlib.pyplot as plt

from matplotlib import rc

# rc("text", usetex=True)
rc("font", **{"family": "serif", "size": 12})
rc("figure", **{"figsize": (5, 3)})


newtons = pickle.load(
    open("physical_video/newton_lists/shp_teaser_newton_list_classic.pkl", "rb")
)

newtons_hybrid = pickle.load(
    open("physical_video/newton_lists/shp_teaser_newton_list_hybrid.pkl", "rb")
)


def draw_newton_plot(cts, savepath=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    start = 0
    end = 40

    plt.axvline(
        x=2,
        color="r",
        linestyle="dashed",
        label=r"$\bf{Well}$ $\bf{opening}$",
        zorder=0,
        linewidth=2,
    )
    plt.axvline(
        x=12,
        color="r",
        linestyle="dotted",
        label=r"$\bf{Well}$ $\bf{closing}$",
        zorder=0,
        linewidth=2,
    )
    plt.scatter(
        range(start + 1, end + 1),
        newtons[start:end],
        marker="x",
        s=100,
        linewidths=4,
        zorder=1,
        label=r"$\bf{Standard}$",
    )

    plt.scatter(
        range(start + 1, end + 1),
        newtons_hybrid[start:end],
        marker="x",
        s=100,
        linewidths=4,
        zorder=1,
        color="green",
        label=r"$\bf{Hybrid}$",
    )

    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    plt.xlabel(r"$\bf{Time-step}$ $\bf{number}$", fontsize=18)
    plt.ylabel(r"$\bf{Number}$ $\bf{of}$ $\bf{Newton}$ $\bf{iterations}$", fontsize=18)

    ax.set_ylim([-1, 42])
    xtick_loc = [0, 2, 5, 10, 12, 15, 20, 25, 30, 35, 40]
    ytick_loc = [0, 2, 4, 6, 8, 10, 20, 30, 40]
    ax.set_xticks(xtick_loc)
    ax.set_yticks(ytick_loc)

    plt.legend(loc="upper right", fontsize=20)
    plt.tight_layout()
    return fig


draw_newton_plot(0)
plt.savefig(f"article_hybrid_standard_example.pdf", bbox_inches="tight")
plt.close()
