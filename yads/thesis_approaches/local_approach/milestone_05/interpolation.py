import copy

import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt


def full_reconstruction(S_global, S_local, well_cell_idx, dist, plot=False):
    S_local = S_local.clip(min=1e-14)
    S_pred = copy.deepcopy(S_global)

    S_pred[
        well_cell_idx - int((dist - 1) / 2) : well_cell_idx + int((dist - 1) / 2 + 1)
    ] = S_local
    # create polynomial left and right model based on S_local
    left_model, right_model = poly_model(S_local, well_cell_idx)
    # find the index of intersections between the polynomial models and the global initial saturation (0.)
    left_intersect, right_intersect, lp_S_local_rp = intersection(
        S_local, well_cell_idx, left_model, right_model, dist, len(S_pred)
    )
    # reconstruct the whole solution
    S_reconstructed = reconstruct(
        S_global, lp_S_local_rp, left_intersect, right_intersect
    )

    if plot:
        # Plot
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(24, 24))

        ax1.title.set_text(f"S initial")
        ax1.scatter(range(0, len(S_global)), S_global, s=5, c="b")

        ax2.title.set_text(f"S predicted")
        ax2.scatter(range(0, len(S_local)), S_local, s=5, c="g")

        ax3.title.set_text(f"First Saturation reconstruction")
        ax3.scatter(range(0, len(S_pred)), S_pred, s=5, c="b")
        ax3.scatter(
            range(
                well_cell_idx - int((dist - 1) / 2) - 1,
                well_cell_idx + int((dist - 1) / 2),
            ),
            S_local,
            s=5,
            c="g",
        )

        ax4.title.set_text(
            f"First Saturation reconstruction \n and Linear interpolation"
        )
        ax4.plot(S_global, c="b")

        ax4.scatter(range(0, len(lp_S_local_rp)), lp_S_local_rp, s=5, c="orange")
        ax4.scatter(
            range(
                well_cell_idx - int((dist - 1) / 2),
                well_cell_idx + int((dist - 1) / 2) + 1,
            ),
            S_local,
            s=5,
            c="g",
        )

        ax5.title.set_text(f"Final Saturation reconstructed")
        ax5.scatter(range(0, len(S_reconstructed)), S_reconstructed, s=5, c="r")
        return fig, S_reconstructed
    else:
        return S_reconstructed


def poly_model(S_local, well_pos, fit_d=None):
    d = int((len(S_local) - 1) / 2)

    left_global_domain = range(well_pos - (d + 1), well_pos)
    left_local_domain = range(0, d + 1)

    left_model = np.poly1d(
        np.polyfit(left_global_domain, S_local[left_local_domain], 1)
    )

    right_local_domain = range(d, len(S_local))
    right_global_domain = range(well_pos, well_pos + d + 1)

    right_model = np.poly1d(
        np.polyfit(right_global_domain, S_local[right_local_domain], 1)
    )
    return left_model, right_model


def intersection(S_local, well_cell_idx, left_model, right_model, d, nb_cells):
    d = int((d - 1) / 2)

    right_poly = right_model(range(well_cell_idx + d + 1, nb_cells))
    left_poly = left_model(range(0, well_cell_idx - d))
    lp_S_local_rp = np.concatenate([left_poly, S_local, right_poly])
    intersections = np.where(np.diff(np.sign(lp_S_local_rp)) != 0)[0] + 1
    left_intersect, right_intersect = intersections[0], intersections[-1] - 1
    return left_intersect, right_intersect, lp_S_local_rp


def reconstruct(S_global, global_view, left_intersect, right_intersect):
    S_left = S_global[0:left_intersect]
    S_right = S_global[right_intersect:-1]
    S_mid = global_view[left_intersect : right_intersect + 1]
    return np.concatenate([S_left, S_mid, S_right])


def main():
    test = pd.read_csv(
        "sci_pres/data/train_well_extension_10.csv",
        converters={
            "S_local": literal_eval,
            "P_imp_local": literal_eval,
            "Pb": literal_eval,
        },
        sep="\t",
        nrows=100,
    )

    S_global = np.full(200, 0.0)
    S_local = np.array(test["S_local"][12])

    well_cell_idx = 99

    dist = len(S_local)
    S_reconstructed = full_reconstruction(
        S_global=S_global,
        S_local=S_local,
        well_cell_idx=well_cell_idx,
        dist=dist,
        plot=False,
    )

    from matplotlib import rc

    rc("text", usetex=False)
    rc("font", **{"family": "serif", "size": 12})
    rc("figure", **{"figsize": (5, 3)})

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    init_cells = list(range(0, 82)) + list(range(114, 200))
    ax.scatter(
        init_cells,
        np.zeros_like(init_cells),
        s=10,
        zorder=2,
        label="Initial saturation",
    )

    ax.scatter(
        range(89, 89 + len(S_local)), S_local, s=10, zorder=1, label="Local prediction"
    )
    ax.scatter(
        range(0, len(S_reconstructed)),
        S_reconstructed,
        s=5,
        zorder=0,
        label="Linear interpolation",
    )
    ax.set_ylabel("Saturation")
    ax.legend()
    plt.savefig(
        "local_approach_test_case_1_extension_10_pred_interp_example.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()

    plt.show()
