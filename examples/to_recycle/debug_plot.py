import matplotlib.pyplot as plt
import numpy as np
import json


def plot_P_S(json_path, figures_save_path):
    with open(json_path, "r") as f:
        states = json.load(f)
    sim_data = states["newton_step_data"]
    sim_metadata = states["metadata"]
    centers = sim_metadata["cell_centers"]

    # figure
    for step in sim_data.keys():
        fig = plt.figure(figsize=(12, 12))
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)

        ax1.scatter(centers, sim_data[step]["P_i_plus_1"], s=1)
        ax2.scatter(centers, sim_data[step]["S_i_plus_1"], s=1)
        ax3.scatter(centers, sim_data[step]["Residual"][len(centers) :], s=1)
        ax4.scatter(centers, sim_data[step]["Residual"][: len(centers)], s=1)

        R_p = np.linalg.norm(sim_data[step]["Residual"][len(centers) :], ord=2)
        R_s = np.linalg.norm(sim_data[step]["Residual"][: len(centers)], ord=2)
        ax1.title.set_text("Pressure through the domain")
        ax2.title.set_text("Saturation through the domain")
        ax3.title.set_text(f"Pressure residual through the domain\n norm={R_p:0.2E}")
        ax4.title.set_text(f"Water residual through the domain\n norm={R_s:0.2E}")

        ax2.set_ylim([-0.1, 1.1])

        ax1.set(ylabel="P", xlabel="x")
        ax2.set(ylabel="S", xlabel="x")
        ax3.set(ylabel="Pressure residual", xlabel="x")
        ax4.set(ylabel="Water residual", xlabel="x")

        plt.savefig(figures_save_path + "_" + str(step), dpi=150)
        plt.close(fig)
    return


if __name__ == "__main__":
    plot_P_S("./saves/debug/vanilla/vanilla_18.json", "./saves/vanilla_18")
    plot_P_S("./saves/debug/cheating_P/cheating_P_18.json", "./saves/cheating_P_18")
    plot_P_S("./saves/debug/cheating_S/cheating_S_18.json", "./saves/cheating_S_18")
    plot_P_S(
        "./saves/debug/cheating_P_and_S/cheating_P_and_S_18.json",
        "./saves/cheating_P_and_S_18",
    )
