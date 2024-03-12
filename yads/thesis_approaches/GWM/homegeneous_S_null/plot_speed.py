import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval

from matplotlib.cm import ScalarMappable

from yads.mesh.two_D import create_2d_cartesian


def visualize(grid, df):
    """
    visualization of the board
    2 plots, one for the cells index visualization and one for the faces index visualization
    :return:
    """

    nb_cells = int(np.sqrt(grid.nb_cells))
    P_imp = np.array(df["P_imp"].loc[0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    plt.rc("grid", linestyle="-", color="black")

    pos1 = axes[0][0].imshow(
        P_imp.reshape(nb_cells, nb_cells).T,
        vmin=10e6,
        vmax=20e6,
        extent=(0, nb_cells, nb_cells, 0),
    )
    plt.colorbar(pos1, ax=axes[0][0])
    # cell plot
    axes[0][0].set_xticks(np.arange(0, nb_cells + 1, 1))
    axes[0][0].set_yticks(np.arange(0, nb_cells + 1, 1))
    axes[0][0].set_xlim(left=0, right=nb_cells)
    axes[0][0].set_ylim(bottom=0, top=nb_cells)
    axes[0][0].grid(True)

    axes[0][0].title.set_text(r"$P_{IMP}$")

    axes[0][1].set_xticks(np.arange(0, nb_cells + 1, 1))
    axes[0][1].set_yticks(np.arange(0, nb_cells + 1, 1))
    axes[0][1].set_xlim(left=-0.25, right=nb_cells + 0.25)
    axes[0][1].set_ylim(bottom=-0.25, top=nb_cells + 0.25)
    axes[0][1].set_aspect("equal")
    plt.setp(axes[0][1], xticklabels=[], yticklabels=[])

    valeurs = np.array(df["F_init"].loc[0])
    colormap = plt.cm.get_cmap("viridis")
    Fg = valeurs[: grid.nb_faces]
    Fw = valeurs[grid.nb_faces : 2 * grid.nb_faces]
    sm = ScalarMappable(cmap=colormap)

    lines = get_grid_face_line_coords(grid)

    sm.set_array(np.log10(abs(Fw)))
    couleurs = sm.to_rgba(np.log10(abs(Fw)))
    plt.colorbar(sm, ax=axes[0][1])
    lc = matplotlib.collections.LineCollection(lines, colors=couleurs, linewidths=4)

    # axes[1].scatter(nb_cells/2, nb_cells/2)
    axes[0][1].add_collection(lc)

    axes[0][1].title.set_text(r"$F(P_{IMP}, S)=\lambda(S)T(K)\nabla P$")

    axes[1][0].set_xticks(np.arange(0, nb_cells + 1, 1))
    axes[1][0].set_yticks(np.arange(0, nb_cells + 1, 1))
    axes[1][0].set_xlim(left=-0.25, right=nb_cells + 0.25)
    axes[1][0].set_ylim(bottom=-0.25, top=nb_cells + 0.25)
    axes[1][0].set_aspect("equal")
    plt.setp(axes[1][0], xticklabels=[], yticklabels=[])

    grad_P = np.array(df["grad_P"].loc[0])
    colormap = plt.cm.get_cmap("viridis")
    sm = ScalarMappable(cmap=colormap)

    lines = get_grid_face_line_coords(grid)

    sm.set_array(np.log10(np.abs(grad_P)))
    couleurs = sm.to_rgba(np.log10(np.abs(grad_P)))
    plt.colorbar(sm, ax=axes[1][0])
    lc = matplotlib.collections.LineCollection(lines, colors=couleurs, linewidths=4)
    axes[1][0].add_collection(lc)
    axes[1][0].title.set_text(r"$T(K)\nabla P$")

    axes[1][1].set_xticks(np.arange(0, nb_cells + 1, 1))
    axes[1][1].set_yticks(np.arange(0, nb_cells + 1, 1))
    axes[1][1].set_xlim(left=-1.0, right=nb_cells + 1.0)
    axes[1][1].set_ylim(bottom=-1.0, top=nb_cells + 1.0)
    axes[1][1].set_aspect("equal")
    plt.setp(axes[1][1], xticklabels=[], yticklabels=[])

    pos1 = axes[1][1].imshow(
        P_imp.reshape(nb_cells, nb_cells).T,
        vmin=10e6,
        vmax=20e6,
        extent=(0, nb_cells, nb_cells, 0),
    )

    X_quiver = grid.centers(item="face")[:, 0] / Lx * nb_cells
    Y_quiver = grid.centers(item="face")[:, 1] / Ly * nb_cells
    U, V = get_quiver_arrows(grid, grad_P)
    axes[1][1].quiver(X_quiver, Y_quiver, U, V)
    axes[1][1].grid(True)
    plt.show()


def get_grid_face_line_coords(grid):
    lines = []
    L = np.max(grid.node_coordinates[:, 0])
    half_f_l = 1 / 2
    nb_cells = int(np.sqrt(grid.nb_cells))
    for f in range(grid.nb_faces):
        coord = nb_cells * grid.centers(item="face")[f] / L
        # vertical line
        if coord[1] % 1 == 0.0:
            f_line = [(coord[0] - half_f_l, coord[1]), (coord[0] + half_f_l, coord[1])]
            lines.append(f_line)

        # horizontal line
        if coord[0] % 1 == 0.0:
            f_line = [(coord[0], coord[1] - half_f_l), (coord[0], coord[1] + half_f_l)]
            lines.append(f_line)
    return lines


def get_quiver_arrows(grid, grad_P):
    U = np.empty(len(grad_P))
    V = np.empty((len(grad_P)))
    for f in range(grid.nb_faces):
        group = grid.group(f)[0]
        if group == "0":
            front, back = grid.face_to_cell(f, face_type="inner")
            # same x
            if (
                grid.centers(item="cell")[front][0]
                == grid.centers(item="cell")[back][0]
            ):
                U[f] = 0
                V[f] = np.sign(grad_P[f])
            # same y
            elif (
                grid.centers(item="cell")[front][1]
                == grid.centers(item="cell")[back][1]
            ):
                U[f] = np.sign(grad_P[f])
                V[f] = 0
        else:
            if grid.group(f)[0] in ["left", "right"]:
                U[f] = np.sign(grad_P[f])
                V[f] = 0
            else:
                U[f] = 0
                V[f] = np.sign(grad_P[f])
    return U, V


if __name__ == "__main__":
    df = pd.read_csv(
        "idc/idc_1.csv",
        sep="\t",
        converters={
            "Pb": literal_eval,
            "P_imp": literal_eval,
            "S": literal_eval,
            "F_init": literal_eval,
            "F_final": literal_eval,
            "grad_P": literal_eval,
        },
    )

    raw_df = pd.read_csv(
        "raw_idc/idc_0.csv",
        sep="\t",
        converters={
            "Pb": literal_eval,
            "P_imp": literal_eval,
            "S": literal_eval,
            "F_init": literal_eval,
            "F_final": literal_eval,
            "grad_P": literal_eval,
        },
    )

    Lx, Ly = 555, 555
    Nx, Ny = 11, 11
    grid = create_2d_cartesian(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
    visualize(grid=grid, df=df)
    visualize(grid=grid, df=raw_df)
