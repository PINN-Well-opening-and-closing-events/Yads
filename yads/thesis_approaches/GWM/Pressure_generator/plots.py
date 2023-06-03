import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def plot_circle_P(grid, P, cart_coords, circle_coords, ax):
    nb_bd_faces = grid.nb_boundary_faces
    Nx = Ny = nb_bd_faces / 4
    Lx = Ly = grid.measures(item="face")[0] * Nx
    radius = Lx / 2

    # get center of boundary faces
    face_center_coords = []
    for group in grid.face_groups:
        if group in ["left", "right", "lower", "upper"]:
            cell_idxs = grid.face_groups[group][:, 0]
            for coord in grid.centers(item="face")[cell_idxs]:
                face_center_coords.append(list(coord))
    rect = matplotlib.patches.Rectangle(
        (-Lx / 2, -Ly / 2), Lx, Ly, linewidth=1, edgecolor="black", facecolor="none"
    )
    circle = plt.Circle((0.0, 0.0), radius * np.sqrt(2), color="r", fill=False)
    ax.add_patch(circle)
    ax.add_patch(rect)

    ax.scatter(
        x=np.array(face_center_coords)[:, 0] - Lx / 2,
        y=np.array(face_center_coords)[:, 1] - Ly / 2,
    )
    x_cart_coords = np.array([coords[0] for coords in cart_coords])
    y_cart_coords = np.array([coords[1] for coords in cart_coords])

    ax.scatter(x=x_cart_coords - Lx / 2, y=y_cart_coords - Ly / 2)

    x_circle_coords = np.array([coords[0] for coords in circle_coords])
    y_circle_coords = np.array([coords[1] for coords in circle_coords])
    ax.scatter(x_circle_coords, y_circle_coords)

    for i, p in enumerate(P):
        ax.text(x_circle_coords[i], y_circle_coords[i], s=f"{p / 1e6:.1f}")

    ax.scatter(0, 0)
    ax.set_xlim([-Lx, Lx])
    ax.set_ylim([-Ly, Ly])

    ax.set_xticks([])
    ax.set_yticks([])
    return


def plot_circle_interp_P(grid, P_interp, P, x_interp, x, ax):
    nb_bd_faces = grid.nb_boundary_faces
    Nx = Ny = nb_bd_faces / 4
    Lx = Ly = grid.measures(item="face")[0] * Nx
    radius = Lx / 2

    # get center of boundary faces
    face_center_coords = []
    for group in grid.face_groups:
        if group != "0":
            cell_idxs = grid.face_groups[group][:, 0]
            for coord in grid.centers(item="face")[cell_idxs]:
                face_center_coords.append(list(coord))

    rect = matplotlib.patches.Rectangle(
        (-Lx / 2, -Ly / 2), Lx, Ly, linewidth=1, edgecolor="black", facecolor="none"
    )

    circle = plt.Circle((0.0, 0.0), radius * np.sqrt(2), color="r", fill=False)
    ax.add_patch(circle)
    ax.add_patch(rect)
    ax.scatter(x_interp[:, 0], x_interp[:, 1])
    ax.scatter(x[:, 0], x[:, 1])

    for i, p in enumerate(P_interp):
        ax.text(x_interp[i, 0], x_interp[i, 1], s=f"{p / 1e6:.1f}")

    for i, p in enumerate(P):
        ax.text(x[i, 0], x[i, 1], s=f"{p / 1e6:.1f}")

    ax.scatter(0, 0)
    ax.set_xlim([-Lx, Lx])
    ax.set_ylim([-Ly, Ly])

    ax.set_xticks([])
    ax.set_yticks([])
    return


def plot_P_imp(grid, P_imp, ax, Pmax, Pmin):
    Nx = Ny = int(np.sqrt(grid.nb_cells))
    pos = ax.imshow(P_imp.reshape(Nx, Ny).T, vmax=Pmax, vmin=Pmin)
    plt.colorbar(pos, ax=ax)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    return
