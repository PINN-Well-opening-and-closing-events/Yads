import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from yads.thesis_approaches.GWM.Pressure_generator.P_generator import (
    P_generator_wrapper,
)
from yads.thesis_approaches.GWM.Pressure_generator.cov_mats import (
    cov_matrix_P_dist,
    cov_matrix_exp,
    cov_matrix_Id,
)
from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian
from yads.thesis_approaches.GWM.Pressure_generator.plots import plot_circle_P
from yads.thesis_approaches.GWM.Pressure_generator.P_interpolation import P_interp

if __name__ == "__main__":
    Lx, Ly = 3, 3
    Nx, Ny = 5, 5

    nb_samples = 2
    radius = Lx / 2
    nb_boundaries = 3

    P_min = 10e6
    P_max = 20e6
    grid = create_2d_cartesian(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)

    cor_ds = [3.0 / grid.nb_boundary_faces, 3]
    seed = 3

    P1, cart_coords, circle_coords = P_generator_wrapper(
        grid=grid,
        nb_boundaries=nb_boundaries,
        nb_samples=nb_samples,
        cov_mat=cov_matrix_Id,
        cor_ds=cor_ds,
        P_min=P_min,
        P_max=P_max,
        seed=seed,
    )

    P2, _, _ = P_generator_wrapper(
        grid=grid,
        nb_boundaries=nb_boundaries,
        nb_samples=nb_samples,
        cov_mat=cov_matrix_P_dist,
        cor_ds=cor_ds,
        P_min=P_min,
        P_max=P_max,
        seed=seed,
    )

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    plot_circle_P(
        grid=grid,
        P=P1[0, :],
        cart_coords=cart_coords,
        circle_coords=circle_coords,
        ax=ax1,
    )
    ax1.title.set_text(r"$cov(X_i, X_j) = \mathbf{I_n}  $")
    plot_circle_P(
        grid=grid,
        P=P2[0, :],
        cart_coords=cart_coords,
        circle_coords=circle_coords,
        ax=ax2,
    )
    ax2.title.set_text(
        r"$cov(Xi, Xj) = sign(P(X_j) - P(X_i)) "
        r"\exp(- (\frac{d(X_i, X_j)}{\Theta} + \frac{\mu}{|P(X_i) - P(X_j)|})) $"
    )

    (P_interp, P), (x_interp, x), (circle_interp, circle_no_interp) = P_interp(
        P1[0, :], circle_coords, grid=grid
    )

    ax4.scatter(x_interp, P_interp)
    ax4.scatter(x, P)

    rect = matplotlib.patches.Rectangle(
        (-Lx / 2, -Ly / 2), Lx, Ly, linewidth=1, edgecolor="black", facecolor="none"
    )

    circle = plt.Circle((0.0, 0.0), radius * np.sqrt(2), color="r", fill=False)
    ax3.add_patch(circle)
    ax3.add_patch(rect)
    ax3.scatter(circle_interp[:, 0], circle_interp[:, 1])
    ax3.scatter(circle_coords[:, 0], circle_coords[:, 1])

    for i, p in enumerate(P_interp):
        ax3.text(circle_interp[i, 0], circle_interp[i, 1], s=f"{p / 1e6:.1f}")

    for i, p in enumerate(P):
        ax3.text(circle_no_interp[i, 0], circle_no_interp[i, 1], s=f"{p / 1e6:.1f}")

    ax3.scatter(0, 0)
    ax3.set_xlim([-Lx, Lx])
    ax3.set_ylim([-Ly, Ly])

    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.show()
