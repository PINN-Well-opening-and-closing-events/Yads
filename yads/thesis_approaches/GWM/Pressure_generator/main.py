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
from yads.thesis_approaches.GWM.Pressure_generator.plots import (
    plot_circle_P,
    plot_circle_interp_P,
)
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
    seed = 5

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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
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

    (P_interp, P), (x_interp, x), (circle_interp, circle_no_interp), _ = P_interp(
        P1[0, :], circle_coords, grid=grid
    )

    plot_circle_interp_P(grid, P_interp, P, circle_interp, circle_no_interp, ax=ax3)
    plt.show()
