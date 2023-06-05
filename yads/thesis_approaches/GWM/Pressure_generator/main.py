import copy

import numpy as np
from matplotlib import pyplot as plt

from yads.numerics.physics import calculate_transmissivity
from yads.numerics.solvers import implicit_pressure_solver
from yads.thesis_approaches.GWM.Pressure_generator.P_generator import (
    P_generator_wrapper,
    P_imp_generator,
)
from yads.thesis_approaches.GWM.Pressure_generator.P_interp_to_P_imp import (
    create_Pb_groups,
)
from yads.thesis_approaches.GWM.Pressure_generator.cov_mats import (
    cov_matrix_P_dist,
    cov_matrix_Id,
)
from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian
from yads.thesis_approaches.GWM.Pressure_generator.plots import (
    plot_circle_P,
    plot_circle_interp_P,
    plot_P_imp,
)
from yads.thesis_approaches.GWM.Pressure_generator.P_interpolation import P_interp

if __name__ == "__main__":
    Lx, Ly = 5, 5
    Nx, Ny = 5, 5

    nb_samples = 100
    radius = Lx / 2
    nb_boundaries = 4

    P_min = 10e6
    P_max = 20e6
    grid = create_2d_cartesian(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)

    phi = 0.2

    # Porosity
    phi = np.full(grid.nb_cells, phi)
    # Diffusion coefficient (i.e Permeability)
    K = np.full(grid.nb_cells, 100.0e-15)

    T = calculate_transmissivity(grid, K)
    # gaz saturation initialization
    S = np.full(grid.nb_cells, 0.0)
    # Pressure initialization
    P0 = np.full(grid.nb_cells, 100.0e5)
    mu_w = 0.571e-3
    mu_g = 0.0285e-3

    kr_model = "quadratic"
    # BOUNDARY CONDITIONS #

    cor_ds = [3.0 / grid.nb_boundary_faces, 3]
    seed = 2

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
        cart_coords=cart_coords[0, :],
        circle_coords=circle_coords[0, :],
        ax=ax1,
    )
    ax1.title.set_text(r"$cov(X_i, X_j) = \mathbf{I_n}  $")
    plot_circle_P(
        grid=grid,
        P=P2[0, :],
        cart_coords=cart_coords[0, :],
        circle_coords=circle_coords[0, :],
        ax=ax2,
    )
    ax2.title.set_text(
        r"$cov(Xi, Xj) = sign(P(X_j) - P(X_i)) "
        r"\exp(- (\frac{d(X_i, X_j)}{\Theta} + \frac{\mu}{|P(X_i) - P(X_j)|})) $"
    )

    (P_inter, P_no_inter), (x_interp, x), (circle_interp, circle_no_interp) = P_interp(
        P2[0, :], circle_coords[0, :], grid=grid
    )

    plot_circle_interp_P(
        grid, P_inter, P_no_inter, circle_interp, circle_no_interp, ax=ax3
    )

    ax3.title.set_text(r"Linear interpolation")

    all_P = np.concatenate([P_inter, P_no_inter[:-1]])
    all_circles_coords = np.concatenate([circle_interp, circle_no_interp[:-1]])
    groups, Pb_dict = create_Pb_groups(grid, all_P, all_circles_coords)
    for group in groups:
        grid.add_face_group_by_line(*group)
    # Saturation
    Sb_d = copy.deepcopy(Pb_dict)
    Sb_n = copy.deepcopy(Pb_dict)
    for group in Sb_d.keys():
        Sb_d[group] = 0.0
        Sb_n[group] = None
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}
    P_imp = implicit_pressure_solver(
        grid=grid,
        K=K,
        T=T,
        P=P0,
        S=S,
        Pb=Pb_dict,
        Sb_dict=Sb_dict,
        mu_g=mu_g,
        mu_w=mu_w,
    )
    plot_P_imp(grid, P_imp, ax4, Pmax=P_max, Pmin=P_min)
    ax4.title.set_text(r"$P_{IMP}$")
    plt.show()

    grid = create_2d_cartesian(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)

    savepath = f"nb_samples_{nb_samples}_nb_boundaries_{nb_boundaries}"
    P_imp_generator(
        grid,
        nb_samples,
        nb_boundaries,
        P_min,
        P_max,
        cov_matrix_P_dist,
        cor_ds,
        seed,
        savepath,
    )
