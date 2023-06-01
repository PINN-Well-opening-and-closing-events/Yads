from matplotlib import pyplot as plt

from yads.thesis_approaches.GWM.Pressure_generator.P_generator import P_generator_wrapper
from yads.thesis_approaches.GWM.Pressure_generator.cov_mats import cov_matrix_P_dist, cov_matrix_exp, cov_matrix_Id
from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian
from yads.thesis_approaches.GWM.Pressure_generator.plots import plot_circle_P
from yads.thesis_approaches.GWM.Pressure_generator.P_interpolation import P_interp

if __name__ == "__main__":
    Lx, Ly = 3, 3
    Nx, Ny = 5, 5

    nb_samples = 2
    radius = Lx / 2
    nb_boundaries = 4

    P_min = 10e6
    P_max = 20e6
    grid = create_2d_cartesian(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)

    cor_ds = [3./grid.nb_boundary_faces, 3]
    seed = 5
    P1, cart_coords, circle_coords = P_generator_wrapper(grid=grid,
                                                         nb_boundaries=nb_boundaries,
                                                         nb_samples=nb_samples,
                                                         cov_mat=cov_matrix_Id,
                                                         cor_ds=cor_ds,
                                                         P_min=P_min,
                                                         P_max=P_max,
                                                         seed=seed)

    P2, _, _ = P_generator_wrapper(grid=grid,
                                   nb_boundaries=nb_boundaries,
                                   nb_samples=nb_samples,
                                   cov_mat=cov_matrix_P_dist,
                                   cor_ds=cor_ds,
                                   P_min=P_min,
                                   P_max=P_max,
                                   seed=seed)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # plot_circle_P(grid=grid,
    #               P=P1[0, :],
    #               cart_coords=cart_coords,
    #               circle_coords=circle_coords,
    #               ax=ax1)
    # ax1.title.set_text(r'$cov(X_i, X_j) = \mathbf{I_n}  $')
    # plot_circle_P(grid=grid,
    #               P=P2[0, :],
    #               cart_coords=cart_coords,
    #               circle_coords=circle_coords,
    #               ax=ax2)
    # ax2.title.set_text(r'$cov(Xi, Xj) = sign(P(X_j) - P(X_i)) '
    #                    r'\exp(- (\frac{d(X_i, X_j)}{\Theta} + \frac{\mu}{|P(X_i) - P(X_j)|})) $')
    # plt.show()

    P_linear, x_linear = P_interp(P1[0, :], circle_coords, grid=grid)
    print(len(P_linear), len(x_linear))
    fig = plt.subplots(1, 1, figsize=(6, 6))
    plt.scatter(x_linear, P_linear)
    plt.show()
