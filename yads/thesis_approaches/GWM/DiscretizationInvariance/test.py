from models.FNO import FNO2d, UnitGaussianNormalizer
import torch
import pickle
from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian
import numpy as np
from yads.numerics.physics import calculate_transmissivity
from yads.numerics.solvers import solss_newton_step, implicit_pressure_solver
from yads.wells import Well
import matplotlib.pyplot as plt
import matplotlib

S_model = model = FNO2d(modes1=12, modes2=12, width=64, n_features=4)
S_model.load_state_dict(
    torch.load(
        "models/GWM_3100_checkpoint_2500.pt",
        map_location=torch.device("cpu"),
    )["model"]
)

GWM_q_normalizer = pickle.load(open("models/GWM_q_normalizer.pkl", "rb"))
GWM_P_imp_normalizer = pickle.load(open("models/GWM_P_imp_normalizer.pkl", "rb"))
GWM_dt_normalizer = pickle.load(open("models/GWM_dt_normalizer.pkl", "rb"))

#  grid_discr = [9,15, 25]
grid_discr = [21]

def place_image_at_center(larger_image, smaller_image):
    # Calculate the center coordinates of the larger image
    n = larger_image.shape[0]
    center_row = n // 2
    center_col = n // 2

    # Calculate the starting and ending indices to place the smaller image
    start_row = center_row - smaller_image.shape[0] // 2
    end_row = start_row + smaller_image.shape[0]
    start_col = center_col - smaller_image.shape[1] // 2
    end_col = start_col + smaller_image.shape[1]

    # Place the smaller image at the calculated position in the larger image
    result_image = np.copy(larger_image)
    result_image[start_row:end_row, start_col:end_col] = smaller_image

    return result_image


def apply_irregular_normalizer(irr_data, normalizer, shape):
    
    image_size = shape
    sub_image_size = 9
    result_image = np.zeros((image_size, image_size))
    mask = np.zeros((image_size, image_size))
    reg_data = irr_data[:, :, 0].numpy()
    for i in range(image_size - sub_image_size + 1):
        for j in range(image_size - sub_image_size + 1):
            # Extract sub-image
            sub_image = reg_data[i:i+sub_image_size, j:j+sub_image_size]
            
            # Apply your operator on the sub-image
            processed_sub_image = np.array(normalizer.encode(torch.from_numpy(np.reshape(sub_image, (9, 9, 1)))))  
            processed_sub_image = np.reshape(processed_sub_image, (9, 9))
            # Place the processed sub-image back into the result image
            result_image[i:i+sub_image_size, j:j+sub_image_size] += processed_sub_image
            mask[i:i+sub_image_size, j:j+sub_image_size] += 1
    result_image /= mask
    irr_data_n = torch.from_numpy(
    np.array(np.reshape(result_image, (image_size, image_size, 1)))
    )
    return irr_data_n

for discr in grid_discr:
    Nx, Ny = discr, discr
    Lx, Ly = 450, 450
    grid = create_2d_cartesian(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
    # Porosity
    phi = 0.2
    phi = np.full(grid.nb_cells, phi)
    # Diffusion coefficient (i.e Permeability)
    K = np.full(grid.nb_cells, 100.0e-15)

    T = calculate_transmissivity(grid, K)
    # Pressure initialization
    P = np.full(grid.nb_cells, 100.0e5)
    S = np.full(grid.nb_cells, 0.2)
    mu_w = 0.571e-3
    mu_g = 0.0285e-3
    kr_model = "quadratic"

    # BOUNDARY CONDITIONS #
    # Pressure
    Pb = {"left": 100.0e5, "right": 100.0e5}
    # Saturation
    Sb_d = {"left": 0.2, "right": 0.2}
    Sb_n = {"left": None, "right": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}
    dt = 2 * (60 * 60 * 24 * 365.25)  # in years
    q = -5e-5

    well_co2 = Well(
            name="well co2",
            cell_group=np.array([[Lx/2, Ly/2]]),
            radius=0.1,
            control={"Neumann": q},
            s_inj=1.0,
            schedule=[
                [0.0, dt],
            ],
            mode="injector",
        )
    # P_imp = implicit_pressure_solver(
    #         grid=grid,
    #         K=K,
    #         T=T,
    #         P=P,
    #         S=S,
    #         Pb=Pb,
    #         Sb_dict=Sb_dict,
    #         mu_g=mu_g,
    #         mu_w=mu_w,
    #         kr_model=kr_model,
    #         wells=[well_co2],
    #     )
    
    # shape prep
    # shape = Nx

    # log_q_tmp = np.zeros((9 * 9))
    # log_q_tmp[40] =  -np.log10(-q)
    # log_q_tmp = torch.from_numpy(np.reshape(log_q_tmp, (9, 9, 1)))
    # log_q_tmp_n = np.reshape(GWM_q_normalizer.encode(log_q_tmp), (9, 9))
    # log_q_n = place_image_at_center(np.zeros((shape , shape)), log_q_tmp_n)
    # log_q_n =  torch.from_numpy(np.reshape(log_q_n, (shape, shape, 1)))
    
    # log_dt = torch.from_numpy(np.full((shape, shape, 1), np.log(dt)))
    # S_n = torch.from_numpy(np.array(np.reshape(S, (shape, shape, 1))))
    # P_imp_n = torch.from_numpy(
    #     np.array(np.reshape(P_imp, (shape, shape, 1)))
    # )
    
    # log_dt_n = np.round(apply_irregular_normalizer(log_dt, GWM_dt_normalizer, shape=shape), 4)
    # P_imp_n = apply_irregular_normalizer(np.log10(P_imp_n), GWM_P_imp_normalizer, shape=shape)

    # # fig, axs = plt.subplots(2, 4)
    # # axs[0][0].imshow(log_q_n)
    # # axs[1][0].imshow(S_n)

    # # axs[0][1].imshow(np.round(log_dt_n, 4))
    # # axs[1][1].imshow(log_dt)

    # # axs[0][2].imshow(P_imp_n)
    # # axs[1][2].imshow(P_imp.reshape(shape, shape))


    # # plt.show()

    # #
    # x = torch.cat([log_q_n, log_dt_n, S_n, P_imp_n], 2).float()
    # x = x.reshape(1, shape, shape, 4)
    # S_pred = model(x)
    # S_pred = S_pred.detach().numpy()
    # S_pred = np.reshape(S_pred, (shape * shape))
    
    # P_i_plus_1, S_i_plus_1, dt_sim, nb_newton, norms = solss_newton_step(
    #     grid=grid,
    #     P_i=P,
    #     S_i=S,
    #     Pb=Pb,
    #     Sb_dict=Sb_dict,
    #     phi=phi,
    #     K=K,
    #     T=T,
    #     mu_g=mu_g,
    #     mu_w=mu_w,
    #     dt_init=dt,
    #     dt_min=dt,
    #     wells=[well_co2],
    #     max_newton_iter=200,
    #     eps=1e-6,
    #     kr_model=kr_model,
    #     P_guess=P_imp,
    #     S_guess=S,
    # )

    # # fig, axs = plt.subplots(1, 2)
    
    # # axs[0].imshow(S_i_plus_1.reshape(shape, shape).T, vmin=0., vmax=1.)
    # # axs[1].imshow(S_pred.reshape(shape, shape).T, vmin=0., vmax=1.)
    # # for ax in axs:
    # #     ax.invert_yaxis()
    # #     ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    # #     ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    # # plt.show()
    # print(np.linalg.norm(S_pred - S_i_plus_1, 2))
    

# q_flat_zeros = np.zeros((shape * shape))
# q_flat_zeros[grid.cell_groups['well co2']] = -np.log10(-q)
# log_q = torch.from_numpy(np.reshape(q_flat_zeros, (shape, shape, 1)))