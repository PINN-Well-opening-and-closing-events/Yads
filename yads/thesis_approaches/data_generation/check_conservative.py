from typing import Union, List

import numpy as np

from yads.mesh import Mesh
from yads.numerics import peaceman_radius
from yads.wells import Well

from yads.physics.relative_permeability import kr

# import tensorflow as tf


def check_conservative(
    grid: Mesh,
    S_i,
    Pb,
    Sb_dict,
    phi,
    K,
    T,
    mu_g,
    mu_w,
    dt_init,
    wells,
    kr_model,
    P_guess,
    S_guess,
):
    V = grid.measures(item="cell")
    if wells:
        for well in wells:
            grid.connect_well(well)
    P = P_guess
    S = S_guess

    # system construction for the Residual
    B = np.zeros(2 * grid.nb_cells)

    for cell in range(grid.nb_cells):
        # accumulation term
        value = V[cell] * phi[cell] / dt_init
        accu_g = value * (S[cell] - S_i[cell])
        accu_w = value * ((1.0 - S[cell]) - (1.0 - S_i[cell]))

        B[cell] = accu_g
        B[grid.nb_cells + cell] = accu_w

    # inner faces
    for f in grid.faces(group="0", with_nodes=False):
        # an inner face has always 2 adjacent cells
        # cells[0] -> front, cells[1] -> back
        front, back = grid.face_to_cell(f, face_type="inner")
        # upwinding
        value = T[f] * (P[front] - P[back])
        if P[front] >= P[back]:
            # Flux
            F_g = kr(S[front], model=kr_model) / mu_g * value
            F_w = kr(1.0 - S[front], model=kr_model) / mu_w * value

        else:
            # Flux
            F_g = kr(S[back], model=kr_model) / mu_g * value
            F_w = kr(1.0 - S[back], model=kr_model) / mu_w * value

        B[front] += F_g
        B[grid.nb_cells + front] += F_w
        B[back] -= F_g
        B[grid.nb_cells + back] -= F_w

    # boundary faces
    for group in Pb.keys():
        for f in grid.faces(group=group, with_nodes=False):
            cell = grid.face_to_cell(f, face_type="boundary")
            value = T[f] * (P[cell] - Pb[group])
            if P[cell] >= Pb[group]:
                # Flux
                F_g = kr(S[cell], model=kr_model) / mu_g * value
                F_w = kr(1.0 - S[cell], model=kr_model) / mu_w * value

            else:
                if Sb_dict["Dirichlet"][group] is not None:
                    F_g = kr(Sb_dict["Dirichlet"][group], model=kr_model) / mu_g * value
                    F_w = (
                        kr(1.0 - Sb_dict["Dirichlet"][group], model=kr_model)
                        / mu_w
                        * value
                    )
                else:
                    ######### RISKY NULL FLUX #####
                    F_g = 0.0
                    F_w = 0.0

            B[cell] += F_g
            B[grid.nb_cells + cell] += F_w

    if wells:
        for well in wells:
            ip = well.ip
            if "Neumann" in well.control:
                for c in grid.cell_groups[well.name]:
                    B[c] += well.control["Neumann"]

            elif "Dirichlet" in well.control:
                for c in grid.cell_groups[well.name]:
                    if ip < 0:
                        dx, dy = well.dx, well.dy
                        re = peaceman_radius(dx, dy)
                        ip = 2.0 * np.pi * K[c] / np.log(re / well.radius)
                        well.set_ip(ip)

                    value = ip * (P[c] - well.control["Dirichlet"])
                    # upwinding
                    if P[c] >= well.control["Dirichlet"]:
                        # Productor
                        if well.is_injector:
                            continue
                        F_g = kr(S[c], model=kr_model) / mu_g * value
                        F_w = kr((1.0 - S[c]), model=kr_model) / mu_w * value

                    else:
                        # Injector
                        if well.is_productor:
                            continue
                        F_g = (
                            kr(well.injected_saturation, model=kr_model) / mu_g * value
                        )

                        F_w = (
                            kr(1.0 - well.injected_saturation, model=kr_model)
                            / mu_w
                            * value
                        )

                    B[c] += F_g
                    B[grid.nb_cells + c] += F_w

    # Algebric correction of jacobian in case of injection phase disappears
    for i in range(grid.nb_cells):
        # injection phase lines
        B[i] += B[grid.nb_cells + i]

    norm = np.linalg.norm(B, ord=2)
    # print(f"norm: {norm:0.2E}")
    return norm


"""
def tf_check_cons(
        grid: Mesh,
        S_i,
        Pb,
        Sb_dict,
        phi,
        K,
        T,
        mu_g,
        mu_w,
        dt_init,
        wells,
        kr_model,
        P_guess,
        S_guess
):
    def check_cons(S_input):
        return check_conservative(grid=grid, S_i=S_i,
                                  Pb=Pb, Sb_dict=Sb_dict,
                                  phi=phi, K=K, T=T,
                                  mu_g=mu_g, mu_w=mu_w,
                                  dt_init=dt_init,
                                  wells=wells,
                                  kr_model=kr_model,
                                  P_guess=P_guess,
                                  S_guess=S_input)

    print("second call", check_cons(S_guess))

    @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
    def tf_check_conservative(input1):
        print(input1)
        r = tf.numpy_function(check_cons, [input1], tf.float32)
        return r

    return tf_check_conservative(S_guess)


if __name__ == "__main__":
    import yads.mesh as ym
    from yads.numerics.calculate_transmissivity import calculate_transmissivity

    grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 256, 1)

    phi = 0.2

    # Porosity
    phi = np.full(grid.nb_cells, phi)
    # Diffusion coefficient (i.e Permeability)
    K = np.full(grid.nb_cells, 100.0e-15)
    T = calculate_transmissivity(grid=grid, K=K)
    # gaz saturation initialization
    S = np.full(grid.nb_cells, 0.0)
    # Pressure initialization
    P = np.full(grid.nb_cells, 100.0e5)

    mu_w = 0.571e-3
    mu_g = 0.0285e-3

    kr_model = "quadratic"
    # BOUNDARY CONDITIONS #
    # Pressure
    Pb = {"left": 110.0e5, "right": 100.0e5}

    # Saturation

    Sb_d = {"left": None, "right": None}
    Sb_n = {"left": None, "right": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

    dt = 1 * (60 * 60 * 24 * 365.25)  # in years
    total_sim_time = 30 * (60 * 60 * 24 * 365.25)
    max_newton_iter = 20

    injector_two = Well(
        name="injector 2",
        cell_group=np.array([[7500.0, 500.0]]),
        radius=0.1,
        control={"Dirichlet": 105.0e5},
        s_inj=0.0,
        schedule=[[0.0, total_sim_time]],
        mode="injector",
    )

    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[3000.0, 500.0]]),
        radius=0.1,
        control={"Neumann": -0.02},
        s_inj=1.0,
        schedule=[
            [0.0, total_sim_time],
        ],
        mode="injector",
    )
    print("first call", check_conservative(grid,
                             S,
                             Pb,
                             Sb_dict,
                             phi,
                             K,
                             T,
                             mu_g,
                             mu_w,
                             dt_init=dt,
                             wells=[well_co2, injector_two],
                             kr_model=kr_model,
                             P_guess=P,
                             S_guess=S))

    S_guess = tf.convert_to_tensor(S, dtype=tf.float32)
    tf_check_cons(grid,
                  S,
                  Pb,
                  Sb_dict,
                  phi,
                  K,
                  T,
                  mu_g,
                  mu_w,
                  dt_init=dt,
                  wells=[well_co2, injector_two],
                  kr_model=kr_model,
                  P_guess=P,
                  S_guess=S)
                  
"""
