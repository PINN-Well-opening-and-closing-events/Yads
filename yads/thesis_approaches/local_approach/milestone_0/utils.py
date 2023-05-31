import itertools
from typing import Union, List
from yads.mesh import Mesh
from yads.wells import Well
import copy
import pandas as pd


def generate_data_2_elem(q, dt):
    all_combinations = [
        list(zip(each_permutation, dt))
        for each_permutation in itertools.permutations(q, len(dt))
    ]
    return all_combinations


def generate_data_n_elem(list_elem):
    all_combinations = list(itertools.product(*list_elem))
    return all_combinations


def generate_wells_from_q(q_list, well_ref: Well, other_wells: List[Well]):
    well_list = []

    for q in q_list:
        well_save = copy.deepcopy(well_ref)
        well_save.set_control(q)
        well_list.append([well_save] + other_wells)
    return well_list


def data_dict_to_combi(data_dict, combination, mapping):
    for key in mapping.keys():
        data_dict[key] = combination[mapping[key]]
        if key == "S":
            S0 = combination[mapping[key]][0]
            data_dict["Sb_dict"]["Dirichlet"] = {
                "injector_one": S0,
                "injector_two": S0,
                "right": 0.0,
            }
        elif key == "Pb_left":
            data_dict["Pb"]["left"] = combination[mapping[key]]
        elif key == "Pb_right":
            data_dict["Pb"]["right"] = combination[mapping[key]]
    return data_dict


def data_dict_to_combi_hard(data_dict, combination, mapping, n):
    for key in mapping.keys():
        data_dict[key] = combination[mapping[key]]
    data_dict["n"] = n
    return data_dict


def args_to_dict(
    grid: Mesh,
    P,
    S,
    Pb,
    Sb_dict,
    phi,
    K,
    mu_g,
    mu_w,
    dt_init,
    total_sim_time,
    kr_model: str = "cross",
    max_newton_iter=10,
    eps=1e-6,
    wells: Union[List[Well], None] = None,
):
    data_dict = {
        "grid": grid,
        "P": P,
        "S": S,
        "Pb": Pb,
        "Sb_dict": Sb_dict,
        "phi": phi,
        "K": K,
        "mu_g": mu_g,
        "mu_w": mu_w,
        "dt_init": dt_init,
        "total_sim_time": total_sim_time,
        "kr_model": kr_model,
        "max_newton_iter": max_newton_iter,
        "eps": eps,
        "wells": wells,
    }

    return data_dict


def dict_to_args(data_dict):
    grid = data_dict["grid"]
    P = data_dict["P"]
    S = data_dict["S"]
    Pb = data_dict["Pb"]
    Sb_dict = data_dict["Sb_dict"]
    phi = data_dict["phi"]
    K = data_dict["K"]
    mu_g = data_dict["mu_g"]
    mu_w = data_dict["mu_w"]
    dt_init = data_dict["dt_init"]
    total_sim_time = data_dict["total_sim_time"]
    kr_model = data_dict["kr_model"]
    max_newton_iter = data_dict["max_newton_iter"]
    eps = data_dict["eps"]
    wells = data_dict["wells"]
    return (
        grid,
        P,
        S,
        Pb,
        Sb_dict,
        phi,
        K,
        mu_g,
        mu_w,
        dt_init,
        total_sim_time,
        kr_model,
        max_newton_iter,
        eps,
        wells,
    )


def dict_to_args_hard(data_dict):
    grid = data_dict["grid"]
    P = data_dict["P"]
    S = data_dict["S"]
    Pb = data_dict["Pb"]
    Sb_dict = data_dict["Sb_dict"]
    phi = data_dict["phi"]
    K = data_dict["K"]
    mu_g = data_dict["mu_g"]
    mu_w = data_dict["mu_w"]
    dt_init = data_dict["dt_init"]
    total_sim_time = data_dict["total_sim_time"]
    kr_model = data_dict["kr_model"]
    max_newton_iter = data_dict["max_newton_iter"]
    eps = data_dict["eps"]
    wells = data_dict["wells"]
    n = data_dict["n"]
    return (
        grid,
        P,
        S,
        Pb,
        Sb_dict,
        phi,
        K,
        mu_g,
        mu_w,
        dt_init,
        total_sim_time,
        kr_model,
        max_newton_iter,
        eps,
        wells,
        n,
    )


def better_data_to_df(combination_ser, sim_state):
    list_of_dict = []
    well_co2 = combination_ser[1][0]
    q = well_co2["control"]["Neumann"]
    P_imp = sim_state["metadata"]["P_imp"]
    data_dict = sim_state["data"]
    for tot_t in data_dict.keys():
        step = data_dict[tot_t]["step"]
        S = data_dict[tot_t]["S"]
        P = data_dict[tot_t]["P"]
        total_time = data_dict[tot_t]["total_time"]
        dt_init = data_dict[tot_t]["dt_init"]
        dt = data_dict[tot_t]["dt"]
        nb_newton = data_dict[tot_t]["nb_newton"]
        S0 = data_dict[tot_t]["S0"]
        B = data_dict[tot_t]["Res"]
        future_df = {
            "q": q,
            "total_time": total_time,
            "step": step,
            "S": S,
            "dt": dt,
            "P": P,
            "P_imp": P_imp,
            "S0": S0,
            "nb_newton": nb_newton,
            "dt_init": dt_init,
            "Res": B,
        }
        list_of_dict.append(future_df)

    df = pd.DataFrame(list_of_dict)
    return df


def data_to_df(json_obj):
    list_of_dict = []
    for combination_ser, sim_state, case in json_obj:
        well_co2 = combination_ser[1][0]
        q = well_co2["control"]["Neumann"]
        data_dict = sim_state["data"]
        for tot_t in data_dict.keys():
            step = data_dict[tot_t]["step"]
            S = data_dict[tot_t]["S"]
            P = data_dict[tot_t]["P"]
            total_time = data_dict[tot_t]["total_time"]
            dt_init = data_dict[tot_t]["dt_init"]
            dt = data_dict[tot_t]["dt"]
            nb_newton = data_dict[tot_t]["nb_newton"]
            S0 = data_dict[tot_t]["S0"]
            future_df = {
                "q": q,
                "total_time": total_time,
                "step": step,
                "S": S,
                "dt": dt,
                "P": P,
                "S0": S0,
                "nb_newton": nb_newton,
                "dt_init": dt_init,
                "case": case,
            }
            list_of_dict.append(future_df)
    df = pd.DataFrame(list_of_dict)
    return df


def hard_data_to_df(hard_data):
    list_of_dict = []
    for combination_ser, sim_state in hard_data:
        well_co2 = combination_ser[1][0]
        q = well_co2["control"]["Neumann"]
        data_dict = sim_state["data"]
        for tot_t in data_dict.keys():
            step = data_dict[tot_t]["step"]
            S = data_dict[tot_t]["S"]
            P = data_dict[tot_t]["P"]
            total_time = data_dict[tot_t]["total_time"]
            dt = data_dict[tot_t]["dt"]
            nb_newton = data_dict[tot_t]["nb_newton"]
            dt_init = data_dict[tot_t]["dt_init"]
            future_df = {
                "q": q,
                "total_time": total_time,
                "step": step,
                "S": S,
                "dt": dt,
                "P": P,
                "nb_newton": nb_newton,
                "dt_init": dt_init,
            }
            list_of_dict.append(future_df)
    df = pd.DataFrame(list_of_dict)
    return df


def giga_hard_data_to_df(giga_hard_data):
    list_of_dict = []
    for combination_ser, sim_state in giga_hard_data:
        well_co2 = combination_ser[1][0]
        q = well_co2["control"]["Neumann"]
        data_dict = sim_state["data"]
        for tot_t in data_dict.keys():
            step = data_dict[tot_t]["step"]
            S = data_dict[tot_t]["S"]
            P = data_dict[tot_t]["P"]
            total_time = data_dict[tot_t]["total_time"]
            dt = data_dict[tot_t]["dt"]
            dt_init = data_dict[tot_t]["dt_init"]
            nb_newton = data_dict[tot_t]["nb_newton"]
            n = data_dict[tot_t]["n"]
            newtons = data_dict[tot_t]["newtons"]
            future_df = {
                "q": q,
                "total_time": total_time,
                "step": step,
                "S": S,
                "dt": dt,
                "P": P,
                "nb_newton": nb_newton,
                "n": n,
                "dt_init": dt_init,
                "newtons": newtons,
            }
            list_of_dict.append(future_df)
    df = pd.DataFrame(list_of_dict)
    return df


if __name__ == "__main__":
    a = [[1, 2], [3, 4]]
    generate_data_n_elem(a)
