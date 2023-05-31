import numpy as np  # type : ignore
from typing import List


def get_dist(cell_center, face_center) -> float:
    """computes the distance between 2 points
    cell_center and face_center here in order to compute the transmissivity

    Args:
        cell_center: tuple (x, y) or (x, y, z)
        face_center: tuple (x, y) or (x, y, z)

    Returns:
        np.float
    """
    return np.linalg.norm(cell_center - face_center)


# def clipping(S, eps):
#     if not all([1.0 >= sw >= 0.0 for sw in S]):
#         for i, sw in enumerate(S):
#             if eps >= sw - 1 >= 0.0:
#                 S[i] = 1.0
#             if 0.0 >= sw >= -eps:
#                 S[i] = 0.0
#     return S


def clipping_S(S):
    if not all([1.0 >= sw >= 0.0 for sw in S]):
        for i, sw in enumerate(S):
            if sw > 1.0:
                S[i] = 1.0
            if sw < 0.0:
                S[i] = 0.0
    return S


def clipping_P(P, P_min, P_max):
    if not all([P_max >= sw >= P_min for sw in P]):
        for i, p in enumerate(P):
            if p > P_max:
                P[i] = P_max
            if p < P_min:
                P[i] = P_min
    return P


def newton_list_format(newton_list: List, max_newton_iter: int):
    newton_list_reformat = []
    for elt in newton_list:
        if elt != -1:
            newton_list_reformat.append(elt)
        else:
            newton_list_reformat[-1] += max_newton_iter
    return newton_list_reformat


def P_closest_to_dt_in_json(json_path, dt):
    import json

    with open(json_path, "r") as f:
        states = json.load(f)

    min_dist = -1
    dt_state = None

    for t in states["simulation data"].keys():
        dist = abs(float(states["simulation data"][t]["total_time"]) - dt)
        if dist <= min_dist or min_dist < 0:
            # print(f"new min dist found: {dist}, old dist:{min_dist}", dist < min_dist)
            min_dist = dist
            dt_state = states["simulation data"][t]
    P = dt_state["P"]
    return P


def S_closest_to_dt_in_json(json_path, dt):
    import json

    with open(json_path, "r") as f:
        states = json.load(f)

    min_dist = -1
    dt_state = None

    for t in states["simulation data"].keys():
        dist = abs(float(states["simulation data"][t]["total_time"]) - dt)
        if dist <= min_dist or min_dist < 0:
            # print(f"new min dist found: {dist}, old dist:{min_dist}", dist < min_dist)
            min_dist = dist
            dt_state = states["simulation data"][t]
    S = dt_state["S"]
    return S


def from_P_S_to_PS(P, S):
    PS = np.zeros(len(P) + len(S))
    PS[0::2] = P
    PS[1::2] = S
    return PS


def from_PS_to_P_S(PS):
    P = PS[0::2]
    S = PS[1::2]
    return P, S
