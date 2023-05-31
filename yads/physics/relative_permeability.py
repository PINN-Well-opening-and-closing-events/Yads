import numpy as np


def kr(S, model="cross"):
    if model == "cross":
        return S
    elif model == "quadratic":
        return np.square(S)
    else:
        print(f"model not handled yet. got {model}")
    return


def d_kr_ds(S, model="cross", negative=False):
    if model == "cross":
        if negative:
            return -1.0
        else:
            return 1.0
    elif model == "quadratic":
        if negative:
            return -2.0 * S
        else:
            return 2.0 * S
    else:
        print(f"model not handled yet. got {model}")
    return
