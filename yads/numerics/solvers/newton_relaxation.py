import numpy as np


def compute_relaxation(S_obj: float, delta_S: np.array) -> float:
    """Compute relaxation value for Newton update based on saturation variations

    :param S_obj: Max Saturation variation tolerated
    :param delta_S: Saturation variation obtained from linear solver during Newton iteration
    :return:
    """
    if np.max(delta_S) != 0.0:
        value = S_obj / np.max(delta_S)
    else:
        value = 1.0
    return min([value, 1.0])
