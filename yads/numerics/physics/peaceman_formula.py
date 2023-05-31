import numpy as np  # type: ignore
from typing import Union


def peaceman_radius(dx: Union[float, int], dy: Union[float, int]):
    return 0.14 * np.sqrt(np.square(dx) + np.square(dy))
