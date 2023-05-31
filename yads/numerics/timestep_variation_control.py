import numpy as np


def update_dt(S_t: np.ndarray, S_t_plus_1: np.ndarray, auto_dt, dt):
    """

    S:
    auto_dt: [Vtol, Vmax, dr_var, tho1, tho2, dt_min, dt_max]
        with:
            Vtol:
            Vmax:
            dr_var:
            tho1:
            tho2:
            dt_min:
            dt_max:
    dt: timestep
    """
    assert auto_dt is not None
    assert len(auto_dt) == 7

    dt_plus_1 = dt
    Vtol, Vmax, dr_var, tho1, tho2, dt_min, dt_max = auto_dt
    delta_S = max(abs(S_t - S_t_plus_1))

    if Vtol / (1.0 + dr_var) <= delta_S <= dr_var * Vtol:
        dt_plus_1 = dt
    elif delta_S < Vtol / (1.0 + dr_var):
        dt_plus_1 = dt * tho1
    elif dr_var * Vtol < delta_S < Vmax:
        dt_plus_1 = dt / tho2

    # timestep must be in [dt_min, dt_max]
    if dt_plus_1 > dt_max:
        dt_plus_1 = dt_max
    elif dt_plus_1 < dt_min:
        dt_plus_1 = dt_min
    return dt_plus_1
