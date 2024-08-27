import copy

import numpy as np  # type: ignore
from typing import Union, List
from yads.wells import Well

from yads.numerics.physics import calculate_transmissivity

from yads.numerics.numerical_tests import cfl_condition

from yads.numerics.solvers.implicit_pressure_solver import implicit_pressure_solver
from yads.numerics.solvers.explicit_saturation_solver import explicit_saturation_solver
import yads.physics as yp
import yads.mesh as ym
import time


def impes_solver(
    grid: ym.Mesh,
    P: np.ndarray,
    S: np.ndarray,
    Pb: Union[dict, None],
    Sb_dict: Union[dict, None],
    phi: np.ndarray,
    K: np.ndarray,
    mu_w: Union[float, int],
    mu_g: Union[float, int],
    total_sim_time: Union[float, int],
    dt_init: Union[float, int],
    wells: Union[List[Well], None] = None,
    auto_dt: bool = False,
    save: bool = False,
    save_step: int = 1,
    save_path: str = "./auto_save_path_",
    return_dt: bool = False,
) -> Union[List, None]:
    """solver for incompressible two-phase flow using Darcy's equation.
    Uses a 2 points finite volume method.
    Solves implicitly the pressure (imp)  and explicitly the saturation (es)

    Args:
        grid: yads.mesh.Mesh object
        P: pressure, np.ndarray size(grid.nb_cells)
        S: water saturation, np.ndarray size(grid.nb_cells)
        Pb: pressure boundary conditions dict
            example: Pb = {"left":1.0, "right": 2.0}
        Sb_dict: water saturation boundary conditions dict
            example: Sb_dict = {"Neumann":{"left":1.0, "right": 0.2}, "Dirichlet": "left":None, "right":None}
        phi: porosity in each cell, np.ndarray size(grid.nb_cells)
        K: diffusion coefficient (i.e. permeability), np.ndarray size(grid.nb_cells)
        mu_w: water viscosity
        mu_g: oil viscosity
        total_sim_time: total simulation time
        max_iter: maximum number of iteration to reach total_sim_time
        dt_init: initial time step, must not be too high is setting auto_dt as False
        wells: list of Well objects
        auto_dt: automatically set the time step to the CFL condition
        save: if true: save at save_path
        save_step: if save: save every save step the simulation state
        save_path: path where to save simulation
        return_dt: if true: return all CFL values in a list
    """
    init_time = time.time()
    T = calculate_transmissivity(grid, K)
    dt_list = []
    dt = dt_init
    stop = False
    total_time = 0.0
    nb_iter = 0

    if wells:
        for well in wells:
            grid.connect_well(well)

    while not stop:
        start_time = time.time()

        effective_wells = []
        if wells:
            for well in wells:
                for schedule in well.schedule:
                    if schedule[0] <= total_time <= schedule[1]:
                        effective_wells.append(well)

        M = yp.total_mobility(S, mu_w, mu_g)

        P = implicit_pressure_solver(
            grid, K, T, M, P, Pb, Sb_dict, mu_w, mu_g, wells=effective_wells
        )

        S_n = copy.deepcopy(S)

        S, Fl, F_well = explicit_saturation_solver(
            grid, P, S, K, T, phi, Pb, Sb_dict, dt, mu_w, mu_g, wells=effective_wells
        )

        dt_lim, flow_sum = cfl_condition(
            grid,
            phi,
            Fl,
            F_well,
            yp.fractional_flow.dfw_dsw,
            Pb,
            mu_w,
            mu_g,
            effective_wells,
        )

        total_time += dt
        if dt > dt_lim:
            print(
                "CFL conditions not respected at step "
                + str(nb_iter)
                + ": "
                + str(dt)
                + " > "
                + str(dt_lim)
            )
            # update dt and total_time
            total_time -= dt

            dt = dt_lim * 0.9

            S = S_n
            continue

        if save:
            if nb_iter % save_step == 0:
                filename = save_path + str(nb_iter)
                ym.export_vtk(
                    filename,
                    grid=grid,
                    cell_data={"P": P, "S": S, "Flow_sum": flow_sum},
                )

        if return_dt:
            dt_list.append(dt_lim)

        if auto_dt:
            dt = 0.9 * dt_lim

        mid_time = time.time()
        print(f"step: {nb_iter} done in {(mid_time - start_time)} seconds")
        nb_iter += 1
        # stop criterion
        if total_time >= total_sim_time:
            stop = True

    print("total simulation time: ", total_time)
    print("--- %s seconds ---" % (time.time() - init_time))
    if return_dt:
        return dt_list
    return
