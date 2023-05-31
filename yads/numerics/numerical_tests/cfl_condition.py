import numpy as np  # type: ignore
from typing import Union, List, Dict

from yads.wells import Well
from yads.mesh import Mesh


def cfl_condition(
    grid: Mesh,
    phi: np.ndarray,
    F: np.ndarray,
    F_well: Dict,
    dfw,
    Pb: dict,
    mu_w: Union[float, int],
    mu_o: Union[float, int],
    wells: Union[List[Well], None] = None,
):
    """Compute the limit value of time step dt according to CFL (Courant-Friedrichs-Lewy) condition: dt <= dt_lim

    Args:
        grid: yads.mesh.cartesian.Cartesian object
        phi: porosity in each cell, np.ndarray size(grid.nb_cells)
        F: discretization of flux on each face, np.ndarray size(grid.nb_faces)
        F_well:
        dfw: function dfw_dsw
        Pb: pressure boundary conditions dict
            example: Pb = {"left":1.0, "right": 2.0}
        mu_w: water viscosity
        mu_o: oil viscosity
        wells:
    Returns:
        limit time step allowed
    """
    assert len(phi) == grid.nb_cells
    assert len(F) == grid.nb_faces
    if mu_w <= 0.0 or mu_o <= 0.0:
        raise ValueError(f"viscosity must be positive (got mu_w: {mu_w}, mu_o: {mu_o})")
    # Numerator
    cell_vols = grid.measures(item="cell")
    num = min(np.multiply(cell_vols, phi))
    # Denominator
    sw = np.linspace(0.0, 1.0, num=1000)
    sup_dfw = max(dfw(sw, mu_w, mu_o))

    flow_sum = np.zeros(grid.nb_cells)

    # boundary faces
    if Pb:
        for group in Pb.keys():
            for f in grid.faces(group=group, with_nodes=False):
                c = grid.face_to_cell(f, face_type="boundary")
                flow_sum[c] -= min(F[f], 0.0)

    # print("boundary flow:", flow_sum)
    # inner faces
    for f in grid.faces(group="0", with_nodes=False):
        front, back = grid.face_to_cell(f, face_type="inner")
        flow_sum[front] -= min(-F[f], 0.0)
        flow_sum[back] -= min(F[f], 0.0)

    # print("inner flow:", flow_sum)
    # well faces
    if wells:
        for well in wells:
            for c in grid.cell_groups[well.name]:
                flow_sum[c] -= min(F_well[well.name], 0.0)

    # print("well flow:", flow_sum)
    sup_F = max(flow_sum)
    denom = sup_dfw * sup_F
    if denom != 0:
        return num / denom, flow_sum
    return 123456789, flow_sum
