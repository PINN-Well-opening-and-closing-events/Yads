import numpy as np  # type: ignore
from typing import Union, List, Dict

from yads.numerics.utils import clipping_P, clipping_S
from yads.physics import total_mobility
from yads.physics.fractional_flow import fw
from yads.mesh import Mesh
from yads.wells import Well


def explicit_saturation_solver(
    grid: Mesh,
    P: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    phi: np.ndarray,
    Pb: dict,
    Sb_dict: dict,
    dt: Union[float, int],
    mu_w: Union[float, int],
    mu_g: Union[float, int],
    wells: Union[List[Well], None] = None,
    eps: Union[float, int] = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicitly updates saturation

    Args:
        grid: yads.mesh.Mesh object
        P: pressure, np.ndarray size(grid.nb_cells)
        S: water saturation, np.ndarray size(grid.nb_cells)
        K: diffusion coefficient (i.e. permeability), np.ndarray size(grid.nb_cells)
        T: transmissivity , np.ndarray size(grid.nb_faces)
        phi: porosity in each cell, np.ndarray size(grid.nb_cells)
        Pb: pressure boundary conditions dict
            example: Pb = {"left":1.0, "right": 2.0}
        Sb_dict: water saturation boundary conditions dict
            example: Sb_dict = {"Neumann":{"left":1.0, "right": 0.2}, "Dirichlet": "left":None, "right":None}
        dt: time step
        mu_w: water viscosity
        mu_g: gas viscosity
        wells: list of Well object
        eps: clipping tolerance

    Returns:
        updated water saturation
    """
    # inputs checks
    if len(K) != grid.nb_cells:
        raise ValueError(
            f"K length must match grid.nb_cells: {grid.nb_cells} (got: {len(K)})"
        )
    if np.any((K < 0)):
        raise ValueError(f"Permeability K must contain only positive values")
    if len(phi) != grid.nb_cells:
        raise ValueError(
            f"phi length must match grid.nb_cells: {grid.nb_cells} (got: {len(phi)})"
        )
    if not all([1.0 >= sw >= 0.0 for sw in S]):
        print("S:", S)
        print("P:", P)
        raise ValueError(f"Saturation S must have all its values between 0 and 1")
    if np.any((phi < 0)):
        raise ValueError(f"Porosity phi must contain only positive values")
    if dt < 0:
        raise ValueError(f"Time step dt must be positive: (got {dt})")

    # initialize flow, size nb_faces for CFL
    Fl = np.full(grid.nb_faces, None)
    F_well = {}
    M = total_mobility(S, mu_w, mu_g)
    #### Saturation Solver ####
    cell_measures = grid.measures(item="cell")
    # boundary conditions
    if Pb:
        for group in Pb.keys():
            for f in grid.faces(group=group, with_nodes=False):
                F = compute_flow_by_face(f, grid, P, S, T, Pb, Sb_dict, mu_w)
                Fl[f] = F
                # border only connected to one cell c
                c = grid.face_to_cell(f, face_type="boundary")
                S[c] -= dt / (cell_measures[c] * phi[c]) * F

    # inner faces
    for f in grid.faces(group="0", with_nodes=False):
        F = compute_flow_by_face(f, grid, P, S, T, Pb, Sb_dict, mu_w)
        Fl[f] = F

        # inner face is connected to 2 cells
        front, back = grid.face_to_cell(f, face_type="inner")
        # front cell
        S[front] -= dt / (cell_measures[front] * phi[front]) * F
        # back cell
        S[back] += dt / (cell_measures[back] * phi[back]) * F

    # well faces
    if wells:
        F_well = compute_well_flows(grid, P, S, M, wells, mu_w, mu_g)
        for well in wells:
            for c in grid.cell_groups[well.name]:
                S[c] -= dt / (cell_measures[c] * phi[c]) * F_well[well.name]

    S = clipping_S(S)
    return S, Fl, F_well


def compute_flow_by_face(
    f: int,
    grid: Mesh,
    P: np.ndarray,
    S: np.ndarray,
    T: np.ndarray,
    Pb: dict,
    Sb_dict: dict,
    mu_w: Union[float, int],
) -> np.ndarray:
    """computes the flow Flux w.r.t a face f of a grid object

    Args:
        f: face
        grid: yads.mesh.cartesian.Cartesian object
        P: pressure, np.ndarray size(grid.nb_cells)
        S: water saturation, np.ndarray size(grid.nb_cells)
        T: transmissivity , np.ndarray size(grid.nb_faces)
        Pb: pressure boundary conditions dict
            example: Pb = {"left":1.0, "right": 2.0}
        Sb_dict: water saturation boundary conditions dict
            example: Sb_dict = {"Neumann":{"left":1.0, "right": 0.2}, "Dirichlet": "left":None, "right":None}
        mu_w: water viscosity

    Returns:
        updated Flux
    """

    F = None
    group = "error"
    groups = grid.group(f)
    if len(groups) == 1:
        group = groups[0]
    else:
        for g in Pb.keys():
            if g in groups:
                group = g
                break
    ### At this moment, some terms of the saturation may be 0. < S < 1 because of the face assembly
    # inner face

    if group == "0":
        # an inner face has always 2 adjacent cells
        # cells[0] -> front, cells[1] -> front, back
        front, back = grid.face_to_cell(f, face_type="inner")
        # upwinding
        m = None
        if P[front] >= P[back]:
            m = S[front] / mu_w
            F = m * T[f] * (P[front] - P[back])
        else:
            m = S[back] / mu_w
            F = m * T[f] * (P[front] - P[back])

        # Phi = T[f] * m * (P[i] - P[j])
        # F = fw(S[i], mu_w, mu_g) * max(Phi, 0.0) + fw(S[j], mu_w, mu_g) * min(Phi, 0.0)

    elif group in Pb.keys():
        # Boundary conditions
        # a border face is always connected to only one cell c
        if Sb_dict["Dirichlet"][group] is not None:
            c = grid.face_to_cell(f, face_type="boundary")

            # upwinding
            m = None
            if P[c] >= Pb[group]:
                m = S[c] / mu_w
                F = T[f] * m * (P[c] - Pb[group])
            else:
                m = Sb_dict["Dirichlet"][group] / mu_w
                F = T[f] * m * (P[c] - Pb[group])

            # Phi = T[f] * m * (P[c] - Pb[group])
            # F = fw(S[c], mu_w, mu_g) * max(Phi, 0.0) + fw(
            #   Sb_dict["Dirichlet"][group], mu_w, mu_g
            # ) * min(Phi, 0.0)

        elif Sb_dict["Neumann"][group] is not None:
            F = Sb_dict["Neumann"][group]

    return F


def compute_well_flows(
    grid: Mesh,
    P: np.ndarray,
    S: np.ndarray,
    M: np.ndarray,
    wells: Union[List[Well], None],
    mu_w: Union[float, int],
    mu_g: Union[float, int],
) -> Dict:
    """computes the flow Flux w.r.t the effective wells connected to the grid

    Args:
        grid: yads.mesh.cartesian.Cartesian object
        P: pressure, np.ndarray size(grid.nb_cells)
        S: water saturation, np.ndarray size(grid.nb_cells)
        M:
        wells:
        mu_w: water viscosity
        mu_g

    Returns:
        updated Flux
    """
    F_well = {}

    if wells != {}:
        for well in wells:
            if "Neumann" in well.control:
                for _ in grid.cell_groups[well.name]:
                    F_well[well.name] = well.control["Neumann"]
            elif "Dirichlet" in well.control:
                for c in grid.cell_groups[well.name]:
                    # upwinding
                    m = None
                    if P[c] >= well.control["Dirichlet"]:
                        m = M[c]
                    else:
                        m = total_mobility(well.injected_saturation, mu_w, mu_g)

                    ip = well.ip
                    value = ip * m
                    Phi = value * (P[c] - well.control["Dirichlet"])
                    F_well[well.name] = fw(S[c], mu_w, mu_g) * max(Phi, 0.0) + fw(
                        well.water_saturation, mu_w, mu_g
                    ) * min(Phi, 0.0)
    return F_well
