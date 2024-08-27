import numpy as np  # type: ignore


def total_mobility(sw, mu_w, mu_g, model="cross"):
    """Calculates the total mobility which is the sum of each mobility (water + oil here)
    with mobility_i(s_i) = kr_i(s_i)/mu_i
    and kr_w(sw) the relative permeability of water

    Args:
        sw: water saturation in [0,1]
        mu_w: water viscosity
        mu_g: oil viscosity
        model: relative permeability of water model
            cross: kr(sw) = sw
            quadratic: kr(sw) = sw**2
    Returns:
        same type as sw
    """
    if mu_w <= 0.0 or mu_g <= 0.0:
        raise ValueError(f"viscosity must be positive (got mu_w: {mu_w}, mu_g: {mu_g})")
    model_list = ["cross", "quadratic"]
    if model not in model_list:
        raise ValueError(f"Model not handled yet (got: {model}, expected {model_list})")

    m_w, m_o = None, None

    if model == "cross":
        m_w, m_o = sw / mu_w, (1.0 - sw) / mu_g

    elif model == "quadratic":
        m_w, m_o = sw**2 / mu_w, (1.0 - sw) ** 2 / mu_g

    assert m_w is not None
    assert m_o is not None

    return m_w + m_o


def d_total_mobility_ds(sw, mu_w, mu_g, model="cross"):
    """ "
    if isinstance(sw, float):
        if 0.0 > sw or sw > 1.0:
            raise ValueError(f"Saturation sw must be between 0 and 1 (got {sw})")
    else:
        if 0.0 > sw.any() or sw.any() > 1.0:
            raise ValueError(f"Saturation sw must be between 0 and 1 (got {sw})")
    """
    if mu_w <= 0.0 or mu_g <= 0.0:
        raise ValueError(f"viscosity must be positive (got mu_w: {mu_w}, mu_g: {mu_g})")
    model_list = ["cross", "quadratic"]
    if model not in model_list:
        raise ValueError(f"Model not handled yet (got: {model}, expected {model_list})")

    dm_ds = None
    if model == "cross":
        dm_ds = 1.0 / mu_w - 1.0 / mu_g

    elif model == "quadratic":
        dm_ds = 2.0 * sw / mu_w - 2.0 * sw * (1.0 - sw) / mu_g
    return dm_ds


def calculate_mobility(grid, P, S, Pb, Sb_dict, mu_w, mu_g):
    """Calculates mobility with upwinding/décentrement amont

    Args:
        grid: yads.mesh.Mesh object
        P: pressure, np.ndarray size(grid.nb_cells)
        S: water saturation, np.ndarray size(grid.nb_cells)
        Pb: pressure boundary conditions dict
            example: Pb = {"0": 1.0, "1": 2.0}
        Sb_dict: water saturation boundary conditions dict
            example: Sb = {"0": 1.0, "1": 0.2}
        mu_w: water viscosity
        mu_g: oil viscosity

    Returns:
        M: total mobility, np.ndarray size(grid.nb_faces)
    """
    assert len(P) == grid.nb_cells
    assert len(S) == grid.nb_cells
    if mu_w <= 0.0 or mu_g <= 0.0:
        raise ValueError(f"viscosity must be positive (got mu_w: {mu_w}, mu_g: {mu_g})")
    if not all([1.0 >= sw >= 0.0 for sw in S]):
        raise ValueError(r"Saturation S must have all its values between 0 and 1")

    # initialize mobility, size number of faces
    M = np.full(grid.nb_faces, None)

    for group in Pb.keys():
        # boundary faces
        for f in grid.faces(group=group):
            # find the cell connected to the border face
            c = grid.face_to_cell(f, face_type="boundary")
            # upwinding
            if P[c] >= Pb[group]:
                M[f] = total_mobility(S[c], mu_w, mu_g)
            else:
                if Sb_dict["Dirichlet"][group] is not None:
                    M[f] = total_mobility(Sb_dict["Dirichlet"][group], mu_w, mu_g)

    # inner faces
    for f in grid.faces(group="0"):
        # get two adjacent cells of the face
        i, j = grid.face_to_cell(f, face_type="inner")
        # upwinding
        if P[i] >= P[j]:
            M[f] = total_mobility(S[i], mu_w, mu_g)
        else:
            M[f] = total_mobility(S[j], mu_w, mu_g)

    return M
