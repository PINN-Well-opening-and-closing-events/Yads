def fw(sw, mu_w, mu_o, model="cross"):
    """Calculate the fractional flow of water:
    fw = mobility_w(sw)/(mobility_w(sw) + mobility_o(1-sw))
    with mobility(sw) = kr_w(sw)/mu_w
    and kr_w(sw) the relative permeability of water

    Args:
        sw: water saturation in [0,1]
        mu_w: water viscosity
        mu_o: oil viscosity
        model: relative permeability of water model
            cross: kr(sw) = sw
            quadratic: kr(sw) = sw**2
    Returns:
        fw: same type as sw input
    """
    if mu_w <= 0.0 or mu_o <= 0.0:
        raise ValueError(f"viscosity must be positive (got mu_w: {mu_w}, mu_o: {mu_o})")
    model_list = ["cross", "quadratic"]
    if model not in model_list:
        raise ValueError(f"Model not handled yet (got: {model}, expected {model_list})")

    if isinstance(sw, float) or isinstance(sw, int):
        if 0.0 > sw or sw > 1.0:
            raise ValueError(f"Saturation sw must be between 0 and 1 (got {sw})")
    else:
        if 0.0 > sw.any() or sw.any() > 1.0:
            raise ValueError(f"Saturation sw must be between 0 and 1 (got {sw})")

    m_w, m_o = None, None

    if model == "cross":
        m_w, m_o = sw / mu_w, (1.0 - sw) / mu_o

    elif model == "quadratic":
        m_w, m_o = sw**2 / mu_w, (1.0 - sw) ** 2 / mu_o

    assert m_w is not None
    assert m_o is not None
    return m_w / (m_w + m_o)


def dfw_dsw(sw, mu_w, mu_o, model="cross"):
    """Calculate the derivative of the fractional flow of water fw with respect to the water saturation sw
    fw = mobility_w(sw)/(mobility_w(sw) + mobility_o(1-sw))
    with mobility(sw) = kr_w(sw)/mu_w
    and kr_w(sw) the relative permeability of water

    Args:
        sw: water saturation in [0,1]
        mu_w: water viscosity
        mu_o: oil viscosity
        model: relative permeability of water model
            cross: kr(sw) = sw
            quadratic: kr(sw) = sw**2
    Returns:
        dfw_dsw(sw): same type as sw input
    """
    if mu_w <= 0.0 or mu_o <= 0.0:
        raise ValueError(f"viscosity must be positive (got mu_w: {mu_w}, mu_o: {mu_o})")

    if isinstance(sw, float) or isinstance(sw, int):
        if 0.0 > sw or sw > 1.0:
            raise ValueError(f"Saturation sw must be between 0 and 1 (got {sw})")
    else:
        if 0.0 > sw.any() or sw.any() > 1.0:
            raise ValueError(f"Saturation sw must be between 0 and 1 (got {sw})")

    model_list = ["cross", "quadratic"]
    if model not in model_list:
        raise ValueError(f"Model not handled yet (got: {model}, expected {model_list})")

    num, denom = None, None

    if model == "cross":
        m = mu_o / mu_w
        num = 1.0 / m
        denom = (sw + 1.0 / m * (1.0 - sw)) ** 2

    elif model == "quadratic":
        m = mu_o / mu_w
        num = 2.0 * sw / m * (1.0 - sw)
        denom = (sw**2 + 1.0 / m * (1.0 - sw) ** 2) ** 2

    return num / denom
