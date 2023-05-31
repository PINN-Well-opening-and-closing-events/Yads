import pytest

from yads.physics.fractional_flow import fw, dfw_dsw


def test_wrong_inputs():
    sw_down = -1.0
    sw_up = 2.0
    with pytest.raises(ValueError, match=r"Saturation sw must be between 0 and 1"):
        fw(sw=sw_down, mu_o=1.0, mu_w=1.0)
    with pytest.raises(ValueError, match=r"Saturation sw must be between 0 and 1"):
        fw(sw=sw_up, mu_o=1.0, mu_w=1.0)

    with pytest.raises(ValueError, match=r"Saturation sw must be between 0 and 1"):
        dfw_dsw(sw=sw_down, mu_o=1.0, mu_w=1.0)
    with pytest.raises(ValueError, match=r"Saturation sw must be between 0 and 1"):
        dfw_dsw(sw=sw_up, mu_o=1.0, mu_w=1.0)

    with pytest.raises(ValueError, match=f"Model not handled yet"):
        fw(0.5, 0.5, 0.5, model="error")
    with pytest.raises(ValueError, match=f"Model not handled yet"):
        dfw_dsw(0.5, 0.5, 0.5, model="error")

    mu_wrong, mu_ok = -1.0, 1.0
    with pytest.raises(ValueError, match=r"viscosity must be positive"):
        fw(1.0, mu_w=mu_wrong, mu_o=mu_ok)
