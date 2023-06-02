import pytest
import numpy as np

from yads.physics.relative_permeability import d_kr_ds, kr


def test_wrong_input():
    S = np.zeros(10)
    kr(S=S, model="error")
    d_kr_ds(S=S, model="error")


def test_output():
    S = np.zeros(10)
    kr(S=S, model="cross")
    kr(S=S, model="quadratic")
    d_kr_ds(S=S, model="cross")
    d_kr_ds(S=S, model="quadratic")
