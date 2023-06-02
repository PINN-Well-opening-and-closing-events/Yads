import pytest
import numpy as np

from yads.wells import Well


def test_Well_properties():
    well_test = Well(
        name="well test",
        cell_group=np.array([[0.5, 0.5]]),
        radius=0.1,
        control={"Neumann": 0.0},
        s_inj=1.0,
        schedule=[[0, 0]],
        mode="injector",
    )

    assert well_test.control == {"Neumann": 0.0}
    assert well_test.injected_saturation == 1.0
    assert well_test.radius == 0.1
    assert not well_test.is_closed
    assert well_test.is_injector
    assert well_test.dy == well_test.dx == 0.0
    assert well_test.schedule == [[0, 0]]
    assert well_test.ip < 0


def test_Well_methods():
    well_test = Well(
        name="well test",
        cell_group=np.array([[0.5, 0.5]]),
        radius=0.1,
        control={"Neumann": 0.0},
        s_inj=1.0,
        schedule=[[0, 0]],
        mode="injector",
    )
    assert not well_test.is_productor
    well_test.change_mode(mode="productor")
    assert well_test.is_productor
    assert not well_test.is_injector

    well_test.change_mode(mode="closed")
    assert well_test.mode == "closed"
    assert well_test.is_closed

    assert type(well_test.well_to_dict()) == dict

    well_test.change_schedule(schedule=[[0, 1], [2, 3]])
    well_test.set_control(new_control=123456789)

    well_test = Well(
        name="well test",
        cell_group=np.array([[0.5, 0.5]]),
        radius=0.1,
        control={"Dirichlet": 1.0},
        s_inj=1.0,
        schedule=[[0, 0]],
        mode="injector",
    )
    well_test.set_control(new_control=123456789)
    well_test.set_ip(ip=123456789)
