from typing import Union, Dict, List

import numpy as np  # type: ignore


class Well:
    def __init__(
        self,
        name: str,
        cell_group: np.ndarray,
        control: Dict,
        s_inj: Union[float, int],
        radius: Union[float, int],
        schedule: List[List[Union[float, int]]],
        mode: str,
    ) -> None:
        self._name: str = name
        self._cell_group: np.ndarray = cell_group
        self._mode: str = mode
        self._control: Dict = control
        self._s_inj: Union[float, int] = s_inj
        self._radius: Union[float, int] = radius
        self._schedule: List[List[Union[float, int]]] = schedule
        self._ip: Union[float, int] = -999
        self._dx: Union[float, int] = 0.0
        self._dy: Union[float, int] = 0.0

    @property
    def is_closed(self) -> bool:
        if self._mode == "closed":
            return True
        return False

    @property
    def is_injector(self) -> bool:
        if self._mode == "injector":
            return True
        return False

    @property
    def is_productor(self) -> bool:
        if self._mode == "productor":
            return True
        return False

    @property
    def mode(self):
        return self._mode

    @property
    def control(self):
        return self._control

    @property
    def injected_saturation(self):
        return self._s_inj

    @property
    def name(self) -> str:
        return self._name

    @property
    def cell_group(self) -> np.ndarray:
        return self._cell_group

    @property
    def radius(self) -> Union[float, int]:
        return self._radius

    @property
    def schedule(self) -> List[List[int]]:
        return self._schedule

    @property
    def ip(self):
        return self._ip

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    def set_ip(self, ip):
        self._ip = ip
        return

    def change_mode(self, mode: str):
        if mode in ["injector", "productor", "closed"]:
            self._mode = mode
        return

    def change_schedule(self, schedule: List[List[Union[float, int]]]):
        self._schedule = schedule
        return

    def set_dx(self, dx):
        self._dx = dx
        return

    def set_dy(self, dy):
        self._dy = dy
        return

    def set_control(self, new_control):
        if "Neumann" in self._control.keys():
            self._control["Neumann"] = new_control
        elif "Dirichlet" in self._control.keys():
            self._control["Dirichlet"] = new_control
        return

    def well_to_dict(self):
        return {
            "name": self._name,
            "cell_group": self._cell_group.tolist(),
            "mode": self._mode,
            "control": self._control,
            "s_inj": self._s_inj,
            "radius": self._radius,
            "schedule": self._schedule,
            "ip": self._ip,
            "dx": self._dx,
            "dy": self._dy,
        }
