from abc import ABCMeta, abstractmethod
import numpy as np  # type: ignore
from typing import Tuple, Mapping, Iterable, Union, List, Dict
from yads.wells.well import Well


class Mesh(metaclass=ABCMeta):  # pragma: no cover
    @property
    @abstractmethod
    def type(self) -> str:
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @property
    @abstractmethod
    def nb_cells(self) -> int:
        pass

    @property
    @abstractmethod
    def nb_nodes(self) -> int:
        pass

    @property
    @abstractmethod
    def nb_faces(self) -> int:
        pass

    @property
    @abstractmethod
    def node_coordinates(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def cells(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def cell_face_connectivity(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def cell_node_connectivity(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def face_groups(self) -> Mapping[str, np.ndarray]:
        pass

    @abstractmethod
    def centers(self, item: str) -> np.ndarray:
        pass

    @abstractmethod
    def measures(self, item: str) -> np.ndarray:
        pass

    @abstractmethod
    def faces(self, group: str, with_nodes: bool) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def nb_groups(self) -> int:
        pass

    @property
    @abstractmethod
    def nb_boundary_faces(self) -> int:
        pass

    @abstractmethod
    def group(self, face: int) -> str:
        pass

    @abstractmethod
    def face_to_cell(self, face: int, face_type: str) -> Union[Tuple[int, int], int]:
        pass

    @abstractmethod
    def cell_to_face(self, cell) -> List:
        pass

    @property
    @abstractmethod
    def cell_groups(self):
        pass

    @abstractmethod
    def add_cell_group_by_index(self, name: str, cells_idx: List[int]) -> None:
        pass

    @abstractmethod
    def add_cell_group_by_coord(self, name: str, coord: np.ndarray) -> None:
        pass

    @abstractmethod
    def add_cell_group_by_square(
        self, name: str, up_left: Tuple, down_right: Tuple
    ) -> None:
        pass

    @abstractmethod
    def connect_well(self, well: Well):
        pass

    @abstractmethod
    def add_face_group_by_line(self, name: str, point_1, point_2):
        pass

    @abstractmethod
    def change_measures(self, item, new_measure: np.ndarray, return_item=False):
        pass

    @abstractmethod
    def find_cells_inside_square(self, up_left, down_right):
        pass

    @abstractmethod
    def to_json(self, json_path: str):
        pass
