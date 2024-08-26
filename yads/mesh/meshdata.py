import numpy as np  # type: ignore
import json

from typing import Tuple, Union, List, Dict

from yads.mesh import Mesh
from yads.wells import Well


class MeshData(Mesh):
    def __init__(
        self,
        dim: int,
        nb_cells: int,
        nb_nodes: int,
        nb_faces: int,
        node_coordinates: np.ndarray,
        cells: np.ndarray,
        faces: Union[None, np.ndarray],
        cell_centers: np.ndarray,
        face_centers: np.ndarray,
        cell_measures: np.ndarray,
        face_measures: np.ndarray,
        face_groups: Dict[str, np.ndarray],
        cell_face_connectivity,
        cell_node_connectivity,
    ) -> None:
        """
        Args:
            dim: dimension of the mesh
            nb_cells: mesh number of cells
            nb_nodes: mesh number of nodes
            nb_faces: mesh number of faces
            node_coordinates: mesh node coordinate expressed as nd.array depending of the mesh dimension
                ex 1D: np.ndarray([x_n1,x_n2,x_n3,...])
                ex 2D: np.ndarray([(x_n1,y_n1),(x_n2, y_n2)])
            cell_centers: coordinates of cell centers np.ndarray
            face_centers: coordinates of face centers np.ndarray
            cell_measures: volume of cells np.ndarray
            face_measures: surface of faces np.ndarray
            face_groups: group label of faces Mapping[str, np.ndarray]
                ex : {'inner': [...]
                      'bound': [...]}
            cell_node_connectivity: connectivity between cells and nodes np.ndarray
            cell_face_connectivity: connectivity between cells and faces np.ndarray
        """
        self._dim = dim
        self._nb_cells = nb_cells
        self._nb_nodes = nb_nodes
        self._nb_faces = nb_faces
        self._node_coordinates = node_coordinates
        self._cells = cells
        self._faces = faces
        self._cell_centers = cell_centers
        self._face_centers = face_centers
        self._cell_measures = cell_measures
        self._face_measures = face_measures
        self._face_groups = face_groups
        self._cell_face_connectivity = cell_face_connectivity
        self._cell_node_connectivity = cell_node_connectivity
        self._cell_groups: Dict[str : List[int]] = {}
        self._type: str = self.find_type()

    @property
    def type(self):
        return self._type

    @property
    def dim(self):
        return self._dim

    @property
    def nb_cells(self):
        return self._nb_cells

    @property
    def nb_nodes(self):
        return self._nb_nodes

    @property
    def nb_faces(self):
        return self._nb_faces

    @property
    def node_coordinates(self):
        return self._node_coordinates

    @property
    def cells(self):
        return self._cells

    @property
    def cell_face_connectivity(self):
        return self._cell_face_connectivity

    @property
    def cell_node_connectivity(self):
        return self._cell_node_connectivity

    @property
    def face_groups(self):
        return self._face_groups

    @property
    def nb_boundary_faces(self) -> int:
        count = 0
        for group in self.face_groups.keys():
            if group != "0":
                count += len(self.faces(group=group, with_nodes=False))
        return count

    def find_type(self):
        # only cartesian and triangular meshes are handled
        # triangular = 3 faces
        # cartesian = 4 faces
        if len(self.cells) > 0:
            if len(self.cells[0]) == 4:
                return "cartesian"
            return "triangular"
        # no cells -> cartesian with only one cell
        return "cartesian"

    def centers(self, item: str) -> np.ndarray:
        if item == "cell":
            return self._cell_centers
        elif item == "face":
            return self._face_centers
        else:
            raise ValueError(f"centers is valid for 'cell' or 'face' (got: {item})")

    def measures(self, item: str) -> np.ndarray:
        if item == "cell":
            return self._cell_measures
        elif item == "face":
            return self._face_measures
        else:
            raise ValueError(f"measures is valid for 'cell' or 'face' (got: {item})")

    def faces(self, group: str, with_nodes: bool = False) -> np.ndarray:
        try:
            if with_nodes:
                return self._face_groups[group]
            else:
                return self._face_groups[group][:, 0]
        except KeyError:
            raise ValueError(
                f"unknown group : valid are {self._face_groups.keys()} (got: {group})"
            )

    def face_to_cell(self, face: int, face_type: str) -> Union[Tuple[int, int], int]:
        """
        :param face: face index must be 0 <= face <= nb_faces
        :param face_type: inner or boundary
        :return: cell(s) containing the face
                2 cells index if inner face
                1 cell index if boundary face
        """
        if face > self.nb_faces or face < 0:
            raise ValueError(
                f"unknown face : face index out of range (accepted range is [{0}:{self.nb_faces})"
            )
        if face_type not in ["inner", "boundary"]:
            raise ValueError(
                f"invalid face_type, valid face_type are 'inner' and 'boundary'."
            )
        for f in self._cell_face_connectivity:
            if self._type == "cartesian":
                if f[1] == face:
                    if face_type == "inner":
                        return f[0]
                    elif face_type == "boundary":
                        return f[0][0]
            else:
                if f[1] == face:
                    if face_type == "inner":
                        return f[0]
                    elif face_type == "boundary":
                        return f[0][0]
        return -999

    def cell_to_face(self, cell) -> List:
        """

        :param cell: cell index
        :return: list of faces composing the cell
            3 elements if triangular mesh
            4 elements if cartesian mesh
        """
        if cell > self.nb_cells or cell < 0:
            raise ValueError(
                f"unknown cell : cell index out of range (accepted range is [{0}:{self.nb_cells})"
            )
        faces_conn = []
        vcs = self.cells[cell]
        nb_faces_found = 0
        for group in self.face_groups.keys():
            for face in self.face_groups[group]:
                # if 2 vertex in common -> face in common
                if len(np.intersect1d(vcs, face[1:])) == 2:
                    # append the index of the face
                    faces_conn.append(face[0])
                    nb_faces_found += 1
        assert nb_faces_found in [1, 3, 4]
        return faces_conn

    @property
    def nb_groups(self) -> int:
        return len(self._face_groups)

    def group(self, face: int) -> List[str]:
        """

        :param face: face index
        :return: face group(s)
        """
        # we assume that one face can belong to multiple groups
        if face > self.nb_faces or face < 0:
            raise ValueError(
                f"unknown face : face index out of range (accepted range is [{0}:{self.nb_faces})"
            )
        else:
            groups = []
            for group in self._face_groups.keys():
                for f in self._face_groups[group]:
                    if f[0] == face:
                        groups.append(str(group))
        return groups

    @property
    def cell_groups(self):
        return self._cell_groups

    def add_cell_group_by_index(self, name: str, cells_idx: List[int]) -> None:
        """Create a new cell group using cells indexes

        Args:
            name: name of  the cell group
            cells_idx: cell indexes belonging to the new cell group
        """
        assert np.all(
            [0 <= idx <= self._nb_cells - 1 for idx in cells_idx]
        )  # all index must be in [0, nb_cells]

        if name not in self._cell_groups.keys():
            self._cell_groups[name] = cells_idx

        return

    def add_cell_group_by_coord(self, name: str, coord: np.ndarray) -> None:
        """Create a new group using coordinates.
        For each coordinate, we look for the closest cell center and add its cell index to the cell group

        Args:
            name: name of  the cell group
            coord: array of coord
        """
        cells_idx = []
        for c in coord:
            # calculate distances to cell cell centers
            dists = [np.linalg.norm(c - i) for i in self._cell_centers]
            # add the cell index of the minimal distance
            cells_idx.append(np.argmin(dists))

        if name not in self._cell_groups.keys():
            self._cell_groups[name] = cells_idx
        return

    def add_cell_group_by_square(
        self, name: str, up_left: Tuple, down_right: Tuple
    ) -> None:
        """Create a new cell group using a square. All cells within this square are added to the group

        Args:
            name: name of  the cell group
            up_left: coord (x,y) of the up left point of the square
            down_right: coord (x,y) of the down right point of the square
        """
        cells = []
        for cell, coord in enumerate(self._cell_centers):
            # check if cell_center coord is in the square
            if (
                up_left[0] <= coord[0] <= down_right[0]
                and down_right[1] <= coord[1] <= up_left[1]
            ):
                cells.append(cell)
        if name not in self._cell_groups.keys():
            self._cell_groups[name] = cells
        return

    def connect_well(self, well: Well) -> None:
        """

        :param well: Well object
        :return:
        """
        self.add_cell_group_by_coord(well.name, well.cell_group)
        if self._type == "cartesian":
            coord = self.node_coordinates
            Lx = max(coord[:, 0])
            Ly = max(coord[:, 1])
            nx_cells, ny_cells = (
                len(list(dict.fromkeys(coord[:, 0]))) - 1,
                len(list(dict.fromkeys(coord[:, 1]))) - 1,
            )
            dx = Lx / nx_cells
            dy = Ly / ny_cells
            well.set_dx(dx)
            well.set_dy(dy)
        else:
            well.set_dx(1.0)
            well.set_dy(1.0)
        return

    def add_face_group_by_line(self, name: str, point_1, point_2):
        if self._type != "cartesian":
            print("this method only works for cartesian meshes")
            return
        if point_1[0] == point_2[0]:
            line = "vertical"
        elif point_1[1] == point_2[1]:
            line = "horizontal"
        else:
            print(
                "the 2 points must form a horizontal or vertical line (i.e one component in common)."
            )
            return

        faces = []
        if line == "horizontal":
            for group in ["upper", "lower"]:
                for face in self._face_groups[group]:
                    idx = face[0]
                    if (
                        point_1[0] <= self._face_centers[idx][0] <= point_2[0]
                        and point_1[1] == self._face_centers[idx][1]
                    ):
                        faces.append(face)
        else:
            for group in ["left", "right"]:
                for face in self._face_groups[group]:
                    idx = face[0]
                    if (
                        point_1[1] <= self._face_centers[idx][1] <= point_2[1]
                        and point_1[0] == self._face_centers[idx][0]
                    ):
                        faces.append(face)
        if name not in self._face_groups.keys():
            self._face_groups[name] = np.array(faces)
        return

    def change_measures(self, item: str, new_measure: np.ndarray, return_item=False):
        if new_measure.shape != self._cell_measures.shape:
            print(f"new measure must have the same shape as the actual measure ")
        if item == "cell":
            self._cell_measures = new_measure
            if return_item:
                return self._cell_measures
        elif item == "face":
            if return_item:
                return self._face_measures
        else:
            raise ValueError(f"measures is valid for 'cell' or 'face' (got: {item})")
        return

    def find_cells_inside_square(self, up_left, down_right):
        cells = []
        for cell, coord in enumerate(self.centers(item="cell")):
            # check if cell_center coord is in the square
            if (
                up_left[0] <= coord[0] <= down_right[0]
                and down_right[1] <= coord[1] <= up_left[1]
            ):
                cells.append(cell)
        return cells

    def to_json(self, json_path: str):
        save_dict = {
            "dimension": self._dim,
            "nb_cells": self._nb_cells,
            "nb_nodes": self._nb_nodes,
            "nb_faces": self._nb_faces,
            "node_coordinates": self._node_coordinates.tolist(),
            "cells": self._cells.tolist(),
            "faces": self._faces.tolist(),
            "cell_centers": self._cell_centers.tolist(),
            "face_centers": self._face_centers.tolist(),
            "cell_measures": self._cell_measures.tolist(),
            "face_measures": self._face_measures.tolist(),
            "face_groups": {
                key: value.tolist() for (key, value) in self._face_groups.items()
            },
            "cell_face_connectivity": self._cell_face_connectivity,
            "cell_node_connectivity": [
                [item[0], int(item[1])] for item in self.cell_node_connectivity
            ],
            "cell_groups": {
                key: value.tolist() for (key, value) in self._cell_groups.items()
            },
            "type": self._type,
        }
        json.dump(save_dict, open(json_path + ".json", "w"))
