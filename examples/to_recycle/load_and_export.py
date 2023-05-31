from yads.mesh import export_vtk
import numpy as np  # type: ignore
from yads.mesh import load_mesh
from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian
from yads.mesh.utils import load_json

# Create one Dimension example
grid = load_mesh("../meshes/1D/50nc_1d_line.mesh")

# cell property
K = np.random.random_sample((grid.nb_cells,)) + 1.0e-3

# Export with property
export_vtk("./test_1D", grid, cell_data={"K": K})

# Load a 2D mesh
grid = load_mesh("../meshes/2D/Tests/rod_3_1/rod_3_1.mesh")

# Generate some random properties
S = np.random.random_sample((grid.nb_cells,)) + 1.0 - 3
P = np.random.random_sample((grid.nb_cells,)) + 1.0 - 3

# Export with properties
export_vtk("./test_2D", grid, cell_data={"S": S, "P": P})

grid = create_2d_cartesian(3, 2, 3, 2)

# Generate some random properties
S = np.random.random_sample((grid.nb_cells,)) + 1.0 - 3
P = np.random.random_sample((grid.nb_cells,)) + 1.0 - 3

export_vtk("./test_cartesian", grid, cell_data={"S": S, "P": P})

grid = create_2d_cartesian(4750, 3000, 19, 12)
grid.to_json("./test_json")
grid = load_json("./test_json.json")

if __name__ == "__main__":
    import subprocess as sp

    sp.call("rm ./test_2D.vtu", shell=True)
    sp.call("rm ./test_1D.vtr", shell=True)
    sp.call("rm ./test_cartesian.vts", shell=True)
    sp.call("rm ./test_json.json", shell=True)
