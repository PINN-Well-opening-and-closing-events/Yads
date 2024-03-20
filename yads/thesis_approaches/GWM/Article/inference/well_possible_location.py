import matplotlib.pyplot as plt
import numpy as np
from yads.mesh.utils import load_json
from yads.mesh import Mesh

grid = load_json("../../../../../meshes/SHP_CO2/2D/SHP_CO2_2D_S.json")

# Permeability barrier zone creation
barrier_1 = grid.find_cells_inside_square((1000.0, 2000.0), (1250.0, 0))
barrier_2 = grid.find_cells_inside_square((2250.0, 3000.0), (2500.0, 1000.0))
barrier_3 = grid.find_cells_inside_square((3500.0, 3000.0), (3750.0, 500.0))

phi = 0.2
# Porosity
phi = np.full(grid.nb_cells, phi)
# Diffusion coefficient (i.e Permeability)
K = np.full(grid.nb_cells, 100.0e-15)
permeability_barrier = 1.0e-15
K[barrier_1] = permeability_barrier
K[barrier_2] = permeability_barrier
K[barrier_3] = permeability_barrier

well_x, well_y = 1475, 2225
grid_dxy = 50
d = 4

# pre-compute cells the are on boundary (forbidden)
fbd_cells = []


for group in grid.face_groups.keys():
    if group != "0":
        for f in grid.faces(group=group, with_nodes=False):
            c = grid.face_to_cell(f, face_type="boundary")
            fbd_cells.append(c)
print("Forbidden cells computed")


def is_well_loc_ok(well_loc, grid: Mesh, K):
    well_x, well_y = well_loc
    cells_d = grid.find_cells_inside_square(
        (grid_dxy * (well_x / grid_dxy - d), grid_dxy * (well_y / grid_dxy + d)),
        (grid_dxy * (well_x / grid_dxy + d), grid_dxy * (well_y / grid_dxy - d)),
    )

    # Ensure the local domain has the correct size
    if len(cells_d) != (2 * d + 1) * (2 * d + 1):
        return False
    # Ensure local domain does not touch boundary
    for c in cells_d:
        if c in fbd_cells:
            return False
    # Ensure no permeability barrier are reached
    if not np.all(K[cells_d] == 100.0e-15):
        return False
    return True


print(is_well_loc_ok(well_loc=(well_x, well_y), grid=grid, K=K))

cell_centers = grid.centers(item="cell")

well_locs_cell_idxs = [
    i
    for i, _ in enumerate(grid.cells)
    if is_well_loc_ok(grid.centers(item="cell")[i], grid, K)
]
print(len(well_locs_cell_idxs))

np.random.seed(42)
well_loc_samples = np.random.randint(low=0, high=len(well_locs_cell_idxs) - 1, size=200)


well_loc_samples_cell_idxs = [well_locs_cell_idxs[i] for i in well_loc_samples]

well_locs_plot = np.zeros(grid.nb_cells)
well_locs_plot[well_locs_cell_idxs] = 1
well_locs_plot[barrier_1] = 2
well_locs_plot[barrier_2] = 2
well_locs_plot[barrier_3] = 2
well_locs_plot[well_loc_samples_cell_idxs] = 3
well_locs_plot = np.reshape(well_locs_plot, (95, 60)).T

import matplotlib

# extract all colors from the .jet map
cmaplist = [
    (0.298, 0.447, 0.690, 1.0),
    (0.333, 0.658, 0.407, 1.0),
    (0.867, 0.514, 0.322, 1.0),
    (0.768, 0.305, 0.321, 1.0),
]
print(cmaplist)

# create the new map
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist)

fig, ax = plt.subplots(1, 1)
ax.imshow(well_locs_plot, cmap=cmap)

ax.invert_yaxis()
ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
plt.show()
