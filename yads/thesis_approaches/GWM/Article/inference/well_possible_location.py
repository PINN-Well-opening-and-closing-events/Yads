import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib.patches import Patch

import sys

sys.path.append("/home/AD.NORCERESEARCH.NO/anle/Yads")

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
fbd_cells_1 = []
for group in grid.face_groups.keys():
    if group != "0":
        for f in grid.faces(group=group, with_nodes=False):
            c = grid.face_to_cell(f, face_type="boundary")
            fbd_cells_1.append(c)
# pre-compute cells the are two cells away from boundary (forbidden)
fbd_cells_2 = []
for group in grid.face_groups.keys():
    if group == "0":
        for f in grid.faces(group=group, with_nodes=False):
            c1, c2 = grid.face_to_cell(f, face_type="inner")
            # check if one of the cells is on the boundary
            if c1 in fbd_cells_1:
                fbd_cells_2.append(c2)
            elif c2 in fbd_cells_1:
                fbd_cells_2.append(c1)

# Stay far from boundary conditions
fbd_cells_3 = list(grid.find_cells_inside_square((0.0, 1000.0), (1250.0, 0)))
fbd_cells_4 = list(grid.find_cells_inside_square((2500.0, 3000.0), (3500.0, 2500.0)))
fbd_cells_5 = list(grid.find_cells_inside_square((3750.0, 3000.0), (4500.0, 0.0)))

fbd_cells = fbd_cells_1 + fbd_cells_2

fbd_cells = list(dict.fromkeys(fbd_cells))
print("Forbidden cells computed")


def is_well_loc_ok(well_loc, grid: Mesh, K, i):
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

    cells_d_plus_i = grid.find_cells_inside_square(
        (
            grid_dxy * (well_x / grid_dxy - (d + i)),
            grid_dxy * (well_y / grid_dxy + (d + i)),
        ),
        (
            grid_dxy * (well_x / grid_dxy + (d + i)),
            grid_dxy * (well_y / grid_dxy - (d + i)),
        ),
    )
    # Ensure no permeability barrier are reached
    if not np.all(K[cells_d_plus_i] == 100.0e-15):
        return False
    return True


cell_centers = grid.centers(item="cell")

well_locs_cell_idxs = [
    i
    for i, _ in enumerate(grid.cells)
    if is_well_loc_ok(grid.centers(item="cell")[i], grid, K, 1)
]
print(len(well_locs_cell_idxs))

np.random.seed(42)
nb_well_samples = 200
print(f"Sampling {nb_well_samples} well locations in the location pool")
well_loc_samples = np.random.randint(
    low=0, high=len(well_locs_cell_idxs) - 1, size=nb_well_samples
)
well_loc_samples_cell_idxs = [well_locs_cell_idxs[i] for i in well_loc_samples]


well_locs_plot = np.zeros(grid.nb_cells)
well_locs_plot[well_locs_cell_idxs] = 1
well_locs_plot[barrier_1] = 2
well_locs_plot[barrier_2] = 2
well_locs_plot[barrier_3] = 2
well_locs_plot[well_loc_samples_cell_idxs] = 3

well_locs_plot = np.reshape(well_locs_plot, (95, 60)).T

# Define your four colors (colorblind-friendly)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#e5e5e5"]
legend_labels = [
    "No sampling locations",
    "Possible sampling locations",
    "Permeability barriers",
    "Sampled locations",
]
cmap = ListedColormap(colors)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.imshow(well_locs_plot, cmap=cmap)
ax.invert_yaxis()
# Remove tick labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

# Set the color of the axis spine and ticks to black
ax.spines["top"].set_linewidth(2)
ax.spines["right"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)

# Create a custom legend outside the plot
# legend_elements = [Patch(facecolor=color, edgecolor='black', label=legends_label[i]) for i, color in enumerate(colors)]
# Plot empty scatter plot for each label to create legend
for i, label in enumerate(legend_labels):
    ax.scatter([], [], color=colors[i], label=label)
# Add legend below the plot
legend = ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize="large"
)
plt.setp(legend.get_texts(), fontweight="bold", fontsize=16)
legend.get_frame().set_linewidth(2)  # Set legend border width
legend.get_frame().set_edgecolor("black")  # Set legend border color

# Adjust layout to accommodate legend
plt.tight_layout()

plt.savefig("well_possible_locations_2.pdf")
plt.show()
