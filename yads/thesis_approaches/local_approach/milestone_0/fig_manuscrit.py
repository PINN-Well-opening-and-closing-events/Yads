import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import json
from matplotlib import rc
import yads.mesh as ym

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'size': 12})
rc('figure', **{'figsize': (5, 3)})

with open('fig_manuscrit.json', "r") as f:
    debug_dict = json.load(f)
# debug_dict["newton_step_data"][step] = {
#     "P_i_plus_1": P.tolist(),
#     "S_i_plus_1": S.tolist(),
#     "Residual": B.tolist(),
# }
dt = 155103597.1
grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 200, 1)

V = grid.measures(item="cell")
print(debug_dict['newton_step_data']['1'].keys())
norms = []
for i, k in enumerate(debug_dict['newton_step_data'].keys()):
    B = debug_dict['newton_step_data'][k]['Residual']
    norm = np.max(np.abs(B) * dt / np.concatenate([V, V]))
    norms.append(norm)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax2.scatter(range(0, len(norms)), norms, s=15, marker='x', color='black')

for i, k in enumerate(debug_dict['newton_step_data'].keys()):
    if i in [0, 19, 39, 49, 62]:
        ax1.plot(range(0, len(debug_dict['newton_step_data'][k]['S_i_plus_1'])),
                 debug_dict['newton_step_data'][k]['S_i_plus_1'], label=f'iteration {k}')
        ax2.scatter(i, norms[i], s=75, marker='X', label=f'iteration {k}')

# ax1.plot(range(0, len(debug_dict['newton_step_data'][k]['S_i_plus_1'])),
#                  debug_dict['newton_step_data'][k]['S_i_plus_1'], label=f'iteration {k}')
# ax2.scatter(i, norms[i], s=75, marker='X', label=f'iteration {k}')

ax2.set_yscale('log')
ax2.set_title('Residual evolution through iterations')
ax1.set_title('Saturation evolution through iterations')
ax1.legend()
ax2.legend(loc='best')
plt.savefig('local_approach_test_1_sat_res.pdf', bbox_inches='tight')
plt.show()
