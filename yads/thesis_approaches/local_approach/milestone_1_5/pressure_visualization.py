import numpy as np
from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt

save_dir = "P_imp_naive_generation/"
file = f"P_{23}.csv"
df = pd.read_csv(
    save_dir + file, sep="\t", converters={"P": literal_eval, "Pb": literal_eval}
)
P = np.array(df["P"].values[0]).reshape(51, 51)
Pb_dict = df["Pb"][0]

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plt.imshow(P, vmin=100e5, vmax=200e5)
plt.colorbar()
plt.axis("off")

plt.text(20, 52, int(Pb_dict["lower"]), fontdict={"fontsize": 12})
plt.text(20, -1, int(Pb_dict["upper"]), fontdict={"fontsize": 12})
plt.text(-3, 29, int(Pb_dict["left"]), fontdict={"fontsize": 12, "rotation": 90})
plt.text(51, 29, int(Pb_dict["right"]), fontdict={"fontsize": 12, "rotation": -90})
# plt.show()


def grad_P(Pimp, h=20):
    well_x, well_y = np.unravel_index(Pimp.argmax(), Pimp.shape)
    P_right = Pimp[well_y][well_x:]
    P_left = Pimp[well_y][0 : well_x + 1]
    P_up = Pimp.T[well_y][0 : well_x + 1]
    P_down = Pimp.T[well_y][well_x:]
    P_up_left = np.diagonal(P)[: well_x + 1]
    P_down_right = np.diagonal(P)[well_x:]
    P_up_right = np.diagonal(np.fliplr(P))[: well_x + 1]
    P_down_left = np.diagonal(np.fliplr(P))[well_x:]

    def grad_compute(P_dir, d):
        # centered grad
        grad_P_dir = np.mean(
            [(P_dir[i + 1] - P_dir[i - 1]) / (2 * d) for i in range(1, len(P_dir) - 1)]
        )
        # all grads are negative from mid to boundary
        return -np.abs(grad_P_dir)

    grad_P_right = grad_compute(P_right, h)
    grad_P_left = grad_compute(P_left, h)
    grad_P_up = grad_compute(P_up, h)
    grad_P_down = grad_compute(P_down, h)
    grad_P_up_left = grad_compute(P_up_left, h)
    grad_P_down_right = grad_compute(P_down_right, h)
    grad_P_up_right = grad_compute(P_up_right, h)
    grad_P_down_left = grad_compute(P_down_left, h)
    return (
        grad_P_right,
        grad_P_left,
        grad_P_up,
        grad_P_down,
        grad_P_up_left,
        grad_P_down_right,
        grad_P_up_right,
        grad_P_down_left,
    )


print(grad_P(Pimp=P))
plt.show()
