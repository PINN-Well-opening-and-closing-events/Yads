from typing import Tuple
import numpy as np
import subprocess as sp
import os


def sliding_window(
    field: np.array, window_shape: Tuple[int, int], padding: Tuple[int, int]
) -> np.array:
    all_windows = np.lib.stride_tricks.sliding_window_view(
        x=field, window_shape=window_shape
    )
    windows_padded = all_windows[:: padding[0], :: padding[1]]
    return windows_padded


if __name__ == "__main__":
    phi = np.loadtxt("../SPE10/data/spe_phi.dat")
    K = np.loadtxt("../SPE10/data/spe_perm.dat")
    # (z, y, x)
    phi = phi.reshape((85, 220, 60))
    K = K.reshape((3, 85, 220, 60))

    ws = (21, 21)
    pad = (10, 2)
    save_filename = f"window_shape_{ws[0]}_{ws[1]}_padding_{pad[0]}_{pad[1]}"
    save_dir = "../SPE10/slided_K_phi/" + save_filename
    if not os.path.isdir(save_dir):
        sp.call(f"mkdir {save_dir}", shell=True)
        sp.call(f"mkdir {save_dir}/K", shell=True)
        sp.call(f"mkdir {save_dir}/phi", shell=True)

    for layer in range(phi.shape[0]):
        save_dir_K_layer = f"K_{layer}"
        save_dir_phi_layer = f"phi_{layer}"

        full_K_dir = f"{save_dir}/K/{save_dir_K_layer}"
        full_phi_dir = f"{save_dir}/phi/{save_dir_phi_layer}"

        # sp.call(f"mkdir {full_K_dir}", shell=True)
        # sp.call(f"mkdir {full_phi_dir}", shell=True)

        # Kx
        K_layer = K[0, layer, :, :]
        phi_layer = phi[layer, :, :]

        windows_K_layer = sliding_window(field=K_layer, window_shape=ws, padding=pad)
        windows_K_layer = windows_K_layer.reshape(
            windows_K_layer.shape[0] * windows_K_layer.shape[1],
            windows_K_layer.shape[2],
            windows_K_layer.shape[3],
        )

        windows_phi_layer = sliding_window(
            field=phi_layer, window_shape=ws, padding=pad
        )
        windows_phi_layer = windows_phi_layer.reshape(
            windows_phi_layer.shape[0] * windows_phi_layer.shape[1],
            windows_phi_layer.shape[2],
            windows_phi_layer.shape[3],
        )

        with open(f"{full_K_dir}.npy", "wb") as f:
            np.save(f, windows_K_layer)

        with open(f"{full_phi_dir}.npy", "wb") as f:
            np.save(f, windows_phi_layer)
