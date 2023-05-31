import numpy as np

from matplotlib import pyplot as plt
from matplotlib import colors

phi = np.loadtxt("data/spe_phi.dat")
# (x, y, z)
phi = phi.reshape((85, 220, 60))

K = np.loadtxt("data/spe_perm.dat")
# ((Kx, Ky, Kz), z, y, x)
K = K.reshape((3, 85, 220, 60))
# darcy -> m^2
K *= 0.97 * 10e-12

Kmax = np.argmax(K[0, :, :, :].reshape(85 * 220 * 60))
Kmean = np.mean(phi[:, :, :].reshape(85 * 220 * 60))
print(Kmean)
print(Kmax, phi.reshape(85 * 220 * 60)[Kmax])

K = np.log10(K)

layer = 84
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 24), sharey="all")

ax1.imshow(phi[layer, :, :], cmap="jet", aspect="auto")

ax2.imshow(K[0, layer, :, :], cmap="jet", aspect="auto")

ax3.imshow(K[1, layer, :, :], cmap="jet", aspect="auto")

ax4.imshow(K[2, layer, :, :], cmap="jet", aspect="auto")

fig.axes[1].invert_yaxis()
fig.axes[2].invert_yaxis()
fig.axes[0].invert_yaxis()
fig.axes[3].invert_yaxis()

ax1.title.set_text(f"Porosity of SPE10 layer {layer}")
ax2.title.set_text(r"$log10(K_{x})$")
ax3.title.set_text(r"$log10(K_{y})$")
ax4.title.set_text(r"$log10(K_{z})$")

# plt.colorbar()
# plt.show()
