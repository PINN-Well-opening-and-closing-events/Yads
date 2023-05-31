import copy

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian
import random


def angle_between_points(A, B):
    return np.abs(np.arctan2(B[1], B[0]) - np.arctan2(A[1], A[0]))


def circle_dist(A, B, radius):
    angle = angle_between_points(A, B)
    return angle * radius


def dist(X1, X2):
    return (X1 - X2)**2


def cov_matrix(X, dist_func, cor_d):
    dim = X.shape[0]
    K = np.eye(dim)
    for i in range(dim):
        for j in range(dim):
            if j != i:
                d = dist_func(X[i], X[j], radius=radius*np.sqrt(2))
                K[i][j] = np.exp(- d / cor_d)
    return K


def cov_matrix_P_dist(X, P, dist_func, cor_d):
    dim = X.shape[0]
    K = np.eye(dim)
    for i in range(dim):
        for j in range(dim):
            if j != i:
                d = dist_func(X[i], X[j], radius=radius*np.sqrt(2))
                P_dist = np.square(P[i] - P[j])

                K[i][j] = np.sign(P[j] - P[i]) * np.exp(- ((d / cor_d) + 20e5/P_dist))
    return K


def P_generator(nb_bds, nb_s, coords, cov_mat_fun, cor_d, P_min=10e6, P_max=20e6):
    lhd = lhs(n=nb_bds, samples=nb_s, criterion="maximin")
    all_P_scaled = []
    for i in range(nb_s):
        P_unscaled = np.matmul(cov_mat_fun(coords, dist_func=circle_dist, cor_d=cor_d), lhd[i, :])
        P_scaled = P_min + P_unscaled * (P_max - P_min)
        all_P_scaled.append(P_scaled)
    return all_P_scaled


Lx, Ly = 3, 3
Nx, Ny = 5, 5
nb_bd_faces = 2 * Nx + 2 * Ny
grid = create_2d_cartesian(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
groups = ["lower", "left", "upper", "right"]
# get center of boundary faces
face_center_coords = []
for group in groups:
    cell_idxs = grid.face_groups[group][:, 0]
    for coord in grid.centers(item="face")[cell_idxs]:
        face_center_coords.append(list(coord))

nb_boundaries = 3
boundaries = random.sample(range(4), nb_boundaries)
# mesh a 5 boundary faces on each boundary
random_faces = np.random.randint(0, Nx, size=len(boundaries))
rd_faces_coord = []

for i, bound in enumerate(boundaries):
    rd_face_idx = grid.face_groups[groups[bound]][random_faces[i]][0]
    rd_face_coord = grid.centers(item="face")[rd_face_idx]
    rd_faces_coord.append(rd_face_coord)

nb_samples = 2
radius = Lx/2

circle_coord_x = (np.array(rd_faces_coord)[:, 0] - Lx/2) * radius * np.sqrt(2) / \
                 np.sqrt((np.array(rd_faces_coord)[:, 0] - Lx/2)**2 + (np.array(rd_faces_coord)[:, 1] - Ly/2)**2)
circle_coord_y = (np.array(rd_faces_coord)[:, 1] - Ly/2) * radius * np.sqrt(2) / \
                 np.sqrt((np.array(rd_faces_coord)[:, 0] - Lx/2)**2 + (np.array(rd_faces_coord)[:, 1] - Ly/2)**2)

circle_coords = np.array([[circle_coord_x[i], circle_coord_y[i]] for i in range(len(circle_coord_x))])


# Ps = P_generator(nb_boundaries, nb_s=nb_samples, coords=circle_coords, cov_mat_fun=cov_matrix, P_min=10e6, P_max=20e6)
lhd = lhs(n=nb_boundaries, samples=nb_samples, criterion="maximin")
all_P_scaled = []
for i in range(nb_samples):
    P_unscaled = np.matmul(cov_matrix(circle_coords,
                                      dist_func=circle_dist,
                                      cor_d=2*np.pi*radius*np.sqrt(2)/nb_bd_faces),
                           lhd[i, :])
    P_scaled = 10e6 + P_unscaled * (20e6 - 10e6)
    all_P_scaled.append(P_scaled)

# Ps = P_generator(nb_bds=nb_boundaries,
#                  nb_s=nb_samples,
#                  coords=circle_coords,
#                  cov_mat_fun=cov_matrix,
#                  cor_d=2*np.pi*radius*np.sqrt(2)/nb_bd_faces,
#                  P_min=10e6, P_max=20e6)

Ps = copy.deepcopy(all_P_scaled)

all_P_scaled = []
for i in range(nb_samples):
    P_unscaled = np.matmul(cov_matrix(circle_coords, dist_func=circle_dist, cor_d=1e-6), lhd[i, :])
    P_scaled = 10e6 + P_unscaled * (20e6 - 10e6)
    all_P_scaled.append(P_scaled)
P_nulls = copy.deepcopy(all_P_scaled)

all_P_scaled = []

for i in range(nb_samples):
    P_unscaled = np.matmul(cov_matrix_P_dist(X=circle_coords, P=P_nulls[0],
                                             dist_func=circle_dist,
                                             cor_d=2*np.pi*radius*np.sqrt(2)/nb_bd_faces),
                           lhd[i, :])
    P_scaled = 10e6 + P_unscaled * (20e6 - 10e6)
    all_P_scaled.append(P_scaled)
P_d_p = copy.deepcopy(all_P_scaled)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))


### ax1 plot
rect = matplotlib.patches.Rectangle((-Lx/2, -Ly/2), Lx, Ly, linewidth=1, edgecolor='black', facecolor='none')
ax1.add_patch(rect)

for i, P in enumerate(P_nulls[0]):
    ax1.text(rd_faces_coord[i][0] - Lx/2, rd_faces_coord[i][1] - Ly/2, s=f"{P/1e6:.1f}")


ax1.scatter(x=np.array(face_center_coords)[:, 0] - Lx/2, y=np.array(face_center_coords)[:, 1] - Ly/2)
ax1.scatter(x=np.array(rd_faces_coord)[:, 0] - Lx/2, y=np.array(rd_faces_coord)[:, 1] - Ly/2)

ax1.set_xlim([-Lx, Lx])
ax1.set_ylim([-Ly, Ly])

ax1.title.set_text(r'$cov(Xi, Xj) = \mathbf{I_n}  $')

### ax2 plot
rect = matplotlib.patches.Rectangle((-Lx/2, -Ly/2), Lx, Ly, linewidth=1, edgecolor='black', facecolor='none')
circle = plt.Circle((0., 0.), radius*np.sqrt(2), color='r', fill=False)
ax2.add_patch(circle)
ax2.add_patch(rect)

ax2.scatter(x=np.array(face_center_coords)[:, 0] - Lx/2, y=np.array(face_center_coords)[:, 1] - Ly/2)
ax2.scatter(x=np.array(rd_faces_coord)[:, 0] - Lx/2, y=np.array(rd_faces_coord)[:, 1] - Ly/2)
ax2.scatter(circle_coord_x, circle_coord_y)

for i, P in enumerate(P_nulls[0]):
    ax2.text(circle_coord_x[i], circle_coord_y[i], s=f"{P/1e6:.1f}")

ax2.scatter(0, 0)
ax2.set_xlim([-Lx, Lx])
ax2.set_ylim([-Ly, Ly])

ax2.set_xticks([])
ax2.set_yticks([])

ax2.title.set_text(r'$cov(X_i, X_j) = \mathbf{I_n}  $')
################# ax3

rect = matplotlib.patches.Rectangle((-Lx/2, -Ly/2), Lx, Ly, linewidth=1, edgecolor='black', facecolor='none')
circle = plt.Circle((0., 0.), radius*np.sqrt(2), color='r', fill=False)
ax3.add_patch(circle)
ax3.add_patch(rect)

ax3.scatter(x=np.array(face_center_coords)[:, 0] - Lx/2, y=np.array(face_center_coords)[:, 1] - Ly/2)
ax3.scatter(x=np.array(rd_faces_coord)[:, 0] - Lx/2, y=np.array(rd_faces_coord)[:, 1] - Ly/2)
ax3.scatter(circle_coord_x, circle_coord_y)

for i, P in enumerate(Ps[0]):
    ax3.text(circle_coord_x[i], circle_coord_y[i], s=f"{P/1e6:.1f}")

ax3.scatter(0, 0)
ax3.set_xlim([-Lx, Lx])
ax3.set_ylim([-Ly, Ly])

ax3.set_xticks([])
ax3.set_yticks([])

ax3.title.set_text(r'$cov(X_i, X_j) = \exp(-d(X_i, X_j))$')

################ ax4
rect = matplotlib.patches.Rectangle((-Lx/2, -Ly/2), Lx, Ly, linewidth=1, edgecolor='black', facecolor='none')
circle = plt.Circle((0., 0.), radius*np.sqrt(2), color='r', fill=False)
ax4.add_patch(circle)
ax4.add_patch(rect)

ax4.scatter(x=np.array(face_center_coords)[:, 0] - Lx/2, y=np.array(face_center_coords)[:, 1] - Ly/2)
ax4.scatter(x=np.array(rd_faces_coord)[:, 0] - Lx/2, y=np.array(rd_faces_coord)[:, 1] - Ly/2)
ax4.scatter(circle_coord_x, circle_coord_y)

for i, P in enumerate(P_d_p[0]):
    ax4.text(circle_coord_x[i], circle_coord_y[i], s=f"{P/1e6:.1f}")

ax4.scatter(0, 0)
ax4.set_xlim([-Lx, Lx])
ax4.set_ylim([-Ly, Ly])

ax4.set_xticks([])
ax4.set_yticks([])

ax4.title.set_text(r'$cov(Xi, Xj) = -sign(P(X_i) - P(X_j)) '
                   r'\exp(- (\frac{d(X_i, X_j)}{\Theta} + \frac{\mu}{(P(X_i) - P(X_j))^2})) $')

plt.show()
