import numpy as np


def cov_matrix_exp(X, dist_func, cor_d, radius, grid_nb_faces, **kwargs):
    dim = X.shape[0]
    K = np.eye(dim)
    cor_dist = cor_d[0] * 2*np.pi*radius*np.sqrt(2)/grid_nb_faces
    for i in range(dim):
        for j in range(dim):
            if j != i:
                d = dist_func(X[i], X[j], radius=radius*np.sqrt(2))
                K[i][j] = np.exp(- d / cor_dist)
    return K


def cov_matrix_Id(X, **kwargs):
    dim = X.shape[0]
    K = np.eye(dim)
    for i in range(dim):
        for j in range(dim):
            if j != i:
                K[i][j] = 0.
    return K


def cov_matrix_P_dist(X, lhd, dist_func, cor_d, radius):
    dim = X.shape[0]
    K = np.eye(dim)
    cor_dist = cor_d[0] * 2 * np.pi * radius * np.sqrt(2)
    cor_P = (1/cor_d[1])
    for i in range(dim):
        for j in range(dim):
            if j != i:
                d = dist_func(X[i], X[j], radius=radius*np.sqrt(2))
                P_dist = np.abs(lhd[i] - lhd[j])
                K[i][j] = np.sign(lhd[j] - lhd[i]) * np.exp(- ((d / cor_dist) + (cor_P / P_dist)))
    return K
