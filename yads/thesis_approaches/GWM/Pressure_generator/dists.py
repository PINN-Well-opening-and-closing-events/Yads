import numpy as np


def angle_between_points(A, B):
    return np.abs(np.arctan2(B[1], B[0]) - np.arctan2(A[1], A[0]))


def circle_dist(A, B, radius):
    angle = angle_between_points(A, B)
    return angle * radius


def dist(X1, X2):
    return (X1 - X2)**2
