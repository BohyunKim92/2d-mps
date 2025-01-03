import numpy as np

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mps import Solver, \
    generate_boundary_sample_points_from_path, \
    generate_interior_sample_points_from_path, \
    path_from_vertices  # noqa

alpha = 5 / 4


def decagon_vertices():
    r = 1
    theta = (np.pi / 5) * np.arange(10)
    x, y = r*np.cos(theta), r*np.sin(theta)
    return np.stack((x, y), axis=1)


vertices = decagon_vertices()
pivot_vertices = vertices
path = path_from_vertices(vertices)
rotations = np.pi * np.concatenate((
    np.arange(-3, -5, -1),
    np.arange(5, -3, -1),
)) / 5


def create_solver(m_B_per_edge, m_I, N):
    '''Given the parameters, constructs and returns a Solver'''
    alpha_k = np.array([alpha] * 10).reshape(-1, 1, 1) * (np.arange(N) + 1)
    r_i_boundary, theta_i_boundary = generate_boundary_sample_points_from_path(
        vertices, m_B_per_edge, pivot_vertices, rotations)
    r_i_interior, theta_i_interior = generate_interior_sample_points_from_path(
        vertices, m_I, pivot_vertices, rotations)
    return Solver(
        alpha_k,
        np.concatenate((r_i_boundary, r_i_interior), axis=1),
        np.concatenate((theta_i_boundary, theta_i_interior), axis=1),
        r_i_boundary.shape[1],
    )
