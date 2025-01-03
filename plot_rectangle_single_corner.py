import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.pyplot as plt
import mps
import multiprocessing
import itertools


# pi/alpha is the angle of the domain, which in this case is a rectangle
alpha = 2
rectangle = Path([
    [0, 0],
    [3*np.pi, 0],
    [3*np.pi, 4*np.pi],
    [0, 4*np.pi],
    [0, 0],
], [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY,
])
pivot_vertices = [(0, 0)]
rotations = [0]


def create_solver(m_B_per_edge, m_I, N):
    '''Given the parameters, constructs and returns a Solver'''
    alpha_k = (alpha * (np.arange(N) + 1)).reshape(1, 1, -1)
    r_i_boundary, theta_i_boundary = \
        mps.generate_boundary_sample_points_from_path(
            rectangle, m_B_per_edge, pivot_vertices, rotations)
    r_i_interior, theta_i_interior = \
        mps.generate_interior_sample_points_from_path(
            rectangle, m_I, pivot_vertices, rotations)
    return mps.Solver(
        alpha_k,
        np.concatenate((r_i_boundary, r_i_interior), axis=1),
        np.concatenate((theta_i_boundary, theta_i_interior), axis=1),
        r_i_boundary.shape[1],
    )


def find_Z(eigenmode, solver, X, Y):
    ustar = solver.find_ustar(eigenmode, pivot_vertices, rotations)
    Z = ustar(X, Y)
    Z -= np.nanmin(Z)
    Z /= np.nanmax(Z) / 2
    Z -= 1
    return Z


def plot_sigma():
    lam = np.linspace(1, 5, 1000)
    solver = create_solver(100, 100, 30)

    vec_sigma = np.vectorize(solver.sigma)
    sigma_lam = vec_sigma(lam)
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(lam, sigma_lam)
    ax.set_xlabel(r'$\lambda$')
    ax.set_xlim((0, 5))
    ax.set_ylabel(r'$\sigma(\lambda)$')
    # ax.set_ylim((0, 1))
    plt.savefig('rectangle_sigma_single_corner.png',
                dpi=300, bbox_inches='tight')


def plot_eigenfunctions():
    solver = create_solver(50, 50, 60)
    eigenmodes = solver.find_eigenmodes(11, 1.5, 0.01)
    lam_indices = np.array([1, 3, 7, 2, 5, 9, 4, 6, 11])
    eigenmodes = eigenmodes[lam_indices - 1]

    # the contour plot looks weird on the edges, so we exclude them with [1:-1]
    x = np.linspace(0, 3*np.pi, 200)[1:-1]
    y = np.linspace(0, 4*np.pi, 200)[1:-1]
    X, Y = np.meshgrid(x, y)

    with multiprocessing.Pool() as pool:
        Zs = pool.starmap(
            find_Z,
            zip(
                eigenmodes,
                itertools.repeat(solver),
                itertools.repeat(X),
                itertools.repeat(Y),
            )
        )

    fig, axs = plt.subplots(3, 3, figsize=(12, 12), layout='constrained')
    for i, (lam, lam_idx, Z) in enumerate(zip(eigenmodes, lam_indices, Zs)):
        ax = axs[i // 3][i % 3]
        ax.add_patch(patches.Polygon([
            [0, 0],
            [3*np.pi, 0],
            [3*np.pi, 4*np.pi],
            [0, 4*np.pi],
            [0, 0],
        ], closed=True, edgecolor='black', facecolor='none', linewidth=2))
        ax.contour(X, Y, Z, levels=np.linspace(-1, 1, 9))
        ax.set_title(f'$\\lambda_{{{lam_idx}}}={lam:.8}$')
        ax.set_aspect('equal')
        ax.set_axis_off()

    norm = colors.Normalize(vmin=-1, vmax=1)
    sm = cm.ScalarMappable(norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axs, shrink=0.75, pad=0.07)
    plt.savefig('rectangle_eigenfunctions.png',
                dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 16, 'font.weight': 'bold'})
    plot_sigma()
