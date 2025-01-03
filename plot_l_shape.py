import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mps
import multiprocessing
import itertools


# pi/alpha is the angle of the domain, which in this case is an L shape
alpha = 2/3
vertices = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [-1, 1],
    [-1, -1],
    [0, -1],
])
lshape = mps.path_from_vertices(vertices)
pivot_vertices = [(0, 0)]
rotations = [0]


def create_solver(m_B_per_edge, m_I, N):
    '''Given the parameters, constructs and returns a Solver'''
    alpha_k = (alpha * (np.arange(N) + 1)).reshape(1, 1, -1)
    r_i_boundary, theta_i_boundary = \
        mps.generate_boundary_sample_points_from_path(
            vertices, m_B_per_edge, pivot_vertices, rotations)
    r_i_interior, theta_i_interior = \
        mps.generate_interior_sample_points_from_path(
            vertices, m_I, pivot_vertices, rotations)
    return mps.Solver(
        alpha_k,
        np.concatenate((r_i_boundary, r_i_interior), axis=1),
        np.concatenate((theta_i_boundary, theta_i_interior), axis=1),
        r_i_boundary.shape[1],
    )


def plot_sigma():
    lam = np.linspace(1, 20, 1000)
    solver = create_solver(50, 50, 30)

    vec_sigma = np.vectorize(solver.sigma)
    sigma_lam = vec_sigma(lam)
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(lam, sigma_lam)
    ax.set_xlabel(r'$\lambda$')
    ax.set_xlim((0, 20))
    ax.set_ylabel(r'$\sigma(\lambda)$')
    ax.set_ylim((0, 1))
    plt.savefig('lshaped_sigma.png',
                dpi=300, bbox_inches='tight')


def plot_cond():
    def cond_A(N, lam):
        solver = create_solver(N, N, N)
        A_lambda = solver.A(lam)
        Q, R = np.linalg.qr(A_lambda)
        Q_B = Q[:solver.m_B, :]
        return np.linalg.cond(Q_B)
    vec_cond_A = np.vectorize(cond_A)

    lam_1 = 9.6397238440219
    N = np.arange(1, 61)

    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(N, vec_cond_A(N, lam_1), '.-', label=r'$\lambda=\lambda_1$')
    ax.plot(N, vec_cond_A(N, lam_1 / 2), '.-',
            label=r'$\lambda=\frac{\lambda_1}{2}$')
    ax.set_xlabel('$N$')
    ax.set_ylabel(r'cond $Q_B(\lambda)$')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig('lshaped_cond.png',
                dpi=300, bbox_inches='tight')


def find_eigenmode_and_Z(approx_eigenmode, solver, X, Y):
    exact_eigenmode = solver.find_lambda(
        approx_eigenmode-0.1, approx_eigenmode+0.1),

    ustar = solver.find_ustar(exact_eigenmode, pivot_vertices, rotations)

    Z = ustar(X, Y)
    Z[~lshape.contains_points(
        np.stack((X.reshape(-1), Y.reshape(-1)), axis=1)
    ).reshape(X.shape)] = np.nan
    Z -= np.nanmin(Z)
    Z /= np.nanmax(Z) / 2
    Z -= 1

    return exact_eigenmode, Z


def plot_eigenfunctions():
    solver = create_solver(50, 50, 60)
    lam_indices = [1, 2, 3, 4, 5, 6, 20, 50, 104]
    approx_lams = [9.63972384, 15.19725193, 19.7392088, 29.52148111,
                   31.91263596, 41.47450989, 101.60529408, 250.7854811,
                   493.48022005]

    # the contour plot looks weird on the edges, so we exclude them with [1:-1]
    x = np.linspace(-1, 1, 200)[1:-1]
    y = np.linspace(-1, 1, 200)[1:-1]
    X, Y = np.meshgrid(x, y)

    with multiprocessing.Pool() as pool:
        eigenmodes_and_Z = pool.starmap(
            find_eigenmode_and_Z,
            zip(
                approx_lams,
                itertools.repeat(solver),
                itertools.repeat(X),
                itertools.repeat(Y),
            )
        )

    fig, axs = plt.subplots(3, 3, figsize=(12, 12), layout='constrained')
    for i, (((lam,), Z), idx) in enumerate(zip(eigenmodes_and_Z, lam_indices)):
        ax = axs[i // 3][i % 3]
        # outline of L shape
        ax.add_patch(patches.PathPatch(
            lshape, edgecolor='black', facecolor='none', linewidth=2))
        ax.contour(X, Y, Z, levels=np.linspace(-1, 1, 9))
        ax.set_title(f'$\\lambda_{{{idx}}}={lam:.8}$')
        ax.set_aspect('equal')
        ax.set_axis_off()

    norm = colors.Normalize(vmin=-1, vmax=1)
    sm = cm.ScalarMappable(norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axs, shrink=0.75, pad=0.03)
    plt.savefig('lshaped_eigenfunctions.png',
                dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 16, 'font.weight': 'bold'})
    plot_sigma()
    plot_cond()
    plot_eigenfunctions()
