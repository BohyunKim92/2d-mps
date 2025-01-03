import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import multiprocessing
import base


class ZFinder:
    def __init__(self, solver, X, Y):
        self.solver = solver
        self.X = X
        self.Y = Y

    def find_Z(self, eigenmode):
        ustar = self.solver.find_ustar(
            eigenmode, base.pivot_vertices, base.rotations)

        Z = ustar(self.X, self.Y)
        Z[~base.path.contains_points(
            np.stack((self.X.reshape(-1), self.Y.reshape(-1)), axis=1)
        ).reshape(self.X.shape)] = np.nan
        Z -= np.nanmin(Z)
        Z /= np.nanmax(Z) / 2
        Z -= 1

        return Z


def plot_eigenfunctions():
    solver = base.create_solver(100, 40, 100)
    lam_indices = np.array([1, 2, 3, 4, 5, 6, 10, 25, 50])
    df = pd.read_csv('eigenmodes.csv')
    lams = np.array(df['eigenmode'][lam_indices - 1])

    x = np.linspace(-1, 1, 200)
    y = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(x, y)

    z_finder = ZFinder(solver, X, Y)

    with multiprocessing.Pool() as pool:
        Zs = pool.map(
            z_finder.find_Z,
            lams,
        )

    fig, axs = plt.subplots(3, 3, figsize=(12, 12), layout='constrained')
    for i, (lam, Z, idx) in enumerate(zip(lams, Zs, lam_indices)):
        ax = axs[i // 3][i % 3]
        # outline of L shape
        ax.add_patch(patches.PathPatch(
            base.path, edgecolor='black', facecolor='none', linewidth=2))
        ax.contour(X, Y, Z, levels=np.linspace(-1, 1, 9))
        ax.set_title(f'$\\lambda_{{{idx}}}={lam:.8}$')
        ax.set_aspect('equal')
        ax.set_axis_off()

    norm = colors.Normalize(vmin=-1, vmax=1)
    sm = cm.ScalarMappable(norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axs, shrink=0.75, pad=0.03)
    plt.savefig('eigenfunctions.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 16, 'font.weight': 'bold'})
    plot_eigenfunctions()
