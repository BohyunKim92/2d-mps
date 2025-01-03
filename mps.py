import numpy as np
import scipy
from matplotlib.path import Path


def A(lam, alpha_k, r_i, theta_i):
    '''Evaluates A(lambda)

    Parameters:
    lam (float): the value of lambda
    alpha_k (array_like with shape (s, 1, N)): which multiples of alpha to use
    r (array_like with shape (s, m, 1)): the radii of the sample points to use
    theta (array_like with shape (s, m, 1)): the angles of the sample points to
        use

    Returns:
    An m x s*N matrix
    '''
    stacked = scipy.special.jv(alpha_k, np.sqrt(lam)*r_i) * \
        np.sin(alpha_k*theta_i)
    stacked[np.isclose(theta_i, 0)[:, :, 0]] = 0
    stacked[np.isclose(theta_i, 2*np.pi)[:, :, 0]] = 0
    stacked[np.isclose(theta_i, np.pi / alpha_k)[:, :, 0]] = 0
    # flattens the stacked matrices by putting them side-by-side in a single
    # matrix
    return stacked.transpose(1, 0, 2).reshape(r_i.shape[1], -1)


class Solver:
    '''A simple object holding the parameters for the approximation

    Attributes:
        alpha_k (array_like with shape (s, 1, N)): which multiples of alpha to
            use
        r (array_like with shape (s, m, 1)): the radii of the sample points to
            use
        theta (array_like with shape (s, m, 1)): the angles of the sample
            points to use
        m_B (float): the number of sample points on the boundary
    '''

    def __init__(self, alpha_k, r, theta, m_B):
        alpha_k = np.asarray(alpha_k)
        r = np.asarray(r)
        theta = np.asarray(theta)

        # alpha_k should be s x 1 x N
        assert len(alpha_k.shape) == 3
        assert alpha_k.shape[1] == 1

        # r and theta should be s x m x 1
        assert len(r.shape) == 3
        assert r.shape[2] == 1
        assert r.shape == theta.shape

        # s should match for alpha_k, r, and theta
        assert alpha_k.shape[0] == r.shape[0]

        # m_B should at most the number of sample points we got
        assert m_B <= r.shape[1]

        self.alpha_k = alpha_k
        self.r = r
        self.theta = theta
        self.m_B = m_B

    @property
    def m_I(self):
        '''The number of sample points on the interior'''
        return self.m - self.m_B

    @property
    def m(self):
        '''The total number of sample points'''
        return self.r.shape[1]

    @property
    def N(self):
        '''The number of Fourier-Bessel functions being used in the
        approximation'''
        return self.alpha_k.shape[2]

    @property
    def s(self):
        '''The number of singular points'''
        return self.alpha_k.shape[0]

    def A(self, lam):
        '''Evaluates A(lambda) for the specified approximation parameters

        Parameters:
        lam (float): the value of lambda

        Returns:
        An mxN matrix
        '''
        return A(lam, self.alpha_k, self.r, self.theta)

    def __column_pivoted_qr(self, lam):
        '''Returns A(lambda), Q_B, R, P'''
        A_lambda = self.A(lam)
        Q, R, P = scipy.linalg.qr(A_lambda, pivoting=True)
        Q_B = Q[:self.m_B, :A_lambda.shape[1]]
        Q_B = Q_B[:, np.abs(np.diag(R)) >= 1e-8]
        R = R[:Q_B.shape[1], :Q_B.shape[1]]
        P = P[:Q_B.shape[1]]
        return Q_B, R, P

    def sigma(self, lam):
        '''Finds the minimum singular value of Q_B(lambda)

        Parameters:
        lam (float): the value of lambda

        Returns:
        A float
        '''
        Q_B, _, _ = self.__column_pivoted_qr(lam)

        sing_vals = np.linalg.svd(Q_B, compute_uv=False, hermitian=False)
        return sing_vals[-1]

    def find_lambda(self, lower_bound, upper_bound):
        '''Approximates the eigenmode between lower_bound and upper_bound of
        the Laplace operator on the domain described by the given sample points

        Parameters:
        lower_bound (float): the lower bound of the search space
        upper_bound (float): the upper bound of the search space

        Returns:
        A float
        '''
        result = scipy.optimize.minimize(
            self.sigma, ((lower_bound+upper_bound)/2,), method='Nelder-Mead',
            options={'maxfev': 2000}, tol=1e-13,
            bounds=((lower_bound, upper_bound),))
        if not result.success:
            raise Exception(f"couldn't find the eigenmode: {result.message}")
        # # false positive
        # if result.fun > 1e-7:
        #     return None

        return result.x[0]

    def find_eigenmodes(self, num_eigenmodes,
                        search_window=20.0, search_interval=0.2):
        '''Finds the first num_eigenmodes eigenmodes of the domain described by
        the given sample points

        Parameters:
        num_eigenmodes (int): The number of eigenmodes to compute
        search_window (float): The width the window to use when searching for
            eigenmodes. The window will be shifted to the right until the
            requested number of eigenmodes has been found
        search_interval (float): Within a search window, the interval to use
            between samples of sigma

        Returns:
        A numpy array of the first num_eigenmodes eigenmodes
        '''
        vec_sigma = np.vectorize(self.sigma)

        eigenmodes = []
        lambdas = np.arange(
            0, search_window, search_interval) + search_interval
        while len(eigenmodes) < num_eigenmodes:
            sigmas = vec_sigma(lambdas)
            # argrelmin returns a tuple with one element, which is why we have
            # [0]
            local_minima_idx = scipy.signal.argrelmin(sigmas)[0]
            for local_minimum_idx in local_minima_idx:
                eigenmode = self.find_lambda(
                    lambdas[local_minimum_idx - 1],
                    lambdas[local_minimum_idx + 1])
                if eigenmode is None:
                    continue
                eigenmodes.append(eigenmode)
            lambdas += search_window

        return np.array(eigenmodes[:num_eigenmodes])

    def find_ustar(self, eigenmode, pivot_vertices, rotations):
        '''Given an eigenmode of the domain described by the given sample
        points, finds and returns u*

        Parameters:
        eigenmode (float): the eigenmode to use
        pivot_vertices (array_like with shape(s, 2)): the pivot vertices to
            use
        rotations (array_like with shape (s,)): the rotations to put the edges
            formed by the corresponding pivot vertex at theta = 0 and
            theta = pi/alpha

        Returns:
        A function which takes r, theta and returns u*(r, theta)
        '''
        pivot_vertices = np.asarray(pivot_vertices)
        rotations = np.asarray(rotations)
        assert pivot_vertices.shape == (self.s, 2)
        assert np.all((-np.pi <= rotations) & (rotations <= np.pi))
        assert rotations.shape == (self.s,)

        Q_B, R, P = self.__column_pivoted_qr(eigenmode)
        U, S, VT = np.linalg.svd(Q_B)
        v_tilde = VT[-1, :]
        c = scipy.linalg.solve_triangular(R, v_tilde)

        def ustar(x, y):
            x = np.asarray(x)
            y = np.asarray(y)

            final_shape = np.broadcast_shapes(x.shape, y.shape)
            x = np.broadcast_to(x, final_shape).reshape(-1, 1)
            y = np.broadcast_to(y, final_shape).reshape(-1, 1)

            x = x - pivot_vertices[:, 0].reshape(-1, 1, 1)
            y = y - pivot_vertices[:, 1].reshape(-1, 1, 1)

            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            theta += rotations.reshape(-1, 1, 1)
            theta[theta < 0] += 2*np.pi

            matrix = A(eigenmode, self.alpha_k, r, theta)[:, P]
            return (matrix @ c).reshape(final_shape)

        return ustar


def iter_segments(vertices):
    '''Iterates over the segments in an array of vertices

    Parameters:
    vertices (array_like with shape (n, 2)): the vertices of the polygon
    '''
    points = iter(vertices)
    first_point = next(points)
    for second_point in points:
        yield first_point, second_point
        first_point = second_point
    yield first_point, vertices[0]


def generate_boundary_sample_points_from_path(
        vertices, m_B_per_edge, pivot_vertices, rotations):
    '''Generates sample points on the boundary

    Parameters:
    vertices (array_like with shape (n, 2)): the vertices of the polygon
    m_B_per_edge (int): the number of sample points to generate on each edge
        of the L-shaped domain
    pivot_vertices (array_like with shape(s, 2)): the pivot vertices to use
    rotations (array_like): the rotations to put the edges formed by the
        corresponding pivot vertex at theta = 0 and theta = pi/alpha

    Returns:
    stacked column vectors r_i, theta_i
    '''
    vertices = np.asarray(vertices)
    assert len(vertices.shape) == 2
    assert vertices.shape[1] == 2

    pivot_vertices = np.asarray(pivot_vertices)
    assert len(pivot_vertices.shape) == 2
    assert pivot_vertices.shape[1] == 2
    assert pivot_vertices.shape[0] <= vertices.shape[0]

    rotations = np.asarray(rotations)
    assert np.all((-np.pi <= rotations) & (rotations <= np.pi))
    assert len(rotations.shape) == 1
    assert rotations.shape[0] == pivot_vertices.shape[0]

    xy = np.concatenate([
        (
            (end - start) *
            np.linspace(0, 1, m_B_per_edge + 2)[1:-1].reshape(-1, 1)
        ) + start
        for i, (start, end) in enumerate(iter_segments(vertices))
    ], axis=0)
    xy = xy - pivot_vertices.reshape(-1, 1, 2)

    r_i = np.sqrt(np.sum(xy**2, axis=2, keepdims=True))
    theta_i = np.arctan2(xy[:, :, [1]], xy[:, :, [0]])
    theta_i += rotations.reshape(-1, 1, 1)
    theta_i[theta_i < 0] += 2*np.pi

    return r_i, theta_i


def generate_interior_sample_points_from_path(
        vertices, m_I, pivot_vertices, rotations):
    '''Generates sample points on the interior

    Parameters:
    vertices (array_like with shape (n, 2)): the vertices of the polygon
    m_I (int): the number of sample points to generate
    pivot_vertices (array_like with shape(s, 2)): the pivot vertices to use
    rotations (array_like): the rotations to put the edges formed by the
        corresponding pivot vertex at theta = 0 and theta = pi/alpha

    Returns:
    stacked column vectors r_i, theta_i
    '''
    vertices = np.asarray(vertices)
    assert len(vertices.shape) == 2
    assert vertices.shape[1] == 2

    pivot_vertices = np.asarray(pivot_vertices)
    assert len(pivot_vertices.shape) == 2
    assert pivot_vertices.shape[1] == 2
    assert pivot_vertices.shape[0] <= vertices.shape[0]

    rotations = np.asarray(rotations)
    assert np.all((-np.pi <= rotations) & (rotations <= np.pi))
    assert len(rotations.shape) == 1
    assert rotations.shape[0] == pivot_vertices.shape[0]

    rng = np.random.default_rng()

    x_min = np.min(vertices[:, 0])
    x_max = np.max(vertices[:, 0])
    y_min = np.min(vertices[:, 1])
    y_max = np.max(vertices[:, 1])

    path = path_from_vertices(vertices)

    x = np.array([])
    y = np.array([])
    while len(x) < m_I or len(y) < m_I:
        new_x = rng.uniform(x_min, x_max, m_I)
        new_y = rng.uniform(y_min, y_max, m_I)
        is_in_domain = path.contains_points(np.stack((new_x, new_y), axis=1))
        x = np.concatenate((x, new_x[is_in_domain]))
        y = np.concatenate((y, new_y[is_in_domain]))
    x = x[:m_I, np.newaxis] - pivot_vertices[:, 0].reshape(-1, 1, 1)
    y = y[:m_I, np.newaxis] - pivot_vertices[:, 1].reshape(-1, 1, 1)

    r_i = np.sqrt(x**2 + y**2)
    theta_i = np.arctan2(y, x)
    theta_i += rotations.reshape(-1, 1, 1)
    theta_i[theta_i < 0] += 2*np.pi

    return r_i, theta_i


def path_from_vertices(vertices):
    '''Generates a matplotlib Path from an array of vertices

    Parameters:
    vertices (array_like with shape (n, 2)): the vertices of the polygon

    Returns:
    Path
    '''
    vertices = np.asarray(vertices)
    assert len(vertices.shape) == 2
    assert vertices.shape[1] == 2

    return Path(
        np.concatenate((vertices, [vertices[0]]), axis=0),
        [Path.MOVETO] + ([Path.LINETO] * (vertices.shape[0] - 1)
                         ) + [Path.CLOSEPOLY]
    )
