import numpy as np
from .base import TestFunction


class QUBO(TestFunction):
    """
    Quadratic Unconstrained Binary Optimization problem.
    Minimizes x^T Q x where Q is a random symmetric matrix.
    """
    problem_type = 'categorical'

    def __init__(self, dim=10, normalize=False, seed=None):
        super(QUBO, self).__init__(normalize)
        self.dim = dim
        self.dim_bi = dim
        self.n_vertices = np.full(dim, 2)
        self.config = self.n_vertices
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)
        np.random.seed(seed)
        self.Q = np.random.randn(dim, dim)
        self.Q = (self.Q + self.Q.T) / 2  # Ensure symmetry

        if self.normalize:
            self.mean, self.std = self.sample_normalize()
        else:
            self.mean, self.std = None, None

    def compute(self, X, normalize=True):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        res = np.array([x.T @ self.Q @ x for x in X])
        if normalize and self.mean is not None:
            res = (res - self.mean) / self.std
        return res

    def sample_normalize(self, size=None):
        if size is None:
            size = 2 * self.dim + 1
        y = []
        for _ in range(size):
            x = np.random.randint(0, 2, size=self.dim)
            y.append(self.compute(x, normalize=False))
        y = np.array(y)
        return np.mean(y), np.std(y)


class TrapFunction(TestFunction):
    """
    Deceptive trap function where all zeros is a local optimum
    and all ones is the global optimum.
    """
    problem_type = 'categorical'

    def __init__(self, dim=20, normalize=False):
        super(TrapFunction, self).__init__(normalize)
        self.dim = dim
        self.dim_bi = dim
        self.n_vertices = np.full(dim, 2)
        self.config = self.n_vertices
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        if self.normalize:
            self.mean, self.std = self.sample_normalize()
        else:
            self.mean, self.std = None, None

    def compute(self, X, normalize=True):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        sum_x = np.sum(X, axis=1)
        res = np.where(
            sum_x == 0, 1.0, np.where(
                sum_x == 1, 0.8, sum_x / self.dim
            )
        )
        if normalize and self.mean is not None:
            res = (res - self.mean) / self.std
        return -res

    def sample_normalize(self, size=None):
        if size is None:
            size = 2 * self.dim + 1
        y = []
        for _ in range(size):
            x = np.random.randint(0, 2, size=self.dim)
            y.append(self.compute(x, normalize=False))
        y = np.array(y)
        return np.mean(y), np.std(y)


class MaxCut(TestFunction):
    """
    Max-Cut problem on a randomly generated graph.
    The weight matrix W is symmetric with random entries.
    """
    problem_type = 'categorical'

    def __init__(self, dim=10, normalize=False, seed=None):
        super(MaxCut, self).__init__(normalize)
        self.dim = dim
        self.dim_bi = dim
        self.n_vertices = np.full(dim, 2)
        self.config = self.n_vertices
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)
        np.random.seed(seed)
        self.W = np.random.rand(dim, dim)  # Random weight matrix
        self.W = (self.W + self.W.T) / 2   # Symmetrize

        if self.normalize:
            self.mean, self.std = self.sample_normalize()
        else:
            self.mean, self.std = None, None

    def compute(self, X, normalize=True):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        res = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            res[i] = -np.sum(self.W * np.outer(x, (1 - x)))  # Minimize negative cut weight
        if normalize and self.mean is not None:
            res = (res - self.mean) / self.std
        return -res

    def sample_normalize(self, size=None):
        if size is None:
            size = 2 * self.dim + 1
        y = []
        for _ in range(size):
            x = np.random.randint(0, 2, size=self.dim)
            y.append(self.compute(x, normalize=False))
        y = np.array(y)
        return np.mean(y), np.std(y)