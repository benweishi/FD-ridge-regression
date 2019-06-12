import numpy as np
from numpy.linalg import inv, pinv


class ProjectionRegression():

    def __init__(self, d, ell):
        self.d = d
        self.ell = ell
        self.X_sketch = np.zeros((self.ell, self.d))
        self.y_sketch = np.zeros(ell)
        self.cache_X = np.empty((self.ell, self.d))
        self.cache_y = np.empty(self.ell)
        self.cache_idx = 0

    def partial_fit(self, X, y):
        n = len(y)
        self.cache_X[self.cache_idx:self.cache_idx+n] = X
        self.cache_y[self.cache_idx:self.cache_idx+n] = y
        self.cache_idx += n
        if self.cache_idx >= self.ell:
            self._sketch(self.cache_X, self.cache_y)
            self.cache_idx = 0
        return self

    def compute_coef(self, gamma):
        return inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * gamma) @ self.X_sketch.T @ self.y_sketch


class RandomProjections(ProjectionRegression):

    def __init__(self, d, ell):
        ProjectionRegression.__init__(self, d, ell)
        self.scale = 1 / np.sqrt(self.ell)

    def _sketch(self, X, y):
        randomMatrix = np.random.choice([-1, 1], (self.ell, len(X)))
        self.X_sketch += self.scale * randomMatrix @ X
        self.y_sketch += self.scale * y @ randomMatrix.T


class Hashing(ProjectionRegression):

    def _sketch(self, X, y):
        n = len(X)
        randomMatrix = np.zeros((self.ell, n))
        randomMatrix[np.random.randint(0, n, n), range(n)] = np.random.choice([-1.0, 1.0], n)
        self.X_sketch += randomMatrix @ X
        self.y_sketch += y @ randomMatrix.T
