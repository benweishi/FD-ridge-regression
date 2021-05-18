import numpy as np
from numpy.linalg import inv, pinv


class ProjectionRegression():

    def __init__(self, d, ell):
        self.d = d
        self.ell = ell
        self.X_sketch = np.zeros((self.ell, self.d))
        self.y_sketch = np.zeros(ell)

    def partial_fit(self, X, y):
        return self

    def partial_fit_finish(self):
        return self

    def compute_coef(self, gamma):
        return inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * gamma) @ self.X_sketch.T @ self.y_sketch


class RandomProjections(ProjectionRegression):

    def __init__(self, d, ell):
        ProjectionRegression.__init__(self, d, ell)
        self.scale = 1 / np.sqrt(self.ell)

    def partial_fit(self, X, y):
        randomMatrix = np.random.choice([-1, 1], (self.ell, len(y)))
        self.X_sketch += self.scale * randomMatrix @ X
        self.y_sketch += self.scale * y @ randomMatrix.T
        return self

class Hashing(ProjectionRegression):

    def partial_fit(self, X, y):
        n = len(y)
        randomMatrix = np.zeros((self.ell, n))
        randomMatrix[np.random.randint(0, self.ell, n), range(n)] = np.random.choice([-1, 1], n)
        self.X_sketch += randomMatrix @ X
        self.y_sketch += randomMatrix @ y
        return self

