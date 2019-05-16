import numpy as np
from numpy.linalg import inv
from models.linear_regression import LinearRegression


class ProjectionRegression(LinearRegression):

    def __init__(self, d, ell, gamma=0):
        LinearRegression.__init__(self, d, ell, gamma)
        self.X_sketch = np.zeros((self.ell, self.d))
        self.y_sketch = np.zeros(ell)

    # abstractmethod
    def _randomMatrix(self):
        # Shrink sketch rank
        pass

    def _coef(self, gamma):
        return inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * gamma) @ self.X_sketch.T @ self.y_sketch


class RandomProjections(ProjectionRegression):

    def __init__(self, gamma, d, ell):
        ProjectionRegression.__init__(self, d, ell, gamma)
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
