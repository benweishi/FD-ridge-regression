import numpy as np
from numpy.linalg import inv
from scipy.linalg import hadamard
import time


class RandomProjections():

    def __init__(self, gamma, d, ell):
        self.gamma = gamma
        self.d = d
        self.ell = ell
        self.scale = 1 / np.sqrt(self.ell)
        self.X_sketch = np.zeros((self.ell, self.d))
        self.y_sketch = np.zeros(ell)
        self.train_time = 0

    def partial_fit(self, X, y, sample_weight=None):
        start_time = time.time()
        n = len(X)
        randomVector = np.random.choice([-1, 1], (self.ell, n))
        self.X_sketch += self.scale * randomVector @ X
        self.y_sketch += self.scale * y @ randomVector.T
        self.train_time += time.time() - start_time
        return self

    def get_params(self):
        coef = inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * self.gamma) @ self.X_sketch.T @ self.y_sketch
        return coef

    def predict(self, X):
        coef = inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * self.gamma) @ self.X_sketch.T @ self.y_sketch
        return X @ coef


class Hashing():

    def __init__(self, gamma, d, ell):
        self.gamma = gamma
        self.d = d
        self.ell = ell
        self.X_sketch = np.zeros((self.ell, self.d))
        self.y_sketch = np.zeros(ell)
        self.train_time = 0

    def partial_fit(self, X, y, sample_weight=None):
        start_time = time.time()
        n = len(X)
        randomVector = np.zeros((self.ell, n))
        randomVector[np.random.randint(0, n, n), range(n)] = np.random.choice([-1.0, 1.0], n)
        self.X_sketch += randomVector @ X
        self.y_sketch += y @ randomVector.T
        self.train_time += time.time() - start_time
        return self

    def get_params(self):
        coef = inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * self.gamma) @ self.X_sketch.T @ self.y_sketch
        return coef

    def predict(self, X):
        coef = inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * self.gamma) @ self.X_sketch.T @ self.y_sketch
        return X @ coef
