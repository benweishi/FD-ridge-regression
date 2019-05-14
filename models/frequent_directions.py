import numpy as np
from numpy.linalg import svd, inv
import time


class FrequentDirections():

    def __init__(self, gamma, d, ell):
        self.d = d
        self.ell = ell
        self.gamma = gamma
        self.Xt_y = np.zeros(d)
        self.X_sketch = np.zeros((0, d))
        self.train_time = 0

    def partial_fit(self, X, y, sample_weight=None):
        start_time = time.time()
        self.X_sketch = np.concatenate((self.X_sketch, X))
        _, s, Vt = svd(self.X_sketch)
        self.Xt_y += y @ X
        if len(s) > self.ell:
            delt2 = s[self.ell]**2
            s = np.sqrt(s[:self.ell]**2 - delt2)
            self.X_sketch = Vt[:self.ell] * s.reshape(-1, 1)
        self.train_time += time.time() - start_time
        return self

    def get_params(self):
        coef = inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * self.gamma) @ self.Xt_y
        return coef

    def predict(self, X):
        coef = inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * self.gamma) @ self.Xt_y
        return X @ coef


class RobustFrequentDirections():

    def __init__(self, gamma, d, ell):
        self.d = d
        self.ell = ell
        self.alpha = gamma
        self.Xt_y = np.zeros(d)
        self.X_sketch = np.zeros((0, d))
        self.train_time = 0

    def partial_fit(self, X, y, sample_weight=None):
        start_time = time.time()
        self.X_sketch = np.concatenate((self.X_sketch, X))
        _, s, Vt = svd(self.X_sketch)
        self.Xt_y += y @ X
        if len(s) > self.ell:
            delt2 = s[self.ell]**2
            self.alpha += delt2/2
            s = np.sqrt(s[:self.ell]**2 - delt2)
            self.X_sketch = Vt[:self.ell] * s.reshape(-1, 1)
        self.train_time += time.time() - start_time
        return self

    def get_params(self):
        coef = inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * self.alpha) @ self.Xt_y
        return coef

    def predict(self, X):
        coef = inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * self.alpha) @ self.Xt_y
        return X @ coef


class ISVD():

    def __init__(self, gamma, d, ell):
        self.d = d
        self.ell = ell
        self.gamma = gamma
        self.Xt_y = np.zeros(d)
        self.X_sketch = np.zeros((0, d))
        self.train_time = 0

    def partial_fit(self, X, y, sample_weight=None):
        start_time = time.time()
        self.X_sketch = np.concatenate((self.X_sketch, X))
        _, s, Vt = svd(self.X_sketch)
        self.Xt_y += y @ X
        if len(s) > self.ell:
            self.X_sketch = Vt[:self.ell] * s[:self.ell].reshape(-1, 1)
        self.train_time += time.time() - start_time
        return self

    def get_params(self):
        coef = inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * self.gamma) @ self.Xt_y
        return coef

    def predict(self, X):
        coef = inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * self.gamma) @ self.Xt_y
        return X @ coef
