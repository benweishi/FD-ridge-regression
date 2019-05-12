import numpy as np
from numpy.linalg import svd, inv


class FrequentDirections():

    def __init__(self, gamma, d, ell):
        self.d = d
        self.ell = ell
        self.gamma = gamma
        self.Xt_y = np.zeros(d)
        self.X_sketch = np.zeros((0, d))

    def partial_fit(self, X, y, sample_weight=None):
        self.X_sketch = np.concatenate((self.X_sketch, X))
        _, s, Vt = svd(self.X_sketch)
        self.Xt_y += y @ X
        if len(s) <= self.ell:
            return self
        delt2 = s[self.ell]**2
        s = np.sqrt(s[:self.ell]**2 - delt2)
        self.X_sketch = Vt[:self.ell] * s.reshape(-1, 1)
        return self

    def get_params(self):
        coef = inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * self.gamma) @ self.Xt_y
        return coef

    def predict(self, X):
        coef = inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * self.gamma) @ self.Xt_y
        return X @ coef
