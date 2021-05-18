import numpy as np
from numpy.linalg import inv


class RR():

    name = 'RR'

    def __init__(self, d, ell=1):
        self.d = d
        self.reset()

    def reset(self, ell=None, state=None):
        self.Xt_X = np.zeros((self.d, self.d))
        self.Xt_y = np.zeros(self.d)

    def fit(self, X, y):
        self.Xt_X += X.T @ X
        self.Xt_y += X.T @ y
        return self

    def coefs(self, gamma=1):
        coef = inv(self.Xt_X + np.identity(self.d) * gamma) @ self.Xt_y
        return coef

    def __str__(self):
        return f'Ridge(d={self.d})'
