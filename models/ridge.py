import numpy as np
from numpy.linalg import inv


class Ridge():

    def __init__(self, d):
        self.d = d
        self.Xt_X = np.zeros((d, d))
        self.Xt_y = np.zeros(d)

    def partial_fit(self, X, y):
        self.Xt_X += X.T @ X
        self.Xt_y += X.T @ y
        return self

    def fit(self, X, y):
        self.Xt_X = X.T @ X
        self.Xt_y = y @ X
        return self

    def compute_coef(self, gamma=1):
        coef = self.Xt_y @ inv(self.Xt_X + np.identity(self.d) * gamma)
        return coef

    def predict(self, X, gamma=None):
        coef = self.get_coef(gamma)
        return X @ coef
