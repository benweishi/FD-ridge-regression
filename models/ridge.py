import numpy as np
from numpy.linalg import inv


class Ridge():

    def __init__(self, d, gamma=0):
        self.gamma = gamma
        self.d = d
        self.Xt_X = np.zeros((d, d))
        self.Xt_y = np.zeros(d)

    def partial_fit(self, X, y):
        self.Xt_X += X.T @ X
        self.Xt_y += X.T @ y.reshape((-1, 1))
        return self

    def fit(self, X, y):
        self.Xt_X = X.T @ X
        self.Xt_y = y @ X
        return self

    def get_params(self, gamma=None):
        if gamma is None:
            gamma = self.gamma
        coef = self.Xt_y @ inv(self.Xt_X + np.identity(self.d) * gamma)
        return coef

    def predict(self, X, gamma=None):
        coef = self.get_params(gamma)
        return X @ coef
