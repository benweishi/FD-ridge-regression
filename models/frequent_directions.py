import numpy as np
from numpy.linalg import svd, inv, pinv
from models.linear_regression import LinearRegression


class IterativeRegression():

    def __init__(self, d, ell):
        self.d = d
        self.ell = ell
        self.alpha = 0
        self.Xt_y = np.zeros(d)
        self.Vt = np.zeros((ell, d))
        self.s = np.zeros(ell)
        self.cache_X = np.empty((2*self.ell, self.d))
        self.cache_y = np.empty(2*self.ell)
        self.cache_idx = 0

    def partial_fit(self, X, y):
        self.Xt_y += y @ X
        n = len(y)
        self.cache_X[self.cache_idx:n] = X
        self.cache_idx += n
        if self.cache_idx == 2*self.ell:
            _, s, Vt = svd(self.X_sketch, full_matrices=False)
            self.s = self._shrink(s)
            self.Vt = Vt[:self.ell]
            self.cache_X[:self.ell] = self.Vt * self.s.reshape(-1, 1)
            self.cache_idx = self.ell
        return self

    # abstractmethod
    def _shrink(self, s):
        # Shrink sketch rank
        pass

    def compute_coef(self, gamma, robust=True):
        alpha = self.alpha if robust else 0
        v_space_part = self.Vt @ self.Xt_y
        rest_part = (self.Xt_y - self.Vt.T @ v_space_part) / (gamma + alpha)
        v_space_part = self.Vt.T @ (v_space_part / (self.s**2 + gamma + alpha))
        return v_space_part + rest_part


class FrequentDirections(IterativeRegression):

    def _shrink(self, s):
        delt2 = s[self.ell]**2
        self.alpha += delt2/2
        return np.sqrt(s[:self.ell]**2 - delt2)


class ISVD(IterativeRegression):

    def _shrink(self, s):
        self.alpha += s[self.ell]**2
        return s[:self.ell]
