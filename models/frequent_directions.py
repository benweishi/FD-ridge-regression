import numpy as np
from numpy.linalg import svd, inv, pinv
from models.linear_regression import LinearRegression


class IterativeRegression(LinearRegression):

    def __init__(self, d, ell, gamma=0):
        LinearRegression.__init__(self, d, ell, gamma)
        self.alpha = 0
        self.Xt_y = np.zeros(d)
        self.Vt = np.zeros((ell, d))
        self.s = np.zeros(ell)
        self.X_sketch = np.zeros((1, d))

    def _sketch(self, X, y):
        self.X_sketch = self.Vt * self.s.reshape(-1, 1)
        self.X_sketch = np.concatenate((self.X_sketch, X))
        self.Xt_y += y @ X
        _, s, Vt = svd(self.X_sketch, full_matrices=False)
        self.s = self._shrink(s)
        self.Vt = Vt[:self.ell]

    # abstractmethod
    def _shrink(self, s):
        # Shrink sketch rank
        pass

    def _coef(self, gamma):
        v_space_part = self.Vt @ self.Xt_y
        rest_part = (self.Xt_y - self.Vt.T @ v_space_part) / (gamma + self.alpha)
        v_space_part = self.Vt.T @ (v_space_part / (self.s**2 + gamma + self.alpha))
        return v_space_part + rest_part


class FrequentDirections(IterativeRegression):

    def _shrink(self, s):
        return np.sqrt(s[:self.ell]**2 - s[self.ell]**2)


class RobustFrequentDirections(IterativeRegression):

    def _shrink(self, s):
        delt2 = s[self.ell]**2
        self.alpha += delt2/2
        return np.sqrt(s[:self.ell]**2 - delt2)


class ISVD(IterativeRegression):

    def _shrink(self, s):
        return s[:self.ell]
