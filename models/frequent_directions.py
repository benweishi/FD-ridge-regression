import numpy as np
from numpy.linalg import svd, inv
from models.linear_regression import LinearRegression


class IterativeRegression(LinearRegression):

    def __init__(self, d, ell, gamma=0):
        LinearRegression.__init__(self, d, ell, gamma)
        self.alpha = 0
        self.Xt_y = np.zeros(d)
        self.X_sketch = np.zeros((1, d))

    def _sketch(self, X, y):
        self.X_sketch = np.concatenate((self.X_sketch, X))
        self.Xt_y += y @ X
        _, s, Vt = svd(self.X_sketch)
        s = self._shrink(s)
        self.X_sketch = Vt[:self.ell] * s.reshape(-1, 1)

    # abstractmethod
    def _shrink(self, s):
        # Shrink sketch rank
        pass

    def _coef(self, gamma):
        return inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * (gamma + self.alpha)) @ self.Xt_y


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
