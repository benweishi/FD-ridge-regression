import numpy as np
from numpy.linalg import svd, inv, pinv


class IterativeRegression():

    def __init__(self, d, ell):
        self.d = d
        self.ell = ell
        self.alpha = 0
        self.Xt_y = np.zeros(d)
        self.Vt = np.zeros((ell, d))
        self.s = np.zeros(ell)
        self.X_sketch = np.zeros((2*self.ell, self.d))
        self.cache_y = np.zeros(self.ell)
        self.cache_idx = 0

    def partial_fit(self, X, y):
        n = len(y)
        self.X_sketch[self.cache_idx+self.ell:self.ell+self.cache_idx+n] = X
        self.cache_y[self.cache_idx:self.cache_idx+n] = y
        self.cache_idx += n
        if self.cache_idx >= self.ell:
            self._sketch()
        return self

    def _sketch(self):
        self.Xt_y += self.cache_y @ self.X_sketch[self.ell:]
        self.cache_y *= 0
        # TODO: try sklearn.decomposition.TruncatedSVD
        _, s, Vt = svd(self.X_sketch, full_matrices=False)
        self.s = self._shrink(s)
        self.Vt = Vt[:self.ell]
        self.X_sketch[:self.ell] = self.Vt * self.s.reshape(-1, 1)
        self.cache_idx = 0

    # abstractmethod
    def _shrink(self, s):
        # Shrink sketch rank
        pass

    def compute_coef(self, gamma, robust=False):
        if self.cache_idx != 0:
            self._sketch()
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
