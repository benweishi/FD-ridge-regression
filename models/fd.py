import numpy as np
from numpy.linalg import svd, inv, pinv
from scipy import linalg
import datetime


class FD():

    def __init__(self, d, ell=8):
        self.d = d
        self.reset(ell=ell)

    def reset(self, ell=None):
        if ell is not None:
            self.ell = ell
        self.X = np.zeros((self.ell, self.d))
        self.Vt = np.zeros((self.ell, self.d))
        self.s2 = np.zeros(self.ell)  # Sigma^2

    def feed(self, X):
        self.X = np.concatenate((self.X, X), axis=0)
        # TODO: try sklearn.decomposition.TruncatedSVD
        try:
            _, s, self.Vt = svd(self.X, full_matrices=False)
        except np.linalg.LinAlgError:
            np.save(f'./np_svd_fail{datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")}.npy', self.X)
            _, s, self.Vt = linalg.svd(self.X, full_matrices=False, lapack_driver='gesvd')
        self.s2 = s**2  # Sigma^2
        self._shrink()  # shrink self.s2 and self.Vt
        self.X = self.Vt * np.sqrt(self.s2).reshape(-1, 1)
        return self

    def result(self):
        return self.X, self.s2, self.Vt


class FFD(FD):

    def _shrink(self):
        self.s2 = self.s2[:self.ell] - self.s2[self.ell]
        self.Vt = self.Vt[:self.ell]


class iSVD(FD):

    def _shrink(self):
        self.s2 = self.s2[:self.ell]
        self.Vt = self.Vt[:self.ell]

# Robust FD
class RFD(FD):

    def __init__(self, d, ell):
        super(RFD, self).__init__(d, ell)
        self.delta = 0

    def reset(self, ell=None):
        super(RFD, self).reset(ell)
        self.delta = 0

    def _shrink(self):
        self.delta += self.s2[self.ell]/2
        self.s2 = self.s2[:self.ell] - self.s2[self.ell]
        self.Vt = self.Vt[:self.ell]

    def result(self):
        return self.X, self.s2, self.Vt[:self.ell], self.delta

# Near Optimal FD
class NOFD(FD):

    def __init__(self, d, ell=8, state=None):
        # ell = ell / 2, k = ell/3
        self.FD2 = FFD(d, ell//2)
        super(NOFD, self).__init__(d, ell//2)

    def reset(self, ell=None, state=None):
        ell = ell if ell is None else ell//2
        super(NOFD, self).reset(ell)
        self.FD2.reset(ell)  # 2nd level sketch, eps = 1/(ell-k)
        self.eps2 = (3/(2*self.FD2.ell))**2
        self.C = np.zeros((0, self.d))  # cache for 2nd level sketch
        self.F = 0

    def _shrink(self):
        rs2 = np.copy(self.s2)
        rs2[:self.ell] = self.s2[self.ell]
        self.F += np.sum(rs2)
        p = rs2 / self.F / self.eps2
        choice = np.random.uniform(len(p)) < p
        Vts = self.Vt[choice]
        s2s = rs2[choice]
        ps = p[choice]
        Cs = Vts * (np.sqrt(s2s) / np.sqrt(ps)).reshape(-1, 1)
        self.C = np.concatenate((self.C, Cs), axis=0)
        if len(self.C) >= self.FD2.ell:
            self.FD2.feed(self.C)
            self.C = np.zeros((0, self.d))
        self.s2 = self.s2[:self.ell] - self.s2[self.ell]
        self.Vt = self.Vt[:self.ell]

    def result(self):
        if len(self.C > 0):
            self.FD2.feed(self.C)
            self.C = np.zeros((0, self.d))
        Q, Qs2, QVt = self.FD2.result()
        return self.X, Q, self.s2, self.Vt[:self.ell], Qs2, QVt
