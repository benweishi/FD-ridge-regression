import numpy as np
from numpy.linalg import svd, inv, pinv
from models.fd import FFD, iSVD, RFD, NOFD

class FDRR():

    def __init__(self, d, ell):
        self.d = d
        self.ell = ell
        self.Xt_y = np.zeros(d)

    def reset(self, ell=None, state=None):
        self.Xt_y = np.zeros(self.d)
        if ell is not None:
            self.ell = ell
        self.fd.reset(self.ell)
        return self

    def fit(self, X, y):
        self.Xt_y += X.T @ y
        self.fd.feed(X)
        return self

    def coefs(self, gamma):
        return None

    def __str__(self):
        return f'{self.name}(d={self.d}, ell={self.ell})'


class FFDRR(FDRR):

    name = 'FDRR'

    def __init__(self, d, ell=8):
        super(FFDRR, self).__init__(d, ell)
        self.fd = FFD(d, ell)

    def coefs(self, gamma):
        _, s2, Vt = self.fd.result()
        v_part = Vt @ self.Xt_y
        rest_part = (self.Xt_y - Vt.T @ v_part) / gamma
        v_part = Vt.T @ (v_part / (s2 + gamma))
        return v_part + rest_part


class iSVDRR(FDRR):

    name = 'iSVDRR'

    def __init__(self, d, ell=8):
        super(iSVDRR, self).__init__(d, ell)
        self.fd = iSVD(d, ell)

    def coefs(self, gamma):
        _, s2, Vt = self.fd.result()
        v_part = Vt @ self.Xt_y
        rest_part = (self.Xt_y - Vt.T @ v_part) / gamma
        v_part = Vt.T @ (v_part / (s2 + gamma))
        return v_part + rest_part


class RFDRR(FDRR):

    name = 'RFDRR'

    def __init__(self, d, ell=8):
        super(RFDRR, self).__init__(d, ell)
        self.fd = RFD(d, ell)

    def coefs(self, gamma):
        _, s2, Vt, delta = self.fd.result()
        v_part = Vt @ self.Xt_y
        rest_part = (self.Xt_y - Vt.T @ v_part) / (gamma + delta)
        v_part = Vt.T @ (v_part / (s2 + gamma + delta))
        return v_part + rest_part


class NOFDRR(FDRR):

    name = '2LFDRR'

    def __init__(self, d, ell=8):
        super(NOFDRR, self).__init__(d, ell)
        self.fd = NOFD(d, ell)

    def coefs(self, gamma):
        _, _, s2, Vt, s22, V2t = self.fd.result()
        '''
        if any(np.isnan(s2)):
            print("s2 is nan")
        if any(np.isnan(Vt)):
            print("Vt is nan")
        if any(np.isnan(s22)):
            print("s22 is nan")
        if any(np.isnan(V2t)):
            print("V2t is nan")
        '''
        M = (Vt.T * s2) @ Vt + (V2t.T * s22) @ V2t + gamma * np.identity(self.d)
        try:
            coefs = inv(M) @ self.Xt_y
        except np.linalg.LinAlgError as err:
            print(err)
            np.save('./errorM.npy', M)
        return coefs
