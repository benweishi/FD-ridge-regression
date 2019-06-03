import numpy as np
from numpy.linalg import svd, inv
import time


class LinearRegression():

    def __init__(self, d, ell, gamma=0):
        self.d = d
        self.ell = ell
        self.gamma = gamma
        self.train_time = 0
        self.coef = None

    def partial_fit(self, X, y):
        start_time = time.time()
        self._sketch(X, y)
        self.train_time += time.time() - start_time
        return self

    # abstractmethod
    def get_coef(self, gamma=None):
        if gamma is None:
            gamma = self.gamma
        if self.coef is not None and gamma == self.gamma:
            return self.coef
        self.gamma = gamma
        self.coef = self._coef(gamma)
        return self.coef

    def predict(self, X, gamma=None):
        coef = self.get_coef(gamma)
        return X @ coef

    # abstractmethod
    def _sketch(self, X, y):
        # Shrink sketch rank
        pass

    # abstractmethod
    def _coef(self, gamma):
        # Shrink sketch rank
        pass

    def get_time(self):
        start_time = time.time()
        self._coef(1)
        return self.train_time + (time.time() - start_time)
