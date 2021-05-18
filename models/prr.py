import numpy as np
from numpy.linalg import inv, pinv
from sklearn.utils import check_random_state


class PRR():

    def __init__(self, d, ell):
        self.d = d
        self.ell = ell
        self.state = None
        self.random = check_random_state(None)
        self.reset(None, None)

    def reset(self, ell=None, state=None):
        if ell is not None:
            self.ell = ell
        if self.ell is not None:
            self.X_sketch = np.zeros((self.ell, self.d))
            self.y_sketch = np.zeros(self.ell)
        if state is not None:
            self.state = state
        if self.state is not None:
            self.random = check_random_state(state)

    def fit(self, X, y):
        return self

    def coefs(self, gamma):
        return inv(self.X_sketch.T @ self.X_sketch + np.identity(self.d) * gamma) @ self.X_sketch.T @ self.y_sketch

    def __str__(self):
        return f'{self.name}(d={self.d}, ell={self.ell})'


class RPRR(PRR):

    name = 'RPRR'

    def __init__(self, d, ell):
        PRR.__init__(self, d, ell)
        self.scale = 1 / np.sqrt(self.ell)

    def reset(self, ell=None, state=None):
        PRR.reset(self, ell, state)
        if ell is not None:
            self.scale = 1 / np.sqrt(self.ell)

    def fit(self, X, y):
        randomMatrix = self.random.choice([-1, 1], (self.ell, len(y)))
        self.X_sketch += self.scale * randomMatrix @ X
        self.y_sketch += self.scale * y @ randomMatrix.T
        return self

class CSRR(PRR):

    name = 'CSRR'

    def fit(self, X, y):
        n = len(y)
        randomMatrix = np.zeros((self.ell, n))
        randomMatrix[self.random.randint(0, self.ell, n), range(n)] = self.random.choice([-1, 1], n)
        self.X_sketch += randomMatrix @ X
        self.y_sketch += randomMatrix @ y
        return self

class LSRR():
    """Leverage Score sampling ridge regression.
    Online Row Sampling. Michael B. Cohen MIT∗, Cameron Musco, Jakub Pachocki

    Args:
        d (type): Description of parameter `d`.
        gamma (type): Description of parameter `gamma`.

    Attributes:
        random (type): Description of parameter `random`.
        reset (type): Description of parameter `reset`.
        name (type): Description of parameter `name`.
        d

    """
    name = 'CSRR'

    def __init__(self, d, ell, gamma):
        self.d = d
        self.random = check_random_state(None)
        self.reset(None, None)
        self.name = 'LSRR'

    def reset(self, ell=None, state=None):
        if ell is not None:
            self.ell = ell
        if self.ell is not None:
            self.X_sketch = np.zeros((self.ell, self.d))
            self.y_sketch = np.zeros(self.ell)
        if state is not None:
            self.state = state
        if self.state is not None:
            self.random = check_random_state(state)

    def fit(self, X, y):
        n = len(y)
        randomMatrix = np.zeros((self.ell, n))
        randomMatrix[self.random.randint(0, self.ell, n), range(n)] = self.random.choice([-1, 1], n)
        self.X_sketch += randomMatrix @ X
        self.y_sketch += randomMatrix @ y
        return self
