import numpy as np
from sklearn.utils import check_array, check_random_state
from scipy.stats import ortho_group, norm
import math


class LowRankRegression():
    def __init__(self, n_features=100, effective_rank=0.1, correlation=2.0, noise=0.2, random_state=None):
        self.d = n_features
        if effective_rank < 1:
            self.effective_rank = int(n_features * effective_rank)
        else:
            self.effective_rank = effective_rank
        self.generator = check_random_state(random_state)
        self.ortho = ortho_group.rvs(self.d, random_state=self.generator)
        self.sigmas = np.exp(-(np.arange(0, self.d, 1) / self.effective_rank)**2)
        self.coefs = self.generator.normal(size=self.effective_rank)
        self.coefs /= np.linalg.norm(self.coefs)
        self.coefs *= correlation
        self.noise = noise

    def sampleData(self, n_samples=100):
        X = self.generator.normal(scale=self.sigmas, size=(n_samples, self.d))
        y = self.coefs @ X[:, :self.effective_rank].T + self.generator.normal(scale=self.noise, size=n_samples)
        return X @ self.ortho, y

    def get_coefs(self):
        return self.coefs @ self.ortho[:self.effective_rank]


if __name__ == '__main__':
    regression = LowRankRegression(10, 0.2, random_state=0)
    X, y = regression.sampleData(5)
    print(X, y)
