import numpy as np
from sklearn.utils import check_array, check_random_state
from scipy.stats import ortho_group, norm
from scipy.stats import special_ortho_group
from scipy.fftpack import dct
import math


class LowRankRegression():
    def __init__(self, n_features=100, n_samples=400, eval_size=100, test_size=100, effective_rank=0.1, correlation=2.0, noise=0.2, random_state=None, rotate='dct4', name='synthetic'):
        self.d = n_features
        self.n_train = n_samples
        self.n_eval = eval_size
        self.n_test = test_size
        self.rotate = rotate
        if effective_rank < 1:
            self.effective_rank = int(n_features * effective_rank)
        else:
            self.effective_rank = effective_rank
        assert self.effective_rank > 0, "effective_rank {} is too small".format(effective_rank)
        self.sigmas = np.exp(-(np.arange(0, self.d, 1) / self.effective_rank)**2)
        self.noise = noise
        self.name = name
        self.reset(random_state)

    def reset(self, random_state=None):
        self.generator = check_random_state(random_state)
        if random_state is None:
            self.generator_X = check_random_state(random_state)
            self.generator_y = check_random_state(random_state)
        else:
            self.generator_X = check_random_state(random_state+1)
            self.generator_y = check_random_state(random_state+2)
        if self.rotate == 'random':
            self.ortho = ortho_group.rvs(self.d, random_state=self.generator)
        self.coefs = self.generator.normal(size=self.effective_rank)
        self.coefs /= np.linalg.norm(self.coefs)
        self.rest_train_num = self.n_train

    def sampleData(self, n_samples=100):
        if n_samples == 0:
            return np.empty((0, self.d)), np.empty(0)
        X = self.generator_X.normal(scale=self.sigmas, size=(n_samples, self.d))
        y = self.coefs @ X[:, :self.effective_rank].T + self.generator_y.normal(scale=self.noise, size=n_samples)
        if self.rotate == 'dct4':
            X = dct(X, 4, norm='ortho')
        elif self.rotate == 'random':
            X = X @ self.ortho
        return X, y

    def get_coefs(self):
        return self.coefs @ self.ortho[:self.effective_rank]

    def singular_values(self):
        return self.sigmas

    def trainData(self, n_samples=100):
        if self.rest_train_num >= n_samples:
            self.rest_train_num -= n_samples
            return self.sampleData(n_samples)
        else:
            self.rest_train_num = 0
            return self.sampleData(self.rest_train_num)

    def evalData(self):
        return self.sampleData(self.n_eval)

    def testData(self):
        return self.sampleData(self.n_test)

if __name__ == '__main__':
    regression = LowRankRegression(10, 0.2, random_state=0)
    X, y = regression.sampleData(5)
    print(X, y)
