import numpy as np
from sklearn.utils import check_array, check_random_state
from scipy.stats import ortho_group, norm
from scipy.stats import special_ortho_group
from scipy.fftpack import dct
import math


class KSLC():
    def __init__(self, n_features=100, n_samples=400, test_size=100, random_state=None, random=True, name='KSLC'):
        # data sequence may include NaN, ordered from old to new
        data_sequence = np.genfromtxt('./datasets/KSLC.2019-07-15.csv', delimiter=',', skip_header=12, usecols=2)
        np.flip(data_sequence)  # ordered from new to old
        nans_mask = np.isnan(data_sequence)
        valid_i = (~nans_mask).nonzero()[0]
        # remove head and tail nan
        min_valid_i = valid_i.min()
        max_valid_i = valid_i.max()
        data_sequence = data_sequence[min_valid_i:max_valid_i+1]
        nans_mask = nans_mask[min_valid_i:max_valid_i+1]
        valid_i -= min_valid_i
        # interpolate middle nan
        nan_i = nans_mask.nonzero()[0]
        valid_v = data_sequence[~nans_mask]
        data_sequence[nans_mask] = np.interp(nan_i, valid_i, valid_v)
        # difference sequence
        self.diff_seq = data_sequence[:-1] - data_sequence[1:]
        n_total = len(self.diff_seq)
        self.d = n_features
        self.n = n_total - self.d
        if n_samples + test_size > self.n:
            raise ValueError(f'n_train + n_test = {n_samples} + {test_size} = {n_samples+test_size} > {self.n} = n_data.')
        self.n_train = n_samples
        self.n_test = test_size
        self.random = random
        self.name = name
        self.reset(random_state)

    def reset(self, random_state=None):
        if random_state is not None:
            random_state += 20
        self.generator = check_random_state(random_state)
        self.idx_seqs = np.arange(self.n)
        if self.random:
            self.generator.shuffle(self.idx_seqs)
        self.train_i_seqs = self.idx_seqs[:self.n_train]
        self.test_i_seqs = self.idx_seqs[-self.n_test:]

    def sampleData(self, idx_array):
        y = self.diff_seq[idx_array]
        x_head_idx = idx_array.reshape(-1, 1) + 1
        X = self.diff_seq[x_head_idx + np.arange(self.d)]
        return X, y

    def trainData(self, n_samples=100):
        if len(self.train_i_seqs) >= n_samples:
            idx_array = self.train_i_seqs[:n_samples]
            self.train_i_seqs = self.train_i_seqs[n_samples:]
        else:
            idx_array = self.train_i_seqs
            self.train_i_seqs = self.train_i_seqs[-len(self.idx_seqs):]
        return self.sampleData(idx_array)

    def testData(self):
        return self.sampleData(self.test_i_seqs)

if __name__ == '__main__':
    dataset = KSLC(5, 10, 5, random_state=0)
    X_train, y_train = dataset.trainData(10)
    X_test, y_test = dataset.testData()
    print(X_train, y_train, X_test, y_test)
