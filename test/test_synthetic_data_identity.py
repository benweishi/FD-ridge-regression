from unittest import TestCase
import numpy as np
from datasets.low_rank_regression import LowRankRegression


class TestLowRankRegression(TestCase):

    def setUp(self):
        pass

    def testDataIdentitySameBatchSize(self):
        p = 2
        random_state = 0
        make_data_params = dict(n_features=2**p,
                                n_samples=2**(p+2),
                                eval_size=2**p,
                                test_size=2**p,
                                effective_rank=0.5,
                                noise=2,
                                correlation=1,
                                random_state=random_state,
                                rotate='dct4')
        dataset1 = LowRankRegression(**make_data_params)
        X1, y1 = dataset1.sampleData(10)
        dataset2 = LowRankRegression(**make_data_params)
        X2, y2 = dataset2.sampleData(10)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def testDataIdentityDifferentBatchSize(self):
        p = 2
        random_state = 0
        make_data_params = dict(n_features=2**p,
                                n_samples=2**(p+2),
                                eval_size=2**p,
                                test_size=2**p,
                                effective_rank=0.5,
                                noise=2,
                                correlation=1,
                                random_state=random_state,
                                rotate='dct4')
        dataset1 = LowRankRegression(**make_data_params)
        X11, y11 = dataset1.sampleData(5)
        X12, y12 = dataset1.sampleData(5)
        X1 = np.concatenate([X11, X12], axis=0)
        y1 = np.concatenate([y11, y12], axis=0)
        dataset2 = LowRankRegression(**make_data_params)
        X2, y2 = dataset2.sampleData(10)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def testSpeed(self):
        p = 8
        random_state = 0
        make_data_params = dict(n_features=2**p,
                                n_samples=2**(p+2),
                                eval_size=2**p,
                                test_size=2**p,
                                effective_rank=0.1,
                                noise=2,
                                correlation=1,
                                random_state=random_state,
                                rotate='dct4')
        dataset = LowRankRegression(**make_data_params)
        for i in range(1000):
            X, y = dataset.sampleData(100)
