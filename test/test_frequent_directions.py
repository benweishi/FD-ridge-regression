from unittest import TestCase
import numpy as np
import random
from numpy.linalg import svd, inv
from models.frequent_directions import FrequentDirections, RobustFrequentDirections, ISVD


class TestISVD(TestCase):

    def setUp(self):
        pass

    def testRandom1BatchSmallData(self):
        d = 3
        ell = 2
        for i in range(1, 100, 1):
            X = np.random.rand(ell*d).reshape((ell, d)) * i
            y = np.random.rand(ell) * i
            regressor = ISVD(d, ell, 0)
            regressor.partial_fit(X, y)
            Xs = regressor.X_sketch
            np.testing.assert_array_almost_equal(X.T @ X, Xs.T @ Xs)
            np.testing.assert_array_almost_equal(regressor.Xt_y, y @ X)
            gamma = i/10
            coef = y @ X @ inv(X.T @ X + np.identity(d) * gamma)
            coef_ = regressor.get_coef(gamma)
            np.testing.assert_array_almost_equal(coef_, coef)

    def testSketchRandom1BatchMediaData(self):
        d = 30
        ell = 20
        for i in range(100):
            X = np.random.rand(ell*d).reshape((ell, d)) * i
            y = np.random.rand(ell) * i
            regressor = ISVD(d, ell, 0)
            regressor.partial_fit(X, y)
            Xs = regressor.X_sketch
            np.testing.assert_array_almost_equal(X.T @ X, Xs.T @ Xs)
            np.testing.assert_array_almost_equal(regressor.Xt_y, y @ X)

    def testSketchRandom2BatchMediaData(self):
        d = 30
        ell = 20
        for i in range(100):
            X = np.random.rand(2*ell*d).reshape((2*ell, d)) * i
            y = np.random.rand(2*ell) * i
            regressor = ISVD(d, ell, 0)
            regressor.partial_fit(X[:ell], y[:ell])
            regressor.partial_fit(X[ell:], y[ell:])
            _, s, Vt = svd(X, full_matrices=False)
            Xs = regressor.X_sketch
            err = np.linalg.norm(s[ell:])**2
            self.assertAlmostEqual(np.linalg.norm(X.T @ X - Xs.T @ Xs, ord='nuc'), err, places=6)
            np.testing.assert_array_almost_equal(regressor.Xt_y, y @ X)
            np.testing.assert_array_almost_equal(regressor.Xt_y, y @ X)


class TestFrequentDirections(TestCase):

    def setUp(self):
        pass

    def testRandom1BatchSmallData(self):
        d = 3
        ell = 2
        for i in range(1, 100, 1):
            X = np.random.rand(ell*d).reshape((ell, d)) * i
            y = np.random.rand(ell) * i
            regressor = FrequentDirections(d, ell, 0)
            regressor.partial_fit(X, y)
            Xs = regressor.X_sketch
            np.testing.assert_array_almost_equal(X.T @ X, Xs.T @ Xs)
            np.testing.assert_array_almost_equal(regressor.Xt_y, y @ X)
            gamma = i/10
            coef = y @ X @ inv(X.T @ X + np.identity(d) * gamma)
            coef_ = regressor.get_coef(gamma)
            np.testing.assert_array_almost_equal(coef_, coef)

    def testRandom1BatchMediaData(self):
        d = 30
        ell = 20
        for i in range(100):
            X = np.random.rand(ell*d).reshape((ell, d)) * i
            y = np.random.rand(ell) * i
            regressor = FrequentDirections(d, ell, 0)
            regressor.partial_fit(X, y)
            Xs = regressor.X_sketch
            np.testing.assert_array_almost_equal(X.T @ X, Xs.T @ Xs)
            np.testing.assert_array_almost_equal(regressor.Xt_y, y @ X)

    def testRandom2BatchMediaData(self):
        d = 30
        ell = 20
        for i in range(100):
            X = np.random.rand(2*ell*d).reshape((2*ell, d)) * i
            y = np.random.rand(2*ell) * i
            regressor = FrequentDirections(d, ell, 0)
            regressor.partial_fit(X[:ell], y[:ell])
            regressor.partial_fit(X[ell:], y[ell:])
            _, s, Vt = svd(X, full_matrices=False)
            Xs = regressor.X_sketch
            err = np.linalg.norm(s[ell:])**2 + ell * s[ell]**2
            self.assertAlmostEqual(np.linalg.norm(X.T @ X - Xs.T @ Xs, ord='nuc'), err, places=6)
            np.testing.assert_array_almost_equal(regressor.Xt_y, y @ X)

    def testSketchRandomMix(self):  # theoretical test
        for i in range(10):
            d = random.randint(10, 200)
            ell = d // random.randint(2, 10)
            b = random.randint(1, 20)
            X = np.random.rand(b*ell*d).reshape((b*ell, d)) * i * 10
            y = np.random.rand(b*ell) * i * 10
            regressor = FrequentDirections(d, ell, 0)
            for j in range(b):
                regressor.partial_fit(X[j*ell:(j+1)*ell], y[j*ell:(j+1)*ell])
            _, s, Vt = svd(X, full_matrices=False)
            Xs = regressor.X_sketch
            for k in range(ell):
                self.assertLessEqual(np.linalg.norm(X.T @ X - Xs.T @ Xs, ord=2), np.linalg.norm(s[k:])**2 / (ell-k))
            np.testing.assert_array_almost_equal(regressor.Xt_y, y @ X)


class TestRobustFrequentDirections(TestCase):

    def setUp(self):
        pass

    def testRandom1BatchSmallData(self):
        d = 3
        ell = 2
        for i in range(1, 100, 1):
            X = np.random.rand(ell*d).reshape((ell, d)) * i
            y = np.random.rand(ell) * i
            regressor = RobustFrequentDirections(d, ell, 0)
            regressor.partial_fit(X, y)
            Xs = regressor.X_sketch
            np.testing.assert_array_almost_equal(X.T @ X, Xs.T @ Xs)
            np.testing.assert_array_almost_equal(regressor.Xt_y, y @ X)
            gamma = i/10
            coef = y @ X @ inv(X.T @ X + np.identity(d) * gamma)
            coef_ = regressor.get_coef(gamma)
            np.testing.assert_array_almost_equal(coef_, coef)

    def testRandom1BatchMediaData(self):
        d = 30
        ell = 20
        for i in range(100):
            X = np.random.rand(ell*d).reshape((ell, d)) * i
            y = np.random.rand(ell) * i
            regressor = RobustFrequentDirections(d, ell, 0)
            regressor.partial_fit(X, y)
            Xs = regressor.X_sketch
            np.testing.assert_array_almost_equal(X.T @ X, Xs.T @ Xs)
            np.testing.assert_array_almost_equal(regressor.Xt_y, y @ X)

    def testRandom2BatchMediaData(self):
        d = 30
        ell = 20
        for i in range(100):
            X = np.random.rand(2*ell*d).reshape((2*ell, d)) * i
            y = np.random.rand(2*ell) * i
            regressor = RobustFrequentDirections(d, ell, 0)
            regressor.partial_fit(X[:ell], y[:ell])
            regressor.partial_fit(X[ell:], y[ell:])
            _, s, Vt = svd(X, full_matrices=False)
            Xs = regressor.X_sketch
            err = np.linalg.norm(s[ell:])**2 + ell * s[ell]**2
            self.assertAlmostEqual(np.linalg.norm(X.T @ X - Xs.T @ Xs, ord='nuc'), err, places=6)
            np.testing.assert_array_almost_equal(regressor.Xt_y, y @ X)

    def testSketchRandomMix(self):  # theoretical test
        for i in range(10):
            d = random.randint(10, 200)
            ell = d // random.randint(2, 10)
            b = random.randint(1, 20)
            X = np.random.rand(b*ell*d).reshape((b*ell, d)) * i * 10
            y = np.random.rand(b*ell) * i * 10
            regressor = RobustFrequentDirections(d, ell, 0)
            for j in range(b):
                regressor.partial_fit(X[j*ell:(j+1)*ell], y[j*ell:(j+1)*ell])
            _, s, Vt = svd(X, full_matrices=False)
            Xs = regressor.X_sketch
            alpha = regressor.alpha
            for k in range(ell):
                self.assertLessEqual(np.linalg.norm(X.T @ X - Xs.T @ Xs, ord=2) - alpha, np.linalg.norm(s[k:])**2 / (ell-k) / 2)
            np.testing.assert_array_almost_equal(regressor.Xt_y, y @ X)
