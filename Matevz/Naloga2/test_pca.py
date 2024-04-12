import unittest
from unittest import mock

import numpy as np

from hw_pca import PCA


# Kot je doloceno v navodilih, porabite potencno metodo, ne pa ze narejenih razcepov
mock.patch("numpy.linalg.eig", None).start()
mock.patch("numpy.linalg.eigh", None).start()
mock.patch("numpy.linalg.eigvals", None).start()
mock.patch("numpy.linalg.eigvalsh", None).start()
svdmock = mock.patch("numpy.linalg.svd", None)
svdmock.start()


class PCATest(unittest.TestCase):

    def get_data(self, size= 500000, mean=np.array([0,0,0])):
        rand = np.random.RandomState(0)

        eigenvectors = np.array([[1, 2, 1], [1, -1, 1], [1, 0, -1]])
        eigenvalues = np.array([6, 3, 2])
        svdmock.stop()
        X = rand.multivariate_normal([0, 0, 0], np.diag([1, 1, 1]), size=size)
        svdmock.start()
        data = X.dot(eigenvectors) + mean

        return data, eigenvectors, eigenvalues

    def test_pca_types(self):
        data, eigenvectors, eigenvalues = self.get_data(size=50)

        pca = PCA(n_components=3)
        self.assertEqual(pca.eigenvalues, [])
        self.assertEqual(pca.eigenvectors, [])

        pca_fit = pca.fit(data)
        self.assertIsNone(pca_fit)

        pca_data = pca.transform(data)
        self.assertIsNotNone(pca_data)
        self.assertEqual(type(pca_data), np.ndarray)
        self.assertEqual(data.shape, pca_data.shape)

        self.assertEqual(len(pca.get_explained_variance()), 3)

    def test_pca_fit(self):
        data, eigenvectors, _ = self.get_data(size=500000)

        pca = PCA(n_components=3)
        pca.fit(data)

        pca_vectors = np.array(pca.eigenvectors)
        pca_eigenvectors = pca_vectors[
            np.argsort(np.abs(pca_vectors).sum(axis=1))[::-1]
        ]

        cosine = np.sort(
            [
                np.dot(i, j) / (np.linalg.norm(i, ord=2) * np.linalg.norm(j, ord=2))
                for i in pca_eigenvectors
                for j in eigenvectors
            ]
        )

        np.testing.assert_almost_equal(cosine[-3:], np.ones(3), decimal=3)

    def test_pca_centering(self):
        data, _, _ = self.get_data(size=50, mean=[0, 0, 0])
        pca = PCA(n_components=3)
        pca.fit(data)

        data2, _, _ = self.get_data(size=50, mean=[0, 42, 0])
        pca2 = PCA(n_components=3)
        pca2.fit(data2)

        np.testing.assert_almost_equal(pca.eigenvectors, pca2.eigenvectors)
        np.testing.assert_almost_equal(pca.eigenvalues, pca2.eigenvalues)

        t = pca.transform(data)
        t2 = pca2.transform(data2)

        np.testing.assert_almost_equal(t, t2)

    def test_pca_transform_inverse_transform(self):
        data, eigenvectors, _ = self.get_data(size=5000, mean=[0, 42, 0])

        pca = PCA(n_components=3)
        pca.fit(data)

        pca_data = pca.transform(data)
        result = pca.inverse_transform(pca_data)

        np.testing.assert_almost_equal(result, data, decimal=5)

    def test_pca_explained_variance(self):
        data, _, eigenvalues = self.get_data(size=500000)

        pca = PCA(n_components=3)
        pca.fit(data)

        result = pca.get_explained_variance()

        np.testing.assert_almost_equal(result, eigenvalues, decimal=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
