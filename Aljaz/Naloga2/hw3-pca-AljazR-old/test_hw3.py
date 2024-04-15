import os
import unittest
import numpy as np
from pkg_resources import resource_listdir

from helper_functions import (
    read_text_file,
    preprocess_text,
    words_into_kmers,
    words_into_bag_of_words,
    words_into_phrases,
    term_frequency,
    inverse_document_frequency,
    tf_idf,
    PCA,
)

from test_variables import *


class ProcessingTest(unittest.TestCase):
    def test_reading_files(self):

        text = read_text_file("test_besedilo.txt")

        self.assertEqual(text, original_text)

    def test_special_characters(self):

        text = "+*.,#!$%&/'()=+?andsoon..."
        expected = ["andsoon"]

        result = preprocess_text(text)
        self.assertEqual(expected, result)

    def test_uppercase(self):

        text = "adFadgfadsfASdgGDF"
        expected = ["adfadgfadsfasdggdf"]

        result = preprocess_text(text)
        self.assertEqual(expected, result)

    def test_split_into_words(self):

        text = "e.g. +*.,#!$%/&'()=+? and so on     ..."
        expected = ["e", "g", "and", "so", "on"]

        result = preprocess_text(text)
        self.assertEqual(expected, result)

    def test_preprocess_pipeline(self):

        result = preprocess_text(read_text_file("test_besedilo.txt"))

        self.assertEqual(result, words)


class PreprocessingTest(unittest.TestCase):
    def sort_dict(self, dictionary: dict):
        return dict(sorted(dictionary.items(), key=lambda x: x[0]))

    def test_words_into_kmers_2(self):

        result = words_into_kmers(words, 2)
        result = self.sort_dict(result)
        expected = self.sort_dict(kmers)

        self.assertEqual(list(result.keys()), list(expected.keys()))
        for key in result.keys():
            self.assertAlmostEqual(
                result[key],
                expected[key],
                5,
                msg=f"Missmatch at key: {key}.",
            )

    def test_words_into_kmers_3(self):

        result = words_into_kmers(words, 3)
        result = self.sort_dict(result)
        expected = self.sort_dict(kmers_3)

        self.assertEqual(list(result.keys()), list(expected.keys()))
        for key in result.keys():
            self.assertAlmostEqual(
                result[key],
                expected[key],
                5,
                msg=f"Missmatch at key: {key}.",
            )

    def test_words_into_bag_of_words(self):

        result = words_into_bag_of_words(words)
        result = self.sort_dict(result)
        expected = self.sort_dict(bag_of_words)

        self.assertEqual(list(result.keys()), list(expected.keys()))
        for key in result.keys():
            self.assertAlmostEqual(
                result[key],
                expected[key],
                5,
                msg=f"Missmatch at key: {key}.",
            )

    def test_words_into_phrases_101(self):

        result = words_into_phrases(words, [1, 0, 1])
        result = self.sort_dict(result)
        expected = self.sort_dict(phrases_101)

        self.assertEqual(list(result.keys()), list(expected.keys()))
        for key in result.keys():
            self.assertAlmostEqual(
                result[key],
                expected[key],
                5,
                msg=f"Missmatch at key: {key}.",
            )

    def test_words_into_phrases_0110(self):

        result = words_into_phrases(words, [0, 1, 1, 0])
        result = self.sort_dict(result)
        expected = self.sort_dict(phrases_0110)

        self.assertEqual(list(result.keys()), list(expected.keys()))
        for key in result.keys():
            self.assertAlmostEqual(
                result[key],
                expected[key],
                5,
                msg=f"Missmatch at key: {key}.",
            )


class TermFrequencyTest(unittest.TestCase):
    def generate_documents(self, bag_of_words: dict):

        np.random.seed(0)
        words, word_counts = list(bag_of_words.keys()), list(bag_of_words.values())
        all_words = [i for x in [[k, k + "x"] for k in words] for i in x]

        documents = []
        for i in range(10):
            np.random.shuffle(all_words)
            np.random.shuffle(word_counts)
            documents += [{k: v for k, v in zip(all_words[: len(words)], word_counts)}]

        return documents

    def test_term_frequency(self):
        result = term_frequency(bag_of_words)

        result = self.sort_dict(result)
        expected = self.sort_dict(term_freq_expected)

        self.assertEqual(list(result.keys()), list(expected.keys()))
        for key in result.keys():
            self.assertAlmostEqual(
                result[key],
                expected[key],
                5,
                msg=f"Missmatch at key: {key}.",
            )

    def sort_dict(self, dictionary: dict):
        return dict(sorted(dictionary.items(), key=lambda x: x[0]))

    def test_inverse_document_frequency(self):
        documents = self.generate_documents(bag_of_words)
        result = inverse_document_frequency(documents)

        result = self.sort_dict(result)
        expected = self.sort_dict(idf_expected)

        self.assertEqual(list(result.keys()), list(expected.keys()))
        for key in result.keys():
            self.assertAlmostEqual(
                result[key],
                expected[key],
                5,
                msg=f"Missmatch at key [{key}], Expected {expected[key]}, found: {result[key]}.",
            )

    def test_tf_idf(self):
        result = tf_idf(term_freq_expected, idf_expected)

        result = self.sort_dict(result)
        expected = self.sort_dict(tf_idf_expected)

        self.assertEqual(list(result.keys()), list(expected.keys()))
        for key in result.keys():
            self.assertAlmostEqual(
                result[key],
                expected[key],
                5,
                msg=f"Missmatch at key [{key}], Expected {expected[key]}, found: {result[key]}.",
            )


class PCATest(unittest.TestCase):
    def get_data(self, size: int = 500000):
        np.random.seed(0)

        eigenvectors = np.array([[1, 2, 1], [1, -1, 1], [1, 0, -1]])
        eigenvalues = np.array([6, 3, 2])
        X = np.random.multivariate_normal([0, 0, 0], np.diag([1, 1, 1]), size=size)
        data = X.dot(eigenvectors)

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

        cosine = np.sort([
                np.abs(
                    np.dot(i, j) / (np.linalg.norm(i, ord=2) * np.linalg.norm(j, ord=2))
                ) 
                for i in pca_eigenvectors
                for j in eigenvectors
        ])

        np.testing.assert_almost_equal(cosine[-3:], np.ones(3), decimal=3)

    def test_pca_transform_inverse_transform(self):

        data, eigenvectors, _ = self.get_data(size=5000)

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


class ImageTest(unittest.TestCase):
    def test_image_jeziki(self):
        image = "jeziki.png"

        self.assertIn(image, os.listdir("."))

    def test_image_top100(self):
        image = "top100.png"

        self.assertIn(image, os.listdir("."))


if __name__ == "__main__":
    unittest.main(verbosity=2)
