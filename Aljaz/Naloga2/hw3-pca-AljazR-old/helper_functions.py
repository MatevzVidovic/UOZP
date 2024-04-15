import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# do not import any non-python native libraries because github tests might fail
# consult with TAs if you find them crucial for solving the exercise


def read_text_file(filename: str) -> str:
    """
    Read text file as a long string

    Arguments
    ---------
    filename: str
        Name of the input text file with '.txt' suffix.

    Return
    ------
    str
        A string of characters from the file
    """
    with open(filename) as file:
        lines = file.read()
    return lines


def preprocess_text(text: str, all_letters=True) -> "list[str]":
    """
    Preprocess text string by:
        1. removing any character that is not a letter (e.g. +*.,#!"$%&/'()=+? and so on ...)
        2. convert any uppercase letters into lowercase
        3. split a string into a list of words

        Caution: There should be no empty strings in the list of words.

    Arguments
    ---------
    text: str
        A string of text

    Return
    ------
    list['str']
        A list of words as strings in the same order as in the original document.
    """
    if all_letters:
        text = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿčćšđžČĆŠĐŽ]+", ' ', text)
    else:
        text = re.sub(r"[^A-Za-z]+", ' ', text)
    text = text.lower()
    text = text.split()
    return text


def words_into_kmers(words: "list[str]", k: int) -> dict:
    """
    Convert a list of words into a dictionary of {k-mer : number of k-mers in words} with k as a parameter.
    If a word is shorter than k characters, discard it.

    Arguments
    ---------
    words: list['str']
        A list of words as strings.
    k: int
        The length of k-mers.

    Return
    ------
    dict
        Dictionary with keys k-mers (string) and values number of occurances (int)
    """
    kmers = {}
    for word in words:
        if len(word) >= k:
            for i in range(len(word) - k + 1):
                kmer = word[i : i + k]
                if kmer in kmers:
                    kmers[kmer] += 1
                else:
                    kmers[kmer] = 1
    return kmers


def words_into_bag_of_words(words) -> dict:
    """
    Convert a list of words into a dictonary of {word : number of words in text}.

    Arguments
    ---------
    words: list['str']
        A list of words as strings.

    Return
    ------
    dict
        Dictionary with keys words (string) and values number of occurances (int)
    """
    bag_of_words = {}
    for word in words:
        if word in bag_of_words:
            bag_of_words[word] += 1
        else:
            bag_of_words[word] = 1
    
    return bag_of_words


def words_into_phrases(words: "list[str]", phrase_encoding: "list[int]") -> "dict[str]":
    """
    Convert a list of words in to a dictonary of {phrase : number of phrases in text}.

    Phrase encoding is a list of integers where 1 means the word is in and 0 that it is
    not in the phrase. We encode a phrase by joining words in the phrase with "-" sign
    and use "/" to represent ommited word in the phrase.

    Example text 1: "it is raning man".
    Using encoding [1,1] we get three phrases ("it-is", "is-raning", "raining-man").
    Using encoding [1, 0, 1] we get only two phrases ("it-/-raining", "is-/-man").

    Example text 2: "we did not start the fire"
    Using encoding [1, 0, 1] we get four phrases ("we-/-not", "did-/-start", "not-/-the", "start-/-fire").
    Using encoding [0, 1, 0, 1] we get three phrases ("/-did-/-start", "/-not-/-the", "/-start-/-fire").

    As you can see, phrase encoding does not have to start or end with 1.

    Arguments
    ---------
    words: list['str']
        A list of words as strings.
    phrase_encoding: list['int']
        Phrases are consecutive words where 1 means a word is in the phrase and 0 that
        the word is not in the phrase. Example is above.

    Return
    ------
    dict
        Dictionary with keys phrase (string) and values number of occurances (int)
    """
    phrases = {}

    if len(words) < len(phrase_encoding):
        return phrases

    for i in range(len(words) - len(phrase_encoding) + 1):
        phrase = ""
        for j in range(len(phrase_encoding)):
            if phrase_encoding[j] == 1:
                phrase += words[i + j] + "-"
            else:
                phrase += "/-"
        phrase = phrase[:-1]

        if phrase in phrases:
            phrases[phrase] += 1
        else:
            phrases[phrase] = 1

    return phrases


def term_frequency(encoding: dict) -> dict:
    """
    Calculate the frequency of each term in the encoding.

    Arguments
    ---------
    encoding: dict
        Dictonary with keys strings (kmers, words, phrases) and values (number of occurances).

    Return
    ------
    dict
        Dictonary with keys strings (kmers, words, phrases) and values (FREQUENCY of occurances in this document).
    """
    sum_all = sum(encoding.values())

    out = {}
    for key in encoding:
        out[key] = encoding[key] / sum_all

    return out


def inverse_document_frequency(documents: "list[dict]"):
    """
    Calculate inverse document frequency (idf) of all terms in the encoding of a document.
    Use the corrected formula for the idf (lecture notes page 36):
        idf(t) = log(|D| / (1 + |{d : t in d}|)),
    where |D| is the number of documents and |{d : t in d}| is the number of documents with the term t.
    Use natrual logarithm.

    Arguments
    ---------
    documents: list['dict']
        List of encodings for all documents in the study.

    Return
    ------
    dict
        Dictonary with keys strings (kmers, words, phrases) and values (FREQUENCY of occurances in this document.
    """
    idf = {}

    for document in documents:
        for key in document:
            if key in idf:
                idf[key] += 1
            else:
                idf[key] = 1
    
    for key in idf:
        idf[key] = np.log(len(documents) / (1 + idf[key]))
    
    return idf    


def tf_idf(encoding: dict, term_importance_idf: dict) -> dict:
    """
    Calculate term frequency - inverse document frequency (tf-idf) using precomputed idf (with your function)
    and term_frequency function you implemented above (use it in this function).

    The output should contain only the terms that are listed inside the term_importance_idf dictionary.
    If the term does not exist in the document, asign it a value 0.
    Filter terms AFTER you calculated term frequency.

    Arguments
    ---------
    encoding: dict
        Dictonary with keys strings (kmers, words, phrases) and values (frequency of occurances).
    term_importance_idf: dict
        Term importance as an output of inverse_document_frequency function.

    Return
    ------
    dict
        Dictonary with keys strings (kmers, words, phrases) and values (tf-idf value).
        Includes only keys from the IDF dictionary.
    """
    tf = term_frequency(encoding)
    out_tf_idf = {}

    for key in term_importance_idf:
        if key in tf:
            out_tf_idf[key] = tf[key] * term_importance_idf[key]
        else:
            out_tf_idf[key] = 0

    return out_tf_idf


def cosine_similarity(vektor_a: np.array, vektor_b: np.array) -> float:
    """
    Cosine similariy between vectors a and b.

    Arguments
    ---------
    vector_a, vector_b: np.array
        Vector of a document in the feature space

    Return
    ------
    float
        Cosine similarity
    """
    return np.dot(vektor_a, vektor_b) / (np.linalg.norm(vektor_a) * np.linalg.norm(vektor_b))
    # np.dot(np.sqrt(np.square(vektor_a)), np.sqrt(np.square(vektor_b)))


def jaccard_similarity(vektor_a, vektor_b) -> float:
    """
    Jaccard similarity

    Arguments
    ---------
    vector_a, vector_b: np.array
        Vector of a document in the feature space

    Return
    ------
    float
        Jaccard similarity
    """
    return np.intersect1d(vektor_a, vektor_b).size / np.union1d(vektor_a, vektor_b).size


class PCA:
    def __init__(
        self,
        n_components: int,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        rnd_seed: int = 0,
    ):
        assert (
            type(n_components) == int
        ), f"n_components is not of type int, but {type(n_components)}"
        assert (
            n_components > 0
        ), f"n_components has to be greater than 0, but found {n_components}"

        self.n_components = n_components
        self.eigenvectors = []
        self.eigenvalues = []

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.rnd_seed = rnd_seed


    def fit(self, X: np.ndarray) -> None:
        """
        Fit principle component vectors.
        Center the data around zero.

        Arguments
        ---------
        X: np.ndarray
            Data matrix with shape (n_samples, n_features)
        """
        X = X - np.mean(X, axis=0)

        M = np.cov(X.T)

        for _ in range(self.n_components):
            eigenvector, eigenvalue = self.potencna_metoda(M, np.random.rand(X.shape[1]))
            self.eigenvectors.append(eigenvector)
            self.eigenvalues.append(eigenvalue)
            M = M - eigenvalue * eigenvector * eigenvector[np.newaxis].T


    def potencna_metoda(
        self, M: np.ndarray, vector: np.array, iteration: int = 0
    ) -> tuple:
        """
        Perform the power method for calculating the eigenvector with the highest corresponding
        eigenvalue of the covariance matrix.
        This should be a recursive function. Use 'max_iterations' and 'tolerance' to terminate
        recursive call when necessary.

        Arguments
        ---------
        M: np.ndarray
            Covariance matrix of the zero centered data.
        vector: np.array
            Candidate eigenvector in the iteration.
        iteration: int
            Index of the consecutive iteration for termination purpose of the

        Return
        ------
        np.array
            The unit eigenvector of the covariance matrix.
        float
            The corresponding eigenvalue of the covariance matrix.
        """ 
        x = M @ vector
        x = x / np.linalg.norm(x)

        if iteration + 1 == self.max_iterations or np.linalg.norm(vector - x) < self.tolerance:
            return x, (x[np.newaxis] @ M @ x[np.newaxis].T)[0, 0]
        else:
            return self.potencna_metoda(M, x, iteration + 1)


    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data (X) using fitted eigenvectors

        Arguments
        ---------
        X: np.ndarray
            New data with the same number of features as the fitting data.

        Return
        ------
        np.ndarray
            Transformed data with the shape (n_samples, n_components).
        """
        X_pca = np.array(self.eigenvectors) @ X.T

        return X_pca.T
        
            
    def get_explained_variance(self):
        """
        Return the explained variance ratio of the principle components.
        Prior to calling fit() function return None.
        Return only the ratio for the top 'n_components'.

        Return
        ------
        np.array
            Explained variance for the top 'n_components'.
        """
        if len(self.eigenvalues) == 0:
            return None

        return np.array(self.eigenvalues)


    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data from the principle component space into
        the real space.

        Arguments
        ---------
        X: np.ndarray
            Data  in PC space with the same number of features as
            the fitting data.

        Return
        ------
        np.ndarray
            Transformed data in original space with
            the shape (n_samples, n_components).
        """
        X = np.array(self.eigenvectors).T @ X.T
        
        return X.T


def dual_pca(data):
    N = data.shape[0]
    # calculate mean values of each axis
    mean = np.sum(data, axis=0) / N
    # center data
    centered_points = data - mean
    # compute covariance matrix
    C = 1 / (N - 1) * centered_points @ centered_points.T
    # compute SVD and get eigenvalues and eigenvectors
    U, eigenvalues, _ = np.linalg.svd(C)
    # make sure eigenvalues are non-zero
    eigenvalues += 10e-15
    # get eigenvectors
    eigenvectors = centered_points.T @ U @ np.diag(np.sqrt(1 / (eigenvalues * (N - 1))))

    return eigenvectors[:,:2].T, eigenvalues[:2]


def plot_Gutenberg_jeziki():
    path = "Gutenberg-jeziki/Gutenberg-jeziki-preprocessed"
    languages_arr = []
    files_arr = []
    all_languages = os.listdir(path)

    # read all text files grouped by language and grouped by file
    for folder in all_languages:
        language_str = ""
        for file in os.listdir(f"{path}/{folder}"):
            file_str = read_text_file(f"{path}/{folder}/{file}")
            language_str += file_str
            files_arr.append(file_str)
        languages_arr.append(language_str)


    # preprocess text files
    files_preprocessed = []
    for file in files_arr:
        files_preprocessed.append(preprocess_text(file, all_letters=False))

    # preprocess language files
    languages_preprocessed = []
    for language in languages_arr:
         languages_preprocessed.append(preprocess_text(language))
     

    # bag of words for text files
    group_txt = []
    for file in files_preprocessed:
        group_txt.append(words_into_bag_of_words(file))

    # bag of words for language files
    group_lang = []
    for lang in languages_preprocessed:
        group_lang.append(words_into_bag_of_words(lang))


    # k-mers for text files
    for i, file in enumerate(files_preprocessed):
        group_txt[i].update(words_into_kmers(file, 2))
        group_txt[i].update(words_into_kmers(file, 3))
        group_txt[i].update(words_into_kmers(file, 4))
        group_txt[i].update(words_into_kmers(file, 5))

    # k-mers for language files
    for i, lang in enumerate(languages_preprocessed):
        group_lang[i].update(words_into_kmers(lang, 2))
        group_lang[i].update(words_into_kmers(lang, 3))
        group_lang[i].update(words_into_kmers(lang, 4))
        group_lang[i].update(words_into_kmers(lang, 5))
    

    # # languages into phrases
    # for i, lang in enumerate(languages_preprocessed):
    #     group_lang[i].update(words_into_phrases(lang, [1, 1, 1]))
    #     group_lang[i].update(words_into_phrases(lang, [1, 1]))
    
    # # files into phrases
    # for i, file in enumerate(files_preprocessed):
    #     group_txt[i].update(words_into_phrases(file, [1, 1, 1]))
    #     group_txt[i].update(words_into_phrases(file, [1, 1]))


    # inverse document frequency
    idf = inverse_document_frequency(group_txt)


    # sort by idf for each language
    idf_sort = {}
    count = np.zeros(len(all_languages))
    bigget_n = []
    for i in range(len(all_languages)) :
        bigget_n.append([])
    for tup in sorted(zip(idf.values(), idf.keys())):
        is_in = False
        for i in range(len(all_languages)):
            if tup[1] in group_lang[i] and count[i] < 20:
                if is_in == False:
                    is_in = True
                    idf_sort[tup[1]] = tup[0]
                    bigget_n[i].append(tup[1])
                    count[i] += 1
        if np.all(count == 20):
            break
    idf = idf_sort

    # # sort by idf for each txt file
    # idf_sort = {}
    # count = np.zeros(len(group_txt))
    # bigget_n = []
    # for i in range(len(group_txt)):
    #     bigget_n.append([])
    # for tup in sorted(zip(idf.values(), idf)):
    #     is_in = False
    #     for i in range(len(group_txt)):
    #         if tup[1] in group_txt[i] and count[i] < 5:
    #             if is_in == False:
    #                 is_in = True
    #                 idf_sort[tup[1]] = tup[0]
    #             bigget_n[i].append(tup[1])
    #             count[i] += 1
    #     if np.all(count == 5):
    #         break
    # idf = idf_sort


    ## sort by idf
    # idf_sort = {}
    # for tup in sorted(zip(idf.values(), idf.keys()))[:100]:
    #     idf_sort[tup[1]] = tup[0]
    # idf = idf_sort


    # ti-idf
    tf_idf_dict = []
    for language in group_lang:
        tf_idf_dict.append(tf_idf(language, idf))

    # ti-idf matrix
    tf_idf_matrix = []
    for language in tf_idf_dict:
        tf_idf_matrix.append(np.array(list(language.values())))
    tf_idf_matrix = np.array(tf_idf_matrix)

    # PCA
    pca = PCA(n_components=2)
    
    # direct PCA
    pca.fit(tf_idf_matrix)

    # dual PCA
    # eigenvectors, eigenvalues = dual_pca(tf_idf_matrix)
    # pca.eigenvectors = eigenvectors
    # pca.eigenvalues = eigenvalues
    
    # transform
    pca_data = pca.transform(tf_idf_matrix - np.mean(tf_idf_matrix, axis=0))

    # get main features
    features_indexX = np.argsort((-pca.eigenvectors[0]))[:5]
    features_indexY = np.argsort((-pca.eigenvectors[1]))[:5]
    featuresX = ''
    featuresY = ''

    for i in features_indexX:
        featuresX += "'" + list(tf_idf_dict[0].keys())[i] + "', "
    print()
    for i in features_indexY:
        featuresY += "'" + list(tf_idf_dict[0].keys())[i] + "', "
    featuresX = featuresX[:-2]
    featuresY = featuresY[:-2]

    # plot
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gold', 
          'gray', 'purple', 'pink', 'orange', 'brown', 'olive', 'navy', 'teal']
    for i in range(len(pca_data)):
        plt.scatter(pca_data[i, 0], pca_data[i, 1], color=colors[i], label=all_languages[i])
        plt.text(pca_data[i, 0], pca_data[i, 1], all_languages[i])
        plt.xlabel(featuresX)
        plt.ylabel(featuresY)
    plt.title('k-mers (2,3,4,5) and bag of words; IDF on text files + filtered only minimum 20 values for each language')
    plt.savefig('jeziki.png')
    plt.show()


def plot_Gutenberg_100():
    path = "Gutenberg-100/Gutenberg-top100-preprocessed"
    files_arr = []
    all_files = os.listdir(path)

    # read all text files
    for file in os.listdir(path):
        file_str = read_text_file(f"{path}/{file}")
        files_arr.append(file_str)

    # preprocess text files
    files_preprocessed = []
    for file in files_arr:
        files_preprocessed.append(preprocess_text(file, all_letters=False))
    
    # bag of words
    group_txt = []
    for file in files_preprocessed:
        group_txt.append(words_into_bag_of_words(file))

    # files into phrases
    for i, file in enumerate(files_preprocessed):
        group_txt[i].update(words_into_phrases(file, [1, 1, 1]))
        group_txt[i].update(words_into_phrases(file, [1, 0, 1]))
        group_txt[i].update(words_into_phrases(file, [1, 0, 1, 0, 1]))
        group_txt[i].update(words_into_phrases(file, [1, 0, 0, 1]))

    # inverse document frequency
    idf = inverse_document_frequency(group_txt)

    # sort by idf for each txt file
    idf_sort = {}
    count = np.zeros(len(group_txt))
    bigget_n = []
    for i in range(len(group_txt)):
        bigget_n.append([])
    for tup in sorted(zip(idf.values(), idf.keys())):
        is_in = False
        for i in range(len(group_txt)):
            if tup[1] in group_txt[i] and count[i] < 10:
                if is_in == False:
                    is_in = True
                    idf_sort[tup[1]] = tup[0]
                    bigget_n[i].append(tup[1])
                    count[i] += 1
        if np.all(count == 10):
            break
    idf = idf_sort

    # ti-idf
    tf_idf_dict = []
    for language in group_txt:
        tf_idf_dict.append(tf_idf(language, idf))

    # ti-idf matrix
    tf_idf_matrix = []
    for language in tf_idf_dict:
        tf_idf_matrix.append(np.array(list(language.values())))
    tf_idf_matrix = np.array(tf_idf_matrix)

    # PCA
    pca = PCA(n_components=2)
    pca.fit(tf_idf_matrix)
    
    # transform
    pca_data = pca.transform(tf_idf_matrix - np.mean(tf_idf_matrix, axis=0))

    # get main features
    features_indexX = np.argsort((-pca.eigenvectors[0]))[:5]
    features_indexY = np.argsort((-pca.eigenvectors[1]))[:5]
    featuresX = ''
    featuresY = ''

    for i in features_indexX:
        featuresX += "'" + list(tf_idf_dict[0].keys())[i] + "', "
    print()
    for i in features_indexY:
        featuresY += "'" + list(tf_idf_dict[0].keys())[i] + "', "
    featuresX = featuresX[:-2]
    featuresY = featuresY[:-2]

    # plot
    for i in range(len(pca_data)):
        plt.scatter(pca_data[i, 0], pca_data[i, 1], label=all_files[i])
        plt.text(pca_data[i, 0], pca_data[i, 1], all_files[i])
        plt.xlabel(featuresX)
        plt.ylabel(featuresY)
    plt.title('bag of words and words into phrases; IDF filtered only minimum 10 values per file')
    plt.savefig('top100.png')
    plt.show()



if __name__ == "__main__":
    plot_Gutenberg_jeziki()
    plot_Gutenberg_100()