import numpy as np
import yaml
import vispy

class PCA:
    def __init__(
        self,
        n_components: int,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        rnd_seed: int = 0,
        mean: np.ndarray = None,
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
        Fit PCA. Center the data around zero and use
        potencna_metoda to obtain eigenvectors and eigenvalues
        for later use in other functions.

        Arguments
        ---------
        X: np.ndarray
            Data matrix with shape (n_samples, n_features)
        """
        self.mean = np.mean(X, axis=0)
        X = X - self.mean


        # M = np.cov(X.T)
        M = X.T @ X / X.shape[0]

        for _ in range(self.n_components):
            eigenvector, eigenvalue = self.potencna_metoda(M)
            self.eigenvectors.append(eigenvector)
            self.eigenvalues.append(eigenvalue)
            M = M - eigenvalue * eigenvector * eigenvector[np.newaxis].T

    def potencna_metoda(self, M: np.ndarray) -> tuple:
        """
        Perform the power method for calculating the eigenvector with the highest corresponding
        eigenvalue of the covariance matrix.

        Arguments
        ---------
        M: np.ndarray
            Covariance matrix.

        Return
        ------
        np.array
            The unit eigenvector of the covariance matrix.
        float
            The corresponding eigenvalue for the covariance matrix.
        """
        n = M.shape[0]
        np.random.seed(self.rnd_seed)
        x = np.random.rand(n)
        x = x / np.linalg.norm(x, ord=2)

        for _ in range(self.max_iterations):
            x_new = M @ x
            x_new = x_new / np.linalg.norm(x_new, ord=2)

            if np.linalg.norm(x - x_new, ord=2) < self.tolerance:
                tolerance_break = True
                break

            x = x_new

            # Ensure the first element is always positive, so the sign of the eigenvector is consistent
            if x[0] < 0:
                x = -x

        eigenvalue = x @ M @ x

        return x, eigenvalue
      
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
        X = X - self.mean
        X_pca = np.array(self.eigenvectors) @ X.T

        return X_pca.T

    def get_explained_variance(self):
        """
        Return the explained variance of the principle components.
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
        X = X.T + self.mean
        
        return X


###### 3. hw ######
def preprocess_data(file_path):
    # Open file
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)

    # Delete duplicated keywords in each article
    for item in data:
        item["gpt_keywords"] = list(set(item["gpt_keywords"]))

    # Save only unique keywords for each line
    for item in data:
        for i in range(len(item["gpt_keywords"])):
            item["gpt_keywords"][i] = item["gpt_keywords"][i].split(" ")
            item["gpt_keywords"][i] = list(set(item["gpt_keywords"][i]))
            item["gpt_keywords"][i] = " ".join(item["gpt_keywords"][i])

    # Count the number of times each keyword appears
    count = {}
    for item in data:
        for keyword in item["gpt_keywords"]:
            if keyword not in count:
                count[keyword] = 1
            else:
                count[keyword] += 1
    
    # Remove keywords that appear less than 20 times from data
    processed_data = []
    for item in data:
        keywords = [keyword for keyword in item["gpt_keywords"] if count[keyword] > 20]
        processed_data.append({"gpt_keywords": keywords, "title": item["title"], "url": item["url"]})
    
    # Remove keywords that appear less than 20 times from count
    encoding = {keyword: count[keyword] for keyword in count if count[keyword] > 20}
    
    # Remove empty articles
    processed_data = [item for item in processed_data if len(item["gpt_keywords"]) > 0]

    return processed_data, encoding

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


if __name__ == "__main__":

    # Preprocess data
    data, encoding = preprocess_data("rtvslo.yaml")

    # Save preporcessed data
    import pickle
    with open("data.pkl", "wb") as file:
        pickle.dump(data, file)
    with open("encoding.pkl", "wb") as file:
        pickle.dump(encoding, file)

    # Open preprocessed data
    # import pickle
    # with open("data.pkl", "rb") as file:
    #     data = pickle.load(file)
    # with open("encoding.pkl", "rb") as file:
    #     encoding = pickle.load(file)

    