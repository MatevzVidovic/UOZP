import numpy as np
import yaml
import plotly.express as px
import pickle

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
def preprocess_articles(file_path):
    # Open file
    with open(file_path, "r") as file:
        articles = yaml.safe_load(file)

    # All keywords to lowercase
    for article in articles:
        article["gpt_keywords"] = [keyword.lower() for keyword in article["gpt_keywords"]]
    
        # Delete duplicated keywords in each article
    for article in articles:
        article["gpt_keywords"] = list(set(article["gpt_keywords"]))

    # Save only unique keywords for each line
    for article in articles:
        for i in range(len(article["gpt_keywords"])):
            article["gpt_keywords"][i] = article["gpt_keywords"][i].split(" ")
            article["gpt_keywords"][i] = list(set(article["gpt_keywords"][i]))
            article["gpt_keywords"][i] = " ".join(article["gpt_keywords"][i])

    # Count the number of times each keyword appears
    count = {}
    for article in articles:
        for keyword in article["gpt_keywords"]:
            if keyword not in count:
                count[keyword] = 1
            else:
                count[keyword] += 1
    
    # Remove keywords that appear less than 20 times from data
    processed_articles = []
    for article in articles:
        keywords = [keyword for keyword in article["gpt_keywords"] if count[keyword] >= 20]
        processed_articles.append({"gpt_keywords": keywords, "title": article["title"], "url": article["url"]})
    
    # Remove keywords that appear less than 20 times from count
    count = {keyword: count[keyword] for keyword in count if count[keyword] > 20}
    
    # Remove empty articles
    processed_articles = [item for item in processed_articles if len(item["gpt_keywords"]) > 0]

    # Make dictionary of titles as keys and keywords as values
    processed_articles = {item["title"]: item["gpt_keywords"] for item in processed_articles}

    return processed_articles, count

def pickle_save(articles, count):
    with open("articles.pkl", "wb") as file:
        pickle.dump(articles, file)
    with open("count.pkl", "wb") as file:
        pickle.dump(count, file)

def pickle_load():
    with open("articles.pkl", "rb") as file:
        data = pickle.load(file)
    with open("count.pkl", "rb") as file:
        encoding = pickle.load(file)

    return data, encoding

def inverse_document_frequency(articles, count):
    idf = {}
    for keyword in count:
        idf[keyword] = (np.log(len(articles) / count[keyword]))
    
    return idf

def tf_idf(data, count, idf):
    tf = 1 / len(data)
    out_tf_idf = []

    for key in idf:
        if key in data:
            out_tf_idf.append(tf * idf[key])
        else:
            out_tf_idf.append(0)

    return out_tf_idf

if __name__ == "__main__":

    # Preprocess data
    # articles, count = preprocess_articles("rtvslo.yaml")
    # pickle_save(articles, count)
    articles, count = pickle_load()
    
    # Make td-idf matrix
    tf_idf_matrix = []
    idf = inverse_document_frequency(articles, count)
    for title in articles:
        tf_idf_matrix.append(tf_idf(articles[title], count, idf))

    tf_idf_matrix = np.array(tf_idf_matrix)

    # Fit PCA
    pca = PCA(n_components=3)
    pca.fit(tf_idf_matrix)
    pca_data = pca.transform(tf_idf_matrix)
  
    # Plot pca_data
    import plotly.express as px
    df = {"pca1": pca_data[:, 0], "pca2": pca_data[:, 1], "pca3": pca_data[:, 2]}
    fig = px.scatter_3d(df, x='pca1', y='pca2', z='pca3')
    fig.show()
