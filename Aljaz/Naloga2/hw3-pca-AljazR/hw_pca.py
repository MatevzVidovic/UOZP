import numpy as np
import yaml
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd

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


if __name__ == "__main__":

    # Open file
    with open("rtvslo.yaml", "r") as file:
        articles = yaml.safe_load(file)

    # Make dictionary of titles as keys and keywords as values
    articles = {article["title"]: article["gpt_keywords"] for article in articles}

    # Remove duplicated keywords in each article and count keywords
    count = {}
    for article in articles:
        articles[article] = pd.Series(articles[article]).drop_duplicates().to_list()
        for keyword in articles[article]:
            if keyword in count:
                count[keyword] += 1
            else:
                count[keyword] = 1

    # Calculate idf
    idf = {}
    for keyword in count:
        if count[keyword] >= 20:
            idf[keyword] = (np.log2(len(articles) / count[keyword]))

    # Make td-idf matrix
    tf_idf_matrix = {}
    for title in articles:
        tf = 1 / len(articles[title])
        tf_idf_matrix[title] = np.zeros(len(idf))

        for i, keyword in enumerate(idf):
            if keyword in articles[title]:
                tf_idf_matrix[title][i] = tf * idf[keyword]

    tf_idf_matrix = np.array([tf_idf_matrix[title] for title in tf_idf_matrix])

    # Fit PCA
    pca = PCA(n_components=3, tolerance=1e-10, max_iterations=1000)
    pca.fit(tf_idf_matrix)
    pca_data = pca.transform(tf_idf_matrix)
    
    # Plot pca_data
    df = {"pca1": pca_data[:, 0], "pca2": pca_data[:, 1], "pca3": pca_data[:, 2]}
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=df["pca1"], y=df["pca2"], z=df["pca3"], mode="markers", marker=dict(size=3, color="#1f77b4"))) # 17becf
    fig.update_layout(scene=dict(xaxis_title="PCA1", yaxis_title="PCA2", zaxis_title="PCA3"))
      
    # Ready coordinates and labels for loading plot
    top = 20
    labels = []
    coordinates = []
    for eigenvector in pca.eigenvectors:
        sorted_eigenvector = sorted(zip(eigenvector, range(len(eigenvector))), key=lambda x: x[0], reverse=True)

        top_zip = sorted_eigenvector[:top]
        top_indeces = [x[1] for x in top_zip]
        top_keywords = [[k for k in idf.keys()][x] for x in top_indeces]

        for i in range(top):
            print(f"{i+1}. {top_keywords[i]}({top_indeces[i]}) - {top_zip[i][0]:.2f}")
        print()

        # Get coordinates of the top 20 keywords
        for i in top_indeces:
            coordinates.append([pca.eigenvectors[0][i], pca.eigenvectors[1][i], pca.eigenvectors[2][i]])
        labels += top_keywords

    # fig.add_trace(go.Scatter3d(x=[x[0] for x in coordinates], y=[x[1] for x in coordinates], z=[x[2] for x in coordinates], mode="markers+text", text=labels, textposition="top center", marker=dict(color="#1f77b4")))
    # fig.update_layout(scene=dict(xaxis_title="PCA1", yaxis_title="PCA2", zaxis_title="PCA3"))
    # for coordinate in coordinates:
    #     fig.add_trace(go.Scatter3d(x=[0, coordinate[0]], y=[0, coordinate[1]], z=[0, coordinate[2]], mode="lines", line=dict(color="#1f77b4")))
    
    # Clusters we got from gpt-4
    clusters = {
    "Šport": ["tekma", "zmaga", "trener", "ekipa", "poraz", "točke", "obramba", "gol", "košarka", "napad", "reprezentanca", "nogomet", "igralec", "igralci", "sezona", "turnir", "liga", "kvalifikacije", "rezultat", "vratar", "strelec", "zadetek"],
    "Umetnost in kultura": ["umetnost", "razstava", "glasba", "festival", "film", "nagrada", "igralec", "režiser", "umetnik", "literatura", "gledališče", "kultura", "koncert", "življenje", "zgodovina", "predstava", "umetniki", "premiera", "ustvarjanje", "ljubezen"],
    "Gospodarstvo": ["inflacija", "gospodarstvo", "rast", "vlagatelji", "podjetja", "recesija"]
    }
    
    # Loading plot
    for cluster in clusters:
        mean_points = []
        for keyword in clusters[cluster]:
            index = [i for i, x in enumerate(labels) if x == keyword]
            if len(index) > 0:
                mean_points.append(coordinates[index[0]])
        
        mean_point = np.mean(mean_points, axis=0)
        # use color 214, 39, 40
        fig.add_trace(go.Scatter3d(x=[mean_point[0]], y=[mean_point[1]], z=[mean_point[2]], mode="markers+text", text=cluster, textposition="top center", marker=dict(size=12, color="Black"), textfont=dict(size=35, color="Black", ))) # d62728
        fig.add_trace(go.Scatter3d(x=[0, mean_point[0]], y=[0, mean_point[1]], z=[0, mean_point[2]], mode="lines", line=dict(color="Black", width=4), showlegend=False)) # d62728

    fig.update_layout(showlegend=False)
    fig.show()

    # Bar plot of explained variance
    explained_variance = pca.get_explained_variance()
    plt.bar(range(1, 4), explained_variance)
    plt.xticks([1,2,3])
    plt.show()

