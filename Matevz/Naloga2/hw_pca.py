import numpy as np
import yaml
import os
import pickle

class PCA:
    def __init__(
        self,
        n_components: int,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
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

        self.centroid = None

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
        
        np.random.seed(self.rnd_seed)

        # print(5*"\n")
        # print("X")
        # print(X)
        # print("np.mean(X, axis=0)")
        # print(np.mean(X, axis=0))
        # print(5*"\n")

        self.centroid = np.mean(X, axis=0)
        X = X - self.centroid
        M = np.cov(X.T)
        # M = np.cov(X)


        for _ in range(self.n_components):
            e_max, lambda_max = self.potencna_metoda(M)
            self.eigenvectors.append(e_max)
            self.eigenvalues.append(lambda_max)

            # print(5*"\n")
            # print("e_max")
            # print(e_max)
            # print(5*"\n")

            # A je tole ziher rpov?
            # M = M - lambda_max * np.outer(e_max, e_max)

            for col_ix in range(M.shape[1]):
                M[:, col_ix] = M[:, col_ix] - np.dot(e_max, M[:, col_ix]) * e_max # * lambda_max

        # Dovimo eigenvectorje kot vrstice.
        self.eigenvectors = np.array(self.eigenvectors)
        self.eigenvalues = np.array(self.eigenvalues)

        # print(5*"\n")
        # print("self.eigenvectors")
        # print(self.eigenvectors)
        # print(5*"\n")
        

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
        # print(5*"\n")
        # print("M")
        # print(M)
        # print("M.shape")
        # print(M.shape)
        # print(5*"\n")

        e_max = np.random.rand(M.shape[0])
        e_max = e_max / np.linalg.norm(e_max)
        lambda_max = 0

        for _ in range(self.max_iterations):

            e_max_next =  M @ e_max
            lambda_max_next = np.linalg.norm(e_max_next)
            e_max_next = e_max_next / lambda_max_next

            # cond_1 = np.abs(lambda_max_next - lambda_max) < self.tolerance
            # cond_2 = np.linalg.norm(e_max_next - e_max) < self.tolerance
            # if cond_2 and cond_1:
            #     e_max = e_max_next
            #     lambda_max = lambda_max_next
            #     break

            e_max = e_max_next
            lambda_max = lambda_max_next

            if e_max[0] < 0:
                e_max = -e_max


        return e_max, lambda_max



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

        X = X - self.centroid

        # E is a matrix with eigenvectors as columns
        E = np.array(self.eigenvectors).T

        # print(5*"\n")
        # print("self.eigenvectors")
        # print(self.eigenvectors)
        # print("self.eigenvectors.shape")
        # print(self.eigenvectors.shape)
        # print("E")
        # print(E)
        # print("E.shape")
        # print(E.shape)
        # print(5*"\n")
        
        return X @ E
        

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
        
        # returner = self.eigenvalues[:self.n_components] / np.sum(self.eigenvalues)
        
        # returner = self.eigenvalues[:self.n_components]
        # returner /= np.sum(returner)

        returner = self.eigenvalues[:self.n_components]

        return returner
        # return self.eigenvalues / np.sum(self.eigenvalues)

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
        
        # E is a matrix with eigenvectors as columns
        E = np.array(self.eigenvectors).T

        X = X @ E.T

        return X + self.centroid
    




class Article:
    def __init__(self, title, url, gpt_keywords):
        self.title = title
        self.url = url
        self.gpt_keywords = gpt_keywords

    def __str__(self):
        return f"Title: {self.title}\nURL: {self.url}\nGPT Keywords: {self.gpt_keywords}\n"

    def __repr__(self):
        return f"Title: {self.title}\nURL: {self.url}\nGPT Keywords: {self.gpt_keywords}\n"

    def __eq__(self, other):
        return self.title == other.title and self.url == other.url and self.gpt_keywords == other.gpt_keywords

    def __hash__(self):
        return hash((self.title, self.url, self.gpt_keywords))

class Keywords:

    def __init__(self):
        self.keyword2count = dict()
    
    def __eq__(self, value: object) -> bool:
        return self.keyword2count == value.keyword2count
    
    def add_keyword(self, keyword):

        if keyword not in self.keyword2count:
            self.keyword2count[keyword] = 0

        self.keyword2count[keyword] += 1
    
    def add_article_keywords(self, article_list):
        for article in article_list:
            for keyword in article.gpt_keywords:
                self.add_keyword(keyword)
    
    def trim_keywords(self, threshold):

        keywords_to_del = []
        for keyword, count in self.keyword2count.items():
            if count < threshold:
                keywords_to_del.append(keyword)

        for keyword in keywords_to_del: 
            self.keyword2count.pop(keyword)
    
    # def get_keywords(self):
    #     return list(self.keyword2count.keys())


if __name__ == "__main__":

    create_important_keywords_file = False
    do_tfidf = False
    fit_PCA = False
    printout = True

    if create_important_keywords_file:
        # Open the YAML file and load its contents
        with open("rtvslo.yaml", 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)

        articles = []

        
        # Process the YAML data
        for item in yaml_data:
            gpt_keywords = item.get('gpt_keywords', [])
            title = item.get('title', '')
            url = item.get('url', '')

            articles.append(Article(title, url, gpt_keywords))
        
        keywords = Keywords()
        keywords.add_article_keywords(articles)
        keywords.trim_keywords(20)
        acceptable_keywords = list(keywords.keyword2count.keys())
        
        article_keywords_list = []
        for article in articles:
            article_keywords = [keyword for keyword in article.gpt_keywords if keyword in acceptable_keywords]
            article_keywords_list.append(" ".join(article_keywords))


        


        try:
            os.mkdir("data")
        except:
            pass
        

        
        with open("data/article_keywords_list.pkl", 'wb') as file:
            pickle.dump(article_keywords_list, file)


    # Load the list of objects from the file
    with open("data/article_keywords_list.pkl", 'rb') as file:
        article_keywords_list_loaded = pickle.load(file)
    
    if create_important_keywords_file:
        print("article_keywords_list_loaded")
        print(article_keywords_list_loaded)
        print("article_keywords_list_loaded == article_keywords_list")
        print(article_keywords_list_loaded == article_keywords_list)



        """
        # If you go about saving the dictionary, which doesn't work for tf_idf then:
        
        with open("data/keywords_object.pkl", 'wb') as file:
            pickle.dump(keywords, file)


    # Load the list of objects from the file
    with open("data/keywords_object.pkl", 'rb') as file:
        important_keywords_loaded = pickle.load(file)
        assert type(important_keywords_loaded) == Keywords, f"Expected Keywords, but got {type(important_keywords_loaded)}"
        
        
    if create_important_keywords_file:    
        print(important_keywords_loaded)
        print("important_keywords_loaded == keywords")
        print(important_keywords_loaded == keywords)
        """

        """
        # How to do it, if we store strings:

        important_keywords = keywords.get_keywords()
        print(important_keywords)

        with open("data/important_keywords.yaml", 'w') as yaml_file:
            yaml.dump(important_keywords, yaml_file)


    with open("data/important_keywords.yaml", 'r') as yaml_file:
        important_keywords_loaded = yaml.safe_load(yaml_file)
        """
    










    if do_tfidf:

        from sklearn.feature_extraction.text import TfidfVectorizer


        tfidf_vectorizer = TfidfVectorizer()
        articles_tfidf_sparse = tfidf_vectorizer.fit_transform(article_keywords_list_loaded)
        
        if printout:
            # Print the TF-IDF matrix
            print("articles_tfidf_sparse")
            print(articles_tfidf_sparse)

            # Print the feature names (words)
            print("tfidf_vectorizer.get_feature_names_out()")
            print(tfidf_vectorizer.get_feature_names_out())

        with open("data/articles_tfidf_sparse.pkl", 'wb') as file:
            pickle.dump(articles_tfidf_sparse, file)


    # Load the list of objects from the file
    with open("data/articles_tfidf_sparse.pkl", 'rb') as file:
        articles_tfidf_sparse_loaded = pickle.load(file)


    if do_tfidf:
        # print("articles_tfidf_sparse_loaded != articles_tfidf_sparse")
        # print(articles_tfidf_sparse_loaded != articles_tfidf_sparse)

        print("Checking if articles_tfidf_sparse_loaded \n is equal to articles_tfidf_sparse:")
        print("matrix1.shape == matrix2.shape and np.array_equal(matrix1.data, matrix2.data) and np.array_equal(matrix1.indices, matrix2.indices) and np.array_equal(matrix1.indptr, matrix2.indptr)")
        print(articles_tfidf_sparse_loaded.shape == articles_tfidf_sparse.shape and np.array_equal(articles_tfidf_sparse_loaded.data, articles_tfidf_sparse.data) and np.array_equal(articles_tfidf_sparse_loaded.indices, articles_tfidf_sparse.indices) and np.array_equal(articles_tfidf_sparse_loaded.indptr, articles_tfidf_sparse.indptr))





    articles_tfidf = articles_tfidf_sparse_loaded.toarray()
    # Print the TF-IDF matrix shape
    # print("articles_tfidf.shape")
    # print(articles_tfidf.shape)

    # Now articles are the rows and the keywords are the columns
    # This is in line with how we built the PCA
    articles_tfidf = articles_tfidf.T




    if fit_PCA:

        PCA_model = PCA(n_components=3)
        PCA_model.fit(articles_tfidf)

        with open("data/PCA_model.pkl", 'wb') as file:
            pickle.dump(PCA_model, file)


    # Load the list of objects from the file
    with open("data/PCA_model.pkl", 'rb') as file:
        PCA_model = pickle.load(file)





    
    




    articles_tfidf_transformed = PCA_model.transform(articles_tfidf)

    print("articles_tfidf_transformed")
    print(articles_tfidf_transformed)

    if True:
        # use vispy to plot the data in 3D
        from vispy import scene, app #, visuals

        # canvas = scene.SceneCanvas(keys='interactive', show=True, app='pyqt5')
        # canvas = scene.SceneCanvas(keys='interactive', show=True, app='egl')
        canvas = scene.SceneCanvas(keys='interactive', show=True, app='pyqt6')
        # canvas = scene.SceneCanvas(keys='interactive', show=True)

        view = canvas.central_widget.add_view()

        scatter = scene.visuals.Markers()
        scatter.set_data(articles_tfidf_transformed, edge_color=None, face_color=(1, 1, 1, .5), size=5)
        view.add(scatter)

        view.camera = 'turntable'
        canvas.show()
        # print("here")
        app.run()





    









    

