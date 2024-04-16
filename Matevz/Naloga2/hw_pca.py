import numpy as np
import yaml
import os
import pickle
from timeit import default_timer as timer
import pandas as pd
import math

PRINTOUT = True

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

    def _fit(self, X: np.ndarray) -> None:
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

        self.centroid = np.mean(X, axis=0)
        X = X - self.centroid

        # PCA transpose trick
        # https://stats.stackexchange.com/questions/392087/what-are-conditions-to-apply-the-transpose-trick-in-pca
        # M = np.cov(X.T)
        # M = np.cov(X)
        M = X.T @ X / X.shape[0]
        # if PRINTOUT:
        #     print(5*"\n")
        #     print("X.shape")
        #     print(X.shape)
        #     print("M.shape")
        #     print(M.shape)
        #     print(5*"\n")

        e_max, lambda_max = self.potencna_metoda(M)

        # if PRINTOUT:
        #     print(5*"\n")
        #     print("e_max.shape")
        #     print(e_max.shape)
        #     print("lambda_max.shape")
        #     print(lambda_max)
        #     print("e_max")
        #     print(e_max)

        for i in range(self.n_components):
            self.eigenvectors.append(list(e_max[:, i].reshape(-1)))
            self.eigenvalues.append(lambda_max[i])


            """
            # This is supposed to make the eigenvectors orthogonal
            # I think it is not working as intended
            for col_ix in range(M.shape[1]):
                M[:, col_ix] = M[:, col_ix] - np.dot(e_max, M[:, col_ix]) * e_max # * lambda_max
            """
        # Dobimo eigenvectorje kot vrstice.
        self.eigenvectors = np.array(self.eigenvectors)
        self.eigenvalues = np.array(self.eigenvalues)

        # print(5*"\n")
        # print("self.eigenvectors")
        # print(self.eigenvectors)
        # print(5*"\n")
    




    def _orthogonalise_and_keep_true(self, eigenvectors: np.ndarray) -> np.ndarray:
        
        for curr_col_ix in range(eigenvectors.shape[1]):

            for col_ix in range(curr_col_ix):
                sub_vec = eigenvectors[:, col_ix]
                # print("np.dot(eigenvectors[:, curr_col_ix], sub_vec)")
                # print(np.dot(eigenvectors[:, curr_col_ix], sub_vec))
                eigenvectors[:, curr_col_ix] = eigenvectors[:, curr_col_ix] - np.dot(eigenvectors[:, curr_col_ix], sub_vec) / np.dot(sub_vec, sub_vec) * sub_vec
            
            if eigenvectors[0, curr_col_ix] < 0:
                    eigenvectors[0, curr_col_ix] = -eigenvectors[0, curr_col_ix]

        return eigenvectors

    def _potencna_metoda(self, M: np.ndarray) -> tuple:
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

        e_max = np.random.rand(M.shape[0], self.n_components)
        # print("e_max")
        # print(e_max)
        # self._orthogonalise(e_max)
        # print("e_max")
        # print(e_max)
        e_max = self._orthogonalise_and_keep_true(e_max)
        e_max = e_max / np.linalg.norm(e_max, axis=0)
        # print("e_max")
        # print(e_max)
        lambda_max = np.zeros(self.n_components)

        # initializetional phase
        for i in range(10):
            e_max = M @ e_max
            # print("e_max before orthogonalisation")
            # print(e_max)
            e_max = self._orthogonalise_and_keep_true(e_max)
            lambda_max = np.linalg.norm(e_max, axis=0)
            e_max = e_max / lambda_max
            # print("e_max after orthogonalisation and scaling")
            # print(e_max)

        for _ in range(self.max_iterations):

            e_max_next =  M @ e_max
            lambda_max_next = np.linalg.norm(e_max_next, axis=0)
            e_max_next = self._orthogonalise_and_keep_true(e_max_next)
            new_norms = np.linalg.norm(e_max_next, axis=0)
            e_max_next = e_max_next / new_norms
            

            cond_1 = np.all(np.abs(lambda_max_next - lambda_max) < self.tolerance)
            cond_2 = np.all(np.linalg.norm(e_max_next - e_max, axis=0) < self.tolerance)
            if cond_1 and cond_2:
            # if False:
                e_max = e_max_next
                lambda_max = lambda_max_next
                if PRINTOUT:
                    # print(5*"\n")
                    print("tolerance reached")
                break

            e_max = e_max_next
            lambda_max = lambda_max_next

            # for i in range(self.n_components):
            #     if e_max[0, i] < 0:
            #         e_max[:,i] = -e_max[:,i]


        return e_max, lambda_max





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

            """
            # This is supposed to make the eigenvectors orthogonal
            # I think it is not working as intended
            for col_ix in range(M.shape[1]):
                M[:, col_ix] = M[:, col_ix] - np.dot(e_max, M[:, col_ix]) * e_max # * lambda_max
            """
        # Dovimo eigenvectorje kot vrstice.
        self.eigenvectors = np.array(self.eigenvectors)
        self.eigenvalues = np.array(self.eigenvalues)

        # print(5*"\n")
        # print("self.eigenvectors")
        # print(self.eigenvectors)
        # print(5*"\n")

    
    def _orthogonalise(self, eigenvec: np.ndarray) -> np.ndarray:
        eigenvec_to_return = eigenvec.copy()
        for vec in self.eigenvectors:
            # print("np.dot(eigenvec, vec)")
            # print(np.dot(eigenvec, vec))
            eigenvec_to_return = eigenvec_to_return - np.dot(eigenvec_to_return, vec) * vec
        return eigenvec_to_return
    

    
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
        # print("e_max")
        # print(e_max)
        # self._orthogonalise(e_max)
        # print("e_max")
        # print(e_max)
        e_max = self._orthogonalise(e_max)
        # print("e_max")
        # print(e_max)
        e_max = e_max / np.linalg.norm(e_max)
        lambda_max = 0

        for _ in range(self.max_iterations):

            e_max_next =  M @ e_max
            e_max_next = self._orthogonalise(e_max_next)
            lambda_max_next = np.linalg.norm(e_max_next)
            e_max_next = e_max_next / lambda_max_next

            cond_1 = np.abs(lambda_max_next - lambda_max) < self.tolerance
            cond_2 = np.linalg.norm(e_max_next - e_max) < self.tolerance
            if cond_1 and cond_2:
                e_max = e_max_next
                lambda_max = lambda_max_next
                if PRINTOUT:
                    print("tolerance reached")
                break

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

    def keep_only_acceptable_keywords(self, acceptable_keywords_list):
        new_gpt_keywords = []
        for keyword in self.gpt_keywords:
            if keyword in acceptable_keywords_list:
                new_gpt_keywords.append(keyword)

        self.gpt_keywords = new_gpt_keywords
    
    def __init__(self, gpt_keywords):
        self.gpt_keywords = gpt_keywords

    def __eq__(self, other):
        return self.gpt_keywords == other.gpt_keywords

class Keywords:

    def __init__(self):
        self.keyword2article_count = dict()
    
    def __eq__(self, value: object) -> bool:
        return self.keyword2article_count == value.keyword2article_count
    
    def add_keyword(self, keyword):

        if keyword not in self.keyword2article_count:
            self.keyword2article_count[keyword] = 0

        self.keyword2article_count[keyword] += 1
    
    def add_article_keywords(self, article_list):

        for article in article_list:

            for keyword in article.gpt_keywords:
                self.add_keyword(keyword)
        
    
    def trim_keywords(self, threshold):

        keywords_to_del = []
        for keyword, count in self.keyword2article_count.items():
            if count < threshold:
                keywords_to_del.append(keyword)

        for keyword in keywords_to_del: 
            self.keyword2article_count.pop(keyword)
    
    # def get_keywords(self):
    #     return list(self.keyword2article_count.keys())


if __name__ == "__main__":

    start = timer()

    # 1000 iters, tol 10e-50, samo cond_2, pa kjer keyword_count ni 1
    # 0 deluje slabo
    # 3 deluje dobro

    # DELUJE ČE PONOVLJENE BESEDE COUNTAMO VEČKRAT - TU SE PA OUTER_PCA ČISTO RAZSUJE.
    # ČE JIH LE 1KRAT PA TRETJI VEKTOR SPET NE DELUJE
    # No, RND_SEED = 3, TOL ((1e-80)**2 / 1959)
    # in MAX_ITERS = 1000 je dober. Pa sta oba conditiona uporabljena.
    # To je po originalnem pca pristopu, kjer je en vektor naenkrat v potencni metodi.
    # Pri prvem vektorju celo reacha tolerance.
    RND_SEED = 3
    MAX_ITERS = 5000
    TOL = ((1e-80)**2 / 1959)    # 1e-50  ((1e-30)**2 * 1959)**(1/2)   # 1959 je keywordov

    # OUTER_PCA deluje dobro, pa dobi isti graf. Samo tretji vektor je pa res dober, kot sem jaz dobil v tistem enem primeru. 
    OUTER_PCA = False

    OUTER_TFIDF = False

    KEYWORDS_AND_TFIDF = True
    FIT_PCA = True
    DATA_STORE = False
    
    if DATA_STORE:
        try:
            os.mkdir("data")
        except:
            pass
    
    """
    If this is True, we will load the data from the file.
    So if e.g. KEYWORDS_AND_TFIDF is True, but DATA_STORE is false,
    we will end up overriding what KEYWORDS_AND_TFIDF did
    when we load the data from the file.""" 
    DATA_LOAD = False
    

    if KEYWORDS_AND_TFIDF:
        # Open the YAML file and load its contents
        # with open("rtvslo.yaml", 'r') as yaml_file:
        #     yaml_data = yaml.safe_load(yaml_file)
        
        yaml_data = yaml.load(open("rtvslo.yaml", "rt"), yaml.CLoader)

        articles = []

        # Process the YAML data
        for item in yaml_data:
            gpt_keywords = item['gpt_keywords']

            # gpt_keywords_to_keep = pd.Series(item["gpt_keywords"]).drop_duplicates().to_list()

            gpt_keywords_to_keep = []

            keywords_unique = set()
            for keyword in enumerate(gpt_keywords):
                # print(keyword)
                keyword_to_add = keyword[1].lower()
                if keyword_to_add not in keywords_unique:
                    keywords_unique.add(keyword_to_add)
                    gpt_keywords_to_keep.append(keyword_to_add)

            articles.append(Article(gpt_keywords_to_keep))
        

        # Find all keywords which appear in at least 20 articles
        # Trim them away.
        keywords = Keywords()
        keywords.add_article_keywords(articles)

        keyword2idf = dict()
        for keyword, article_count in keywords.keyword2article_count.items():
            if article_count >= 20:
                idf = math.log2(len(articles) / article_count)
                keyword2idf[keyword] = idf
        

        acceptable_keywords = list(keyword2idf.keys())
        # keyword2ix = {acceptable_keywords[i] : i for i in range(len(acceptable_keywords))}
        # acceptable_keywords.index(keyword)

        article_keywords_tfs = []# dict()
        for ix, article in enumerate(articles):
            article_keywords_tfs.append(np.zeros(len(acceptable_keywords)))
            for keyword in article.gpt_keywords:
                if keyword in acceptable_keywords:
                    article_keywords_tfs[ix][acceptable_keywords.index(keyword)] += 1
            article_keywords_tfs[ix] /= len(articles[ix].gpt_keywords)


        # data_tfidf = np.array([arcticle_ix2keywords_tfs[x] for x in arcticle_ix2keywords_tfs.keys()])
        articles_tfidf = np.array(article_keywords_tfs)
        for keyword in acceptable_keywords:
            articles_tfidf[:, acceptable_keywords.index(keyword)] *= keyword2idf[keyword]
        
        # articles_tfidf = data_tfidf.T
        articles_tfidf = articles_tfidf.T







        
        # keywords.trim_keywords(20)
        # acceptable_keywords = list(keywords.keyword2article_count.keys())
        
        # # Trim them from articles also (this is needed for tf)
        # for article in articles:
        #     article.keep_only_acceptable_keywords(acceptable_keywords)

        # keywords = Keywords()
        # keywords.add_article_keywords(articles)
        # acceptable_keywords = list(keywords.keyword2article_count.keys())


        # articles_tfidf = np.zeros((len(acceptable_keywords), len(articles)))
        
        # acceptable_keywords_idf = []
        # for keyword in acceptable_keywords:
        #     article_count = keywords.keyword2article_count[keyword]
        #     idf = math.log2(len(articles) / (article_count))
        #     acceptable_keywords_idf.append(idf)
        
        # for article_ix, article in enumerate(articles):

        #     for keyword in article.gpt_keywords:
        #         keyword_ix = acceptable_keywords.index(keyword)
        #         # keyword_count = article.gpt_keywords.count(keyword)
        #         keyword_count = 1 # apparently je asistent rekel, da je to ok
        #         tf = keyword_count / len(article.gpt_keywords)
        #         articles_tfidf[keyword_ix, article_ix] = tf * acceptable_keywords_idf[keyword_ix]
        
        if DATA_STORE:        
            with open("data/articles_tfidf.pkl", 'wb') as file:
                pickle.dump((articles_tfidf, acceptable_keywords), file)


    if DATA_LOAD:
        with open("data/articles_tfidf.pkl", 'rb') as file:
            articles_tfidf_loaded, acceptable_keywords_loaded = pickle.load(file)

            if KEYWORDS_AND_TFIDF and PRINTOUT:
                print(5*"\n")
                print("np.array_equal(articles_tfidf, articles_tfidf_loaded)")
                print(np.array_equal(articles_tfidf, articles_tfidf_loaded))
                print("articles == articles_loaded")
                print(acceptable_keywords == acceptable_keywords)
                print(5*"\n")
                
            articles_tfidf = articles_tfidf_loaded
            acceptable_keywords = acceptable_keywords_loaded
    

    if OUTER_TFIDF:

        import math
        # %%
        data = yaml.load(open("rtvslo.yaml", "rt"), yaml.CLoader)

        # %%

        # TF-IDF vektorizacija

        num_articles = len(data)
        articles = dict()
        keywords_df = dict()
        for x in data:
            articles[x["title"]] = x["gpt_keywords"]
            keywords_unique = pd.Series(x["gpt_keywords"]).drop_duplicates().to_list()
            for w in keywords_unique:
                if w in keywords_df.keys():
                    keywords_df[w] += 1
                else:
                    keywords_df[w] = 1

        keywords_idf = dict()
        for w in keywords_df.keys():
            if keywords_df[w] >= 20:
                keywords_idf[w] = math.log2(num_articles / keywords_df[w])

        keywords = [k for k in keywords_idf.keys()]
        num_keywords = len(keywords)
        keywords_idx = {keywords[i] : i for i in range(len(keywords))}

        keywords_tf = dict()
        for x in articles.keys():
            keywords_tf[x] = np.zeros(num_keywords)
            for w in articles[x]:
                if w in keywords:
                    keywords_tf[x][keywords_idx[w]] += 1
            keywords_tf[x] /= len(articles[x])

        data_tfidf = np.array([keywords_tf[x] for x in keywords_tf.keys()])
        for w in keywords:
            data_tfidf[:, keywords_idx[w]] *= keywords_idf[w]


        articles_tfidf = data_tfidf.T
        acceptable_keywords = keywords
    
    





    articles_tfidf = articles_tfidf.T
    # Now articles are the rows and the keywords are the columns
    # This is in line with how we built the PCA

    if PRINTOUT:
        print(5*"\n")
        print("articles_tfidf")
        print(articles_tfidf)
        print(5*"\n")




    if FIT_PCA:

        PCA_model = PCA(n_components=3, max_iterations=MAX_ITERS, tolerance=TOL, rnd_seed=RND_SEED)
        PCA_model.fit(articles_tfidf)



        if DATA_STORE:
            with open("data/PCA_model.pkl", 'wb') as file:
                pickle.dump(PCA_model, file)


    if DATA_LOAD:
        # Load the list of objects from the file
        with open("data/PCA_model.pkl", 'rb') as file:
            PCA_model = pickle.load(file)



    articles_tfidf_transformed = PCA_model.transform(articles_tfidf)



    if OUTER_PCA:
        # Step 1: Compute the covariance matrix
        covariance_matrix = np.cov(articles_tfidf, rowvar=False)

        # Step 2: Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Step 3: Sort the eigenvectors by decreasing eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Step 4: Choose the number of principal components (e.g., n_components)
        n_components = 3  # choose the number of components you want to keep

        # Step 5: Select the first n_components eigenvectors
        principal_components = eigenvectors[:, :n_components]

        # Step 6: Project the data onto the principal components to get transformed data
        transformed_data = np.dot(articles_tfidf, principal_components)

        # get explained variance
        explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)

        # transformed_data now contains your data in the PCA space

        # print(5*"\n")
        # print("np.all(np.abs(transformed_data - articles_tfidf_transformed) < 1e-5)")
        # print("np.max(np.abs(  (articles_tfidf_transformed - transformed_data)  ))")
        # print(np.max(np.abs(  (articles_tfidf_transformed - transformed_data)  )))
        # print("np.max(np.abs(  (articles_tfidf_transformed - transformed_data) / (transformed_data+1e-10)  ))")
        # print(np.max(np.abs(  (articles_tfidf_transformed - transformed_data) / (transformed_data+1e-10)  )))
        # print(5*"\n")
        print("principal_components.shape")
        print(principal_components.shape)
     
        articles_tfidf_transformed = transformed_data
        PCA_model.eigenvectors = principal_components.T # To make it compatible with the original implementation.
        # PCA_model.eigenvectors = [principal_components[:, i] for i in range(n_components)]
        PCA_model.eigenvalues = eigenvalues[:n_components]





    if PRINTOUT:
        print(5*"\n")
        print("articles_tfidf_transformed")
        print(articles_tfidf_transformed)



    # use vispy to plot the data in 3D
    from vispy import scene, app #, visuals

    # canvas = scene.SceneCanvas(keys='interactive', show=True, app='pyqt5')
    # canvas = scene.SceneCanvas(keys='interactive', show=True, app='egl')
    # canvas = scene.SceneCanvas(keys='interactive', show=True, app='pyqt6')
    canvas = scene.SceneCanvas(keys='interactive', show=True)

    view = canvas.central_widget.add_view()

    scatter = scene.visuals.Markers()
    scatter.set_data(articles_tfidf_transformed, edge_color=None, face_color=(1, 1, 1, .5), size=5)
    view.add(scatter)

    view.camera = 'turntable'
    canvas.show()

    end = timer()
    if PRINTOUT:
        print(5*"\n")
        print("Elapsed time in seconds:")
        print(end - start)
    # print("here")

    explained_var = PCA_model.get_explained_variance()
    if OUTER_PCA:
        explained_var = explained_variance
    print(5*"\n")
    print("PCA_model.get_explained_variance()")
    print(explained_var)


    # Test for what the eigenvectors are
    for i in range(3):
        # print(PCA_model.eigenvectors[i])
        to_sort = [(ix, val) for ix, val in enumerate(PCA_model.eigenvectors[i])]
        sorted_list = sorted(to_sort, key= lambda a: a[1], reverse=True)
        best_10 = sorted_list[:10]
        next_10 = sorted_list[10:20]
        worst_10 = sorted_list[-10:]
        print(5*"\n")
        print(f"PCA_model.eigenvectors[{i}]")
        print("best_10")
        print(best_10)
        print("next_10")
        print(next_10)
        print("worst_10")
        print(worst_10)

        best_10_keywords = [acceptable_keywords[ix] for ix, _ in best_10]
        next_10_keywords = [acceptable_keywords[ix] for ix, _ in next_10]
        worst_10_keywords = [acceptable_keywords[ix] for ix, _ in worst_10]
        print("best_10_keywords")
        print(best_10_keywords)
        print("next_10_keywords")
        print(next_10_keywords)
        print("worst_10_keywords")
        print(worst_10_keywords)


    



    app.run()





    









    

