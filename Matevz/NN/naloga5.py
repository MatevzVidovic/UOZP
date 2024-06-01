




"""

UOZP naloga:
- prebrat podatke in jih dat v parseano obliko
- delitev na testno, validacijsko in učno. Pa mogoče kar nek CV izvajat ker pomoje to hočejo.

- izbrane kategorije od spodaj navzgor:
(tfidf := stemming/NLTK za slo. Potem tfidf in rezanje manj pojavljenih/manj pomembnih po feature selectionu.)

- gpt_keywords - tfidf. Teh baje pogosto ni. So za: target_topics = [“gospodarstvo”, “sport”, “kultura”, 
“zabava-in-slog”]. Tako da samo za te modele jih upoštevaj.
- keywords -  tfidf
- !!!topics!!! - preverit koliko jih je. Če jih je več, naredit urejen tuple kot key. Vse te možnosti se
 potem one-hot encode-a. Pa odreže se te, ki nimajo nekega velikega vpliva po feature selectionu, oziroma
   ki se premalo pojavijo. Ne naredim tega, da enega od možnih ne one-ho-encodeaš, da bi bil tvoj baseline
     in je pravilno parametriziran model - ker moje baseline je, da pač ni ujemanja topica s temi, ki so
       one-ho encodani in je to potem baseline.
- lead - tfidf
- naredit number of figures kot svojo značilko in samo dat kot šrtevilko
- ??paragraphs??
- ??title??
- date - dobit ven leto, mesec, in uro. Leto vstavit kot številko in le dobit linearen trend.
 Mesec tudi kot številko ampak ne le linearno (dodat sinusoido, kosinusoido, koren, in kvadrat
   (da bolje ujamem nihanja skozi leto). Ure pa dat v neke buckete po recimo 3 ure in to one-hot encodeat
     - to je pomembno ker od 22.00 ni več komentarjev do jutra in zato to zelo vpliva za pozne članke in tako.
- !!!authors!!! - isto kot topics.
- !!!URL!!! - parseat ker se v njem skriva glavna kategorija in tudi sub-categories.
 Potem za te subcategories isto kot za topics.

- mogoče za vsak topic in (1 kategorija za pomanjkanje topica) naredit svoj model. 
Zna biti zelo uporabno. Lahko model poponoma prilagodi topicu
 (tako da vsak ta model potrebuje veliko manj parametrov pa imamo v celoti višji accuracy.
   In je hitreje za trenirat. Je pa problem, da tako zelo zmanjšaš število training primerov
     ki applyjajo na posamezen model. Zato naredi to samo za večje topice in poskrbi, da je v tej "no topic"
       kategoriji vseeno dosti primerov.

- distribucija števila komentarjev je nabasana proti nižjemu številu in ima potem zelo dolg rep. 
To mi zna zjebat model, ker bojo za kvadratne napake ekstra vlekli v svojo smer. Mogoče odstranit te osamelce.
 Mogoče apllyat logaritem na število komentarjev. Mogoče samo za komentarje nad nekim številom 
 (recimo v 90. percentilu) narediti mapping v: 90. percentil št. kom. + log (pravo število komentarjev). 
 Tako bodo še vedno lepo uporabni za treniranje in jih bo fittalo kot visoko število komentarjev, 
 pa ne bodo uničili treniranja.

- mogoče supervised PCA, ampak se mi zdajle ne da. Pa ne vem če ima to kakšna od dovoljenih knjižnic.

- potem vse te matrike tfidfja in one-hot encodinga, ki sem jih prej naredil, concatat v eno matriko
 in jih dat v linear regression model z L1 regularizacijo.
- z validacijsko množico določit stopnjo regularizacije - kot nek procent R2 ko je regularizacija skoraj nična.

- testirat na testni

- s tem procesom delat 2CV5.
- potem zgradit model z deljenjem samo na train in valid - in to je moj končni model.
 (kot smo za CV govorili na predavanjih)
 """




import gzip
import json

import pickle

from data_preparation import prepare_data, DataTopic

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import mutual_info_regression

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from scipy.stats import spearmanr
from scipy.stats import kendalltau


from scipy.sparse.linalg import norm



PRINTOUT = False
MAIN_PRINTOUT = True
PLOTS = False


YEALD_MAT_PARAMS = {
    "normalizer" : "standardize", # "dont" or "L2". 
    # Or it is the actual Normalizer object - for the test data.

    "comment_func" : "root", # za 500 je izboljšanje
    # can be:
    # None for no cap - great for running on test data
    # integer for absolute cap, 
    # "perc_and_root" for perc percentile value + (number - perc percentile value)^(1/root).
    # "root" for number^(1/root)
    # "log" for np.log(number+1)
    
    # "perc" : 98,
    "root" : 2,
    
    # This is not at all supported yet. Keep it False.
    # The big problem is that how to pass parameters into testing calls of yeald_mat...
    # and not have it do the capping.
    # "pca" : False,
    # "pca_n": 100, # num of pca components
}


# ohe_cutoff = 150
# tfidf_cutoff = 150

# HYPER_PARAMETERS = {

#     "max_iter" : 10,

#     "URLs" : ohe_cutoff,
#     "authors" : ohe_cutoff,
#     "leads" : ohe_cutoff,
#     "keywords" : tfidf_cutoff,
#     "gpt_keywords" : tfidf_cutoff,
#     "topics" : tfidf_cutoff,
#     "alpha" : 0.0005,
#     "method" : "Lasso", #Ridge in Basic ne delata # "Basic", "Ridge" or "Lasso"
#     # or LassoCV, RidgeCV, ElasticNetCV
# }


def mutual_info_best_ixs(X, y, n_best):

    mutual_infos = mutual_info_regression(X, y, n_neighbors=20)

    selector = SelectKBest(mutual_info_regression, k=n_best)
    selector.fit(X, y)
    ixs = selector.get_support(indices=True)

    return ixs, mutual_infos

def plot_me(mutual_infos, label=""):
    shower = sorted(mutual_infos)
    plt.plot(shower, label=label)




def get_new_idf_count_from_tfidf_matrix(tfidf_matrix):
    return np.sum(tfidf_matrix > 0, axis=0)

def tfidf_word_importances(tfidf_matrix):

    return np.sum(tfidf_matrix, axis=0)

def word_importances(matrix, y):
    
    # using pearson correlation:

    # print("here 1")

    returner = np.zeros(matrix.shape[1])

    for ix in range(matrix.shape[1]):
        if isinstance(matrix, np.ndarray):
            # with np.errstate(divide='ignore', invalid='ignore'):
            returner[ix] = np.corrcoef(matrix[:, ix].reshape(-1), y.reshape(-1))[0, 1]
            # print("here 1.1")
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                returner[ix] = np.corrcoef(matrix[:, ix].toarray().reshape(-1), y.reshape(-1))[0, 1]
            # print("here 1.2")

    returner = np.abs(returner)

    # print("here 2")

    sorted_ixs = np.argsort(returner)
    sorted_returner = returner[sorted_ixs]
    return sorted_ixs, sorted_returner


def build_matrix_from_data_topic_and_change_DT_accordingly(data_topic, model_type="single_topic", hyper_parameters=None):
    # Be aware that this changes the data topic!
    # for URLs, authors, topics_encoded it trims them to the best 150 features. 
    # type can be "single_topic", "grouped_topic", "unrecognised_topic", "all_topics_together"

    try:
        assert type(data_topic) == DataTopic
    except:
        print(type(data_topic))
        assert type(data_topic) == DataTopic


    if False:
        URLs_best_ixs, URLs_sorted_pearson_corrs = word_importances(data_topic.URLs, data_topic.num_of_comments)
        if hyper_parameters is None:
            URLs_chosen_ixs = URLs_best_ixs[-150:]
        else:
            URLs_chosen_ixs = URLs_best_ixs[-hyper_parameters["URLs"]:]
        
        if PRINTOUT:
            print("URLs_best_ixs: ")
            print(URLs_best_ixs)
            print("data_topic.URL_names: ")
            print(data_topic.URL_names)

        authors_best_ixs, authors_sorted_pearson_corrs = word_importances(data_topic.authors, data_topic.num_of_comments)
        if hyper_parameters is None:
            authors_chosen_ixs = authors_best_ixs[-150:]
        else:
            authors_chosen_ixs = authors_best_ixs[-hyper_parameters["authors"]:]



        # samo razlika, da single in unrecognised ne delata z topics in jih zato ne režemo
        # Zato imata primera pač drugačen trimming

        if model_type == "grouped_topic" or model_type == "all_topics_together":

            topics_best_ixs, topics_sorted_pears_corrs = word_importances(data_topic.topics_encoded, data_topic.num_of_comments)
            if hyper_parameters is None:
                topics_chosen_ixs = topics_best_ixs[-150:]
            else:
                topics_chosen_ixs = topics_best_ixs[-hyper_parameters["topics"]:]

            if PRINTOUT:
                plot_me(topics_sorted_pears_corrs, label="topics")
        
        elif model_type == "single_topic" or model_type == "unrecognised_topic":

            topics_chosen_ixs = None


        data_topic.one_hot_encoded_chosen_ixs_trim(URLs_chosen_ixs, authors_chosen_ixs, topics_chosen_ixs)
















        # print("leads 1")
        leads_best_ixs, leads_sorted_pearson_corrs = word_importances(data_topic.leads_tfidf, data_topic.num_of_comments)
        # print("leads 2")
        keywords_best_ixs, keywords_sorted_pearson_corrs = word_importances(data_topic.keywords_tfidf, data_topic.num_of_comments)
        gpt_keywords_best_ixs, gpt_keywords_sorted_pearson_corrs = word_importances(data_topic.gpt_keywords_tfidf, data_topic.num_of_comments)
        

        if hyper_parameters is None:
            leads_chosen_ixs = leads_best_ixs[-150:]
            keywords_chosen_ixs = keywords_best_ixs[-150:]
            gpt_keywords_chosen_ixs = gpt_keywords_best_ixs[-150:]

        else:
            leads_chosen_ixs = leads_best_ixs[-hyper_parameters["leads"]:]
            keywords_chosen_ixs = keywords_best_ixs[-hyper_parameters["keywords"]:]
            gpt_keywords_chosen_ixs = gpt_keywords_best_ixs[-hyper_parameters["gpt_keywords"]:]


        data_topic.tfidf_chosen_ixs_trim(leads_chosen_ixs, keywords_chosen_ixs, gpt_keywords_chosen_ixs)

        chosen_ixs_dict = {
            "leads_chosen_ixs" : leads_chosen_ixs,
            "keywords_chosen_ixs" : keywords_chosen_ixs,
            "gpt_keywords_chosen_ixs" : gpt_keywords_chosen_ixs}




        if PRINTOUT:
            plot_me(URLs_sorted_pearson_corrs, label="URLs")
            plot_me(authors_sorted_pearson_corrs, label="authors")
            plot_me(leads_sorted_pearson_corrs, label="leads")
            plot_me(keywords_sorted_pearson_corrs, label="keywords")
            plot_me(gpt_keywords_sorted_pearson_corrs, label="gpt_keywords")
            plt.legend(loc="upper center")
            plt.title(str(data_topic.topic) + ", " + str(model_type))
            plt.show(block=False)

            plt.figure()
    else:
        chosen_ixs_dict = None
            



    data_matrix, y, _, normalizer = data_topic.yeald_complete_matrix(model_type=model_type, params=YEALD_MAT_PARAMS)



    # if PRINTOUT:
    #     input("Waiting...")
    #     plt.clf()






    # This is going to have to be diferent for:
    # - topic data topics
    # - and for the grouped data topic where "stevilke" and "kolumne" and "znanost in tehnologija" are grouped
    # - and also, make a model without topics with all data topics. - this is meant for unrecognised data topics.


    # DONE with pearson corr:
    # do feature selection on the leads, keywords and gpt_keywords
    # Do mutual informaion from sklearn and choose best 150 or sth.
    # Make a graph of mutual information and show it so we can gauge it.

    # concat the data into one matrix

    # do an L2 normalization on all columns.
    # The one-hot colmns remain interpretable (two values for how much in pos or neg we go)
    # Tfidf needs it, because previous "l2" was for rows, not columns. It was for documents.
    # Month functions can get normalized also. Their scale wont be from func(0) to func(1) anymore, but scaled. But who cares.
    # Its just a function with a constant in front now.
    # And all of this allows us to use regularization that doesn't discriminate towards larger normed columns.
    # So these are small prices to pay.

    # Iterate:
    # build the model with an L1 regularization
    # check against validation set
    # make a formula for the tradoff between:
    # loss of R2 compared to the best model, and the number of parameters that got reduced..

    # Then join all these models into one class that makes the entire prediction for a list of test cases.
    # Choosing the model based on the topic, transforming the test example into the correct one-ot encodings and such, and putting it in the model.
    





    return data_matrix, y, chosen_ixs_dict, normalizer





# import torch
from model import RunModel


class WrapperModel:

    def __init__(self, json_list, hyper_parameters=None):


        _, _, all_together_DT, vectorizers, _, _= prepare_data(json_list)
        
        _, _, self.all_together_model_chosen_ixs, self.all_together_normalizer = build_matrix_from_data_topic_and_change_DT_accordingly(all_together_DT, model_type="all_topics_together", hyper_parameters=hyper_parameters)

        self.all_together_DT = all_together_DT.get_DT_keeping_only_names()
        self.vectorizers = vectorizers

        self.yeald_mat_params = YEALD_MAT_PARAMS.copy()
        self.yeald_mat_params["normalizer"] = self.all_together_normalizer

        # self.all_together_matrix, self.all_together_y, 

        # # Get cpu, gpu or mps device for training.
        # self.device = (
        #     "cuda"
        #     if torch.cuda.is_available()
        #     else "mps"
        #     if torch.backends.mps.is_available()
        #     else "cpu"
        # )
        # print(f"Using {self.device} device")


    # def give_data(self):
    #     return self.all_together_matrix, self.all_together_y


    # def load_model(self, model_path):

    #     model_parameters = {
    #         # layer sizes
    #         "chosen_num_of_features" : 5000,
    #         "second_layer_size" : 1000,
    #         "middle_layer_size" : 700,

    #         # general parameters
    #         "dropout" : 0.1,
    #         "leaky_relu_alpha" : 0.1,
    #         "learning_rate" : 1e-3,
    #     }
            
    #     model = Unet(**model_parameters)

    #     model = model.to(self.device)

    #     model.load_state_dict("data/latest.pth")

    #     self.model = model
    
    def get_X_y_from_list_of_jsons(self, cases_list):
        _, _, all_data_topic, _, _, _ = prepare_data(cases_list, all_vectorizers=self.vectorizers, is_test_data=True)


        all_data_topic.tfidf_chosen_ixs_trim(self.all_together_model_chosen_ixs)
        # print("all_data_topic")
        # print(all_data_topic.topics_encoded)
        all_data_topic.one_hot_encoding(self.all_together_DT)
        # print("all_data_topic")
        # print(all_data_topic.topics_encoded)
        X_test, y_test, _, _ = all_data_topic.yeald_complete_matrix(model_type="all_topics_together", params=self.yeald_mat_params, give_names=True)
        return X_test, y_test

    def train_me(self, train_cases_list):

        X_train, y_train = self.get_X_y_from_list_of_jsons(train_cases_list)
        model = RunModel(X_train, y_train)
        model.load_train_test_and_save()



    def predict(self, test_cases_list):

        original_comm_func = self.yeald_mat_params["comment_func"]

        self.yeald_mat_params["comment_func"] = None
        X_test, y_test = self.get_X_y_from_list_of_jsons(test_cases_list)
        
        model = RunModel(X_test, y_test)
        y_pred = model.predict()

        if original_comm_func == "root":
            y_pred = y_pred ** self.yeald_mat_params["root"]
        elif original_comm_func == "log":
            y_pred = np.exp(y_pred) - 1
            y_pred[y_pred < 0] = 0
            
        
        return X_test, y_test, y_pred



def compare_y_test_and_pred(y_test, y_pred):
    


    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    rmse = (mean_squared_error(y_test, y_pred))**(1/2)
    mae = mean_absolute_error(y_test, y_pred)
    spearman = spearmanr(y_test, y_pred)
    kendall = kendalltau(y_test, y_pred)

    returner = [r2, rmse, mae, spearman.statistic, kendall.statistic]

    if MAIN_PRINTOUT:
        print(f"R²: {r2}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"Spearman: {spearman}")
        print(f"Kendall: {kendall}")
            
    return returner

