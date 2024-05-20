




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


import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from data_preparation import prepare_data, DataTopic

PRINTOUT = False











from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import mutual_info_regression

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


YEALD_MAT_PARAMS = {

    "cap_comment_n" : "perc_and_root", # za 500 je izboljšanje
    # can be None for no cap, integer for absolute cap, or "perc_and_root" for
    # perc percentile value + (number - perc percentile value)^(1/root).
    "perc" : 99,
    "root" : 4,
    
    # This is not at all supported yet. Keep it False.
    # The big problem is that how to pass parameters into testing calls of yeald_mat...
    # and not have it do the capping.
    "pca" : False,
    "pca_n": 100, # num of pca components
}



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

    returner = np.zeros(matrix.shape[1])

    for ix in range(matrix.shape[1]):
        if isinstance(matrix, np.ndarray):
            returner[ix] = np.corrcoef(matrix[:, ix].reshape(-1), y.reshape(-1))[0, 1]
        else:
            returner[ix] = np.corrcoef(matrix[:, ix].toarray().reshape(-1), y.reshape(-1))[0, 1]


    returner = np.abs(returner)

    sorted_ixs = np.argsort(returner)
    sorted_returner = returner[sorted_ixs]

    return sorted_ixs, sorted_returner


def build_model_from_data_topic(data_topic, model_type="single_topic", hyper_parameters=None):
    # Be aware that this changes the data topic!
    # for URLs, authors, topics_encoded it trims them to the best 150 features. 
    # type can be "single_topic", "grouped_topic", "unrecognised_topic", "all_topics_together"

    try:
        assert type(data_topic) == DataTopic
    except:
        print(type(data_topic))
        assert type(data_topic) == DataTopic



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


















    leads_best_ixs, leads_sorted_pearson_corrs = word_importances(data_topic.leads_tfidf, data_topic.num_of_comments)
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

        




    data_matrix, y = data_topic.yeald_complete_matrix(model_type=model_type, params=YEALD_MAT_PARAMS)



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
    


    print(10*"hej!\n")


    # model = LinearRegression()
    # model = Ridge(alpha=0.5)
    if hyper_parameters is None:
        model = Lasso(alpha=0.5)
    elif hyper_parameters["method"] == "Basic":
        model = LinearRegression()
    elif hyper_parameters["method"] == "Ridge":
        model = Ridge(alpha=hyper_parameters["alpha"], max_iter=hyper_parameters["max_iter"])
    elif hyper_parameters["method"] == "Lasso":
        model = Lasso(alpha=hyper_parameters["alpha"], max_iter=hyper_parameters["max_iter"])

    model.fit(data_matrix, y)

    return model, chosen_ixs_dict








class Model:

    def __init__(self, topic_2_train_DT, grouped_topics_DT, all_together_DT, vectorizers, hyper_parameters=None):

        self.topic_2_train_DT = topic_2_train_DT
        self.grouped_topics_DT = grouped_topics_DT
        self.all_together_DT = all_together_DT
        self.vectorizers = vectorizers

        self.single_topic_models = {}
        self.single_topic_models_chosen_ixs = {}
        for topic in topic_2_train_DT.keys():
            self.single_topic_models[topic], self.single_topic_models_chosen_ixs[topic] = build_model_from_data_topic(topic_2_train_DT[topic], model_type="single_topic", hyper_parameters=hyper_parameters)
        
        if grouped_topics_DT is not None:
            self.grouped_topic_model, self.grouped_topic_model_chosen_ixs = build_model_from_data_topic(grouped_topics_DT, model_type="grouped_topic", hyper_parameters=hyper_parameters)
            self.grouped_topic_model_topics = grouped_topics_DT.topic.split(",")
        else:
            self.grouped_topic_model = None
            self.grouped_topic_model_chosen_ixs = None
            self.grouped_topic_model_topics = []

        self.unrecognised_topic_model, self.unrecognised_topic_model_chosen_ixs = build_model_from_data_topic(all_together_DT, model_type="unrecognised_topic", hyper_parameters=hyper_parameters)

        self.all_together_model, self.all_together_model_chosen_ixs = build_model_from_data_topic(all_together_DT, model_type="all_topics_together", hyper_parameters=hyper_parameters)


    def predict(self, test_cases_list, all_together_model=False):

        topic_2_train_DT, _, all_data_topic, _, topic_2_ixs, remaining_original_article_ixs = prepare_data(test_cases_list, all_vectorizers=self.vectorizers, is_test_data=True)

        if all_together_model:
            all_data_topic.tfidf_chosen_ixs_trim(**self.all_together_model_chosen_ixs)
            all_data_topic.one_hot_encoding(self.all_together_DT)
            X_test, y_test = all_data_topic.yeald_complete_matrix(model_type="all_topics_together")
            y_test = np.reshape(y_test, (-1))
            y_pred = self.all_together_model.predict(X_test)
            return X_test, y_test, y_pred
        

        X_test_final = None
        y_test_final = np.zeros(len(test_cases_list))
        y_pred_final = np.zeros(len(test_cases_list))

        for topic, DT in topic_2_train_DT.items():

            assert type(DT) == DataTopic


            original_ixs = remaining_original_article_ixs[topic_2_ixs[topic]]

            if topic in self.grouped_topic_model_topics:
                DT.tfidf_chosen_ixs_trim(**self.grouped_topic_model_chosen_ixs)
                DT.one_hot_encoding(self.grouped_topics_DT)
                X_test, y_test = DT.yeald_complete_matrix(model_type="grouped_topic")
                y_test = np.reshape(y_test, (-1))
                y_pred = self.grouped_topic_model.predict(X_test)
                
            elif topic in self.single_topic_models:
                DT.tfidf_chosen_ixs_trim(**self.single_topic_models_chosen_ixs[topic])
                DT.one_hot_encoding(self.topic_2_train_DT[topic])
                X_test, y_test = DT.yeald_complete_matrix(model_type="single_topic")
                y_test = np.reshape(y_test, (-1))
                y_pred = self.single_topic_models[topic].predict(X_test)

            else:
            
                DT.tfidf_chosen_ixs_trim(**self.unrecognised_topic_model_chosen_ixs)
                DT.one_hot_encoding(self.all_together_DT)
                X_test, y_test = DT.yeald_complete_matrix(model_type="unrecognised_topic")
                y_test = np.reshape(y_test, (-1))
                y_pred = self.unrecognised_topic_model.predict(X_test)
            
            y_test_final[original_ixs] = y_test
            y_pred_final[original_ixs] = y_pred


        return X_test_final, y_test_final, y_pred_final




def test_model(model, test_cases_list, all_topics_together=False):
    # model type can be None or "all_topics_together"
    # If None, Model automatically uses "single_topic", "grouped_topic", "unrecognised_topic" depending on
    # matches in the training data.

    assert type(model) == Model

    if all_topics_together:
        X_test, y_test, y_pred = model.predict(test_cases_list, all_together_model=True)
    else:
        X_test, y_test, y_pred = model.predict(test_cases_list)

    
    


    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    rmse = (mean_squared_error(y_test, y_pred))**(1/2)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R²: {r2}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")


    # Plot predicted vs actual values
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual Values")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')

    # Plot residuals
    residuals = y_test - y_pred
    if PRINTOUT:
        print("residuals: ", residuals)

    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values")
    plt.axhline(y=0, color='red', linestyle='--')

    plt.tight_layout()
    plt.show()


    if all_topics_together:
        
        # Get the coefficients
        coefficients = model.all_together_model.coef_
        # print("Coefficients:", coefficients)

        # Plot coefficients
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(coefficients)), coefficients)
        plt.xlabel("Coefficient Index")
        plt.ylabel("Coefficient Value")
        plt.title("Lasso Coefficients")
        plt.show()

    return model


if __name__ == "__main__":

    # Path to your .json.gzip file
    file_path = './data/rtvslo_train.json.gzip'

    # Open the gzip file
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        # Read and parse the JSON data
        data = json.load(f)

    if PRINTOUT:
        print("len(data): " + str(len(data)))
        print("data[0].keys(): " + str(data[0].keys()))


    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    topic_2_train_DT, grouped_topics_DT, all_together_DT, vectorizers, _, _ = prepare_data(train_data)

    model_pickle_path = "./data/model.pickle"
    
    if True:

        ohe_cutoff = 150
        tfidf_cutoff = 150

        hyper_parameters = {

            "max_iter" : 1000,

            "URLs" : ohe_cutoff,
            "authors" : ohe_cutoff,
            "leads" : ohe_cutoff,
            "keywords" : tfidf_cutoff,
            "gpt_keywords" : tfidf_cutoff,
            "topics" : tfidf_cutoff,
            "alpha" : 0.05,
            "method" : "Lasso", #Ridge in Basic ne delata # "Basic", "Ridge" or "Lasso"
        }
    
        curr_model = Model(topic_2_train_DT, grouped_topics_DT, all_together_DT, vectorizers, hyper_parameters=hyper_parameters)
        
        try:
            with open(model_pickle_path, "wb") as f:
                pickle.dump(curr_model, f)
        except:
            print("Couldn't save the model." + 2*"!\n")
            pass    

    else:

        with open(model_pickle_path, "rb") as f:
            curr_model = pickle.load(f)




    test_model(curr_model, test_data, all_topics_together=True)
    test_model(curr_model, test_data, all_topics_together=False)






























