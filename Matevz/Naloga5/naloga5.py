



import hw4

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

# import streaming_pickle as pickle
import pickle


import numpy as np
import matplotlib.pyplot as plt

import tfidf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from lemmagen3 import Lemmatizer

from overloading import overload





class DataClass:
    def __init__(self, data):
        self.URLs = []
        self.authors = []
        self.dates = []
        self.nums_of_figs = []
        self.leads = []
        self.topics = []
        self.keywords = []
        self.gpt_keywords = []
        self.num_of_comments = []

        self.length = 0
        self.num_of_ommited_by_topic = 0

        self.extract_data(data)

    def extract_data(self, data):
        for article in data:
            try:
                self.topics.append(article['topics'])
            except KeyError:
                # print("Article has no topics. Ommiting it. URL:")
                # print(article['url'])
                self.num_of_ommited_by_topic += 1
                continue

            
            self.URLs.append(article['url'])
            self.authors.append(article['authors'])
            self.dates.append(article['date'])
            self.nums_of_figs.append(len(article['figures']))
            self.leads.append(article['lead'])
            self.keywords.append(article['keywords'])
            self.gpt_keywords.append(article['gpt_keywords'])
            self.num_of_comments.append(article['n_comments'])

            self.length += 1






if True:

    # Path to your .json.gzip file
    file_path = './data/rtvslo_train.json.gzip'

    # Open the gzip file
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        # Read and parse the JSON data
        data = json.load(f)


    print("len(data): " + str(len(data)))
    print("data[0].keys(): " + str(data[0].keys()))


    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    data_class = DataClass(train_data)

    print("data_class.URLs[0]: " + str(data_class.URLs[0]))
    print("data_class.length: " + str(data_class.length))
    print("data_class.num_of_ommited_by_topic: " + str(data_class.num_of_ommited_by_topic))














class DataPrepared:

    def __init__(self):
        self.URLs = []
        self.URL_names = []

        self.authors = []
        self.authors_names = []

        self.years = []
        self.month_functions = []
        self.month_functions_names = []
        # self.hours = []
        self.parts_of_day = []
        self.parts_of_day_names = []

        self.nums_of_figs = []
        
        self.topics = []
        self.topics_encoded = []
        self.topics_names = []
        
        self.num_of_comments = []


        self.leads_tfidf = []
        self.leads_names = []

        self.keywords_tfidf = []
        self.keywords_names = []

        self.gpt_keywords_tfidf = []
        self.gpt_keywords_names = []
        







def one_hot_encode(list_of_vals):
    possible_vals = list(set(list_of_vals))

    returner = np.zeros((len(list_of_vals), len(possible_vals)))

    for ix, val in enumerate(list_of_vals):
        returner[ix, possible_vals.index(val)] = 1
    
    return returner, possible_vals

def make_list_1D(list_of_lists):
    returner = []
    for list_ in list_of_lists:
        returner.extend(list_)
    return returner

def one_hot_encode_URLs(list_of_lists_of_urls):
    possible_vals = make_list_1D(list_of_lists_of_urls)
    possible_vals = list(set(possible_vals))

    returner = np.zeros((len(list_of_lists_of_urls), len(possible_vals)))

    for ix, list_of_urls in enumerate(list_of_lists_of_urls):
        for url in list_of_urls:
            returner[ix, possible_vals.index(url)] = 1
    
    return returner, possible_vals


def months_into_functions(months):

    func_names = ["month", "sin(month)", "cos(month)", "sqrt(month), squared(month)"]
    returner = np.zeros((len(months), 5))

    for ix, month in enumerate(months):
        month = month % 12
        month = month / 12
        returner[ix, 0] = month
        returner[ix, 1] = np.sin(2 * np.pi * month)
        returner[ix, 2] = np.cos(2 * np.pi * month)
        returner[ix, 3] = np.sqrt(month)
        returner[ix, 4] = month ** 2
    
    return returner, func_names





URL_KEEP_TOPIC = False

URL_start_ix = 3 if URL_KEEP_TOPIC else 4



if True:

    data_prepared = DataPrepared()

    
    LEMMATIZE = True
    
    lemmatizer = Lemmatizer('sl')

    for curr_ix in range(data_class.length):

        curr_URL = data_class.URLs[curr_ix]
        split_curr_URL = curr_URL.split('/')
        useful_split_curr_URL = split_curr_URL[URL_start_ix:-2]
        data_prepared.URLs.append(useful_split_curr_URL)

        curr_authors = data_class.authors[curr_ix]
        useful_curr_authors = tuple(sorted(curr_authors))
        data_prepared.authors.append(useful_curr_authors)

        curr_date = data_class.dates[curr_ix]
        # print(curr_date)
        # print(type(curr_date))
        year, month, day_hour = curr_date.split("-")
        day, hour = day_hour.split("T")
        hour = hour.split(":")[0]
        # conv to ints in one line
        year, month, day, hour = map(int, [year, month, day, hour])
        part_of_day = hour // 3

        data_prepared.years.append(year)
        data_prepared.month_functions.append(month)
        data_prepared.parts_of_day.append(part_of_day)
        # print(year, month, day, hour, part_of_day)

        curr_num_of_figs = data_class.nums_of_figs[curr_ix]
        # print(curr_num_of_figs)
        data_prepared.nums_of_figs.append(curr_num_of_figs)
        
        curr_topics = data_class.topics[curr_ix]
        # print(curr_topics)
        data_prepared.topics.append(curr_topics)

        curr_num_of_comments = data_class.num_of_comments[curr_ix]
        # print(curr_num_of_comments)
        data_prepared.num_of_comments.append(curr_num_of_comments)




        if LEMMATIZE:

            curr_lead = [lemmatizer.lemmatize(x.lower()) for x in data_class.leads[curr_ix].split(" ")]
            curr_lead = " ".join(curr_lead)
            data_prepared.leads_tfidf.append(curr_lead)

            curr_keywords = [lemmatizer.lemmatize(x.lower()) for x in data_class.keywords[curr_ix]]
            curr_keywords = " ".join(curr_keywords)
            data_prepared.keywords_tfidf.append(curr_keywords)


            # Has to be done in a different way because:
            # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc5 in position 114: unexpected end of data
            curr_gpt_keywords = []
            for x in data_class.gpt_keywords[curr_ix]:
                try:
                    curr_gpt_keywords.append(lemmatizer.lemmatize(x.lower()))
                except:
                    pass
            curr_gpt_keywords = " ".join(curr_gpt_keywords)
            data_prepared.gpt_keywords_tfidf.append(curr_gpt_keywords)


        else:
            
            curr_lead = [x.lower() for x in data_class.leads[curr_ix].split(" ")]
            curr_lead = " ".join(curr_lead)
            data_prepared.leads_tfidf.append(curr_lead)

            curr_keywords = [x.lower() for x in data_class.keywords[curr_ix]]
            curr_keywords = " ".join(curr_keywords)
            data_prepared.keywords_tfidf.append(curr_keywords)

            curr_gpt_keywords = [x.lower() for x in data_class.gpt_keywords[curr_ix]]
            curr_gpt_keywords = " ".join(curr_gpt_keywords)
            data_prepared.gpt_keywords_tfidf.append(curr_gpt_keywords)












    data_prepared.URLs, data_prepared.URL_names = one_hot_encode_URLs(data_prepared.URLs)

    data_prepared.authors, data_prepared.authors_names = one_hot_encode(data_prepared.authors)


    data_prepared.years = np.array(data_prepared.years).reshape(-1, 1)

    data_prepared.month_functions, data_prepared.month_functions_names = months_into_functions(data_prepared.month_functions)

    data_prepared.parts_of_day, data_prepared.parts_of_day_names = one_hot_encode(data_prepared.parts_of_day)


    data_prepared.nums_of_figs = np.array(data_prepared.nums_of_figs).reshape(-1, 1)
    

    data_prepared.topics_encoded, data_prepared.topics_names = one_hot_encode(data_prepared.topics)

    data_prepared.num_of_comments = np.array(data_prepared.num_of_comments).reshape(-1, 1)









    vectorizers = {
        "leads": None,
        "keywords": None,
        "gpt_keywords": None
    }

    tfidf_args = {
        "max_features": None,
        "max_df": 0.85,
        "min_df": 0.001, # za 20 je zelo podobno število na koncu
        "norm": "l2",
        "stop_words": None,
        "smooth_idf": True
    }
    

    vectorizer = TfidfVectorizer(**tfidf_args)
    vectorizer.fit(data_prepared.leads_tfidf)
    data_prepared.leads_tfidf = vectorizer.transform(data_prepared.leads_tfidf)
    data_prepared.leads_names = vectorizer.get_feature_names_out()
    vectorizers["leads"] = vectorizer
    
    
    vectorizer = TfidfVectorizer(**tfidf_args)
    vectorizer.fit(data_prepared.keywords_tfidf)
    data_prepared.keywords_tfidf = vectorizer.transform(data_prepared.keywords_tfidf)
    data_prepared.keywords_names = vectorizer.get_feature_names_out()
    vectorizers["keywords"] = vectorizer


    vectorizer = TfidfVectorizer(**tfidf_args)
    vectorizer.fit(data_prepared.gpt_keywords_tfidf)
    data_prepared.gpt_keywords_tfidf  = vectorizer.transform(data_prepared.gpt_keywords_tfidf)
    data_prepared.gpt_keywords_names = vectorizer.get_feature_names_out()
    vectorizers["gpt_keywords"] = vectorizer
    
    
    
    
    
    # # Path to save the pickled file
    # pickle_file_path = './data/data_prepared.pkl'

    # # Pickle the data
    # with open(pickle_file_path, 'wb') as pf:
    #     pickle.dump(data_prepared, pf)

else:

    # Path to your pickle file
    pickle_file_path = './data/data_prepared.pkl'

    # Open the pickle file and load the data
    with open(pickle_file_path, 'rb') as pf:
        data_prepared = pickle.load(pf)


print("data_prepared.leads_tfidf.shape: " + str(data_prepared.leads_tfidf.shape))
print("data_prepared.keywords_tfidf.shape: " + str(data_prepared.keywords_tfidf.shape))
print("data_prepared.gpt_keywords_tfidf.shape: " + str(data_prepared.gpt_keywords_tfidf.shape))

print("data_prepared.leads_names[:10]: " + str(data_prepared.leads_names[:10]))
print("data_prepared.keywords_names[:10]: " + str(data_prepared.keywords_names[:10]))
print("data_prepared.gpt_keywords_names[:10]: " + str(data_prepared.gpt_keywords_names[:10]))

print("data_prepared.leads_names[30:40]: " + str(data_prepared.leads_names[30:40]))
print("data_prepared.keywords_names[30:40]: " + str(data_prepared.keywords_names[30:40]))
print("data_prepared.gpt_keywords_names[30:40]: " + str(data_prepared.gpt_keywords_names[30:40]))



print("data_prepared.leads_names[50:90]: " + str(data_prepared.leads_names[50:90]))
print("data_prepared.keywords_names[50:90]: " + str(data_prepared.keywords_names[50:90]))
print("data_prepared.gpt_keywords_names[50:90]: " + str(data_prepared.gpt_keywords_names[50:90]))


print("data_prepared.leads_names: " + str(data_prepared.leads_names))





def vertical_concat(firs_np, second_np):
    returner = np.vstack((firs_np, second_np))
    return returner


def copy_list(list_):
    returner = []
    returner.extend(list_)
    return returner

class DataTopic:

    def __init__(self, topic=None, data_prepared=None, ixs=None):

        if topic is None:
            self.URLs = []
            self.URL_names = []

            self.authors = []
            self.authors_names = []

            self.years = []
            self.month_functions = []
            self.month_functions_names = []
            # self.hours = []
            self.parts_of_day = []
            self.parts_of_day_names = []

            self.nums_of_figs = []
            
            self.topics = [] 
            self.topics_encoded = []
            self.topics_names = []
            
            self.num_of_comments = []


            self.leads_tfidf = []
            self.leads_names = []

            self.keywords_tfidf = []
            self.keywords_names = []

            self.gpt_keywords_tfidf = []
            self.gpt_keywords_names = []
        
        else:

            self.topic = topic

            self.URLs = data_prepared.URLs[ixs]
            self.URL_names = data_prepared.URL_names

            self.authors = data_prepared.authors[ixs]
            self.authors_names = data_prepared.authors_names

            self.years = data_prepared.years[ixs]
            self.month_functions = data_prepared.month_functions[ixs]
            self.month_functions_names = data_prepared.month_functions_names
            # self.hours = []
            self.parts_of_day = data_prepared.parts_of_day[ixs]
            self.parts_of_day_names = data_prepared.parts_of_day_names

            self.nums_of_figs = data_prepared.nums_of_figs[ixs]
            
            self.topics = [data_prepared.topics[ix] for ix in ixs] 
            self.topics_encoded = data_prepared.topics_encoded[ixs]
            self.topics_names = data_prepared.topics_names
            
            self.num_of_comments = data_prepared.num_of_comments[ixs]


            self.leads_tfidf = data_prepared.leads_tfidf[ixs]
            self.leads_names = data_prepared.leads_names

            self.keywords_tfidf = data_prepared.keywords_tfidf[ixs]
            self.keywords_names = data_prepared.keywords_names

            self.gpt_keywords_tfidf = data_prepared.gpt_keywords_tfidf[ixs]
            self.gpt_keywords_names = data_prepared.gpt_keywords_names



    def concat(self, other_data_topic):
        new_data_topic = DataTopic()


        new_data_topic.URLs = vertical_concat(self.URLs, other_data_topic.URLs)
        new_data_topic.URL_names = self.URL_names

        new_data_topic.authors = vertical_concat(self.authors, other_data_topic.authors)
        new_data_topic.authors_names = self.authors_names

        new_data_topic.years = vertical_concat(self.years, other_data_topic.years)
        new_data_topic.month_functions = vertical_concat(self.month_functions, other_data_topic.month_functions)
        new_data_topic.month_functions_names = self.month_functions_names
        # new_data_topic.hours = []
        new_data_topic.parts_of_day = vertical_concat(self.parts_of_day, other_data_topic.parts_of_day)
        new_data_topic.parts_of_day_names = self.parts_of_day_names

        new_data_topic.nums_of_figs = vertical_concat(self.nums_of_figs, other_data_topic.nums_of_figs)
        
        new_data_topic.topics = copy_list(self.topics).extend(other_data_topic.topics)
        new_data_topic.topics_encoded = vertical_concat(self.topics_encoded, other_data_topic.topics_encoded)
        new_data_topic.topics_names = self.topics_names
        
        new_data_topic.num_of_comments = vertical_concat(self.num_of_comments, other_data_topic.num_of_comments)


        new_data_topic.leads_tfidf = vertical_concat(self.leads_tfidf, other_data_topic.leads_tfidf)
        new_data_topic.leads_names = self.leads_names

        new_data_topic.keywords_tfidf = vertical_concat(self.keywords_tfidf, other_data_topic.keywords_tfidf)
        new_data_topic.keywords_names = self.keywords_names

        new_data_topic.gpt_keywords_tfidf = vertical_concat(self.gpt_keywords_tfidf, other_data_topic.gpt_keywords_tfidf)
        new_data_topic.gpt_keywords_names = self.gpt_keywords_names

        return new_data_topic
        
    def copy(self):
        new_data_topic = DataTopic()

        new_data_topic.URLs = np.copy(self.URLs)
        new_data_topic.URL_names = copy_list(self.URL_names)

        new_data_topic.authors = np.copy(self.authors)
        new_data_topic.authors_names = copy_list(self.authors_names)

        new_data_topic.years = np.copy(self.years)
        new_data_topic.month_functions = np.copy(self.month_functions)
        new_data_topic.month_functions_names = copy_list(self.month_functions_names)
        # new_data_topic.hours = []
        new_data_topic.parts_of_day = np.copy(self.parts_of_day)
        new_data_topic.parts_of_day_names = copy_list(self.parts_of_day_names)

        new_data_topic.nums_of_figs = np.copy(self.nums_of_figs)

        new_data_topic.topics = copy_list(self.topics)
        new_data_topic.topics_encoded = np.copy(self.topics_encoded)
        new_data_topic.topics_names = copy_list(self.topics_names)

        new_data_topic.num_of_comments = np.copy(self.num_of_comments)

        new_data_topic.leads_tfidf = np.copy(self.leads_tfidf)
        new_data_topic.leads_names = copy_list(self.leads_names)

        new_data_topic.keywords_tfidf = np.copy(self.keywords_tfidf)
        new_data_topic.keywords_names = copy_list(self.keywords_names)

        new_data_topic.gpt_keywords_tfidf = np.copy(self.gpt_keywords_tfidf)
        new_data_topic.gpt_keywords_names = copy_list(self.gpt_keywords_names)

        return new_data_topic
    


    
    def __str__(self):
        out_str = f"""
        URLs: {self.URLs}
        URLs_names: {self.URL_names}
        authors: {self.authors}
        authors_names: {self.authors_names}
        years: {self.years}
        month_functions: {self.month_functions}
        month_functions_names: {self.month_functions_names}
        parts_of_day: {self.parts_of_day}
        parts_of_day_names: {self.parts_of_day_names}
        nums_of_figs: {self.nums_of_figs}
        topics: {self.topics}
        topics_encoded: {self.topics_encoded}
        topics_names: {self.topics_names}
        num_of_comments: {self.num_of_comments}
        leads_tfidf: {self.leads_tfidf}
        leads_names: {self.leads_names}
        keywords_tfidf: {self.keywords_tfidf}
        keywords_names: {self.keywords_names}
        gpt_keywords_tfidf: {self.gpt_keywords_tfidf}
        gpt_keywords_names: {self.gpt_keywords_names}
        """
        return out_str



# Data splitting by topic

prepared_data_len = len(data_prepared.URLs)
possible_topics = list(set(data_prepared.topics))
topic_2_ixs = dict()
for ix, topic in enumerate(data_prepared.topics):
    if topic not in topic_2_ixs:
        topic_2_ixs[topic] = [ix]
    else:
        topic_2_ixs[topic].append(ix)


print("possible_topics: " + str(possible_topics))
print("len(possible_topics): " + str(len(possible_topics)))
print("topic_2_ixs.keys(): " + str(topic_2_ixs.keys()))
print("len(topic_2_ixs): " + str(len(topic_2_ixs)))
print(topic_2_ixs["stevilke"])


topic_2_data_topic = dict()
for topic in possible_topics:
    topic_2_data_topic[topic] = DataTopic(topic, data_prepared, topic_2_ixs[topic])


print(topic_2_data_topic["stevilke"])

grouped_topics = topic_2_data_topic["stevilke"].copy().concat(topic_2_data_topic["kolumne"])
print(grouped_topics)




from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

def build_model_from_data_topic(data_topic):

    # This is going to have to be diferent for:
    # - topic data topics
    # - and for the grouped data topic where "stevilke" and "kolumne" and "znanost in tehnologija" are grouped
    # - and also, make a model without topics with all data topics. - this is meant for unrecognised data topics.


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





    # model = LinearRegression()
    # model = Ridge(alpha=0.5)
    model = Lasso(alpha=0.5)

    X = np.hstack((data_topic.years, data_topic.month_functions, data_topic.parts_of_day, data_topic.nums_of_figs, data_topic.topics_encoded, data_topic.leads_tfidf, data_topic.keywords_tfidf, data_topic.gpt_keywords_tfidf))
    y = data_topic.num_of_comments

    model.fit(X, y)

    return model
































