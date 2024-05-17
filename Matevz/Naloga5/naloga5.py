



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

"""
if True:
    import pickle

    # Path to your .json.gzip file
    file_path = './data/rtvslo_train.json.gzip'

    # Open the gzip file
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        # Read and parse the JSON data
        data = json.load(f)


    # Path to save the pickled file
    pickle_file_path = './data/rtvslo_train.pkl'

    # Pickle the data
    with open(pickle_file_path, 'wb') as pf:
        pickle.dump(data, pf)

else:

    # Path to your pickle file
    pickle_file_path = './data/rtvslo_train.pkl'

    # Open the pickle file and load the data
    with open(pickle_file_path, 'rb') as pf:
        data = pickle.load(pf)
"""




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

    data_class = DataClass(data)


    # Path to save the pickled file
    pickle_file_path = './data/data_class.pkl'

    # Pickle the data
    with open(pickle_file_path, 'wb') as pf:
        pickle.dump(data_class, pf)

else:

    # Path to your pickle file
    pickle_file_path = './data/data_class.pkl'

    # Open the pickle file and load the data
    with open(pickle_file_path, 'rb') as pf:
        data_class = pickle.load(pf)

print("data_class.URLs[0]: " + str(data_class.URLs[0]))
print("data_class.length: " + str(data_class.length))
print("data_class.num_of_ommited_by_topic: " + str(data_class.num_of_ommited_by_topic))














class DataPrepared:

    def __init__(self):
        self.URLs = []
        self.authors = []
        self.times = []
        self.nums_of_figs = []
        self.topics = []
        self.num_of_comments = []


        self.leads_tfidf = []
        self.leads_names = []
        self.leads_counts = []
        self.leads_counts_names = []
        # self.leads_empty_ixs = set()

        self.keywords_tfidf = []
        self.keywords_names = []
        self.keywords_counts = []
        self.keywords_counts_names = []
        # self.keywords_empty_ixs = set()

        self.gpt_keywords_tfidf = []
        self.gpt_keywords_names = []
        self.gpt_keywords_counts = []
        self.gpt_keywords_counts_names = []
        # self.gpt_keywords_empty_ixs = set()



















if True:

    data_prepared = DataPrepared()

    for curr_ix in range(data_class.length):

        curr_URL = data_class.URLs[curr_ix]
        split_curr_URL = curr_URL.split('/')
        useful_split_curr_URL = split_curr_URL[3:-2]
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
        # print(year, month, day, hour, part_of_day)
        data_prepared.times.append([year, month, day, part_of_day])

        curr_num_of_figs = data_class.nums_of_figs[curr_ix]
        # print(curr_num_of_figs)
        data_prepared.nums_of_figs.append(curr_num_of_figs)
        
        curr_topics = data_class.topics[curr_ix]
        # print(curr_topics)
        data_prepared.topics.append(curr_topics)

        curr_num_of_comments = data_class.num_of_comments[curr_ix]
        # print(curr_num_of_comments)
        data_prepared.num_of_comments.append(curr_num_of_comments)




        curr_lead = [x.lower() for x in data_class.leads[curr_ix].split(" ")]
        curr_lead = " ".join(curr_lead)
        data_prepared.leads_tfidf.append(curr_lead)

        curr_keywords = [x.lower() for x in data_class.keywords[curr_ix]]
        curr_keywords = " ".join(curr_keywords)
        data_prepared.keywords_tfidf.append(curr_keywords)

        curr_gpt_keywords = [x.lower() for x in data_class.gpt_keywords[curr_ix]]
        curr_gpt_keywords = " ".join(curr_gpt_keywords)
        data_prepared.gpt_keywords_tfidf.append(curr_gpt_keywords)





    print("Here 1")

    count_vectorizer = CountVectorizer()
    X_counts = count_vectorizer.fit_transform(data_prepared.leads_tfidf)
    data_prepared.leads_counts = X_counts.toarray()
    data_prepared.leads_counts_names = count_vectorizer.get_feature_names_out()

    vectorizer = TfidfVectorizer(max_features=None, norm="l2", stop_words=None, smooth_idf=True)
    vectorizer.fit(data_prepared.leads_tfidf)
    data_prepared.leads_tfidf = vectorizer.transform(data_prepared.leads_tfidf)
    data_prepared.leads_names = vectorizer.get_feature_names_out()
    
    print("Here 2")

    count_vectorizer = CountVectorizer()
    X_counts = count_vectorizer.fit_transform(data_prepared.keywords_tfidf)
    data_prepared.keywords_counts = X_counts.toarray()
    data_prepared.keywords_counts_names = count_vectorizer.get_feature_names_out()

    vectorizer = TfidfVectorizer(max_features=None, norm="l2", stop_words=None, smooth_idf=True)
    vectorizer.fit(data_prepared.keywords_tfidf)
    data_prepared.keywords_tfidf = vectorizer.transform(data_prepared.keywords_tfidf)
    data_prepared.keywords_names = vectorizer.get_feature_names_out()

    print("Here 3")

    count_vectorizer = CountVectorizer()
    X_counts = count_vectorizer.fit_transform(data_prepared.gpt_keywords_tfidf)
    data_prepared.gpt_keywords_counts = X_counts.toarray()
    data_prepared.gpt_keywords_counts_names = count_vectorizer.get_feature_names_out()

    vectorizer = TfidfVectorizer(max_features=None, norm="l2", stop_words=None, smooth_idf=True)
    vectorizer.fit(data_prepared.gpt_keywords_tfidf)
    data_prepared.gpt_keywords_tfidf  = vectorizer.transform(data_prepared.gpt_keywords_tfidf)
    data_prepared.gpt_keywords_names = vectorizer.get_feature_names_out()

    print("Here 4")

    """
    # Leads special, because we had to preprocess it above.
    data_prepared.leads_tfidf, data_prepared.leads_empty_ixs = tfidf.tfidf(data_prepared.leads_tfidf)

    data_prepared.keywords_tfidf, data_prepared.keywords_empty_ixs = tfidf.tfidf(data_class.keywords)
    data_prepared.gpt_keywords_tfidf, data_prepared.keywords_empty_ixs = tfidf.tfidf(data_class.gpt_keywords)
    """

    
    
    
    
    """
    # Path to save the pickled file
    pickle_file_path = './data/data_prepared.pkl'

    # Pickle the data
    with open(pickle_file_path, 'wb') as pf:
        pickle.dump(data_prepared, pf)"""

else:

    # Path to your pickle file
    pickle_file_path = './data/data_prepared.pkl'

    # Open the pickle file and load the data
    with open(pickle_file_path, 'rb') as pf:
        data_prepared = pickle.load(pf)


print("data_prepared.leads_tfidf.shape: " + str(data_prepared.leads_tfidf.shape))
print("data_prepared.keywords_tfidf.shape: " + str(data_prepared.keywords_tfidf.shape))
print("data_prepared.gpt_keywords_tfidf.shape: " + str(data_prepared.gpt_keywords_tfidf.shape))



print("Here 5")


print("np.array_equal(data_prepared.leads_names, data_prepared.leads_counts_names): " + 
      str(np.array_equal(data_prepared.leads_names, data_prepared.leads_counts_names)))

# print("data_prepared.leads_names.shape: " + str(data_prepared.leads_names.shape))
# print("data_prepared.leads_counts_names.shape: " + str(data_prepared.leads_counts_names.shape))

print("Here 6")

print("np.array_equal(data_prepared.keywords_names, data_prepared.keywords_counts_names): " +
        str(np.array_equal(data_prepared.keywords_names, data_prepared.keywords_counts_names)))


print("Here 7")

print("np.array_equal(data_prepared.gpt_keywords_names, data_prepared.gpt_keywords_counts_names): " +
        str(np.array_equal(data_prepared.gpt_keywords_names, data_prepared.gpt_keywords_counts_names)))

print("Here 8")

"""
# This crashes due to RAM

num_of_mismatches = 0

for ix, count_name in enumerate(data_prepared.leads_counts_names):
    if count_name != data_prepared.leads_names[ix]:
        print("count_name != data_prepared.leads_names[ix]")
        print(count_name)
        print(data_prepared.leads_names[ix])
        num_of_mismatches += 1


for ix, count_name in enumerate(data_prepared.keywords_counts_names):
    if count_name != data_prepared.keywords_names[ix]:
        print("count_name != data_prepared.keywords_names[ix]")
        print(count_name)
        print(data_prepared.keywords_names[ix])
        num_of_mismatches += 1


for ix, count_name in enumerate(data_prepared.gpt_keywords_counts_names):
    if count_name != data_prepared.gpt_keywords_names[ix]:
        print("count_name != data_prepared.gpt_keywords_names[ix]")
        print(count_name)
        print(data_prepared.gpt_keywords_names[ix])
        num_of_mismatches += 1

print("num_of_mismatches: " + str(num_of_mismatches))
"""


























