




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

from scipy.sparse import csr_matrix, hstack


URL_KEEP_TOPIC = False
LEMMATIZE = True
PRINTOUT = False





class DataClass:
    def __init__(self):
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

    def extract_data(self, data):

        deleted_ixs = []

        for ix, article in enumerate(data):
            try:
                self.topics.append(article['topics'])
            except KeyError:
                # print("Article has no topics. Ommiting it. URL:")
                # print(article['url'])
                deleted_ixs.append(ix)
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
        
        return deleted_ixs






class DataPrepared:

    def __init__(self, data_class=None):

        self.URLs = []

        self.authors = []

        self.years = [] # becomes np.array() in init
        self.month_functions = [] # becomes np.array() in init
        self.month_functions_names = []
        # self.hours = []
        self.parts_of_day = []

        self.nums_of_figs = [] # becomes np.array() in init
        
        self.topics = []
        
        self.num_of_comments = [] # becomes np.array() in init

        self.leads_tfidf = [] # turns into scipy sparse matrix after tfidf()
        self.leads_names = []

        self.keywords_tfidf = [] # turns into scipy sparse matrix after tfidf()
        self.keywords_names = []

        self.gpt_keywords_tfidf = [] # turns into scipy sparse matrix after tfidf()
        self.gpt_keywords_names = []
        
        if not data_class is None:



            URL_start_ix = 3 if URL_KEEP_TOPIC else 4

            
            
            lemmatizer = Lemmatizer('sl')

            for curr_ix in range(data_class.length):

                curr_URL = data_class.URLs[curr_ix]
                split_curr_URL = curr_URL.split('/')
                useful_split_curr_URL = split_curr_URL[URL_start_ix:-2]
                self.URLs.append(useful_split_curr_URL)

                curr_authors = data_class.authors[curr_ix]
                useful_curr_authors = tuple(sorted(curr_authors))
                self.authors.append(useful_curr_authors)

                curr_date = data_class.dates[curr_ix]
                # print(curr_date)
                # print(type(curr_date))
                year, month, day_hour = curr_date.split("-")
                day, hour = day_hour.split("T")
                hour = hour.split(":")[0]
                # conv to ints in one line
                year, month, day, hour = map(int, [year, month, day, hour])
                part_of_day = hour // 3

                self.years.append(year)
                self.month_functions.append(month)
                self.parts_of_day.append(part_of_day)
                # print(year, month, day, hour, part_of_day)

                curr_num_of_figs = data_class.nums_of_figs[curr_ix]
                # print(curr_num_of_figs)
                self.nums_of_figs.append(curr_num_of_figs)
                
                curr_topics = data_class.topics[curr_ix]
                # print(curr_topics)
                self.topics.append(curr_topics)

                curr_num_of_comments = data_class.num_of_comments[curr_ix]
                # print(curr_num_of_comments)
                self.num_of_comments.append(curr_num_of_comments)









                if LEMMATIZE:

                    curr_lead = [lemmatizer.lemmatize(x.lower()) for x in data_class.leads[curr_ix].split(" ")]
                    curr_lead = " ".join(curr_lead)
                    self.leads_tfidf.append(curr_lead)

                    curr_keywords = [lemmatizer.lemmatize(x.lower()) for x in data_class.keywords[curr_ix]]
                    curr_keywords = " ".join(curr_keywords)
                    self.keywords_tfidf.append(curr_keywords)


                    # Has to be done in a different way because:
                    # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc5 in position 114: unexpected end of data
                    curr_gpt_keywords = []
                    for x in data_class.gpt_keywords[curr_ix]:
                        try:
                            curr_gpt_keywords.append(lemmatizer.lemmatize(x.lower()))
                        except:
                            pass
                    curr_gpt_keywords = " ".join(curr_gpt_keywords)
                    self.gpt_keywords_tfidf.append(curr_gpt_keywords)


                else:
                    
                    curr_lead = [x.lower() for x in data_class.leads[curr_ix].split(" ")]
                    curr_lead = " ".join(curr_lead)
                    self.leads_tfidf.append(curr_lead)

                    curr_keywords = [x.lower() for x in data_class.keywords[curr_ix]]
                    curr_keywords = " ".join(curr_keywords)
                    self.keywords_tfidf.append(curr_keywords)

                    curr_gpt_keywords = [x.lower() for x in data_class.gpt_keywords[curr_ix]]
                    curr_gpt_keywords = " ".join(curr_gpt_keywords)
                    self.gpt_keywords_tfidf.append(curr_gpt_keywords)











            self.years = np.array(self.years).reshape(-1, 1)
            
            self.month_functions, self.month_functions_names = months_into_functions(self.month_functions)

            self.nums_of_figs = np.array(self.nums_of_figs).reshape(-1, 1)

            self.num_of_comments = np.array(self.num_of_comments).reshape(-1, 1)



        
    def tfidf(self, all_vectorizers, tfidf_args):



        new_vectorizers = {
            "leads": None,
            "keywords": None,
            "gpt_keywords": None
        }

        
        if all_vectorizers is None:
            vectorizer = TfidfVectorizer(**tfidf_args)
            vectorizer.fit(self.leads_tfidf)
        else:
            vectorizer = all_vectorizers["leads"]
            
        self.leads_tfidf = vectorizer.transform(self.leads_tfidf).tocsr()
        self.leads_names = vectorizer.get_feature_names_out()
        new_vectorizers["leads"] = vectorizer
        

        if all_vectorizers is None:
            vectorizer = TfidfVectorizer(**tfidf_args)
            vectorizer.fit(self.keywords_tfidf)
        else:
            vectorizer = all_vectorizers["keywords"]

        self.keywords_tfidf = vectorizer.transform(self.keywords_tfidf).tocsr()
        self.keywords_names = vectorizer.get_feature_names_out()
        new_vectorizers["keywords"] = vectorizer


        if all_vectorizers is None:
            vectorizer = TfidfVectorizer(**tfidf_args)
            vectorizer.fit(self.gpt_keywords_tfidf)
        else:
            vectorizer = all_vectorizers["gpt_keywords"]

        self.gpt_keywords_tfidf  = vectorizer.transform(self.gpt_keywords_tfidf).tocsr()
        self.gpt_keywords_names = vectorizer.get_feature_names_out()
        new_vectorizers["gpt_keywords"] = vectorizer

        return new_vectorizers
        
    
    def copy_with_only_chosen_ixs(self, chosen_ixs):


        new_data_prepared = DataPrepared()

        new_data_prepared.URLs = [self.URLs[ix] for ix in chosen_ixs]

        new_data_prepared.authors = [self.authors[ix] for ix in chosen_ixs]

        new_data_prepared.years = self.years[chosen_ixs]
        new_data_prepared.month_functions = self.month_functions[chosen_ixs]
        new_data_prepared.month_functions_names = self.month_functions_names
        # new_data_prepared.hours = []
        new_data_prepared.parts_of_day = [self.parts_of_day[ix] for ix in chosen_ixs]

        new_data_prepared.nums_of_figs = self.nums_of_figs[chosen_ixs]

        new_data_prepared.topics = [self.topics[ix] for ix in chosen_ixs]

        new_data_prepared.num_of_comments = self.num_of_comments[chosen_ixs]


        new_data_prepared.leads_tfidf = self.leads_tfidf.tocsr()[chosen_ixs, :]
        new_data_prepared.leads_names = self.leads_names
        # print("self.leads_tfidf.shape: " + str(self.leads_tfidf.shape))
        # print("chosen_ixs[-10:]: " + str(chosen_ixs[-10:]))
        # print("new_data_prepared.leads_tfidf.shape: " + str(new_data_prepared.leads_tfidf.shape))


        new_data_prepared.keywords_tfidf = self.keywords_tfidf.tocsr()[chosen_ixs, :]
        new_data_prepared.keywords_names = self.keywords_names

        new_data_prepared.gpt_keywords_tfidf = self.gpt_keywords_tfidf.tocsr()[chosen_ixs, :]
        new_data_prepared.gpt_keywords_names = self.gpt_keywords_names

        return new_data_prepared
        
















def one_hot_encode(list_of_vals, list_of_accepted_names=None):

    if list_of_accepted_names is None:
        possible_vals = list(set(list_of_vals))
    else:
        possible_vals = list_of_accepted_names

    returner = np.zeros((len(list_of_vals), len(possible_vals)))

    for ix, val in enumerate(list_of_vals):
        try:
            ix_of_name = possible_vals.index(val)
        except:
            continue
        
        returner[ix, ix_of_name] = 1
    
    return returner, possible_vals

def make_list_1D(list_of_lists):
    returner = []
    for list_ in list_of_lists:
        returner.extend(list_)
    return returner

def one_hot_encode_URLs(list_of_lists_of_urls, list_of_accepted_names=None):

    if list_of_accepted_names is None:
        possible_vals = make_list_1D(list_of_lists_of_urls)
        possible_vals = list(set(possible_vals))
    else:
        possible_vals = list_of_accepted_names
    
    returner = np.zeros((len(list_of_lists_of_urls), len(possible_vals)))

    for ix, list_of_urls in enumerate(list_of_lists_of_urls):
        for url in list_of_urls:
            try:
                ix_of_name = possible_vals.index(url)
            except:
                continue

            returner[ix, ix_of_name] = 1
    
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















def vertical_concat(firs_np, second_np):
    returner = np.vstack((firs_np, second_np))
    return returner


def copy_list(list_):
    returner = []
    returner.extend(list_)
    return returner

class DataTopic:

    def __init__(self, topic=None, data_prepared=None, ixs=None):


        if topic is None and data_prepared is None and ixs is None:
            self.URLs = [] # becomes np.array() in one_hot_encoding()
            self.URL_names = []

            self.authors = [] # becomes np.array() in one_hot_encoding()
            self.authors_names = []

            self.years = [] # is actually np.array()
            self.month_functions = [] # is actually np.array()
            self.month_functions_names = []
            # self.hours = []
            self.parts_of_day = [] # becomes np.array() in one_hot_encoding()
            self.parts_of_day_names = []

            self.nums_of_figs = [] # is actually np.array()
            
            self.topics = [] 
            self.topics_encoded = [] # becomes np.array() in one_hot_encoding()
            self.topics_names = []
            
            self.num_of_comments = [] # is actually np.array()


            self.leads_tfidf = [] # is actually scipy sparse matrix
            self.leads_names = []

            self.keywords_tfidf = [] # is actually scipy sparse matrix
            self.keywords_names = []

            self.gpt_keywords_tfidf = [] # is actually scipy sparse matrix
            self.gpt_keywords_names = []
        


        # This is how DataTopic should be used:
        elif not data_prepared is None:
            self.topic = topic

            self.URLs = copy_list(data_prepared.URLs) # becomes np.array() in one_hot_encoding()
            self.URL_names = [] # gets filled in one_hot_encoding()

            self.authors = copy_list(data_prepared.authors) # becomes np.array() in one_hot_encoding()
            self.authors_names = [] # gets filled in one_hot_encoding()

            self.years = np.copy(data_prepared.years)
            self.month_functions = np.copy(data_prepared.month_functions)
            self.month_functions_names = copy_list(data_prepared.month_functions_names)
            # self.hours = []
            self.parts_of_day = copy_list(data_prepared.parts_of_day) # becomes np.array() in one_hot_encoding()
            self.parts_of_day_names = [] # gets filled in one_hot_encoding()

            self.nums_of_figs = np.copy(data_prepared.nums_of_figs)
            
            self.topics = copy_list(data_prepared.topics)
            self.topics_encoded = [] # gets filled and becomes np.array() in one_hot_encoding()
            self.topics_names = [] # gets filled in one_hot_encoding()
            
            self.num_of_comments = np.copy(data_prepared.num_of_comments)


            self.leads_tfidf = data_prepared.leads_tfidf.copy()
            self.leads_names = copy_list(data_prepared.leads_names)

            self.keywords_tfidf = data_prepared.keywords_tfidf.copy()
            self.keywords_names = copy_list(data_prepared.keywords_names)

            self.gpt_keywords_tfidf = data_prepared.gpt_keywords_tfidf.copy()
            self.gpt_keywords_names = copy_list(data_prepared.gpt_keywords_names)

        
        # I'm not testing this, so be weary
        elif not data_prepared is None and not ixs is None:

            self.topic = topic

            self.URLs = [data_prepared.URLs[ix] for ix in ixs]
            self.URL_names = data_prepared.URL_names

            self.authors = [data_prepared.authors[ix] for ix in ixs]
            self.authors_names = data_prepared.authors_names

            self.years = data_prepared.years[ixs]
            self.month_functions = data_prepared.month_functions[ixs]
            self.month_functions_names = data_prepared.month_functions_names
            # self.hours = []
            self.parts_of_day = [data_prepared.parts_of_day[ix] for ix in ixs]
            self.parts_of_day_names = data_prepared.parts_of_day_names

            self.nums_of_figs = data_prepared.nums_of_figs[ixs]
            
            self.topics = [data_prepared.topics[ix] for ix in ixs]
            try:
                self.topics_encoded = [data_prepared.topics_encoded[ix] for ix in ixs]
            except:
                print("len(data_prepared.topics_encoded): " + str(len(data_prepared.topics_encoded)))
                print("len(ixs): " + str(len(ixs)))
                print("ixs: " + str(ixs))
                print("data_prepared.topics_encoded: " + str(data_prepared.topics_encoded))
            self.topics_names = data_prepared.topics_names
            
            self.num_of_comments = data_prepared.num_of_comments[ixs]


            self.leads_tfidf = data_prepared.leads_tfidf[ixs]
            self.leads_names = data_prepared.leads_names

            self.keywords_tfidf = data_prepared.keywords_tfidf[ixs]
            self.keywords_names = data_prepared.keywords_names

            self.gpt_keywords_tfidf = data_prepared.gpt_keywords_tfidf[ixs]
            self.gpt_keywords_names = data_prepared.gpt_keywords_names


    def yeald_complete_matrix(self, model_type="single_topic") -> csr_matrix :
        # type can be "single_topic", "grouped_topic", "unrecognised_topic", "all_topics_together"
        returner_y = np.copy(self.num_of_comments)

        if model_type == "single_topic" or model_type == "unrecognised_topic":

            returner_matrix = np.hstack((self.URLs, self.authors, self.years, self.month_functions, self.parts_of_day,
                                         self.nums_of_figs))
            returner_matrix = csr_matrix(returner_matrix)
            returner_matrix = hstack([returner_matrix, self.leads_tfidf, self.keywords_tfidf, self.gpt_keywords_tfidf])
        
        elif model_type == "grouped_topic" or model_type == "all_topics_together":
            returner_matrix = np.hstack((self.topics_encoded, self.URLs, self.authors, self.years, self.month_functions, self.parts_of_day,
                                         self.nums_of_figs))
            returner_matrix = csr_matrix(returner_matrix)
            returner_matrix = hstack([returner_matrix, self.leads_tfidf, self.keywords_tfidf, self.gpt_keywords_tfidf])
        
        return returner_matrix, returner_y
    
    def concat(self, other_data_topic):
        new_data_topic = DataTopic()

        new_data_topic.topic = self.topic + "," + other_data_topic.topic


        new_data_topic.URLs = vertical_concat(self.URLs, other_data_topic.URLs)
        new_data_topic.URL_names = [].extend(self.URL_names)

        new_data_topic.authors = vertical_concat(self.authors, other_data_topic.authors)
        new_data_topic.authors_names = [].extend(self.authors_names)

        new_data_topic.years = vertical_concat(self.years, other_data_topic.years)
        new_data_topic.month_functions = vertical_concat(self.month_functions, other_data_topic.month_functions)
        new_data_topic.month_functions_names = [].extend(self.month_functions_names)
        # new_data_topic.hours = []
        new_data_topic.parts_of_day = vertical_concat(self.parts_of_day, other_data_topic.parts_of_day)
        new_data_topic.parts_of_day_names = [].extend(self.parts_of_day_names)

        new_data_topic.nums_of_figs = vertical_concat(self.nums_of_figs, other_data_topic.nums_of_figs)
        
        new_data_topic.topics = copy_list(self.topics).extend(other_data_topic.topics)
        new_data_topic.topics_encoded = vertical_concat(self.topics_encoded, other_data_topic.topics_encoded)
        new_data_topic.topics_names = [].extend(self.topics_names)
        
        new_data_topic.num_of_comments = vertical_concat(self.num_of_comments, other_data_topic.num_of_comments)


        new_data_topic.leads_tfidf = vertical_concat(self.leads_tfidf, other_data_topic.leads_tfidf)
        new_data_topic.leads_names = [].extend(self.leads_names)

        new_data_topic.keywords_tfidf = vertical_concat(self.keywords_tfidf, other_data_topic.keywords_tfidf)
        new_data_topic.keywords_names = [].extend(self.keywords_names)

        new_data_topic.gpt_keywords_tfidf = vertical_concat(self.gpt_keywords_tfidf, other_data_topic.gpt_keywords_tfidf)
        new_data_topic.gpt_keywords_names = [].extend(self.gpt_keywords_names)

        return new_data_topic
        
    def copy(self):
        new_data_topic = DataTopic()

        new_data_topic.topic = self.topic

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
    
    def tfidf_chosen_ixs_trim(self, leads_chosen_ixs, keywords_chosen_ixs, gpt_keywords_chosen_ixs):

        self.leads_tfidf = self.leads_tfidf[:, leads_chosen_ixs]
        self.leads_names = [self.leads_names[ix] for ix in leads_chosen_ixs]

        self.keywords_tfidf = self.keywords_tfidf[:, keywords_chosen_ixs]
        self.keywords_names = [self.keywords_names[ix] for ix in keywords_chosen_ixs]

        self.gpt_keywords_tfidf = self.gpt_keywords_tfidf[:, gpt_keywords_chosen_ixs]
        self.gpt_keywords_names = [self.gpt_keywords_names[ix] for ix in gpt_keywords_chosen_ixs]
    
    def one_hot_encoded_chosen_ixs_trim(self, URLs_chosen_ixs=None, authors_chosen_ixs=None, topics_chosen_ixs=None):
        
        if URLs_chosen_ixs is not None:
            self.URLs = self.URLs[:, URLs_chosen_ixs]
            self.URL_names = [self.URL_names[ix] for ix in URLs_chosen_ixs]

        if authors_chosen_ixs is not None:
            self.authors = self.authors[:, authors_chosen_ixs]
            self.authors_names = [self.authors_names[ix] for ix in authors_chosen_ixs]

        if topics_chosen_ixs is not None:
            self.topics_encoded = self.topics_encoded[:, topics_chosen_ixs]
            self.topics_names = [self.topics_names[ix] for ix in topics_chosen_ixs]


    def one_hot_encoding(self, train_data_topic=None):
            
            assert type(train_data_topic) == DataTopic or train_data_topic is None
            
            if train_data_topic is None:

                self.URLs, self.URL_names = one_hot_encode_URLs(self.URLs)

                self.authors, self.authors_names = one_hot_encode(self.authors)

                self.parts_of_day, self.parts_of_day_names = one_hot_encode(self.parts_of_day)            

                self.topics_encoded, self.topics_names = one_hot_encode(self.topics)

            else:

                self.URLs, self.URL_names = one_hot_encode_URLs(self.URLs, train_data_topic.URL_names)

                self.authors, self.authors_names = one_hot_encode(self.authors, train_data_topic.authors_names)

                self.parts_of_day, self.parts_of_day_names = one_hot_encode(self.parts_of_day, train_data_topic.parts_of_day_names)            

                self.topics_encoded, self.topics_names = one_hot_encode(self.topics, train_data_topic.topics_names)

    

    
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














def prepare_data(list_of_article_objects, all_vectorizers=None, is_test_data=False) -> tuple[dict, DataTopic, DataTopic]:
    # returns: topic_2_data_topic, grouped_topics, all_data_topic

    remaining_original_article_ixs = list(range(len(list_of_article_objects)))

    data_class = DataClass()
    deleted_ixs = data_class.extract_data(list_of_article_objects)

    for ix in sorted(deleted_ixs, reverse=True):
        del remaining_original_article_ixs[ix]



    if PRINTOUT:
        print("data_class.URLs[0]: " + str(data_class.URLs[0]))
        print("data_class.length: " + str(data_class.length))
        print("data_class.num_of_ommited_by_topic: " + str(data_class.num_of_ommited_by_topic))




    if True:

        data_prepared = DataPrepared(data_class)


        tfidf_args = {
            "max_features": None,
            "max_df": 0.85,
            "min_df": 0.001, # za 20 je zelo podobno število na koncu
            "norm": "l2",
            "stop_words": None,
            "smooth_idf": True
        }

        new_vectorizers = data_prepared.tfidf(all_vectorizers, tfidf_args)







        
        
        
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

    if PRINTOUT:
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



















    # Data splitting by topic

    prepared_data_len = len(data_prepared.URLs)
    possible_topics = list(set(data_prepared.topics))
    topic_2_ixs = dict()
    for ix, topic in enumerate(data_prepared.topics):
        if topic not in topic_2_ixs:
            topic_2_ixs[topic] = [ix]
        else:
            topic_2_ixs[topic].append(ix)

    if PRINTOUT:
        print("possible_topics: " + str(possible_topics))
        print("len(possible_topics): " + str(len(possible_topics)))
        print("topic_2_ixs.keys(): " + str(topic_2_ixs.keys()))
        print("len(topic_2_ixs): " + str(len(topic_2_ixs)))
        print(topic_2_ixs["stevilke"])


    topic_2_data_topic = dict()
    for topic in possible_topics:
        topic_2_data_topic[topic] = DataTopic(topic, data_prepared.copy_with_only_chosen_ixs(topic_2_ixs[topic]))
        if not is_test_data:
            topic_2_data_topic[topic].one_hot_encoding()


    try:

        grouped_topics = topic_2_data_topic["stevilke"].copy().concat(topic_2_data_topic["kolumne"])
        if not is_test_data:
            grouped_topics.one_hot_encoding()

        if PRINTOUT:
            print(topic_2_data_topic["stevilke"])
            print(grouped_topics)
    except:
        if PRINTOUT:
            print("stevilke or kolumne not in possible_topics")
        grouped_topics = None


    all_ixs = list(range(prepared_data_len))
    all_data_topic = DataTopic(None, data_prepared.copy_with_only_chosen_ixs(all_ixs))
    if not is_test_data:
        all_data_topic.one_hot_encoding()


    # Just for safety, return the old vectorizers directly
    returning_vectorizers = new_vectorizers if all_vectorizers is None else all_vectorizers

    return topic_2_data_topic, grouped_topics, all_data_topic, returning_vectorizers, topic_2_ixs, np.array(remaining_original_article_ixs)

