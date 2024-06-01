



import pickle

import numpy as np
import matplotlib.pyplot as plt


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from lemmagen3 import Lemmatizer

from scipy.sparse import csr_matrix, hstack

from scipy.sparse.linalg import norm

from datetime import datetime

URL_KEEP_TOPIC = False
LEMMATIZE = True
PRINTOUT = False





class DataClass:
    def __init__(self):
        self.URLs = []
        self.authors = []
        self.dates = []
        self.nums_of_figs = []
        self.topics = []
        
        self.tfidfs = {}
        self.tfidfs["leads"] = []
        self.tfidfs["keywords"] = []
        self.tfidfs["gpt_keywords"] = []
        self.tfidfs["title"] = []
        self.tfidfs["paragraphs"] = []
        self.tfidfs["captions"] = []

        self.num_of_comments = []

        self.length = 0
        self.num_of_ommited_by_topic = 0

    def extract_data(self, data):

        deleted_ixs = []

        for ix, article in enumerate(data):

            # print(article)
            # input("Press Enter to continue...")

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

            try:
                self.tfidfs["leads"].append(article["lead"])
            except:
                self.tfidfs["leads"].append([])
            
            try:
                self.tfidfs["keywords"].append(article["keywords"])
            except:
                self.tfidfs["keywords"].append([])

            try:
                self.tfidfs["gpt_keywords"].append(article["gpt_keywords"])
            except:
                self.tfidfs["gpt_keywords"].append([])
            try:
                self.tfidfs["title"].append(article["title"])
            except:
                self.tfidfs["title"].append([])
            try:
                self.tfidfs["paragraphs"].append(article["paragraphs"])
            except:
                self.tfidfs["paragraphs"].append([])
            try:
                self.tfidfs["captions"].append([fig["caption"] for fig in article["figures"]])
            except:
                self.tfidfs["captions"].append([])

            # try:
            #     self.gpt_keywords.append(article['gpt_keywords'])
            # except:
            #     self.gpt_keywords.append([])



            try:
                self.num_of_comments.append(article['n_comments'])
            except:
                self.num_of_comments.append(0)
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
        self.days_of_week = []

        self.nums_of_figs = [] # becomes np.array() in init
        
        self.topics = []
        
        self.num_of_comments = [] # becomes np.array() in init

        self.tfidfs = {}
        self.tfidf_names = {}

        # self.leads_tfidf = [] # turns into scipy sparse matrix after tfidf()
        # self.leads_names = []

        # self.keywords_tfidf = [] # turns into scipy sparse matrix after tfidf()
        # self.keywords_names = []

        # self.gpt_keywords_tfidf = [] # turns into scipy sparse matrix after tfidf()
        # self.gpt_keywords_names = []
        
        if not data_class is None:



            URL_start_ix = 3 if URL_KEEP_TOPIC else 4

            
            

            for curr_ix in range(data_class.length):

                curr_URL = data_class.URLs[curr_ix]
                split_curr_URL = curr_URL.split('/')
                useful_split_curr_URL = split_curr_URL[URL_start_ix:-2]
                self.URLs.append(useful_split_curr_URL)

                curr_authors = data_class.authors[curr_ix]
                useful_curr_authors = tuple(sorted(curr_authors))
                self.authors.append(useful_curr_authors)

                curr_date = data_class.dates[curr_ix]

                # Convert string to datetime object
                date_obj = datetime.strptime(curr_date, '%Y-%m-%dT%H:%M:%S')
                # Get the day of the week (0=Monday, 6=Sunday)
                day_of_week_index = date_obj.weekday()

                # print(curr_date)
                # print(type(curr_date))
                year, month, day_hour = curr_date.split("-")
                day, hour = day_hour.split("T")
                hour = hour.split(":")[0]
                # conv to ints in one line
                year, month, day, hour = map(int, [year, month, day, hour])
                part_of_day = hour // 1

                self.years.append(year)
                self.month_functions.append(month)
                self.parts_of_day.append(part_of_day)
                self.days_of_week.append(day_of_week_index)
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








            lemmatizer = Lemmatizer('sl')

            if LEMMATIZE:
                
                # Im
                for key in data_class.tfidfs.keys():
                    # data_class.tfidfs[key] je list stringov, list dolgih stringov, list listov stringov.

                    curr = data_class.tfidfs[key]

                    if type(curr) == str:
                        print("curr is str")
                    

                    temp = []
                    for x in curr:

                        if type(x) == list:

                            # if key == "paragraphs":
                            #     try:
                            #         x = [x[0]]
                            #     except:
                            #         pass


                            try:
                                # Because of gpt_keywords it has to be done in a different way because:
                                # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc5 in position 114: unexpected end of data
                                temp_strings = []
                                for y in x:
                                    assert type(y) == str
                                    y = y.split(" ")
                                    for z in y:
                                        try:
                                            temp_strings.append(lemmatizer.lemmatize(z.lower()))
                                        except:
                                            pass
                                temp_strings = " ".join(temp_strings)
                                temp.append(temp_strings)
                                    
                            except:
                                print("Error with lemmatizing.")
                                # print(x)
                                # print(y)
                                # print(curr)
                                # input("Press Enter to continue...")
                            
                        

                        elif type(x) == str:
                            x = x.split(" ")
                            x = [lemmatizer.lemmatize(y.lower()) for y in x]
                            x = " ".join(x)
                            temp.append(x)
                    

                    self.tfidfs[key] = temp





                        # if type(curr) == list:
                            
                        #     temp = []


                        #     # Because of gpt_keywords it has to be done in a different way because:
                        #     # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc5 in position 114: unexpected end of data
                        #     for x in curr:
                        #         try:
                        #             for y in x.split(" "):
                        #                 try:
                        #                     temp.append(lemmatizer.lemmatize(y.lower()))
                        #                 except:
                        #                     pass
                        #         except:
                        #             print(x)
                        #             print(curr)
                        #             input("Press Enter to continue...")
                        #     curr = temp
                        


                        # elif type(curr) == str:
                        #     print(curr)
                        #     input("Here")
                        #     curr = curr.split(" ")
                        #     curr = [lemmatizer.lemmatize(x.lower()) for x in curr]

                        # # za String je le en element in ga splitta

                        # curr = " ".join(curr)


                        # if self.tfidfs.get(key) is None:
                        #     self.tfidfs[key] = []
                        # self.tfidfs[key].append(curr)



                # else:
                    


                #     # Im
                #     for key in data_class.tfidfs.keys():
                #         # data_class.tfidfs[key] je lahko String, list stringov, list dolgih stringov.
                #         curr = data_class.tfidfs[key]
                #         if type(curr) == list:
                #             temp = []


                #             # Because of gpt_keywords it has to be done in a different way because:
                #             # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc5 in position 114: unexpected end of data
                #             for x in curr:
                #                 for y in x.split(" "):
                #                     try:
                #                         temp.append(y.lower())
                #                     except:
                #                         pass
                #             curr = temp
                #         elif type(curr) == str:
                #             curr = curr.split(" ")
                #             curr = [x.lower() for x in curr]

                #         # za String je le en element in ga splitta

                #         curr = " ".join(curr)

                #         if self.tfidfs.get(key) is None:
                #             self.tfidfs[key] = []
                #         self.tfidfs[key].append(curr)







            




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

        for key in self.tfidfs.keys():
            if all_vectorizers is None:
                vectorizer = TfidfVectorizer(**tfidf_args)
                vectorizer.fit(self.tfidfs[key])
            else:
                vectorizer = all_vectorizers[key]

            self.tfidfs[key] = vectorizer.transform(self.tfidfs[key]).tocsr()
            self.tfidf_names[key] = vectorizer.get_feature_names_out()
            new_vectorizers[key] = vectorizer


        
        # if all_vectorizers is None:
        #     vectorizer = TfidfVectorizer(**tfidf_args)
        #     vectorizer.fit(self.leads_tfidf)
        # else:
        #     vectorizer = all_vectorizers["leads"]
            
        # self.leads_tfidf = vectorizer.transform(self.leads_tfidf).tocsr()
        # self.leads_names = vectorizer.get_feature_names_out()
        # new_vectorizers["leads"] = vectorizer
        

        # if all_vectorizers is None:
        #     vectorizer = TfidfVectorizer(**tfidf_args)
        #     vectorizer.fit(self.keywords_tfidf)
        # else:
        #     vectorizer = all_vectorizers["keywords"]

        # self.keywords_tfidf = vectorizer.transform(self.keywords_tfidf).tocsr()
        # self.keywords_names = vectorizer.get_feature_names_out()
        # new_vectorizers["keywords"] = vectorizer


        # if all_vectorizers is None:
        #     vectorizer = TfidfVectorizer(**tfidf_args)
        #     vectorizer.fit(self.gpt_keywords_tfidf)
        # else:
        #     vectorizer = all_vectorizers["gpt_keywords"]

        # self.gpt_keywords_tfidf  = vectorizer.transform(self.gpt_keywords_tfidf).tocsr()
        # self.gpt_keywords_names = vectorizer.get_feature_names_out()
        # new_vectorizers["gpt_keywords"] = vectorizer

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

        new_data_prepared.days_of_week = [self.days_of_week[ix] for ix in chosen_ixs]

        new_data_prepared.nums_of_figs = self.nums_of_figs[chosen_ixs]

        new_data_prepared.topics = [self.topics[ix] for ix in chosen_ixs]

        new_data_prepared.num_of_comments = self.num_of_comments[chosen_ixs]


        for key in self.tfidfs.keys():
            try:
                new_data_prepared.tfidfs[key] = self.tfidfs[key].tocsr()[chosen_ixs, :]
                new_data_prepared.tfidf_names[key] = self.tfidf_names[key]
            except:
                print("key")
                print(key)
                input("Press Enter to continue...")
        # new_data_prepared.leads_tfidf = self.leads_tfidf.tocsr()[chosen_ixs, :]
        # new_data_prepared.leads_names = self.leads_names
        # # print("self.leads_tfidf.shape: " + str(self.leads_tfidf.shape))
        # # print("chosen_ixs[-10:]: " + str(chosen_ixs[-10:]))
        # # print("new_data_prepared.leads_tfidf.shape: " + str(new_data_prepared.leads_tfidf.shape))


        # new_data_prepared.keywords_tfidf = self.keywords_tfidf.tocsr()[chosen_ixs, :]
        # new_data_prepared.keywords_names = self.keywords_names

        # new_data_prepared.gpt_keywords_tfidf = self.gpt_keywords_tfidf.tocsr()[chosen_ixs, :]
        # new_data_prepared.gpt_keywords_names = self.gpt_keywords_names

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









# this is currently made for sparse matrices:
class MyNormalizer:

    def __init__(self, norm=2):
        self.norm = norm
        self.scaling_factors = None

    def fit(self, X):

        if X.ndim < 2:
            print("X.ndim < 2. data_prep, line 579")
            return

        try:
            self.scaling_factors = norm(X, axis=0, ord=self.norm)
            self.scaling_factors = np.where(self.scaling_factors == 0, 1, self.scaling_factors)
        except:
            print("Error with self.scaling_factors = np.linalg.norm(X, axis=0, ord=self.norm).reshape(1, -1). Giving X:")
            print(X)
            self.scaling_factors = norm(X, axis=0, ord=self.norm)
            


    def transform(self, X):

        if X.ndim < 2:
            print("X.ndim < 2. data_prep, line 595")
            return
        
        returner = X / self.scaling_factors
        return returner




class MyStandardizer:

    def __init__(self):
        self.means = None
        self.std_devs = None

    def fit(self, X):

        if X.ndim < 2:
            print("X.ndim < 2. data_prep, line 611")
            return
        
        data = X #.copy()

        self.means = np.mean(data, axis=0)

        self.std_devs = np.std(data, axis=0)

        # Prevents division by zero
        self.std_devs = np.where(self.std_devs == 0, 1, self.std_devs)


    def transform(self, X):

        # if X.ndim < 2:
        #     return
        data = X
        data -= self.means
        data /= self.std_devs
        return data



from sklearn.preprocessing import StandardScaler



def vertical_concat(firs_np, second_np):
    returner = np.vstack((firs_np, second_np))
    return returner


def copy_list(list_, turn_to_strings=False):
    returner = []
    
    if turn_to_strings:
        for item in list_:
            returner.append(str(item))
    else:
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

            self.days_of_week = [] # becomes np.array() in one_hot_encoding()
            self.days_of_week_names = []

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

            self.authors = copy_list(data_prepared.authors, turn_to_strings=True) # becomes np.array() in one_hot_encoding()
            self.authors_names = [] # gets filled in one_hot_encoding()

            self.years = np.copy(data_prepared.years)
            self.month_functions = np.copy(data_prepared.month_functions)
            self.month_functions_names = copy_list(data_prepared.month_functions_names)
            # self.hours = []
            self.parts_of_day = copy_list(data_prepared.parts_of_day) # becomes np.array() in one_hot_encoding()
            self.parts_of_day_names = [] # gets filled in one_hot_encoding()

            self.days_of_week = copy_list(data_prepared.days_of_week) # becomes np.array() in one_hot_encoding()
            self.days_of_week_names = [] # gets filled in one_hot_encoding()

            self.nums_of_figs = np.copy(data_prepared.nums_of_figs)
            
            self.topics = copy_list(data_prepared.topics)
            self.topics_encoded = [] # gets filled and becomes np.array() in one_hot_encoding()
            self.topics_names = [] # gets filled in one_hot_encoding()
            
            self.num_of_comments = np.copy(data_prepared.num_of_comments)

            self.tfidfs = {}
            self.tfidf_names = {}

            for key in data_prepared.tfidfs.keys():
                self.tfidfs[key] = data_prepared.tfidfs[key].copy()
                self.tfidf_names[key] = copy_list(data_prepared.tfidf_names[key])

            # self.leads_tfidf = data_prepared.leads_tfidf.copy()
            # self.leads_names = copy_list(data_prepared.leads_names)

            # self.keywords_tfidf = data_prepared.keywords_tfidf.copy()
            # self.keywords_names = copy_list(data_prepared.keywords_names)

            # self.gpt_keywords_tfidf = data_prepared.gpt_keywords_tfidf.copy()
            # self.gpt_keywords_names = copy_list(data_prepared.gpt_keywords_names)

        
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


    def yeald_complete_matrix(self, model_type="single_topic", params=None, give_names=False) -> csr_matrix :
        # type can be "single_topic", "grouped_topic", "unrecognised_topic", "all_topics_together"
        # normalizer can be None to ignore normalization, "L2" for L2 normalization, or an instance of MyNormalizer
        
        
        constant = np.ones((self.URLs.shape[0], 1))



        if model_type == "single_topic" or model_type == "unrecognised_topic":

            non_tfidf_matrix = np.hstack((constant, self.URLs, self.authors, self.years, self.month_functions, self.parts_of_day, self.days_of_week,
                                         self.nums_of_figs))
            # non_tfidf_matrix = csr_matrix(non_tfidf_matrix)

            non_tfidf_names = self.URL_names + self.authors_names + ["year"] +  self.month_functions_names + self.parts_of_day_names + self.days_of_week_names + ["num_of_figs"]
        
        elif model_type == "grouped_topic" or model_type == "all_topics_together":
            non_tfidf_matrix = np.hstack((constant, self.topics_encoded, self.URLs, self.authors, self.years, self.month_functions, self.parts_of_day,
                                         self.nums_of_figs, self.days_of_week))
            # non_tfidf_matrix = csr_matrix(non_tfidf_matrix)

            non_tfidf_names = self.topics_names + self.URL_names + self.authors_names + ["year"] +  self.month_functions_names + self.parts_of_day_names + ["num_of_figs"] + self.days_of_week_names

        # print("non_tfidf_matrix.shape: " + str(non_tfidf_matrix.shape))


        # Stack tfidf matrices:
        tfidf_matrices = []
        tfidf_names = []
        for key in self.tfidfs.keys():
            tfidf_matrices.append(self.tfidfs[key])
            tfidf_names.append(self.tfidf_names[key])

        tfidf_matrix = hstack(tfidf_matrices)
        tfidf_names = make_list_1D(tfidf_names)

        # print("tfidf_matrix.shape: " + str(tfidf_matrix.shape))




        normalizer = params["normalizer"]
        if type(normalizer) == str:

            if normalizer == "standardize":

                non_tfidf_normalizer = MyStandardizer()
                non_tfidf_normalizer.fit(non_tfidf_matrix)
                non_tfidf_matrix = non_tfidf_normalizer.transform(non_tfidf_matrix)

                tfidf_normalizer = StandardScaler(with_mean=False)
                tfidf_normalizer.fit(tfidf_matrix)
                tfidf_matrix = tfidf_normalizer.transform(tfidf_matrix)


                non_tfidf_matrix = csr_matrix(non_tfidf_matrix)
                returner_matrix = hstack([non_tfidf_matrix, tfidf_matrix])

                # normalizer = StandardScaler()
                # normalizer.fit(returner_matrix)
                # returner_matrix = normalizer.transform(returner_matrix)
                # normalizer = MyStandardizer()
                # normalizer.fit(returner_matrix)
                # returner_matrix = normalizer.transform(returner_matrix)

            elif normalizer == "L2":
                
                # non_tfidf_normalizer = MyNormalizer()
                non_tfidf_normalizer = StandardScaler(with_mean=False)
                non_tfidf_normalizer.fit(non_tfidf_matrix)
                non_tfidf_matrix = non_tfidf_normalizer.transform(non_tfidf_matrix)

                tfidf_normalizer = StandardScaler(with_mean=False)
                tfidf_normalizer.fit(tfidf_matrix)
                tfidf_matrix = tfidf_normalizer.transform(tfidf_matrix)


                non_tfidf_matrix = csr_matrix(non_tfidf_matrix)
                returner_matrix = hstack([non_tfidf_matrix, tfidf_matrix])
            

            elif normalizer == "L2all":
                
                non_tfidf_matrix = csr_matrix(non_tfidf_matrix)
                returner_matrix = hstack([non_tfidf_matrix, tfidf_matrix])
                # non_tfidf_normalizer = MyNormalizer()
                non_tfidf_normalizer = StandardScaler(with_mean=False)
                non_tfidf_normalizer.fit(returner_matrix)
                returner_matrix = non_tfidf_normalizer.transform(returner_matrix)

                tfidf_normalizer = None



            
            normalizer = (non_tfidf_normalizer, tfidf_normalizer)    

        else:

            if normalizer[1] is None:

                non_tfidf_matrix = csr_matrix(non_tfidf_matrix)
                returner_matrix = hstack([non_tfidf_matrix, tfidf_matrix])

                returner_matrix = normalizer[0].transform(returner_matrix)


            else:
                non_tfidf_matrix = normalizer[0].transform(non_tfidf_matrix)
                tfidf_matrix = normalizer[1].transform(tfidf_matrix)


                non_tfidf_matrix = csr_matrix(non_tfidf_matrix)
                returner_matrix = hstack([non_tfidf_matrix, tfidf_matrix])
        









        # print("returner_matrix.shape: " + str(returner_matrix.shape))
        
        


        returner = [returner_matrix, None, None, normalizer]

        if give_names:
            returner_names = non_tfidf_names + tfidf_names

            try:
                returner_names = np.array(returner_names)
            except:
                print("Error with returner_names = np.array(returner_names). Giving names:")
                print(returner_names)
                returner_names = np.array(returner_names)
            
            returner[2] = returner_names
        
            # print("returner[2].shape: " + str(returner[2].shape))
        









        returner_y = np.copy(self.num_of_comments)

        if not params is None:
            if type(params["comment_func"]) == int:
                returner_y = np.minimum(returner_y, params["comment_func"])
            elif params["comment_func"] == "perc_and_root":
                percentile = params["perc"]
                root = params["root"]
                y_percentile = np.percentile(returner_y, percentile)
                
                y_to_cap_ixs = np.where(returner_y > y_percentile)
                returner_y[y_to_cap_ixs] = y_percentile + (returner_y[y_to_cap_ixs] - y_percentile) ** (1 / root)

            elif params["comment_func"] == "root":
                root = params["root"]
                returner_y = returner_y ** (1 / root)
            
            elif params["comment_func"] == "log":
                returner_y = np.log(returner_y + 1)

            # print("returner_y.shape: " + str(returner_y.shape))




            """
            pca_cond = not params is None and params["pca"]        
            if pca_cond:
                from sklearn.decomposition import IncrementalPCA
                pca = IncrementalPCA(n_components=params["pca_n"])
                pca_data = pca.fit_transform(tfidf_matrix)
                pca_data = csr_matrix(pca_data)
                returner_matrix = hstack([returner_matrix, pca_data])

                if give_names:
                    return returner_matrix, returner_y, returner_names, pca
                else:
                    return returner_matrix, returner_y, pca
            """




        returner[1] = returner_y
        
        return returner
    

        """
        ALJAZ = "brez_authorjev_in_mont_funcov" # "brez_authorjev" # 

        if ALJAZ == "brez_authorjev":

            if model_type == "single_topic" or model_type == "unrecognised_topic":

                returner_matrix = np.hstack((self.URLs, self.years, self.month_functions, self.parts_of_day,
                                            self.nums_of_figs))
                returner_matrix = csr_matrix(returner_matrix)
                returner_matrix = hstack([returner_matrix, self.leads_tfidf, self.keywords_tfidf, self.gpt_keywords_tfidf])
            
            elif model_type == "grouped_topic" or model_type == "all_topics_together":
                returner_matrix = np.hstack((self.topics_encoded, self.URLs, self.years, self.month_functions, self.parts_of_day,
                                            self.nums_of_figs))
                returner_matrix = csr_matrix(returner_matrix)
                returner_matrix = hstack([returner_matrix, self.leads_tfidf, self.keywords_tfidf, self.gpt_keywords_tfidf])
            
            return returner_matrix, returner_y
        
        if ALJAZ == "brez_authorjev_in_mont_funcov":

            if model_type == "single_topic" or model_type == "unrecognised_topic":

                returner_matrix = np.hstack((self.URLs, self.years, self.parts_of_day,
                                            self.nums_of_figs))
                returner_matrix = csr_matrix(returner_matrix)
                returner_matrix = hstack([returner_matrix, self.leads_tfidf, self.keywords_tfidf, self.gpt_keywords_tfidf])
            
            elif model_type == "grouped_topic" or model_type == "all_topics_together":
                returner_matrix = np.hstack((self.topics_encoded, self.URLs, self.years, self.parts_of_day,
                                            self.nums_of_figs))
                returner_matrix = csr_matrix(returner_matrix)
                returner_matrix = hstack([returner_matrix, self.leads_tfidf, self.keywords_tfidf, self.gpt_keywords_tfidf])
            
            return returner_matrix, returner_y
        """

    
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

        new_data_topic.days_of_week = vertical_concat(self.days_of_week, other_data_topic.days_of_week)
        new_data_topic.days_of_week_names = [].extend(self.days_of_week_names)

        new_data_topic.nums_of_figs = vertical_concat(self.nums_of_figs, other_data_topic.nums_of_figs)
        
        new_data_topic.topics = copy_list(self.topics).extend(other_data_topic.topics)
        new_data_topic.topics_encoded = vertical_concat(self.topics_encoded, other_data_topic.topics_encoded)
        new_data_topic.topics_names = [].extend(self.topics_names)
        
        new_data_topic.num_of_comments = vertical_concat(self.num_of_comments, other_data_topic.num_of_comments)


        for key in self.tfidfs.keys():
            new_data_topic.tfidfs[key] = vertical_concat(self.tfidfs[key], other_data_topic.tfidfs[key])
            new_data_topic.tfidf_names[key] = [].extend(self.tfidf_names[key])

        # new_data_topic.leads_tfidf = vertical_concat(self.leads_tfidf, other_data_topic.leads_tfidf)
        # new_data_topic.leads_names = [].extend(self.leads_names)

        # new_data_topic.keywords_tfidf = vertical_concat(self.keywords_tfidf, other_data_topic.keywords_tfidf)
        # new_data_topic.keywords_names = [].extend(self.keywords_names)

        # new_data_topic.gpt_keywords_tfidf = vertical_concat(self.gpt_keywords_tfidf, other_data_topic.gpt_keywords_tfidf)
        # new_data_topic.gpt_keywords_names = [].extend(self.gpt_keywords_names)

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

        new_data_topic.days_of_week = np.copy(self.days_of_week)
        new_data_topic.days_of_week_names = copy_list(self.days_of_week_names)

        new_data_topic.nums_of_figs = np.copy(self.nums_of_figs)

        new_data_topic.topics = copy_list(self.topics)
        new_data_topic.topics_encoded = np.copy(self.topics_encoded)
        new_data_topic.topics_names = copy_list(self.topics_names)

        new_data_topic.num_of_comments = np.copy(self.num_of_comments)


        for key in self.tfidfs.keys():
            new_data_topic.tfidfs[key] = np.copy(self.tfidfs[key])
            new_data_topic.tfidf_names[key] = copy_list(self.tfidf_names[key])

        # new_data_topic.leads_tfidf = np.copy(self.leads_tfidf)
        # new_data_topic.leads_names = copy_list(self.leads_names)

        # new_data_topic.keywords_tfidf = np.copy(self.keywords_tfidf)
        # new_data_topic.keywords_names = copy_list(self.keywords_names)

        # new_data_topic.gpt_keywords_tfidf = np.copy(self.gpt_keywords_tfidf)
        # new_data_topic.gpt_keywords_names = copy_list(self.gpt_keywords_names)

        return new_data_topic
    
    def tfidf_chosen_ixs_trim(self, chosen_ixs_dict):

        if chosen_ixs_dict is None:
            return
        
        for key in self.tfidfs.keys():
            chosen_ixs = chosen_ixs_dict[key + "_chosen_ixs"]
            self.tfidfs[key] = self.tfidfs[key][:, chosen_ixs]
            self.tfidf_names[key] = [self.tfidf_names[key][ix] for ix in chosen_ixs]



        # leads_chosen_ixs = chosen_ixs_dict["leads_chosen_ixs"]
        # keywords_chosen_ixs = chosen_ixs_dict["keywords_chosen_ixs"]
        # gpt_keywords_chosen_ixs = chosen_ixs_dict["gpt_keywords_chosen_ixs"]

        # self.leads_tfidf = self.leads_tfidf[:, leads_chosen_ixs]
        # self.leads_names = [self.leads_names[ix] for ix in leads_chosen_ixs]

        # self.keywords_tfidf = self.keywords_tfidf[:, keywords_chosen_ixs]
        # self.keywords_names = [self.keywords_names[ix] for ix in keywords_chosen_ixs]

        # self.gpt_keywords_tfidf = self.gpt_keywords_tfidf[:, gpt_keywords_chosen_ixs]
        # self.gpt_keywords_names = [self.gpt_keywords_names[ix] for ix in gpt_keywords_chosen_ixs]
    
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

                self.days_of_week, self.days_of_week_names = one_hot_encode(self.days_of_week)        

                self.topics_encoded, self.topics_names = one_hot_encode(self.topics)

            else:

                self.URLs, self.URL_names = one_hot_encode_URLs(self.URLs, train_data_topic.URL_names)

                self.authors, self.authors_names = one_hot_encode(self.authors, train_data_topic.authors_names)

                self.parts_of_day, self.parts_of_day_names = one_hot_encode(self.parts_of_day, train_data_topic.parts_of_day_names)

                self.days_of_week, self.days_of_week_names = one_hot_encode(self.days_of_week, train_data_topic.days_of_week_names)        

                self.topics_encoded, self.topics_names = one_hot_encode(self.topics, train_data_topic.topics_names)

    # This is used so a DataTopic can be used to pass into one_hot_encoding, while being lightweight for pickling
    def get_DT_keeping_only_names(self):
        stripped_data_topic = DataTopic()

        stripped_data_topic.URL_names = self.URL_names

        stripped_data_topic.authors_names = self.authors_names

        stripped_data_topic.parts_of_day_names = self.parts_of_day_names

        stripped_data_topic.days_of_week_names = self.days_of_week_names

        stripped_data_topic.topics_names = self.topics_names

        return stripped_data_topic

    
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
        days_of_week: {self.days_of_week}
        days_of_week_names: {self.days_of_week_names}
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














def prepare_data(list_of_article_objects, all_vectorizers=None, is_test_data=False, only_all_together=True) -> tuple[dict, DataTopic, DataTopic]:
    # returns: topic_2_data_topic, grouped_topics, all_data_topic

    remaining_original_article_ixs = list(range(len(list_of_article_objects)))

    data_class = DataClass()
    deleted_ixs = data_class.extract_data(list_of_article_objects)

    # print("Here 1")

    for ix in sorted(deleted_ixs, reverse=True):
        del remaining_original_article_ixs[ix]



    if PRINTOUT:
        print("data_class.URLs[0]: " + str(data_class.URLs[0]))
        print("data_class.length: " + str(data_class.length))
        print("data_class.num_of_ommited_by_topic: " + str(data_class.num_of_ommited_by_topic))




    if True:

        data_prepared = DataPrepared(data_class)

        # print("Here 2")

        tfidf_args = {
            "max_features": None,
            "max_df": 0.85,
            "min_df": 0.001, # za 20 je zelo podobno število na koncu
            "norm": "l2",
            "stop_words": None,
            "smooth_idf": True
        }

        new_vectorizers = data_prepared.tfidf(all_vectorizers, tfidf_args)

        # print("Here 3")





        
        
        
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

















    prepared_data_len = len(data_prepared.URLs)
    possible_topics = list(set(data_prepared.topics))
    topic_2_ixs = dict()

    if not only_all_together:

        # Data splitting by topic

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
    else:
        topic_2_data_topic = None
        grouped_topics = None
    
    # print("Here 4")

    all_ixs = list(range(prepared_data_len))
    all_data_topic = DataTopic(None, data_prepared.copy_with_only_chosen_ixs(all_ixs))
    if not is_test_data:
        all_data_topic.one_hot_encoding()

    # print("Here 5")

    # Just for safety, return the old vectorizers directly
    returning_vectorizers = new_vectorizers if all_vectorizers is None else all_vectorizers

    return topic_2_data_topic, grouped_topics, all_data_topic, returning_vectorizers, topic_2_ixs, np.array(remaining_original_article_ixs)

