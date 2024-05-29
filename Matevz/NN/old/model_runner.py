

from data_preparation import prepare_data

import gzip
import json


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import numpy as np


from naloga5 import Model, test_model

import train_me from trainer



ohe_cutoff = 150
tfidf_cutoff = 150

hyper_parameters = {

    "max_iter" : 400,

    "URLs" : ohe_cutoff,
    "authors" : ohe_cutoff,
    "leads" : ohe_cutoff,
    "keywords" : tfidf_cutoff,
    "gpt_keywords" : tfidf_cutoff,
    "topics" : tfidf_cutoff,
    "alpha" : 8.0, # with Ridge 40 best for all_together (0.238), 5 best for separate 0.32125455 59.4660625  28.07151034, with max_iter 400
    # za Ridge 30 je 0.237 za all_together. Za ridge za 3 je 0.3209 za separate.
    "method" : "Ridge", #Ridge in Basic ne delata # "Basic", "Ridge" or "Lasso"
    # or LassoCV, RidgeCV, ElasticNetCV
}


mode = "train" # "train" or "test"

# Path to your .json.gzip file
file_path = './data/rtvslo_train.json.gzip'

# Open the gzip file
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    # Read and parse the JSON data
    data = json.load(f)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)




if mode == "train":


    topic_2_train_DT, grouped_topics_DT, all_together_DT, vectorizers, _, _ = prepare_data(train_data)
    curr_model = Model(topic_2_train_DT, grouped_topics_DT, all_together_DT, vectorizers, hyper_parameters=hyper_parameters)

    matrix, y = curr_model.give_data()

    train_me(matrix, y)




elif mode == "test":

    topic_2_train_DT, grouped_topics_DT, all_together_DT, vectorizers, _, _ = prepare_data(train_data)
    curr_model = Model(topic_2_train_DT, grouped_topics_DT, all_together_DT, vectorizers, hyper_parameters=hyper_parameters)

    curr_model.load_model()
    
    _, y_test, y_pred, _ = curr_model.predict(test_data)

    test_model(y_test, y_pred)



