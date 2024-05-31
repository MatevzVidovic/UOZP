

from naloga5 import Model, test_model, run
from data_preparation import prepare_data

import gzip
import json


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import numpy as np



# PRINTOUT = False
# MAIN_PRINTOUT = True
# PLOTS = False

# PARAMS = {
#     "col_norm" : "dont", # "dont" or "L2"
# }

# YEALD_MAT_PARAMS = {

#     "cap_comment_n" : "perc_and_root", # za 500 je izboljšanje
#     # can be None for no cap, integer for absolute cap, or "perc_and_root" for
#     # perc percentile value + (number - perc percentile value)^(1/root).
#     "perc" : 99,
#     "root" : 4,
    
#     # This is not at all supported yet. Keep it False.
#     # The big problem is that how to pass parameters into testing calls of yeald_mat...
#     # and not have it do the capping.
#     "pca" : False,
#     "pca_n": 100, # num of pca components
# }


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
    "alpha" : 4.0, # 0.005, 
    
    # 0.8:  (0.11, 0.286)   0.08: (0.13793373, 0.29954335)
    
    # za brez L2 normalizacije:
    # with Ridge 40 best for all_together (0.238), 5 best for separate 0.32125455 59.4660625  28.07151034, with max_iter 400
    # za Ridge 30 je 0.237 za all_together. Za ridge za 3 je 0.3209 za separate.
    "method" : "Ridge", #Ridge in Basic ne delata # "Basic", "Ridge" or "Lasso"
    # or LassoCV, RidgeCV, ElasticNetCV
}


def cv():

    # Path to your .json.gzip file
    file_path = './data/rtvslo_train.json.gzip'

    # Open the gzip file
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        # Read and parse the JSON data
        data = json.load(f)

    scores_1 = None
    scores_2 = None

    print("Hyperparameters: ", hyper_parameters)






    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_index, test_index in kf.split(data):
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]
        curr_scores_1, curr_scores_2 = run(train_data, test_data, hyper_parameters, all_topics_together="both")

        if scores_1 is None:
            scores_1 = np.array(curr_scores_1)
            scores_2 = np.array(curr_scores_2)
        else:
            scores_1 = np.vstack((scores_1, curr_scores_1))
            scores_2 = np.vstack((scores_2, curr_scores_2))


    # Second round of cv:

    # kf = KFold(n_splits=5, shuffle=True, random_state=4)
    
    # for train_index, test_index in kf.split(data):
    #     train_data = [data[i] for i in train_index]
    #     test_data = [data[i] for i in test_index]
    #     curr_scores_1, curr_scores_2 = run(train_data, test_data, hyper_parameters, all_topics_together="both")

    #     if scores_1 is None:
    #         scores_1 = np.array(curr_scores_1)
    #         scores_2 = np.array(curr_scores_2)
    #     else:
    #         scores_1 = np.vstack((scores_1, curr_scores_1))
    #         scores_2 = np.vstack((scores_2, curr_scores_2))

    

    scores_1 = np.mean(scores_1, axis=0)
    print("Scores, all together: ", scores_1)

    scores_2 = np.mean(scores_2, axis=0)
    print("Scores, separate: ", scores_2)





cv()