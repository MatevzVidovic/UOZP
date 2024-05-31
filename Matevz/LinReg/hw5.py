import argparse
import json
import gzip
import os
import numpy as np


def read_json(data_path: str) -> list:
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        return json.load(f)



from naloga5 import Model, test_model, run
from data_preparation import prepare_data

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

ALL_TOPICS_TOGETHER = False


class RTVSlo:

    def __init__(self):
        pass

    def fit(self, train_data: list):
        
        
        topic_2_train_DT, grouped_topics_DT, all_together_DT, vectorizers, _, _ = prepare_data(train_data)

        self.model = Model(topic_2_train_DT, grouped_topics_DT, all_together_DT, vectorizers, hyper_parameters=hyper_parameters)

        
    def predict(self, test_data: list) -> np.array:
        
    
        assert type(self.model) == Model

        if ALL_TOPICS_TOGETHER:
            _, _, returner, _ = self.model.predict(test_data, all_together_model=True)
        else:
            _, _, returner, _ = self.model.predict(test_data)
        
        return returner
    
    
    

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('train_data_path', type=str)
    # parser.add_argument('test_data_path', type=str)
    # args = parser.parse_args()

    train_data = read_json("rtvslo_train.json.gz")
    test_data = read_json("rtvslo_test.json.gz")

    rtv = RTVSlo()
    rtv.fit(train_data)
    predictions = rtv.predict(test_data)

    if os.path.exists('predictions.txt'):
        os.remove('predictions.txt')

    np.savetxt('predictions.txt', predictions)

if __name__ == '__main__':
    main()
