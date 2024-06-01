import argparse
import json
import gzip
import os
import numpy as np

from data_preparation import prepare_data
from naloga5 import WrapperModel

import pickle
import time
TRAIN_ON_THE_SPOT = True
# TRAIN_TIME_IN_SECS = int(0.5 * 60 * 60)

# ohe_cutoff = 150
# tfidf_cutoff = 150

hyper_parameters = {

    # "max_iter" : 400,

    # "URLs" : ohe_cutoff,
    # "authors" : ohe_cutoff,
    # "leads" : ohe_cutoff,
    # "keywords" : tfidf_cutoff,
    # "gpt_keywords" : tfidf_cutoff,
    # "topics" : tfidf_cutoff, 
}


def read_json(data_path: str) -> list:
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        return json.load(f)


class RTVSlo:

    def __init__(self):
        pass

    def fit(self, train_data: list):


        # wrapper_model_creator.py

        # ohe_cutoff = 150
        # tfidf_cutoff = 150

        hyper_parameters = {

            # "max_iter" : 400,

            # "URLs" : ohe_cutoff,
            # "authors" : ohe_cutoff,
            # "leads" : ohe_cutoff,
            # "keywords" : tfidf_cutoff,
            # "gpt_keywords" : tfidf_cutoff,
            # "topics" : tfidf_cutoff, 
        }



        # # Path to your .json.gzip file
        # file_path = './data/rtvslo_train.json.gz'

        # # Open the gzip file
        # with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        #     # Read and parse the JSON data
        #     data = json.load(f)

        curr_model = WrapperModel(train_data, hyper_parameters)

        with open('wrapper_model.pkl', 'wb') as f:
            pickle.dump(curr_model, f)
        





        # start_time = time.time()

        # # trainer.py

        # with open('wrapper_model.pkl', 'rb') as f:
        #     curr_model = pickle.load(f)

        for _ in range(2):
            
            # curr_time = time.time()
            # if curr_time - start_time > TRAIN_TIME_IN_SECS:
            #     break

            curr_model.train_me(train_data)


        
    def predict(self, test_data: list) -> np.array:

        with open('wrapper_model.pkl', 'rb') as f:
            curr_model = pickle.load(f)
        
        _, _, y_pred = curr_model.predict(test_data)
        
        return y_pred
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_path', type=str, nargs='?', default='./data/rtvslo_train.json.gz',
                        help='Path to the training data file')
    parser.add_argument('test_data_path', type=str, nargs='?', default='./data/rtvslo_test.json.gz',
                        help='Path to the test data file')
    args = parser.parse_args()


    train_data = read_json(args.train_data_path)

    # train_data = read_json("./data/rtvslo_train.json.gz")
    # test_data = read_json("./data/rtvslo_test.json.gz")

    rtv = RTVSlo()

    if TRAIN_ON_THE_SPOT:
        rtv.fit(train_data)



    test_data = read_json(args.test_data_path)

    predictions = rtv.predict(test_data)

    if os.path.exists('predictions.txt'):
        os.remove('predictions.txt')

    np.savetxt('predictions.txt', predictions)

if __name__ == '__main__':
    main()
