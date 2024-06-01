
import gzip
import json
import pickle
with open('wrapper_model.pkl', 'rb') as f:
    curr_model = pickle.load(f)


# Path to your .json.gzip file
file_path = './data/rtvslo_train.json.gz'

# Open the gzip file
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    # Read and parse the JSON data
    data = json.load(f)

while True:
    curr_model.train_me(data)
