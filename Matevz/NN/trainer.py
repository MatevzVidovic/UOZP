

import pickle
with open('wrapper_model.pkl', 'rb') as f:
    curr_model = pickle.load(f)

while True:
    curr_model.train_me()
