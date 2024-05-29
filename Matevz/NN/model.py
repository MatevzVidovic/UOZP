import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from timeit import default_timer as timer





preprocess_printout = False

epochs = 200
test_purposes = False


test_num_of_train_rows = 1000
test_num_of_test_rows = 1000



types_dict = {
    0 : "basic",
    1 : "deep_3pure_middle",
    2 : "deep_4pure_middle_dropout_leaky_relu",
    3 : "UOZP"
}

model_params = {
    "model_type" : types_dict[0],
    "chosen_num_of_features" : 5000,
    "second_layer_size" : 1000,
    "middle_layer_size" : 700,

    "dropout" : 0.1,
    "leaky_relu_alpha" : 0.1,

    "learning_rate" : 1e-3
}
"""
!!!!!!!!!!!!!!!!!!!
Res upam da je pri
nn.Softmax(dim=1)
dim pravilno izbran.

Pomoje ne, ker terrible rezultati.

Zdaj dajem na 0.
No, tudi tako dobim zelo zelo slabe rezultate.

Odstranil sem softmax.
!!!!!!!!!!!!!!!!!!!"""




main_folder = "built_models/"
if test_purposes:
    main_folder = "test_models/"

# model_data_path = main_folder + model_type + "_" + str(chosen_num_of_features) + "_" + str(second_layer_size) + "_" + str(middle_layer_size) + "_model/"

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# load_previous_model has been moved down, so that it works with while True










# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.flatten = nn.Flatten()

        model_type = model_params["model_type"]
        chosen_num_of_features = model_params["chosen_num_of_features"]
        second_layer_size = model_params["second_layer_size"]
        middle_layer_size = model_params["middle_layer_size"]

        dropout = model_params["dropout"] 
        leaky_relu_alpha = model_params["leaky_relu_alpha"]
        
        
        if model_type == "basic":
            self.sequential_NN_stack = nn.Sequential(
                nn.Linear(chosen_num_of_features, second_layer_size),
                nn.ReLU(),
                nn.Linear(second_layer_size, middle_layer_size),
                nn.ReLU(),
                nn.Linear(middle_layer_size, 1)
            )
        elif model_type == "deep_3pure_middle":
            self.sequential_NN_stack = nn.Sequential(
              nn.Linear(chosen_num_of_features, second_layer_size),
              nn.ReLU(),
              nn.Linear(second_layer_size, middle_layer_size),
              nn.ReLU(),
              nn.Linear(middle_layer_size, middle_layer_size),
              nn.ReLU(),
              nn.Linear(middle_layer_size, middle_layer_size),
              nn.ReLU(),
              nn.Linear(middle_layer_size, middle_layer_size),
              nn.ReLU(),
              nn.Linear(middle_layer_size, 1)
          )
        elif model_type == "deep_4pure_middle_dropout_leaky_relu":
            self.sequential_NN_stack = nn.Sequential(
              nn.Linear(chosen_num_of_features, second_layer_size),
              nn.ReLU(),
              nn.Linear(second_layer_size, middle_layer_size),
              nn.LeakyReLU(leaky_relu_alpha),
              nn.Linear(middle_layer_size, middle_layer_size),
              nn.ReLU(),
              nn.Dropout(p=dropout),
              nn.Linear(middle_layer_size, middle_layer_size),
              nn.LeakyReLU(leaky_relu_alpha),
              nn.Linear(middle_layer_size, middle_layer_size),
              nn.LeakyReLU(leaky_relu_alpha),
              nn.Linear(middle_layer_size, middle_layer_size),
              nn.ReLU(),
              nn.Linear(middle_layer_size, 1),
              # nn.Softmax(dim=0)
          )


        elif model_type == "UOZP":
            self.sequential_NN_stack = nn.Sequential(

                nn.Linear(chosen_num_of_features, second_layer_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                
                nn.Linear(second_layer_size, middle_layer_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                
                
                nn.Linear(middle_layer_size, 1)

          )
            
            

    def forward(self, x):
        x = self.flatten(x)
        logits = self.sequential_NN_stack(x)

        # m = nn.Softmax(dim=1)
        # print("Softmax(dim=1)")
        # print(m(logits))

        # m = nn.Softmax(dim=0)
        # print("Softmax(dim=0)")
        # print(m(logits))

        return logits








class CustomDataset(Dataset):
    def __init__(self, data_matrix, labels):
        self.data_matrix = data_matrix
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.data_matrix[idx], self.labels[idx]





# import json
# import gzip

# def read_json(data_path: str) -> list:
#     with gzip.open(data_path, 'rt', encoding='utf-8') as f:
#         return json.load(f)



class RunModel:
    
        def __init__(self, data_matrix, given_y) -> None:
            


            model_parameters = model_params
            model_parameters["model_type"] = types_dict[3]
            model_parameters["chosen_num_of_features"] = data_matrix.shape[1]
            model_parameters["second_layer_size"] = 200
            model_parameters["middle_layer_size"] = 50

            model_parameters["dropout"] = 0.0 # nočem ga ker weird reasoning, da je sploh zraven
                                            # 0.1 # pomaga proti overfittingu
            model_parameters["learning_rate"] = 1e-2
            
            # na novo dodane
            model_parameters["weight_decay"] = 0.01 # 0.1 je considered high.
            model_parameters["gradient_clipping_norm"] = 1.0

            self.model_data_path = main_folder + str(model_parameters["model_type"]) + "_" + str(model_parameters["chosen_num_of_features"]) + "_" + str(model_parameters["second_layer_size"]) + "_" + str(model_parameters["middle_layer_size"]) + "_model/"


            print(model_parameters)


            chosen_num_of_features = model_parameters["chosen_num_of_features"]
            learning_rate = model_parameters["learning_rate"]
            weight_decay = model_parameters["weight_decay"]
            gradient_clipping_norm = model_parameters["gradient_clipping_norm"]

            self.chosen_num_of_features = chosen_num_of_features
            self.gradient_clipping_norm = gradient_clipping_norm




            self.batch_size = 64

            self.data = (data_matrix.astype(np.float32), given_y.astype(np.float32))





            # Get cpu, gpu or mps device for training.
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )



            self.model = NeuralNetwork(model_parameters).to(self.device)
            self.model.float() # makes sure the weights are float32
            print(self.model)


            # True if directory already exists
            load_previous_model = os.path.isdir(self.model_data_path)
            # Set to False if you want to rewrite data
            load_previous_model = load_previous_model





            if load_previous_model:
                prev_model_details = pd.read_csv(self.model_data_path + "previous_model_" + str(chosen_num_of_features) + "_details.csv")
                self.prev_serial_num = prev_model_details["previous_serial_num"][0]
                self.prev_cumulative_epochs = prev_model_details["previous_cumulative_epochs"][0]
                self.model.load_state_dict(torch.load(self.model_data_path + "model_" + str(chosen_num_of_features) + "_" + str(self.prev_serial_num) + ".pth"))
            else:
                self.prev_serial_num = 0
                self.prev_cumulative_epochs = 0
                    
                


            # nn.MSELoss()
            # self.loss_fn = nn.CrossEntropyLoss() # good for classification., doesn't work with regression
            self.loss_fn = nn.MSELoss() # good for regression


            # https://pytorch.org/docs/stable/optim.html
            # SGD - stochastic gradient descent
            # imajo tudi Adam, pa sparse adam, pa take.
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)






        def predict(self):

            # X = list(self.data[0])
            # y = list(self.data[1])

            # X, _, y, _ = train_test_split(self.data[0], self.data[1], test_size=0.0, shuffle=False) #, random_state=42)

            # X = self.data[0].toarray()
            # y = self.data[1].toarray()

            # print(type(self.data[0]))
            X = self.data[0].toarray()
            # y = np.array(self.data[1])
            y = self.data[1]


            predict_data = CustomDataset(X, y)
            dataloader = DataLoader(predict_data, batch_size=self.batch_size)
            model = self.model
            
            model.eval()

            predictions = []
            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = model(X)
                    predictions.extend(pred.tolist())
                    # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            predictions = np.array(predictions)
            predictions = np.reshape(predictions, (-1, 1))

            if os.path.exists('run_model_predictions.txt'):
                os.remove('run_model_predictions.txt')

            np.savetxt('run_model_predictions.txt', predictions)

            return predictions










          
        def train(self, dataloader):

            model = self.model
            loss_fn = self.loss_fn
            optimizer = self.optimizer
            gradient_clipping_norm = self.gradient_clipping_norm

            size = len(dataloader.dataset)
            model.train()
            
            start = timer()

            train_times = []

            for batch, (X, y) in enumerate(dataloader):

                X, y = X.to(self.device), y.to(self.device)

                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping_norm)  # Gradient clipping
                optimizer.step()
                optimizer.zero_grad()

                if batch % 100 == 0:
                    end = timer()
                    train_times.append(end - start)
                    start = timer()
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            return train_times
            


        def test(self, dataloader):

            model = self.model
            loss_fn = self.loss_fn


            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            model.eval()
            test_loss = 0
            
            all_y = []
            all_y_pred = []


            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    # print("y")
                    # print(y)
                    # print("pred")
                    # print(pred)
                    # print("loss_fn(pred, y)")
                    # print(loss_fn(pred, y))
                    # input("Press Enter to continue...")
                    all_y.extend(y.tolist())
                    all_y_pred.extend(pred.tolist())

            all_y = np.array(all_y)
            all_y_pred = np.array(all_y_pred)
            
            test_loss /= num_batches
            R2 = 1 - (pred - y).var() / y.var()
            
            print(f"Test Error: \n R2: {(R2):>0.3f}%, Avg loss: {test_loss:>8f} \n")

            
            return R2, test_loss
        

   



        def loop_trainings(self):
            while True:
                self.load_train_test_and_save()
                
                self.prev_serial_num += 1
                self.prev_cumulative_epochs += 1
                


        def load_train_test_and_save(self):


            


            X_train, X_test, y_train, y_test = train_test_split(self.data[0], self.data[1], test_size=0.2) #, random_state=42)

            train_data = CustomDataset(X_train, y_train)
            test_data = CustomDataset(X_test, y_test)


            train_dataloader = DataLoader(train_data, batch_size=self.batch_size)
            test_dataloader = DataLoader(test_data, batch_size=self.batch_size)

            for X, y in test_dataloader:
                print(f"Shape of X [N, C, H, W]: {X.shape}")
                print(f"Shape of y: {y.shape} {y.dtype}")
                break




            R2s = []
            avg_losses = []

            all_train_times = []

            for t in range(epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                train_times = self.train(train_dataloader)
                del train_times[0]
                print(train_times)
                all_train_times.extend(train_times)
                
                R2, loss = self.test(test_dataloader)
                R2s.append(R2)
                avg_losses.append(loss)
            print("Done!")





            try:
                os.mkdir(main_folder)
            except:
                pass

            try:
                os.mkdir(self.model_data_path)
            except:
                pass

            torch.save(self.model.state_dict(), self.model_data_path + "model_" + str(self.chosen_num_of_features) + "_" + str(self.prev_serial_num+1) + ".pth")

            new_df = pd.DataFrame({"previous_serial_num": [self.prev_serial_num+1], "previous_cumulative_epochs": [self.prev_cumulative_epochs+epochs]})
            new_df.to_csv(self.model_data_path + "previous_model_" + str(self.chosen_num_of_features) + "_details.csv")

            """
            Razlog za spodnji nacin kode:
            File "/home/matevzvidovic/.local/lib/python3.10/site-packages/pandas/core/internals/construction.py", line 677, in _extract_index
                raise ValueError("All arrays must be of the same length")
            ValueError: All arrays must be of the same length
            """
            a = {"train_times": all_train_times, "R2s": R2s, "avg_losses": avg_losses}
            new_df = pd.DataFrame.from_dict(a, orient='index')
            new_df = new_df.transpose()
            new_df.to_csv(self.model_data_path + "train_times_" + str(self.prev_serial_num+1) + ".csv")

            print("Saved PyTorch Model State")





