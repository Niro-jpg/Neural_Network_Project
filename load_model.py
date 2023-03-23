import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from Utils import *
import sys
from RNNs import *

models = []
models_name = []

path = "../Archive/DailyDelhiClimateTrain.csv"  
if "-d" in sys.argv:
    index = sys.argv.index("-d")
    path = sys.argv[index + 1]

_,size, dataset, miao = InitDataset(path)

#choosing the model
if "-m" in sys.argv:
    index = sys.argv.index("-m")
    arg = sys.argv[index + 1]

    if "r" in arg:
        model = Load(RNN, "RNN")
        models.append(model)
        models_name.append("RNN")
        print("rnn")


    if "s" in arg:
        model = Load(SRNN, "SRNN")
        models.append(model)
        models_name.append("SRNN")
        print("srnn")

    if "g" in arg:
        model = Load(GRU, "GRU")
        models.append(model)
        models_name.append("GRU")
        print("gru")

    if "l" in arg:
        model = Load(LSTM, "LSTM")
        models.append(model)
        models_name.append("LSTM")
        print("lstm")
        
j = 0
for model in models:
    print(models_name[j])
    j += 1
    print("actual value: ",miao[15])
    output, _ = model.forward(miao[:15].unsqueeze(0).float())
    print("predicted one: ", output)        