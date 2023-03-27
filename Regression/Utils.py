
import torch
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from RNNs import *

# Class we used at the start of the project


class SequentialDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = 0
        self.max = dataset.size()[0]

    def GetItems(self, i):
        if (self.index > self.max - 2):
            self.index = 0
        if (i + self.index > self.max - 1):
            #x = self.dataset[self.index:self.max - 1]
            #y = self.dataset[self.max - 1]
            self.index = 0
            # return x, y
        x = self.dataset[self.index:self.index + i - 1]
        y = self.dataset[self.index + i - 1]
        self.index += i
        return x, y

# Init of the dataset


def InitDataset(path):
    df = pd.read_csv(path, header=0)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    window_size = pd.Timedelta(days=7)
    time_series = df.set_index('date').rolling(window=window_size).mean()
    time_series = time_series.dropna()
    # normalization
    #time_series = (time_series-time_series.mean())/time_series.std()
    time_series_array = torch.from_numpy(time_series.to_numpy())
    column_number = time_series_array.size()[1]
    row_number = time_series_array.size()[0]

    return row_number, column_number, SequentialDataset(time_series_array), time_series_array
# Dataset containing the time series


class Data(Dataset):
    def __init__(self, train_path, sequence_length=100, max_dim=0):

        size, _, data, miao = InitDataset(train_path)

        self.dataset = []
        self.targets = []
        if sequence_length > size:
            sequence_length = 100

        for i in range(miao.size()[0] - sequence_length):
            x = miao[i:sequence_length + i]
            t = miao[i + sequence_length]
            self.dataset.append(x)
            self.targets.append(t)

    def __getitem__(self, index):
        x = self.dataset[index]
        return x, self.targets[index]

    def __len__(self):
        return len(self.dataset)

# function to save the model


def Save(model, PATH):
    torch.save([model.kwargs, model.state_dict()], PATH)

# function to load the saved model


def Load_s(MODEL, PATH):
    kwargs, state = torch.load(PATH)
    model = MODEL(**kwargs)
    model.load_state_dict(state)
    return model
