
import torch
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from RNNs import *

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
      #return x, y
    
    x = self.dataset[self.index:self.index + i - 1]
    y = self.dataset[self.index + i - 1]
    self.index += i
    return x, y

def InitDataset(path):

    df = pd.read_csv( path, header = 0) 

    # converte la colonna temporale in tipo datetime
    df['date'] = pd.to_datetime(df['date'])

    # ordina il dataframe in base alla colonna temporale
    df = df.sort_values(by='date')

    # definisce la finestra temporale di 7 giorni
    window_size = pd.Timedelta(days=7)

    # crea le time series sovrapposte
    time_series = df.set_index('date').rolling(window=window_size).mean()

    # rimuove le righe che contengono dati mancanti
    time_series = time_series.dropna()

    #normalization
    #time_series = (time_series-time_series.mean())/time_series.std()

    # converte le time series in un array numpy
    time_series_array = torch.from_numpy(time_series.to_numpy())
    column_number = time_series_array.size()[1]
    row_number = time_series_array.size()[0]

    return row_number, column_number,SequentialDataset(time_series_array), time_series_array

class Data(Dataset):
    def __init__(self, train_path,sequence_length = 50, max_dim = 0):

        size ,features, data, miao = InitDataset(train_path)   

        self.dataset = []
        self.targets = []
        if sequence_length > size:
            sequence_length = int(size/10)
        self.dataset_size = int(size/sequence_length)

        for i in range(self.dataset_size):
          x, t = data.GetItems(sequence_length)
          self.dataset.append(x)
          self.targets.append(t)

    def __getitem__(self, index):
        x = self.dataset[index]
        return x, self.targets[index]
    
    def __len__(self):
        return len(self.dataset)
    
def Save(model, PATH):
  torch.save([model.kwargs, model.state_dict()], PATH)

def Load(MODEL, PATH):
  kwargs, state = torch.load(PATH)
  model = MODEL(**kwargs)
  model.load_state_dict(state)
  return model
