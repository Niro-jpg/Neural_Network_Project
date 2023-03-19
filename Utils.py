
import torch
import pandas as pd

class SequentialDataset:
  def __init__(self, dataset):
    self.dataset = dataset
    self.index = 0
    self.max = dataset.size()[0]

  def GetItems(self, i):

    if (self.index > self.max - 2):
      self.index = 0

    if (i + self.index > self.max - 1):
      x = self.dataset[self.index:self.max - 1]
      y = self.dataset[self.max - 1]
      self.index = 0 
      return x, y
    
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

    return column_number,SequentialDataset(time_series_array), time_series_array