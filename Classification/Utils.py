
import torch
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from RNNs import *
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from VARIABLES import * 
from sklearn.metrics import accuracy_score
from Utils import *

#nltk.download('punkt')

def accuracy(tensor_t, tensor_y):
  tensor_y = (tensor_y >= 0.5)
  return accuracy_score(tensor_t, tensor_y)
  

def create_words_tensor( max_len, number_of_features, frasi, vocab, tokenizer, dataset_dim = 1000):
    number_of_elements = min(dataset_dim, len(frasi)) 
    X = torch.zeros(number_of_elements, max_len, number_of_features)
    
    for i in range(dataset_dim):
        words = tokenizer(frasi[i])
        iters = min(len(words),max_len)
        for j in range(iters):
            X[i][j] = vocab[words[j].lower()]   
            
    return X     

def InitDataset(path):       

  frase = "ciao sono io flavio"
  arrays = frase.split(" ")

  df = pd.read_csv( ARCHIVE_PATH, header = 0) 

  frasi = np.array(df['review'])
  target = np.array(df['sentiment'])

  tokenizer = get_tokenizer('basic_english')

  def yield_tokens():
    for example in frasi:
        tokens = tokenizer(example)
        yield tokens
        
  token_generator = yield_tokens()  
        
  vocab = build_vocab_from_iterator(token_generator)

  vocab.get_stoi()
  
  max_len = 60
  
  dataset_dim = 1000

  X = create_words_tensor(max_len,1,frasi, vocab, tokenizer, dataset_dim = dataset_dim)

  encoder = LabelEncoder()
  y_encoded = torch.tensor(encoder.fit_transform(target))
  
  return dataset_dim, 1, X, y_encoded

class Data(Dataset):
    def __init__(self, train_path):

        size ,features, X, t = InitDataset(train_path)   

        self.dataset = []
        self.targets = []

        for i in range(size):
          self.dataset.append(X[i])
          self.targets.append(t[i])

    def __getitem__(self, index):
        x = self.dataset[index]
        return x, self.targets[index]
    
    def __len__(self):
        return len(self.dataset)
    
def Save(model, PATH):
  torch.save([model.kwargs, model.state_dict()], PATH)

def Load_s(MODEL, PATH):
  kwargs, state = torch.load(PATH)
  model = MODEL(**kwargs)
  model.load_state_dict(state)
  return model
