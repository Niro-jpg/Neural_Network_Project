import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from VARIABLES import *
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def create_words_tensor(number_of_features, number_of_elements, frasi, vocab):
    X = torch.zeros(number_of_elements, number_of_features)
    
    for i, frase in enumerate(frasi):
        words = tokenizer(frase)
        iters = min(len(words),number_of_features)
        for j in range(iters):
            X[i][j] = vocab[words[j].lower()]   
            
    return X            

#nltk.download('punkt')

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

#X = create_words_tensor(60,len(frasi),frasi, vocab)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(target)
print(type(y_encoded))

print(torch.tensor(y_encoded))