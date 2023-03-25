import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from VARIABLES import *
import chardet
import torchtext

with open(ARCHIVE_PATH, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

data = pd.read_csv(ARCHIVE_PATH)

features = data.iloc[:,:1]
target = data.iloc[:,-1:]

#features_tensor = torch.tensor(features.values)

label_encoder = LabelEncoder()
target_int = label_encoder.fit_transform(target)
target_tensor = torch.tensor(target_int)

#print(features)
print(target_tensor)

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
text_field = torchtext.data.Field(sequential=True, use_vocab=True, tokenize=tokenizer, lower=True, batch_first=True)