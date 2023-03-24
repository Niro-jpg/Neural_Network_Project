import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from VARIABLES import *
import chardet
with open(ARCHIVE_PATH, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

data = pd.read_csv(ARCHIVE_PATH, encoding = 'utf-8')

print(data)