''' fine tuning a bert model in pyt
'''

import gzip 
import shutil
import time

import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torchtext

import transformers
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification


torch.backends.cudnn.deterministic=True
RANDOM_SEED=123
torch.manual_seed(RANDOM_SEED)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE: {DEVICE}')

NUM_EPOCHS=3


url=("https://github.com/rasbt/machine-learning-book/raw/main/ch08/movie_data.csv.gz")
filename=url.split("/")[-1]

with open(filename, "wb") as f:
    r=requests.get(url)
    f.write(r.content)


with gzip.open('movie_data.csv.gz','rb') as f_in:
    with open('movie_data.csv','wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


df=pd.read_csv('movie_data.csv')
print(df.head(3))
