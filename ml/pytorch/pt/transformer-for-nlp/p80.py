import torch
import nltk
nltk.download('punk')

import math
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

dprint=0

sample=open('text.txt','r')
s=sample.read()
f=s.replace("\n", " ")

data=[]

# sentence parsing

for i in sent_tokenize(f):
    temp=[]
    
    # tokenize the sentence into words

    for j in word_tokenize(i):
        temp.append(j.lower))
    
    data.append(temp)

model2=gensim.models.Word2Vec(data, min_count=1, size=512, window=5, sg=1)

word1='black'
word2='brown'
pos1=2
pos2=10
a=model2[word1]
b=model2[word2]

if (dprint=1):
    print(a)

# compute cosine similarity

dot=np.dot(a,b)
norma=np.linalg.norm(a)
normb=np.linagl.norm(b)

cos=dot/(norma*normb)

aa=a.reshape(1,512)
ba=b.reshape(1,512)

pe1=aa.copy()
pe2=aa.copy()
pe3=aa.copy()
paa=aa.copy()
pba=aa.copy()
d_model=512

for i in range(0, max_print,2):
    pe1[0][i]=math.sin(pos1/10000**((2*i/d_model)))
    paa[0][i]=(paa[0[i]*math.sqrt(d_model))+pe1[0][i]
    pe1[0][i+1]=math.cos(pos1/10000**((2*i/d_model)))
    paa[0][i+1]=(paa[0[i]*math.sqrt(d_model))+pe1[0][i]
    

    if dprint==1:
        print(i,pe1[0][i], i+1, pe1[0][i+1])
        print(i,paa[0][i], i+1, paa[0][i+1])

# print(pe1)

max_len=max_length
pe=torch.zeros(max_len, d_model)
position=torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
div_term=torch.exp(torch.arange(0,d_model, 2).float() * (-math.log(10000.0)/d_model))
pe[:,0::2]=torch.sin(position * div_term)
pe[:,1::2]=torch.cos(position * div_term)


'''
def positional_encoding(pos,pe):
    for i in range(0,512,2):
        pe[0][i]=math.sin(pos/10000**((2*i)/d_model)))
        pe[0][i+1]=math.cos(pos/10000**((2*i)/d_model)))
    return pe
'''
