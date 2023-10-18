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

train_texts=df.iloc[:35000]['review'].values
train_labels=df.iloc[:35000]['sentiment'].values
valid_texts=df.iloc[35000:40000]['review'].values
valid_labels=df.iloc[35000:40000]['sentiment'].values
test_texts=df.iloc[40000:]['review'].values
test_labels=df.iloc[40000:]['sentiment'].values

tokenizer=DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings=tokenizer(list(train_texts), truncation=True, padding=True)
valid_encodings=tokenizer(list(train_texts), truncation=True, padding=True)
test_encodings=tokenizer(list(train_texts), truncation=True, padding=True)

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings=encodings
        self.labels=labels

    def __getitem__(self, idx):
        item={key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels']=torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset=IMDBDataset(train_encodings, train_labels)
valid_dataset=IMDBDataset(train_encodings, valid_labels)
test_dataset=IMDBDataset(train_encodings, test_labels)

train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader=torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

model=DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(DEVICE)
model.train()

optim=torch.optim.Adam(model.parameters(), lr=5e-5)

def compute_accuracy(model, data_laoder, device):
    with torch.no_grad():
        correct_pred, num_examples=0,0
        for batch_idx, batch in enumerate(data_loader):

        # Prepare data.

            input_ds = batch['input_ids'].to(device)
            attention_mask=batch['attention_mask'].to(device)
            labels=batch['labels'].to(device)

            outputs=model(input_ids, attention_mask=attention_mask)
            logits=output['logits']
            predicted_labels=torch.argmax(logits,1)
            num_examples+= labels.size(0)
            correct_pred+=(predicted_labels==labels).sum()
    return correct_pred.float()/num_examples * 100

start_time=time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, batch in enumerate(train_loader):

        ## prep data.

        input_ids=batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels=batch['labels'].to(DEVICE)

        ## fwd pass.

        output=model(input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits=output['loss'], output['logits']

        # backward pass.

        optim.zero_grad()
        loss.backward()
        optim.step()

        ## logging.
    
        if not batch_idx % 250:
            print(f'Epoch: {epoch+1:04d}/{NUM_EPOCHS:04d} | Batch {batch_idx:04d} | Loss: {loss:.4f}')

    model.eval()

    with torch.set_grad_enabled(False):
        print(f'Training accuracy: {compute_accuracy(model, train_loader, DEVICE):.2f}%\nValid accuracy: {compute_accuracy(model, valid_loader, DEVICE):.2f}%')

        print(f'Time elapsed: {(time.time() - start_time) / 60:.2f} min')

    
    print(f'Total training time: {(time.time() - start_time)/60:.2f} min')
    print(f'test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%')    



            
    
