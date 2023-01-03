import torch
import torch.nn as nn
import code
from functools import wraps
from datetime import datetime

from torchtext.datasets import IMDB
train_dataset = IMDB(split='train')
test_dataset = IMDB(split='test')
import re

CONFIG_USE_ROCM=0

# 1. create dataset
from torch.utils.data.dataset import random_split

torch.manual_seed(1)
train_dataset, valid_dataset = random_split(list(train_dataset), [20000, 5000])
test_dataset=list(test_dataset)

# 2. find unique tokens

from collections import Counter, OrderedDict

def print_fcn_name(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string, ": ", func.__name__, " entered...")
        result = func(*args, **kwargs)
        return result

    return wrapper

#@print_fcn_name
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(
           '(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ' , text.lower()) + \
        ' '.join(emoticons).replace('-', '')
    
    tokenized = text.split()
    return tokenized

  
token_counts = Counter()
for label, line in train_dataset:
    tokens = tokenizer(line)
    token_counts.update(tokens)
print('Vocab-size:', len(token_counts))

# 3. encoding each unique token into integres

from torchtext.vocab import vocab
sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
vocab = vocab(ordered_dict)
vocab.insert_token('<pad>', 0)
vocab.insert_token('<unk>', 1)
vocab.set_default_index(1)

print([vocab[token] for token in ['this', 'is', 'an', 'example']])

# 3a. define the functions for transformation.

text_pipeline =  lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: 1. if x == 'pos' else 0.

# 3b. wrap the encode and transformation function.

'''
arg: batch - list size of 4. Each list member is a tuple with (integer, text sentence) format assigned to label_ and text_
text_list to padded_text_list: text_list each member may be variable, padding will make equal all lists to longest length
in unpadded text_list.
'''

@print_fcn_name
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))

        # produces encoded text after calling text_pipeline -> tokenizer.

        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))

    if CONFIG_USE_ROCM:
        label_list = torch.tensor(label_list, device='cuda')
        lengths = torch.tensor(lengths, device='cuda')
    else:
        label_list = torch.tensor(label_list)
        lengths = torch.tensor(lengths)

    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    padded_text_list.to('cuda')
    print("- padded_text_list, label_list, lengths: ", padded_text_list.shape, label_list.shape, lengths.shape)
    return padded_text_list, label_list, lengths

# Take a small batch

from torch.utils.data import DataLoader

'''
collate_batch: produces 
1. padded_text_list which goes through text_pipeline/tokenizer and encodes each
word. 
2. label_list (size=4)
3. lengths (each member is a length of padded_text_list.
'''
batch_size=32
train_dl=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
valid_dl=DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
test_dl=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

#       Embedding layer input: 69025, output(embedding size): 20

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn=nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    '''
    [32,683]->embeddeding->[32,683,20], [batch_size, encoding]->[batch_size, encoding, embedding]
    [hidden, cell]->[1,32,64], [1,32,64] = [1, batch_size,lstm output]
    [1,32,64]->out=hidden[-1,:,:]->[32,64]
    [32,64]->fc1->[32,64]
    [32,64]->relu->[32,64]
    [32,64]->fc2->[32,64]
    [32,1]->sigmoid->[32,1]

    '''

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        out, (hidden,cell) = self.rnn(out)
        out = hidden[-1,:,:]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

vocab_size = len(vocab)
embed_dim = 20
rnn_hidden_size=64
fc_hidden_size=64
torch.manual_seed(1)

print("initializing model...")

model=RNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)
if CONFIG_USE_ROCM:
    model.to('cuda')

@print_fcn_name
def train(dataloader):
    model.train()
    total_acc, total_loss=0,0
    for text_batch, label_batch, lengths in dataloader:
        optimizer.zero_grad()
        pred=model(text_batch, lengths)[:,0]
        loss=loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
        total_loss += loss.item() * label_batch.size(0)

    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

@print_fcn_name
def evaluate(dataloader):
    print("evaludate entered...")
    print(dataloader, len(dataloader), type(dataloader))
    model.eval()
    total_acc, total_loss = 0,0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            pred=model(text_batch, lengths)[:,0]
            loss=loss_fn(pred, label_batch)
            total_acc+=((pred>=0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item()*label_batch.size(0)

    #code.interact(local=locals())
    print("total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset): ", total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset))
    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

print("setting loss function + optimizer...")
loss_fn = nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs=5
torch.manual_seed(1)

print("start training...")
for epoch in range(num_epochs):
    print("EPOCH: ", epoch)
    acc_train, loss_train = train(train_dl)
    acc_valid, loss_valid = evaluate(valid_dl)
    print(f'Epoch {epoch} accuracy: {acc_train:.4f}'
            f' val_accuracy: {acc_valid:.4f}')

print("Evaluate...")
acc_test, _ = evaluate(test_dl)
print(f'test accuracy: {acc_test:.4f}')
