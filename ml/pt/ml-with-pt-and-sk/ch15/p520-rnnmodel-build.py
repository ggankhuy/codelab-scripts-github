import torch
import code

from torchtext.datasets import IMDB
train_dataset = IMDB(split='train')
test_dataset =IMDB(split='test')

# 1. create dataset
from torch.utils.data.dataset import random_split

torch.manual_seed(1)
train_dataset, valid_dataset = random_split(list(train_dataset), [20000, 5000])

# 2. find unique tokens

import re
from collections import Counter, OrderedDict

def tokenizer(text):
    text = re.sub('<[^?]', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(\D\P)', text.lower())
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
sorted_by_freq_tuples = sorted(token_coutns.items(), key=lambda x: x[1], reverse=True)
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

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))

    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)

    return padded_text_list, label_list, lengths

# Take a small batch

from torch.util.data import DataLoader
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)

text_batch, label_batch, length_batch = next(iter(dataloader))
print(text_batch)
print(label_batch)
print(length_patch)

batdh_size=32
train_dl=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
valid_dl=DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dl=DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
   
embedding=nn.Embedding(num_embedding=10, embedding_dim=3, padding_idx=0)

# a batch of 2 samples of 4 indices each

text_encoded_input = torch.LongTensor([[1,2,4,5],[4,3,2,0]])
print(embedding(text_encoded_input))

'''
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True)
        # self.rnn = nn.GPU(input_size, hidden_size, num_layers=2, batchfirst=True)
        # self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, batchfirst=True)
        self.fc=nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden  = self.rnn(x)

        # we use the final hidden state from the last hidden layer as the input to the fully 
        # connected layer 

        out = hidden[-1, :, :] 
        out=self.fc(out)
        return out

model=RNN(64,32)
print(model)
model(torch.randn(5,3,64))

'''        
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn=nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        out, (hidden,cell) = self.rnn(out)
        out=self.fc1(out)
        out=self.relu(out)
        out=self.fc2(out)
        out=self.sigmoid(out)
        return out

vocab_size = len(vocab)
embed_dim = 20
rnn_hidden_size=64
fc_hidden_size=64
torch.manual_seed(1)
model=RNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)


def train(dataloader):
    model.train()
    total_acc, total_loss=0,0
    for text_batch, label_batch, lengths in dataloder:
        optimizer.zero_grad()
        pred=model(text_batch, lengths)[:,0]
        loss=loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
        total_loss += loss.item() * label_batch.size(0)

    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

def evaluate(dataloader):
    model.eval()
    total_acc, total_loss = 0,0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            pred=model(text_batch, lengths)[:,0]
            loss=loss_fn(pred, label_batch)
            total_acc+=((pred>=0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item()*label_batch.size(0)
    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

loss_fn = nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs=10
torch.manual_seed(1)
for epoch in range(num_epochs):
        acc_train, loss_train = train(train_dl)
        acc_valid, loss_valid = evaluate(valid_dl)
        print(f'Epoch {epoch} accuracy: {acc_train:.4f}'
            f' val_accuracy: {acc_valid:.4f}')
acc_test, _ = evaluate(test_dl)
print(f'test accuracy: {acc_test:.4f}')

        
