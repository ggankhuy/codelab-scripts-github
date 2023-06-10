import torch
import torch.nn.functional as F

DEBUG=1
CONFIG_COMPARE=1
sentence=torch.tensor([0, 7, 1, 2, 5, 6, 4, 3])
torch.manual_seed(1)

# number of embedding, vocab size: 10.
# number of embedding dimension (vector)=16

embed=torch.nn.Embedding(10, 16)

# generates [8,16], 8 inputs, 16 features.

embedded_sentence=embed(sentence).detach()
print(embedded_sentence.shape)
print(embedded_sentence)

# Create queries size: [1,16]

torch.manual_seed(123)

d = embedded_sentence.shape[1]

U_query = torch.rand(d,d)
U_key = torch.rand(d,d)
U_value = torch.rand(d,d)

x_2 = embedded_sentence[1]
query_2 = U_query.matmul(x_2)
key_2 = U_key.matmul(x_2)
value_2 = U_value.matmul(x_2)

keys = U_key.matmul(embedded_sentence.T).T
values = U_value.matmul(embedded_sentence.T).T

# comparison with pytorch matmul 
'''
keys = U_key.matmul(embedded_sentence.T).T
torch.allclose(key_2, keys[1])
values = U_value.matmul(embedded_sentence.T).T
torch.allclose(value_2, values[1])
'''
    
omega_2 = query_2.matmul(keys.T)
print(omega_2)

attention_weights_2 = F.softmax(omega_2 / d**0.5, dim=0)
print(attention_weights_2)

