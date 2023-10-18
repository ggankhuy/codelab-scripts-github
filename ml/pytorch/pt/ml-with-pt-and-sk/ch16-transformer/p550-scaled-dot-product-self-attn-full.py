# in this section, we implementent the self-attn mechanism scaled dot-product attention 
# it is similar to p550-scaled-product-self-attn.py but instead of taking 2nd vector
# it computes whole embedded sentence. (not sure how appropriate this is though).
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

# Create queries size: [16,16]

torch.manual_seed(123)
d = embedded_sentence.shape[1]

# No. of heads.

h=8

multihead_U_query = torch.rand(h,d,d)
multihead_U_key = torch.rand(h,d,d)
multihead_U_value = torch.rand(h,d,d)

# generates keys, values and queries: [8, 16]

'''
multihead_keys = multihead_U_key.matmul(embedded_sentence.T).T
multihead_values = multihead_U_value.matmul(embedded_sentence.T).T
multihead_queries = multihead_U_query.matmul(embedded_sentence.T).T
'''

stacked_inputs=embedded_sentence.T.repeat(8,1,1)
multihead_keys=torch.bmm(multihead_U_key, stacked_inputs)
print(f'multihead_keys: {multihead_keys.shape}:')

multihead_keys=multihead_keys.permute(0,2,1)

multihead_values=torch.matmul(multhead_U_value, stacked_inputs)
multihead_values=multihead_values.permute(0,2,1)

multihead_z_2=torch.rand(8,16)

linear=torch.nn.Linear(8*16, 16)
context_vector_2=linear(multihead_z_2.flatten())
print(f'context vector: {context_vector_2.shape}')

