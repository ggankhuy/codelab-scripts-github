
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

# No. of heads.

h=8

# should generate [8,8,16]=[heads,embedded_sentence]

multihead_U_query = torch.rand(h,d,d)
multihead_U_key = torch.rand(h,d,d)
multihead_U_value = torch.rand(h,d,d)

# generate embedded_sentence 8 times.
# [8,16,8]

#stacked_inputs=embedded_sentence.T.repeat(8,1,1) # original code.
stacked_inputs=embedded_sentence.T.repeat(h,1,1)

print(f'stacked_inputs: {stacked_inputs.shape}')

# For keys and values [8,16,16] bmmc [8,16,8] => [8,16,8], after permute [16,8,8]

multihead_keys=torch.bmm(multihead_U_key, stacked_inputs)
print(f'multihead_keys: {multihead_keys.shape} (before permute)')
multihead_keys=multihead_keys.permute(0,2,1)
print(f'multihead_keys: {multihead_keys.shape} (before permute)')

multihead_values=torch.bmm(multihead_U_key, stacked_inputs)
print(f'multihead_values: {multihead_values.shape} (before permute)')
multihead_values=multihead_values.permute(0,2,1)
print(f'multihead_values: {multihead_values.shape} (before permute)')

# Create linear layer
# input: 8 * 16
# output: 16

multihead_z_2=torch.rand(8,16) 
linear=torch.nn.Linear(8*16, 16)
context_vector_2=linear(multihead_z_2.flatten())
