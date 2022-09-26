# data dimensions in this example
# embed, nn.Embedding           [10, 16], unique data, features
# embedded_sentence, input      [8, 16]
# omega, importance_weight      [8,8] 
# attention_weights:            [8,8] - norm(omega)
# x_2, second word from input   [16]
# context_vec_2 (for 2nd inp):  [16] - embedded_sentence *(dot) attention_weight[1]
# contect_vec_3 (for 3rd inp):  [16] - embedded_sentence *(dot) attention_weight[2]
# ...
# contect_vec_8 (for 8th inp):  [16] - embedded_sentence *(dot) attention_weight[7]
# ---------------
# self.attn #2 scaled dot product
# U_query, U_key, U_value:      [16, 16]
# query_2=query_2 * (dot) x_2   [16] = [16, 16] *(dot) [16] -> [16]
# key_2=key_2 * (dot) x_2       [16] = [16, 16] *(dot) [16] -> [16]
# value_2=key_2 * (dot) x_2     [16] = [16, 16] *(dot) [16] -> [16]

#keys = U_key.matmul(embedded_sentence.T).T
#values = U_value.matmul(embedded_sentence.T).T
# keys:                          [8, 16] = x[8,16].T->[16,8] *(dot) [16,16] = [8,16]
# values                         [8, 16] = x[8,16].T->[16,8] *(dot) [16,16] = [8,16]

import torch
import torch.nn.functional as F

CONFIG_DEBUG=0
CONFIG_COMPARE=0
sentence=torch.tensor([0, 7, 1, 2, 5, 6, 4, 3])
torch.manual_seed(1)
embed=torch.nn.Embedding(10, 16)

# generates [8,16], 8 inputs, 16 features.

embedded_sentence=embed(sentence).detach()
print(embedded_sentence.shape)
print(embedded_sentence)

# self attention mechanism.
# stage1.  Omage: importance weight computation.  

omega = torch.empty(8,8)
for i, x_i in enumerate(embedded_sentence):
    if CONFIG_DEBUG:
        print("i, x_i: ", i, x_i)
    for j, x_j in enumerate(embedded_sentence):
        if CONFIG_DEBUG:
            print("- j, x_j: ", j, x_j)
        omega[i, j] = torch.dot(x_i, x_j)

if CONFIG_COMPARE:
    omega_mat = embedded_sentence.matmul(embedded_sentence.T)
    torch.allclose(omega_mat, omega)

print("omega:", omega.shape, type(omega))
print(omega)

# stage2. Normalize weights, using softmax.

attention_weights = F.softmax(omega, dim=1)

print("attention_weights (shape/type/sum): ", attention_weights.shape, type(attention_weights))
print(attention_weights.sum(dim=1))
print(attention_weights)
attention_weights_int = attention_weights.to(torch.float16)

print("attention_weights_int (shape/type/sum): ", attention_weights_int.shape, type(attention_weights_int))

# stage3. z, weighted sum???

print(attention_weights_int.sum(dim=1))
print(attention_weights_int)

# Computing context vector (manually) for second word manually.
# context_vector_2[16] = embedded_sentence(x) *(dot) attention_weight[1] = x[0]*attnw[1] + x[1]*attnw[1] + ... + x[N]*attn[1]
# context_vector_3[16] = embedded_sentence(x) *(dot) attention_weight[2] = x[0]*attnw[2] + x[1]*attnw[2] + ... + x[N]*attn[2]

x_2 = embedded_sentence[1, :]
print("x_2: ", x_2.shape, type(x_2))
print(x_2)

context_vec_2 = torch.zeros(x_2.shape)
print("context_vec_2 (init): ", context_vec_2.shape, type(context_vec_2))
print(context_vec_2)

for j in range(8):
    x_j = embedded_sentence[j, :]
    context_vec_2 +=attention_weights[1, j] * x_j

print("context_vec_2:")
print("context_vec_2: ", context_vec_2.shape, type(context_vec_2))

if CONFIG_COMPARE:
    context_vectors = torch.matmul(attention_weights, embedded_sentence)
    torch.allclose(context_vec_2, context_vectors[1])

torch.manual_seed(123)
d = embedded_sentence.shape[1]
U_query = torch.rand(d,d)
U_key = torch.rand(d,d)
U_value = torch.rand(d,d)

# Second input element computation (from embedded_sentencep[1])...

print("U_query/key/value: ", U_query.shape, U_key.shape, U_value.shape)

x_2 = embedded_sentence[1]

query_2 = U_query.matmul(x_2)
key_2 = U_key.matmul(x_2)
value_2 = U_value.matmul(x_2)

keys = U_key.matmul(embedded_sentence.T).T
values = U_value.matmul(embedded_sentence.T).T

if CONFIG_COMPARE:
    keys = U_key.matmul(embedded_sentence.T).T
    torch.allclose(key_2, keys[1])
    values = U_value.matmul(embedded_sentence.T).T
    torch.allclose(value_2, values[1])

omega_23 = query_2.dot(keys[2])
print(omega_23)

omega_2 = query_2.matmul(keys.T)
print(omega_2)

attention_weights_2 = F.softmax(omega_2 / d**0.5, dim=0)
print(attention_weights_2)

context_vector_2 = attention_weights_2.matmul(values)
print(context_vector_2)
