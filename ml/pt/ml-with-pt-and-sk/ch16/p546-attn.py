import torch
sentence=torch.tensor([0, 7, 1, 2, 5, 6, 4, 3])
torch.manual_seed(1)
embed=torch.nn.Embedding(10, 16)
embedded_sentence=embed(sentence).detach()
print(embedded_sentence.shape)

omega = torch.empty(8,8)
for i, x_i in enumerate(embedded_sentence):
    for j, x_j in enumerate(embedded_sentence):
        omega[i, j] = torch.dot(x_i, x_j)

omega_mat = embedded_sentence.matmul(embedded_sentence.T)
torch.allclose(omega_mat, omega)

import torch.nn.functional as F

attention_weights = F.softmax(omega, dim=1)
print(attention_weights.shape)
print(attention_weights.sum(dim=1))
x_2 = embedded_sentence[1, :]
context_vec_2 = torch.zeros(x_2.shape)

for j in range(8):
    x_j = embedded_sentence[j, :]
    context_vec_2 +=attention_weights[1, j] * x_j

print(context_vec_2)

context_vectors = torch.matmul(attention_weights, embedded_sentence)
torch.allclose(context_vec_2, context_vectors[1])



