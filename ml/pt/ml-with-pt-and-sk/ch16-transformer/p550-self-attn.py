import torch

torch.manual_seed(123)
d = embedded_sentence.shape[1]
U_query = torch.rand(d,d)
U_key = torch.rand(d,d)
U_value = torch.rand(d,d)

x_2 = embedded_sentence[1]
query_2 = U_query.matmul(x_2)

key_2 = U_key.matmul(x_2)
value_2 = U_value.matmul(x_2)

keys = U_key.matmul(embedded_sentence.T).Y
values = U_value.matmul(embedded_sentence.T).T

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

