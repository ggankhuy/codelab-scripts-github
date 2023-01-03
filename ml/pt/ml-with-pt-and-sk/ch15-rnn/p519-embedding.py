import torch 
import torch.nn as nn
CONFIG_ERR_OUT_OF_INDEX=0

# embedding layer example.
# input: [2,4] 
# input matrix: 4
# output matrix: 4
# i.e. [1,4]->[4,3], in this case: 4 signifies features, 3 signifies embedding dimension relevant to each feature.

# content:
# embeddings: each value in matrix is up to 10
# output dimension of each matrix: 3 combination vector.

embedding = nn.Embedding(num_embeddings=10, embedding_dim=3, padding_idx=0)

if CONFIG_ERR_OUT_OF_INDEX:
    try:
        text_encoded_input = torch.tensor([[1,2,4,5], [4,3,2,12]])
    except Exception as msg:
        print(msg)
else:
    text_encoded_input = torch.tensor([[1,2,4,5], [4,3,2,9]])
    text_encoded_input = torch.tensor([[1,2,4,4]])

output=embedding(text_encoded_input)
print(embedding(text_encoded_input))
