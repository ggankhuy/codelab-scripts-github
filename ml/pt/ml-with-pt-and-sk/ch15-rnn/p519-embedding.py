import torch 
import torch.nn as nn
CONFIG_ERR_OUT_OF_INDEX=0

embedding = nn.Embedding(num_embeddings=10, embedding_dim=3, padding_idx=0)

if CONFIG_ERR_OUT_OF_INDEX:
    try:
        text_encoded_input = torch.tensor([[1,2,4,5], [4,3,2,12]])
    except Exception as msg:
        print(msg)
else:
    text_encoded_input = torch.tensor([[1,2,4,5], [4,3,2,9]])

print(embedding(text_encoded_input))
