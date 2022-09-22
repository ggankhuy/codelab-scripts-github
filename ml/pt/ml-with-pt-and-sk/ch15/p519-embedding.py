import torch 
import torch.nn as nn
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3, padding_idx=0)
text_encoded_input = torch.tensor([[1,2,4,5], [4,3,2,0]])
print(embedding(text_encoded_input))
