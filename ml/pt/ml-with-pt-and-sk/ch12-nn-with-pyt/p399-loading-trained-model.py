import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x=self.layer1(x)
        x=nn.Sigmoid()(x)
        x=self.layer2(x)
        x=nn.Softmax(dim=1)(x)
        return x

path='iris_classifier.pt'
model_new=Model(input_size, hidden_size, output_size)
model_new=torch.load_state_dict(torch.load(path))

