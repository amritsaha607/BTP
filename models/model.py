import torch
import torch.nn as nn


class BasicModel(nn.Module):

    def __init__(self, input_dim=98):
        super(BasicModel, self).__init__()

        self.size = 256
        self.n_layers = 5

        self.linear1 = nn.Linear(input_dim, self.size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.size, self.size//2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(self.size//2, self.size//4)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(self.size//4, self.size//8)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(self.size//8, self.size//16)
        self.relu5 = nn.ReLU()
        self.linear6 = nn.Linear(self.size//16, self.size//32)
        self.relu6 = nn.ReLU()
        self.linear7 = nn.Linear(self.size//32, 2)

    def forward(self, x):
        y = self.relu1(self.linear1(x))
        y = self.relu2(self.linear2(y))
        y = self.relu3(self.linear3(y))
        y = self.relu4(self.linear4(y))
        y = self.relu5(self.linear5(y))
        y = self.relu6(self.linear6(y))
        y = self.linear7(y)
        return y