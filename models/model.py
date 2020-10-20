import torch
import torch.nn as nn


class BasicModel(nn.Module):

    def __init__(self, input_dim=98):
        super(BasicModel, self).__init__()

        self.lin1 = nn.Linear(input_dim, 100)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(100, 8)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = self.lin1(x)
        y = self.relu1(y)
        y = self.lin2(y)
        y = self.relu2(y)
        return y