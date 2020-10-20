import torch
import torch.nn as nn


class BasicModel(nn.Module):

    def __init__(self, input_dim=98):
        super(BasicModel, self).__init__()

        self.linear1 = nn.Linear(input_dim, 100)
        self.relu1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(100, 64)
        self.relu2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(32, 8)
        # self.relu3 = nn.ReLU()

    def forward(self, x):
        y = self.linear1(x)
        y = self.relu1(y)
        y = self.linear2(y)
        y = self.relu2(y)
        y = self.linear3(y)
        y = self.relu3(y)
        y = self.linear4(y)
        return y