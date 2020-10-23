import torch
import torch.nn as nn
from models.basic_blocks import Dense


class BasicModel(nn.Module):

    def __init__(self, input_dim=98, out_dim=2):
        super(BasicModel, self).__init__()

        self.size = 1024
        self.n_layers = 5
        size = self.size

        self.layers = [Dense(input_dim, self.size)]
        for i in range(self.n_layers):
            self.layers.append(Dense(size, size//2))
            size = size//2
        
        self.layers = nn.ModuleList(self.layers)
        self.out = nn.Linear(size, out_dim)

        # self.linear1 = nn.Linear(input_dim, self.size)
        # # self.relu1 = nn.ReLU()
        # self.sigmoid1 = nn.Sigmoid()
        # self.linear2 = nn.Linear(self.size, self.size//2)
        # # self.relu2 = nn.ReLU()
        # self.sigmoid2 = nn.Sigmoid()
        # self.linear3 = nn.Linear(self.size//2, self.size//4)
        # # self.relu3 = nn.ReLU()
        # self.sigmoid3 = nn.Sigmoid()
        # self.linear4 = nn.Linear(self.size//4, self.size//8)
        # # self.relu4 = nn.ReLU()
        # self.sigmoid4 = nn.Sigmoid()
        # self.linear5 = nn.Linear(self.size//8, self.size//16)
        # # self.relu5 = nn.ReLU()
        # self.sigmoid5 = nn.Sigmoid()
        # # self.linear6 = nn.Linear(self.size//16, self.size//32)
        # # self.relu6 = nn.ReLU()
        # self.linear7 = nn.Linear(self.size//16, out_dim)

    def cuda(self, *args, **kwargs):
        super(BasicModel, self).cuda()
        for i in range(len(self.layers)):
            self.layers[i] = self.layers[i].cuda()

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        y = self.out(y)

        # y = self.relu1(self.linear1(x))
        # y = self.relu2(self.linear2(y))
        # y = self.relu3(self.linear3(y))
        # y = self.relu4(self.linear4(y))
        # y = self.relu5(self.linear5(y))
        # y = self.relu6(self.linear6(y))

        # y = self.sigmoid1(self.linear1(x))
        # y = self.sigmoid2(self.linear2(y))
        # y = self.sigmoid3(self.linear3(y))
        # y = self.sigmoid4(self.linear4(y))
        # y = self.sigmoid5(self.linear5(y))
        # y = self.linear7(y)
        return y