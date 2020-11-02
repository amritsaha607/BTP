import torch
import torch.nn as nn
from models.basic_blocks import Dense, DeepLayer


class BasicModel(nn.Module):

    def __init__(self, input_dim=98, out_dim=2):
        super(BasicModel, self).__init__()

        self.size = 1024
        self.n_layers = 5

        size = self.size

        self.in_ = Dense(input_dim, self.size)
        
        self.layer1 = DeepLayer(
            size=size,
            n_layers=self.n_layers,
            d_factor=2,
            activation='relu',
            bn=False
        )

        size = size//(2**self.n_layers)
        self.layer2 = DeepLayer(
            size=size,
            n_layers=self.n_layers,
            d_factor=1/2,
            activation='relu',
            bn=False
        )
        size = int(size//((1/2)**self.n_layers))

        self.out = nn.Linear(size, out_dim)

    def cuda(self, *args, **kwargs):
        super(BasicModel, self).cuda()

    def forward(self, x):
        y1 = self.in_(x)
        y = self.layer1(y1)
        y = self.layer2(y)
        y = y + y1
        y = self.out(y)
        return y