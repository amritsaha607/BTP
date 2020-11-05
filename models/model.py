import torch
import torch.nn as nn
from models.basic_blocks import Dense, DeepLayer


class BasicModel(nn.Module):

    def __init__(self, input_dim=98, out_dim=2):
        super(BasicModel, self).__init__()

        self.size = 1024
        self.n_layers_1, self.n_layers_2, self.n_layers_3 = 3, 10, 3
        self.d_factor_1, self.d_factor_2, self.d_factor_3 = 2, 1, 1/2

        size = self.size

        self.in_ = Dense(input_dim, self.size)
        
        self.layer1 = DeepLayer(
            size=size,
            n_layers=self.n_layers_1,
            d_factor=self.d_factor_1,
            activation='relu',
            bn=False
        )
        size = int(size//(self.d_factor_1**self.n_layers_1))
        
        self.layer2 = DeepLayer(
            size=size,
            n_layers=self.n_layers_2,
            d_factor=self.d_factor_2,
            activation='relu',
            bn=False
        )
        size = int(size//(self.d_factor_2**self.n_layers_2))

        self.layer3 = DeepLayer(
            size=size,
            n_layers=self.n_layers_3,
            d_factor=self.d_factor_3,
            activation='relu',
            bn=False
        )
        size = int(size//(self.d_factor_3**self.n_layers_3))

        self.out = nn.Linear(size, out_dim)

    def cuda(self, *args, **kwargs):
        super(BasicModel, self).cuda()

    def forward(self, x):
        y = self.in_(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.out(y)
        return y