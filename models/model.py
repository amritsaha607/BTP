import torch
import torch.nn as nn
from models.basic_blocks import Dense, DeepLayer


class BasicModel(nn.Module):

    def __init__(self, input_dim=98, out_dim=2):
        super(BasicModel, self).__init__()

        self.size = 1024
        self.n_layers_1, self.n_layers_2 = 3, 5

        size = self.size

        self.in_ = Dense(input_dim, self.size)
        
        self.layer1 = DeepLayer(
            size=size,
            n_layers=self.n_layers_1,
            d_factor=1,
            activation='relu',
            bn=False
        )
        self.layer2 = DeepLayer(
            size=size,
            n_layers=self.n_layers_2,
            d_factor=2,
            activation='relu',
            bn=False
        )

        self.out = nn.Linear(size//(2**self.n_layers_2), out_dim)

    def cuda(self, *args, **kwargs):
        super(BasicModel, self).cuda()
        # self.layer1 = self.layer1.cuda()
        # self.layer2 = self.layer2.cuda()

    def forward(self, x):

        y = self.in_(x)
        y = self.layer1(y)
        y = self.layer2(y)
        # y = self.layer1_1(y) + self.layer1_2(y)
        # y = self.layer2_1(y) + self.layer2_2(y)
        # y = self.layer(y)
        y = self.out(y)
        return y