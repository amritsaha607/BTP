import torch
import torch.nn as nn
from models.basic_blocks import Dense, DeepLayer


class BasicModel(nn.Module):

    def __init__(self, input_dim=98, out_dim=2):
        super(BasicModel, self).__init__()

        self.size = 512
        self.n_layers = 5

        # self.layers = [Dense(input_dim, self.size)]
        # for i in range(self.n_layers):
        #     self.layers.append(Dense(size, size//2))
        #     size = size//2
        
        # self.layers = nn.ModuleList(self.layers)

        self.in_ = Dense(input_dim, self.size)
        self.layer = DeepLayer(
            size=self.size, 
            n_layers=self.n_layers, 
            d_factor=2,
            activation='relu',
            bn=False
        )
        self.out = nn.Linear(self.size//2**self.n_layers, out_dim)

    def cuda(self, *args, **kwargs):
        super(BasicModel, self).cuda()
        self.layer = self.layer.cuda()

        # for i in range(len(self.layers)):
        #     self.layers[i] = self.layers[i].cuda()

    def forward(self, x):
        # y = x
        # for layer in self.layers:
        #     y = layer(y)

        y = self.in_(x)
        y = self.layer(y)
        y = self.out(y)
        return y