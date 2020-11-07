import torch
import torch.nn as nn
from models.basic_blocks import Dense, DeepLayer


class BasicModel(nn.Module):

    def __init__(self, input_dim=98, out_dim=2, model_id=7):
        super(BasicModel, self).__init__()
        self.model_id = model_id

        if model_id==7:
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

        elif model_id==8:
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
            
            self.layer2_1 = DeepLayer(
                size=size,
                n_layers=self.n_layers_2,
                d_factor=self.d_factor_2,
                activation='relu',
                bn=False
            )
            self.layer2_2 = DeepLayer(
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

        elif model_id==9:
            self.size = 1024
            self.n_layers = [3, 5, 5, 5, 5, 3]
            self.d_factors = [2, 1, 1, 1, 1, 1/2]

            size = self.size

            self.in_ = Dense(input_dim, self.size)

            self.layer1 = DeepLayer(
                size=size,
                n_layers=self.n_layers[0],
                d_factor=self.d_factors[0],
                activation='relu',
                bn=False
            )
            size = int(size // (self.d_factors[0]**self.n_layers[0]))

            self.layer2 = DeepLayer(
                size=size,
                n_layers=self.n_layers[1],
                d_factor=self.d_factors[1],
                activation='relu',
                bn=False
            )
            size = int(size // (self.d_factors[1]**self.n_layers[1]))

            self.layer3 = DeepLayer(
                size=size,
                n_layers=self.n_layers[2],
                d_factor=self.d_factors[2],
                activation='relu',
                bn=False
            )
            size = int(size // (self.d_factors[2]**self.n_layers[2]))

            self.layer4 = DeepLayer(
                size=size,
                n_layers=self.n_layers[3],
                d_factor=self.d_factors[3],
                activation='relu',
                bn=False
            )
            size = int(size // (self.d_factors[3]**self.n_layers[3]))

            self.layer5 = DeepLayer(
                size=size,
                n_layers=self.n_layers[4],
                d_factor=self.d_factors[4],
                activation='relu',
                bn=False
            )
            size = int(size // (self.d_factors[4]**self.n_layers[4]))

            self.layer6 = DeepLayer(
                size=size,
                n_layers=self.n_layers[5],
                d_factor=self.d_factors[5],
                activation='relu',
                bn=False
            )
            size = int(size // (self.d_factors[5]**self.n_layers[5]))

            self.out = nn.Linear(size, out_dim)

    def cuda(self, *args, **kwargs):
        super(BasicModel, self).cuda()

    def forward(self, x):

        if self.model_id==7:
            y = self.in_(x)
            y = self.layer1(y)
            y = self.layer2(y)
            y = self.layer3(y)
            y = self.out(y)
        
        elif self.model_id==8:
            y = self.in_(x)
            y = self.layer1(y)
            y = self.layer2_1(y) + self.layer2_2(y)
            y = self.layer3(y)
            y = self.out(y)

        elif self.model_id==9:
            y = self.in_(x)
            y = self.layer1(y)
            y = self.layer2(y) + y
            y = self.layer3(y) + y
            y = self.layer4(y) + y
            y = self.layer5(y) + y
            y = self.layer6(y)
            y = self.out(y)

        return y