import torch
import torch.nn as nn


class Dense(nn.Module):
    """
        Implements Dense layer
        Linear -> Activation -> BatchNorm(optional)
    """
    def __init__(self, input_dim, output_dim, activation='relu', batch_norm=False):
        super(Dense, self).__init__()

        if isinstance(activation, str):
            if activation=='relu':
                activation = nn.ReLU()
            elif activation=='sigmoid':
                activation = nn.Sigmoid()
            elif activation=='leaky_relu':
                activation = nn.LeakyReLU()
            elif activation=='tanh':
                activation = nn.Tanh()
            else:
                raise AssertionError("Unknown activation {}, please provide with \
                    proper actiavtion function.".format(activation))

        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.batch_norm = nn.BatchNorm1d() if batch_norm else None
        
    def forward(self, x):
        y = self.linear(x)
        y = self.activation(y)
        if self.batch_norm:
            y = self.batch_norm(y)
        return y


class DeepLayer(nn.Module):

    """
        Implements modulelist of Dense layers
        Dense -> Dense -> Dense -> ...
    """
    def __init__(self, size=512, n_layers=5, d_factor=2, activation='relu', bn=False):
        super(DeepLayer, self).__init__()

        if not isinstance(d_factor, list):
            d_factor = [d_factor]*n_layers

        self.size = size
        self.n_layers = n_layers
        self.layers = []
        for i in range(n_layers):
            self.layers.append(Dense(size, int(size//d_factor[i]), activation=activation, batch_norm=bn))
            size = int(size//d_factor[i])
        self.layers = nn.ModuleList(self.layers)
        
    def cuda():
        super(DeepLayer, self).cuda()
        for i in range(self.n_layers):
            self.layers[i] = self.layers[i].cuda()

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y