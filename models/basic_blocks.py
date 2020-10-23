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