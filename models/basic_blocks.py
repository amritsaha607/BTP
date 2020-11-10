import torch
import torch.nn as nn

from utils.operations import makeList


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


class CascadedDeep(nn.Module):
    """
        Implements cascaded deep layer
        Deep -> Deep -> Deep -> ...... -> Deep
    """
    def __init__(self, size, n_layers, d_factors,
                    activations='relu', bns=False):
        super(CascadedDeep, self).__init__()

        if not isinstance(activations, list):
            activations = [activations] * len(n_layers)
        if not isinstance(bns, list):
            bns = [bns] * len(n_layers)

        self.size, self.ret_size = size, size
        self.n_layers = n_layers
        self.d_factors = d_factors
        self.activations = activations
        self.bns = bns

        self.layers = []

        for layer_idx, (n_layer, d_factor, activation, bn) in \
            enumerate(zip(n_layers, d_factors, activations, bns)):

            layer = DeepLayer(
                size=size,
                n_layers=n_layer,
                d_factor=d_factor,
                activation=activation,
                bn=bn
            )
            self.layers.append(layer)
            self.add_module(
                "deeplayer_{}".format(layer_idx+1),
                layer
            )

            size = int(size // (d_factor ** n_layer))

        self.ret_size = size

    def cuda(self):
        super(CascadedDeep, self).cuda()
        self.layers = self.layers.cuda()

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

    def outSize(self):
        return self.ret_size


class InceptionLayer(nn.Module):

    """
        Implements Inception Layer
        Args:
            n_branches : No of parallel branches
            size : Input size
            n_layers : (list / int) contains no of layers for each branch
            d_factors : (list / int) contains d_factor for each branch
            activations : (list / int) contains activations for each branch
            bns : (list / int) contains batch_norm for each branch
            out : 
                'add' / 'sum' => add all the final outputs
                'cat' => concatenate all the final outputs
    """
    def __init__(self, n_branches, size, n_layers, d_factors, 
                activations='relu', bns=False,
                out='add'):

        super(InceptionLayer, self).__init__()

        self.n_branches = n_branches
        self.size = size
        self.n_layers = makeList(n_layers, n_branches)
        self.d_factors = makeList(d_factors, n_branches)
        self.activations = makeList(activations, n_branches)
        self.bns = makeList(bns, n_branches)
        self.out = out

        self.branches = []

        for branch_idx, (d_factor, activation, bn) in enumerate(zip(self.d_factors, self.activations, self.bns)):
            branch_layers, size = [], self.size
            for i in range(self.n_layers[branch_idx]):
                branch_layers.append(Dense(size, int(size//d_factor), activation=activation, batch_norm=bn))
                size = int(size//d_factor)
            self.branches.append(nn.ModuleList(branch_layers))

        for branch_idx, branch in enumerate(self.branches):
            self.add_module(str(branch_idx), branch)

    def cuda():
        super(DeepLayer, self).cuda()
        for i in range(self.branches):
            self.branches[i] = self.branches[i].cuda()

    def forward(self, x):
        y = []
        for branch in self.branches:
            y_ = branch(x)
            y.append(y_)

        if self.out=='add' or self.out=='sum':
            y = sum(y)
        else:
            y = torch.cat(y, dim=0)
        
        return y