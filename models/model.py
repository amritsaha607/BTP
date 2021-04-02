from collections import defaultdict
import torch
import torch.nn as nn
from models.basic_blocks import Dense, DeepLayer, InceptionLayer, CascadedDeep


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

        elif model_id==10:
            self.size = 1024
            self.n_layers_left = [3, 10, 3]
            self.d_factors_left = [2, 1, 1/2]
            self.n_layers_right = [3, 5, 5, 5, 5, 3]
            self.d_factors_right = [2, 1, 1, 1, 1, 1/2]

            size = self.size

            self.in_ = Dense(input_dim, self.size)

            self.layer1 = DeepLayer(
                size=size,
                n_layers=self.n_layers_left[0],
                d_factor=self.d_factors_left[0],
                activation='relu',
                bn=False
            )
            size = int(size//(self.d_factors_left[0]**self.n_layers_left[0]))
            left_size, right_size = size, size


            # Left layers
            self.left_layer2_1 = DeepLayer(
                size=left_size,
                n_layers=self.n_layers_left[1],
                d_factor=self.d_factors_left[1],
                activation='relu',
                bn=False
            )
            self.left_layer2_2 = DeepLayer(
                size=left_size,
                n_layers=self.n_layers_left[1],
                d_factor=self.d_factors_left[1],
                activation='relu',
                bn=False
            )
            left_size = int(left_size//(self.d_factors_left[1]**self.n_layers_left[1]))


            # Right layers
            self.right_layer2 = DeepLayer(
                size=right_size,
                n_layers=self.n_layers_right[1],
                d_factor=self.d_factors_right[1],
                activation='relu',
                bn=False
            )
            right_size = int(right_size // (self.d_factors_right[1]**self.n_layers_right[1]))

            self.right_layer3 = DeepLayer(
                size=right_size,
                n_layers=self.n_layers_right[2],
                d_factor=self.d_factors_right[2],
                activation='relu',
                bn=False
            )
            right_size = int(right_size // (self.d_factors_right[2]**self.n_layers_right[2]))

            self.right_layer4 = DeepLayer(
                size=right_size,
                n_layers=self.n_layers_right[3],
                d_factor=self.d_factors_right[3],
                activation='relu',
                bn=False
            )
            right_size = int(right_size // (self.d_factors_right[3]**self.n_layers_right[3]))

            self.right_layer5 = DeepLayer(
                size=right_size,
                n_layers=self.n_layers_right[4],
                d_factor=self.d_factors_right[4],
                activation='relu',
                bn=False
            )
            right_size = int(right_size // (self.d_factors_right[4]**self.n_layers_right[4]))

            if left_size!=right_size:
                raise ValueError("Left & Right sizes don't match -> {} & {}".format(left_size, right_size))
            
            size = left_size

            self.layer3 = DeepLayer(
                size=size,
                n_layers=self.n_layers_left[-1],
                d_factor=self.d_factors_left[-1],
                activation='relu',
                bn=False
            )
            size = int(size//(self.d_factors_left[-1]**self.n_layers_left[-1]))

            self.out = nn.Linear(size, out_dim)

        elif model_id==11:
            self.size = 1024
            self.n_layers = [4, 20, 4]
            self.d_factors = [2, 1, 1/2]

            size = self.size

            self.in_ = Dense(input_dim, self.size)

            self.layers = []
            for layer_idx, (n_layer, d_factor) in enumerate(zip(self.n_layers, self.d_factors)):
                self.layers.append(
                    DeepLayer(
                        size=size,
                        n_layers=n_layer,
                        d_factor=d_factor,
                        activation='relu',
                        bn=False
                    )
                )
                size = int(size // (d_factor**n_layer))
                self.add_module(str(layer_idx), self.layers[layer_idx])

            self.out = nn.Linear(size, out_dim)

        elif model_id==12:
            self.size = 1024
            self.n_layers = [3, 40, 3]
            self.d_factors = [2, 1, 1/2]

            size = self.size

            self.in_ = Dense(input_dim, self.size)

            self.cascade = CascadedDeep(
                size=self.size,
                n_layers=self.n_layers,
                d_factors=self.d_factors,
                activations='relu',
                bns=False,
            )
            size = self.cascade.outSize()

            self.out = nn.Linear(size, out_dim)

        elif model_id==13:
            self.size = 1024
            self.n_layers = [2, 5, 5, 5, 5, 2]
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

        elif model_id==14:
            self.size = 1024
            self.n_layers_l, self.n_layers_r = [3, 5, 5, 5, 5, 3], [2, 5, 5, 5, 5, 2]
            self.d_factors = [2, 1, 1, 1, 1, 1/2]

            lsize, rsize = self.size, self.size
            self.layers_l, self.layers_r = [], []

            self.in_ = Dense(input_dim, self.size)

            for layer_idx, (n_layer, d_factor) in enumerate(zip(self.n_layers_l, self.d_factors)):
                self.layers_l.append(
                    DeepLayer(
                        size=lsize,
                        n_layers=n_layer,
                        d_factor=d_factor,
                        activation='relu',
                        bn=False
                    )
                )
                lsize = int(lsize // (d_factor**n_layer))
                self.add_module('left_layer_{}'.format(layer_idx), self.layers_l[-1])
            
            for layer_idx, (n_layer, d_factor) in enumerate(zip(self.n_layers_r, self.d_factors)):
                self.layers_r.append(
                    DeepLayer(
                        size=rsize,
                        n_layers=n_layer,
                        d_factor=d_factor,
                        activation='relu',
                        bn=False
                    )
                )
                rsize = int(rsize // (d_factor**n_layer))
                self.add_module('right_layer_{}'.format(layer_idx), self.layers_r[-1])
            
            self.out = nn.Linear(lsize, out_dim)

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

        elif self.model_id==10:
            y = self.in_(x)
            y = self.layer1(y)

            # Left part
            y_l = self.left_layer2_1(y) + self.left_layer2_2(y)

            # Right Part
            y_r = self.right_layer2(y) + y
            y_r = self.right_layer3(y_r) + y_r
            y_r = self.right_layer4(y_r) + y_r
            y_r = self.right_layer5(y_r) + y_r

            # Final Layer
            y = y_l + y_r
            y = self.layer3(y)
            y = self.out(y)

        elif self.model_id==11:
            y = self.in_(x)
            for layer in self.layers:
                y = layer(y)
            y = self.out(y)

        elif self.model_id==12:
            y = self.in_(x)
            y = self.cascade(y)
            y = self.out(y) 

        elif self.model_id==13:
            y = self.in_(x)
            y = self.layer1(y)
            y = self.layer2(y) + y
            y = self.layer3(y) + y
            y = self.layer4(y) + y
            y = self.layer5(y) + y
            y = self.layer6(y)
            y = self.out(y)

        elif self.model_id==14:
            y = self.in_(x)

            yl, yr = y, y
            # Left go
            for layer_idx, layer in enumerate(self.layers_l):
                if layer_idx==0 or layer_idx==len(self.n_layers_l)-1:
                    yl = layer(yl)
                else:
                    yl = layer(yl) + yl

            # Right go
            for layer_idx, layer in enumerate(self.layers_r):
                if layer_idx==0 or layer_idx==len(self.n_layers_r)-1:
                    yr = layer(yr)
                else:
                    yr = layer(yr) + yr

            y = yl + yr
            y = self.out(y)

        return y


class E1Model(nn.Module):

    """
        Model for e1 data
    """

    def __init__(self, classes, model_id, input_dim=98, out_dim=2):

        """
            classes : Different e1 parameter classes (list)
            model_ids : which model to pick for which class (list of BasicModel / BasicModel)
        """

        super(E1Model, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.model_id = model_id
        self.input_dim = input_dim
        self.out_dim = out_dim

        if self.model_id == 1:
            model_ids = 7
            self.bakeModel(model_ids=model_ids)

        elif self.model_id == 2:
            model_ids = 8
            self.bakeModel(model_ids=model_ids)
        
        elif self.model_id == 3:
            model_ids = 9
            self.bakeModel(model_ids=model_ids)

        elif self.model_id == 4:
            model_ids = 10
            self.bakeModel(model_ids=model_ids)

        elif self.model_id == 5:
            model_ids = 11
            self.bakeModel(model_ids=model_ids)

        elif self.model_id == 6:
            model_ids = 12
            self.bakeModel(model_ids=model_ids)

        elif self.model_id == 7:
            model_ids = 13
            self.bakeModel(model_ids=model_ids)

        elif self.model_id == 8:
            model_ids = 14
            self.bakeModel(model_ids=model_ids)

    def cuda(self, *args, **kwargs):
        super(E1Model, self).cuda()
        self.model.cuda()

    def forward(self, x, mat):
        y = self.model[mat](x)
        return y

    def bakeModel(self, model_ids):
        """
            Bake E1Model from BasicModel class
        """
        if not isinstance(model_ids, list):
            model_ids = [model_ids] * self.n_classes

        self.model = {}
        for class_, model_id in zip(self.classes, model_ids):
            self.model[class_] = BasicModel(input_dim=self.input_dim, out_dim=self.out_dim, model_id=model_id)
        self.model = nn.ModuleDict(self.model)


class E1E2Model(nn.Module):

    """
        Model for e1e2 data
    """

    def __init__(self, e1_classes, e2_classes, model_id, input_dim=98, out_dim=2):

        """
            e1_classes : Different e1 parameter classes (list)
            e2_classes : Different e2 parameter classes (list)
            model_id : which model to pick for which class (list of BasicModel / BasicModel)
        """

        super(E1E2Model, self).__init__()
        self.e1_classes = e1_classes
        self.e2_classes = e2_classes
        self.model_id = model_id
        self.input_dim = input_dim
        self.out_dim = out_dim

        if self.model_id == 1:
            model_id = 7
            self.bakeModel(model_id=model_id)

        elif self.model_id == 2:
            model_id = 8
            self.bakeModel(model_id=model_id)
        
        elif self.model_id == 3:
            model_id = 9
            self.bakeModel(model_id=model_id)

        elif self.model_id == 4:
            model_id = 10
            self.bakeModel(model_id=model_id)

        elif self.model_id == 5:
            model_id = 11
            self.bakeModel(model_id=model_id)

        elif self.model_id == 6:
            model_id = 12
            self.bakeModel(model_id=model_id)

        elif self.model_id == 7:
            model_id = 13
            self.bakeModel(model_id=model_id)

        elif self.model_id == 8:
            model_id = 14
            self.bakeModel(model_id=model_id)

    def cuda(self, *args, **kwargs):
        super(E1E2Model, self).cuda()
        self.model.cuda()

    def forward(self, x, e1_mat, e2_mat):
        y = self.model[e1_mat][e2_mat](x)
        return y

    def bakeModel(self, model_id):
        """
            Bake E1E2Model from BasicModel class
        """

        self.model = defaultdict(dict)
        for e1_cls in self.e1_classes:
            for e2_cls in self.e2_classes:
                self.model[e1_cls][e2_cls] = BasicModel(
                    input_dim=self.input_dim,
                    out_dim=self.out_dim,
                    model_id=model_id,
                )
            self.model[e1_cls] = nn.ModuleDict(self.model[e1_cls])
        self.model = nn.ModuleDict(self.model)


class E1E2E3Model(nn.Module):

    """
        Model for e1, e2 & e3 data
    """

    def __init__(self, classes, model_id, input_dim=98, out_dim=2):

        """
            classes : Different e1, e2, e3 parameter classes (list)
            model_ids : which model to pick for which class (list of BasicModel / BasicModel)
        """

        super(E1E2E3Model, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.model_id = model_id
        self.input_dim = input_dim
        self.out_dim = out_dim

        if self.model_id == 1:
            model_ids = 7
            self.bakeModel(model_ids=model_ids)

        elif self.model_id == 2:
            model_ids = 8
            self.bakeModel(model_ids=model_ids)
        
        elif self.model_id == 3:
            model_ids = 9
            self.bakeModel(model_ids=model_ids)

        elif self.model_id == 4:
            model_ids = 10
            self.bakeModel(model_ids=model_ids)

        elif self.model_id == 5:
            model_ids = 11
            self.bakeModel(model_ids=model_ids)

        elif self.model_id == 6:
            model_ids = 12
            self.bakeModel(model_ids=model_ids)

        elif self.model_id == 7:
            model_ids = 13
            self.bakeModel(model_ids=model_ids)

        elif self.model_id == 8:
            model_ids = 14
            self.bakeModel(model_ids=model_ids)

    def cuda(self, *args, **kwargs):
        super(E1E2E3Model, self).cuda()
        self.model.cuda()

    def forward(self, x, e1_mat, e2_mat, e3_mat):
        y = self.model[f'{e1_mat},{e2_mat},{e3_mat}'](x)
        return y

    def bakeModel(self, model_ids):
        """
            Bake E1E2E3Model from BasicModel class
        """
        if not isinstance(model_ids, list):
            model_ids = [model_ids] * self.n_classes

        self.model = {}
        for class_, model_id in zip(self.classes, model_ids):
            self.model[class_] = BasicModel(input_dim=self.input_dim, out_dim=self.out_dim, model_id=model_id)
        self.model = nn.ModuleDict(self.model)
