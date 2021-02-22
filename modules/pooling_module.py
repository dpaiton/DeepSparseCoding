import torch
import torch.nn as nn
import torch.nn.functional as F

import DeepSparseCoding.utils.data_processing as dp


class PoolingModule(nn.Module):
    def setup_module(self, params):
        params.weight_decay = 0 # used by base model; pooling layer never has weight decay
        self.params = params
        if self.params.layer_type == 'fc':
            self.layer = nn.Linear(
                in_features=self.params.layer_channels[0],
                out_features=self.params.layer_channels[1],
                bias=False)
            self.w = self.layer.weight
            self.register_parameter('fc_pool_'+self.params.layer_name+'_w', self.layer.weight)

        elif self.params.layer_type == 'conv':
            self.layer = nn.Conv2d(
                in_channels=self.params.layer_channels[0],
                out_channels=self.params.layer_channels[1],
                kernel_size=self.params.pool_ksize,
                stride=self.params.pool_stride,
                padding=0,
                dilation=1,
                bias=False)
            nn.init.orthogonal_(self.layer.weight) # initialize to orthogonal matrix
            self.w = self.layer.weight
            self.register_parameter('conv_pool_'+self.params.layer_name+'_w', self.layer.weight)

        else:
            assert False, ('layer_type parameter must be "fc", "conv", not %g'%(layer_type))

    def forward(self, x):
        if self.params.layer_type == 'fc':
            x = dp.flatten_feature_map(x)
        return self.layer(x)

    def get_encodings(self, input_tensor):
        return self.forward(input_tensor)
