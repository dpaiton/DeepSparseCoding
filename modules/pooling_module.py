import torch
import torch.nn as nn


class PoolingModule(nn.Module):
    def setup_module(self, params):
        params.weight_decay = 0 # used by base model; pooling layer never has weight decay
        self.params = params
        if self.params.layer_types[0] == 'fc':
            self.layer = nn.Linear(
                in_features=self.params.layer_channels[0],
                out_features=self.params.layer_channels[1],
                bias=False)
            self.weight = self.layer.weight
            #self.register_parameter('fc_pool_'+self.params.layer_name+'_w', self.layer.weight)

        elif self.params.layer_types[0] == 'conv':
            self.layer = nn.Conv2d(
                in_channels=self.params.layer_channels[0],
                out_channels=self.params.layer_channels[1],
                kernel_size=self.params.pool_ksize,
                stride=self.params.pool_stride,
                padding=0,
                dilation=1,
                bias=False)
            nn.init.orthogonal_(self.layer.weight) # initialize to orthogonal matrix
            self.weight = self.layer.weight
            #self.register_parameter('conv_pool_'+self.params.layer_name+'_w', self.layer.weight)

        else:
            assert False, ('layer_types[0] parameter must be "fc", "conv", not %g'%(layer_types[0]))

    def forward(self, x):
        if self.params.layer_types[0] == 'fc':
            x = x.view(x.shape[0], -1) # flat
        return self.layer(x)

    def get_encodings(self, input_tensor):
        return self.forward(input_tensor)
