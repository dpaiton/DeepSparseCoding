import torch
import torch.nn as nn
import torch.nn.functional as F

import DeepSparseCoding.utils.data_processing as dp


class PoolingModule(nn.Module):
    def setup_module(self, params):
        self.params = params
        if self.params.layer_type == 'fc':
            layer = nn.Linear(
                in_features=self.params.layer_channels[0],
                out_features=self.params.layer_channels[1],
                bias=False)
            self.w = layer.weight
            self.register_parameter('fc_pool_'+self.params.pool_name+'_w', layer.weight)

        elif self.params.layer_type == 'conv':
            layer = nn.Conv2d(
                in_channels=self.params.layer_channels[0],
                out_channels=self.params.layer_channels[1],
                kernel_size=self.params.pool_ksize,
                stride=self.params.pool_stride,
                padding=0,
                dilation=1,
                bias=False)
            self.w = layer.weight
            self.register_parameter('conv_pool_'+self.params.pool_name+'_w', layer.weight)

        elif self.params.layer_type == 'orth_conv':
            """
            Based on Orthogonal Convolutional Neural Networks
            https://arxiv.org/abs/1911.12207
            https://github.com/samaonline/Orthogonal-Convolutional-Neural-Networks
            """
            self.w_shape = [
                self.params.layer_channels[1],
                self.params.layer_channels[0],
                self.params.pool_ksize,
                self.params.pool_ksize # assumed square kernel
            ]
            w_init = torch.randn(self.w_shape)
            w_init_normed = dp.l2_normalize_weights(w_init, eps=self.params.eps)
            self.w = nn.Parameter(w_init_normed, requires_grad=True)
            kernel_width = self.params.pool_ksize
            in_channels = self.params.layer_channels[0]
            new_stride = self.params.pool_stride * (kernel_width-1) + kernel_width
            identity = torch.eye(
                n=int(new_stride * new_stride * in_channels),
                requires_grad=True,
                device=self.params.device)
            identity = identity.reshape(
                (new_stride * new_stride * in_channels, in_channels, new_stride, new_stride))
            conv_out = F.conv2d(
                    identity,
                    self.w,
                    stride=self.params.pool_stride,
                    padding=0,
                    dilation=1)
            conv_out = conv_out.reshape((new_stride * new_stride * in_channels, -1))
            Vmat = conv_out[np.floor(new_stride**2 / 2).astype(int)::new_stride**2, :]
            dbt_mask = torch.zeors(in_channels, in_channels * new_stride**2)
            for i in range(in_channels):
                dbt_mask[i, np.floor(new_stride**2 / 2).astype(int) + new_stride**2 * i] = 1
            layer = torch.norm(torch.dot(Vmat, conv_out.transpose()) - dbt_mask, p='fro')

        else:
            assert False, ('layer_type parameter must be "fc", "conv", or "orth_conv", not %g'%(layer_type))

    def forward(self, x):
        if self.params.layer_type == 'fc':
            x = dp.flatten_feature_map(x)
        return layer(x)

    def get_encodings(self, input_tensor):
        return self.forward(input_tensor)
