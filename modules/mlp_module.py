from collections import OrderedDict

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from DeepSparseCoding.modules.activations import activation_picker
from DeepSparseCoding.utils.run_utils import compute_conv_output_shape


class MlpModule(nn.Module):
    def setup_module(self, params):
        self.params = params
        self.act_funcs = [activation_picker(act_func_str)
            for act_func_str in self.params.activation_functions]
        self.layer_output_shapes = [self.params.data_shape] # [channels, height, width]
        self.layers = []
        self.pooling = []
        self.dropout = []
        for layer_index, layer_type in enumerate(self.params.layer_types):
            if layer_type == 'fc':
                if(layer_index > 0 and self.params.layer_types[layer_index-1] == 'conv'):
                    in_features = np.prod(self.layer_output_shapes[-1]).astype(np.int)
                else:
                    in_features = self.params.layer_channels[layer_index]
                layer = nn.Linear(
                    in_features=in_features,
                    out_features=self.params.layer_channels[layer_index + 1],
                    bias=True)
                self.register_parameter('fc'+str(layer_index)+'_w', layer.weight)
                self.register_parameter('fc'+str(layer_index)+'_b', layer.bias)
                self.layers.append(layer)
                self.layer_output_shapes.append(self.params.layer_channels[layer_index + 1])
            elif layer_type == 'conv':
                layer = nn.Conv2d(
                    in_channels=self.params.layer_channels[layer_index],
                    out_channels=self.params.layer_channels[layer_index + 1],
                    kernel_size=self.params.kernel_sizes[layer_index],
                    stride=self.params.strides[layer_index],
                    padding=0,
                    dilation=1,
                    bias=True)
                self.register_parameter('conv'+str(layer_index)+'_w', layer.weight)
                self.register_parameter('conv'+str(layer_index)+'_b', layer.bias)
                self.layers.append(layer)
                output_channels = self.params.layer_channels[layer_index + 1]
                output_height = compute_conv_output_shape(
                    self.layer_output_shapes[-1][1],
                    self.params.kernel_sizes[layer_index],
                    self.params.strides[layer_index],
                    padding=0,
                    dilation=1)
                output_width = compute_conv_output_shape(
                    self.layer_output_shapes[-1][2],
                    self.params.kernel_sizes[layer_index],
                    self.params.strides[layer_index],
                    padding=0,
                    dilation=1)
                self.layer_output_shapes.append([output_channels, output_height, output_width])
            else:
                assert False, ('layer_type parameter must be "fc" or "conv", not %g'%(layer_type))
            if(self.params.max_pool[layer_index] and layer_type == 'conv'):
                self.pooling.append(nn.MaxPool2d(
                    kernel_size=self.params.pool_ksizes[layer_index],
                    stride=self.params.pool_strides[layer_index],
                    padding=0,
                    dilation=1))
                output_channels = self.params.layer_channels[layer_index + 1]
                output_height = compute_conv_output_shape(
                    self.layer_output_shapes[-1][1],
                    self.params.pool_ksizes[layer_index],
                    self.params.pool_strides[layer_index],
                    padding=0,
                    dilation=1)
                output_width = compute_conv_output_shape(
                    self.layer_output_shapes[-1][2],
                    self.params.pool_ksizes[layer_index],
                    self.params.pool_strides[layer_index],
                    padding=0,
                    dilation=1)
                self.layer_output_shapes.append([output_channels, output_height, output_width])
            else:
                self.pooling.append(nn.Identity()) # do nothing
            self.dropout.append(nn.Dropout(p=self.params.dropout_rate[layer_index]))
        conv_module_dict = OrderedDict()
        fc_module_dict = OrderedDict()
        layer_zip = zip(self.params.layer_types, self.layers, self.act_funcs, self.pooling,
            self.dropout)
        for layer_idx, full_layer in enumerate(layer_zip):
            for component_idx, layer_component in enumerate(full_layer[1:]):
                component_id = f'{layer_idx:02}-{component_idx:02}'
                if full_layer[0] == 'fc':
                    fc_module_dict[full_layer[0] + component_id] = layer_component
                else:
                    conv_module_dict[full_layer[0] + component_id] = layer_component
        self.conv_sequential = lambda x: x # identity by default
        self.fc_sequential = lambda x: x # identity by default
        if len(conv_module_dict) > 0:
            self.conv_sequential = nn.Sequential(conv_module_dict)
        if len(fc_module_dict) > 0:
            self.fc_sequential = nn.Sequential(fc_module_dict)

    def preprocess_data(self, input_tensor):
        if self.params.layer_types[0] == 'fc':
            input_tensor = input_tensor.view(input_tensor.size(0), -1) #flat
        return input_tensor

    def forward(self, x):
        x = self.conv_sequential(x)
        x = x.view(x.size(0), -1) #flat
        x = self.fc_sequential(x)
        return x

    def get_encodings(self, input_tensor):
        return self.forward(input_tensor)
