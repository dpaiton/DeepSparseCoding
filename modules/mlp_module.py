import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from DeepSparseCoding.modules.activations import activation_picker
import DeepSparseCoding.utils.data_processing as dp


class MlpModule(nn.Module):
    def setup_module(self, params):
        def compute_conv_output_shape(in_length, kernel_size, stride, padding=0, dilation=1):
            out_shape = ((in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
            return np.floor(out_shape).astype(np.int)
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

    def preprocess_data(self, input_tensor):
        if self.params.layer_types[0] == 'fc':
            input_tensor = input_tensor.view(self.params.batch_size, -1) # flatten input
        return input_tensor

    def forward(self, x):
        layer_zip = zip(self.dropout, self.pooling, self.act_funcs, self.layers)
        for layer_index, (dropout, pooling, act_func, layer) in enumerate(layer_zip):
            prev_layer = self.params.layer_types[layer_index - 1]
            current_layer = self.params.layer_types[layer_index]
            if(layer_index > 0 and current_layer == 'fc' and prev_layer == 'conv'):
                x = dp.flatten_feature_map(x)
            x = dropout(pooling(act_func(layer(x))))
        return x

    def get_encodings(self, input_tensor):
        return self.forward(input_tensor)
