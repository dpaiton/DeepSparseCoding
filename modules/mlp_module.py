import torch.nn as nn
import torch.nn.functional as F

from DeepSparseCoding.modules.activations import activation_picker


class MlpModule(nn.Module):
    def setup_module(self, params):
        self.params = params
        self.act_funcs = [activation_picker(act_func_str)
            for act_func_str in self.params.activation_functions]
        self.layers = []
        self.dropout = []
        for layer_index, layer_type in enumerate(self.params.layer_types):
            if layer_type == 'fc':
                layer = nn.Linear(
                    in_features = self.params.layer_channels[layer_index],
                    out_features = self.params.layer_channels[layer_index+1],
                    bias = True)
            elif layer_type == 'conv':
                w_shape = [
                    self.params.out_channels[layer_index],
                    self.params.in_channels[layer_index],
                    self.params.kernel_size[layer_index],
                    self.params.kernel_size[layer_index]]
                layer = nn.Conv2d(
                    in_channels = self.params.in_channels[layer_index],
                    out_channels = self.params.out_channels[layer_index],
                    kernel_size = w_shape,
                    stride = self.parmas.stride[layer_index],
                    padding = self.params.padding[layer_index],
                    bias=True)
            else:
                assert False, ('layer_type parameter must be "fc", not %g'%(layer_type))
            self.register_parameter(layer_type+str(layer_index)+'_w', layer.weight)
            self.register_parameter(layer_type+str(layer_index)+'_b', layer.bias)
            self.layers.append(layer)
            self.dropout.append(nn.Dropout(p=self.params.dropout_rate[layer_index]))

    def preprocess_data(self, input_tensor):
        input_tensor = input_tensor.view(-1, self.params.layer_channels[0])
        return input_tensor

    def forward(self, x):
        for dropout, act_func, layer in zip(self.dropout, self.act_funcs, self.layers):
            x = dropout(act_func(layer(x)))
        return x

    def get_encodings(self, input_tensor):
        return self.forward(input_tensor)
