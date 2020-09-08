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
                self.register_parameter('fc'+str(layer_index)+'_w', layer.weight)
                self.register_parameter('fc'+str(layer_index)+'_b', layer.bias)
                self.layers.append(layer)
            else:
                assert False, ('layer_type parameter must be "fc", not %g'%(layer_type))
            self.dropout.append(nn.Dropout(p=self.params.dropout_rate[layer_index]))

    def preprocess_data(self, input_tensor):
        input_tensor = input_tensor.view(-1, self.params.layer_channels[0])
        return input_tensor

    def forward(self, x):
        for dropout, act_func, layer in zip(self.dropout, self.act_funcs, self.layers):
            x = dropout(act_func(layer(x)))
        #x = F.log_softmax(x, dim=1)
        return x

    def get_encodings(self, input_tensor):
        return self.forward(input_tensor)
