import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PyTorchDisentanglement.modules.activations import activation_picker
from PyTorchDisentanglement.models.base import BaseModel

class Mlp(BaseModel):
    def setup_model(self):
        self.act_funcs = [activation_picker(act_func_str)
            for act_func_str in self.params.activation_functions]
        self.layers = []
        self.dropout = []
        for layer_index, layer_type in enumerate(self.params.layer_types):
            if layer_type == "fc":
                layer = nn.Linear(
                    in_features = self.params.layer_channels[layer_index],
                    out_features = self.params.layer_channels[layer_index+1],
                    bias = True)
                self.register_parameter("fc"+str(layer_index)+"_w", layer.weight)
                self.register_parameter("fc"+str(layer_index)+"_b", layer.bias)
                self.layers.append(layer)
            else:
                assert False, ("layer_type parameter must be 'fc', not %g"%(layer_type))
            self.dropout.append(nn.Dropout(p=self.params.dropout_rate[layer_index]))

    def preprocess_data(self, input_tensor):
        input_tensor = input_tensor.view(-1, self.params.layer_channels[0])
        return input_tensor

    def get_total_loss(self, input_tuple):
        input_tensor, input_label = input_tuple
        pred = self.forward(input_tensor)
        return F.nll_loss(pred, input_label)

    def forward(self, x):
        for dropout, act_func, layer in zip(self.dropout, self.act_funcs, self.layers):
            x = dropout(act_func(layer(x)))
        x = F.log_softmax(x, dim=1)
        return x

    def get_encodings(self, input_tensor):
        return self.forward(input_tensor)

    def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
        update_dict = super(Mlp, self).generate_update_dict(input_data, input_labels, batch_step)
        epoch = batch_step / self.params.batches_per_epoch
        stat_dict = {
            "epoch":int(epoch),
            "batch_step":batch_step,
            "train_progress":np.round(batch_step/self.params.num_batches, 3)}
        output = self.forward(input_data)
        pred = output.max(1, keepdim=True)[1]
        total_loss = self.get_total_loss((input_data, input_labels))
        correct = pred.eq(input_labels.view_as(pred)).sum().item()
        stat_dict["loss"] = total_loss.item()
        stat_dict["train_accuracy"] = 100. * correct / self.params.batch_size
        update_dict.update(stat_dict)
        return update_dict
