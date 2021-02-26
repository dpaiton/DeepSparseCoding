import torch

import DeepSparseCoding.modules.losses as losses

from DeepSparseCoding.models.base_model import BaseModel
from DeepSparseCoding.modules.pooling_module import PoolingModule

class PoolingModel(BaseModel, PoolingModule):
    """
    TODO: rename pool_ksize and pool_stride to just kernel_size and stride
    """
    def setup(self, params, logger=None):
        self.setup_module(params)
        self.setup_optimizer()
        if params.checkpoint_boot_log != '':
            checkpoint = self.get_checkpoint_from_log(params.checkpoint_boot_log)
            self.module.load_state_dict(checkpoint['model_state_dict'])
            self.module.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def get_total_loss(self, input_tuple):
        def loss_fn(model_output):
            output_loss = losses.trace_covariance(model_output)
            w_stride = self.params.pool_stride
            w_padding = 0
            weight_loss = losses.weight_orthogonality(self.weight, stride=w_stride, padding=w_padding)
            return output_loss + weight_loss
        input_tensor, input_label = input_tuple
        layer_output = self.forward(input_tensor)
        self.loss_fn = loss_fn
        return self.loss_fn(layer_output)

    def generate_update_dict(self, input_data, input_labels=None, batch_step=0, update_dict=None):
        if update_dict is None:
            update_dict = super(PoolinModel, self).generate_update_dict(input_data, input_labels, batch_step)
        stat_dict = dict()
        rep = self.forward(input_data)
        total_loss = self.loss_fn(rep)
        stat_dict['weight_lr'] = self.scheduler.get_lr()[0]
        stat_dict['loss'] = total_loss.item()
        update_dict.update(stat_dict)
        return update_dict
