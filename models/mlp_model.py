import numpy as np

import torch

from DeepSparseCoding.models.base_model import BaseModel
from DeepSparseCoding.modules.mlp_module import MlpModule

class MlpModel(BaseModel, MlpModule):
    def setup(self, params, logger=None):
        super(MlpModel, self).setup(params, logger)
        self.setup_module(params)
        self.setup_optimizer()
        if params.checkpoint_boot_log != '':
            checkpoint = self.get_checkpoint_from_log(params.checkpoint_boot_log)
            self.module.load_state_dict(checkpoint['model_state_dict'])
            self.module.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def get_total_loss(self, input_tuple):
        input_tensor, input_label = input_tuple
        pred = self.forward(input_tensor)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        return self.loss_fn(pred, input_label)

    def generate_update_dict(self, input_data, input_labels=None, batch_step=0, update_dict=None):
        if update_dict is None:
            update_dict = super(MlpModel, self).generate_update_dict(input_data, input_labels, batch_step)
        stat_dict = dict()
        pred = self.forward(input_data)
        total_loss = self.loss_fn(pred, input_labels)
        pred = pred.max(1, keepdim=True)[1]
        correct = pred.eq(input_labels.view_as(pred)).sum().item()
        stat_dict['weight_lr'] = self.scheduler.get_lr()[0]
        stat_dict['loss'] = total_loss.item()
        stat_dict['train_accuracy'] = 100. * correct / self.params.batch_size
        update_dict.update(stat_dict)
        return update_dict
