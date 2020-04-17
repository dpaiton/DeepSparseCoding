#from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PyTorchDisentanglement.models.base import BaseModel
import PyTorchDisentanglement.models.model_loader as ml

class Ensemble(BaseModel):
    def setup_model(self):
        self.models = []
        for model_idx, params in enumerate(self.params.ensemble_params):
            params.epoch_size = self.params.epoch_size
            params.num_val_images = self.params.num_val_images
            params.num_test_images = self.params.num_test_images
            params.data_shape = self.params.data_shape
            model = ml.load_model(params.model_type)
            model.setup(params, self.logger)
            model.to(params.device)
            #model.print_update = self.print_update
            self.models.append(model)

    def forward(self, x):
        for model in models:
            x = model.get_encodings(x) # pre-classifier or pre-generator latent encodings
        return x

    def setup_optimizer(self):
        for model in self.models:
            model.optimizer = self.get_optimizer(
                optimizer_params=model.params,
                trainable_variables=model.parameters())
            model.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                model.optimizer,
                milestones=model.params.optimizer.milestones,
                gamma=model.params.optimizer.lr_decay_rate)

    def get_ensemble_losses(self, input_tuple):
        x = input_tuple[0]
        losses = []
        for model in self.models:
            x = model.get_encodings(x)
            losses.append(model.get_total_loss((x, input_tuple[1])))
        return losses

    def get_total_loss(self, input_tuple):
        total_loss = self.get_ensemble_losses(input_tuple)
        return torch.stack(total_loss, dim=0).sum()

    def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
        update_dict = super(Ensemble, self).generate_update_dict(input_data,
            input_labels, batch_step)
        x = input_data
        for model in self.models:
            model_update_dict = model.generate_update_dict(x, input_labels, batch_step)
            for key, value in model_update_dict.items():
                key = model.params.model_type+"_"+key
                update_dict[key] = value
            x = model.get_encodings(x)
        return update_dict
