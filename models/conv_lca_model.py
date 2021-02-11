import numpy as np
import torch

from DeepSparseCoding.models.base_model import BaseModel
from DeepSparseCoding.modules.conv_lca_module import ConvLcaModule
import DeepSparseCoding.modules.losses as losses


class ConvLcaModel(BaseModel, ConvLcaModule):
    def setup(self, params, logger=None):
        super(ConvLcaModel, self).setup(params, logger)
        self.setup_module(params)
        self.setup_optimizer()
        if params.checkpoint_boot_log != '':
            checkpoint = self.get_checkpoint_from_log(params.checkpoint_boot_log)
            self.module.load_state_dict(checkpoint['model_state_dict'])
            self.module.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def get_total_loss(self, input_tuple):
        input_tensor, input_labels = input_tuple
        latents = self.get_encodings(input_tensor)
        recon = self.get_recon_from_latents(latents)
        recon_loss = losses.half_squared_l2(input_tensor, recon)
        sparse_loss = self.params.sparse_mult * losses.l1_norm(latents)
        total_loss = recon_loss + sparse_loss
        return total_loss

    def generate_update_dict(self, input_data, input_labels=None, batch_step=0, update_dict=None):
        if update_dict is None:
            update_dict = super(ConvLcaModel, self).generate_update_dict(input_data, input_labels, batch_step)
        latents = self.get_encodings(input_data)
        recon = self.get_recon_from_latents(latents)
        recon_loss = losses.half_squared_l2(input_data, recon).item()
        sparse_loss = self.params.sparse_mult * losses.l1_norm(latents).item()
        stat_dict['weight_lr'] = self.scheduler.get_lr()[0]
        stat_dict['loss_recon'] = recon_loss
        stat_dict['loss_sparse'] = sparse_loss
        stat_dict['loss_total'] = recon_loss + sparse_loss
        stat_dict['input_max_mean_min'] = [
                input_data.max().item(), input_data.mean().item(), input_data.min().item()]
        stat_dict['recon_max_mean_min'] = [
                recon.max().item(), recon.mean().item(), recon.min().item()]
        latent_nnz = torch.sum(latents != 0).item() # TODO: github issue 23907 requests torch.count_nonzero
        stat_dict['latents_fraction_active'] = latent_nnz / latents.numel()
        update_dict.update(stat_dict)
        return update_dict
