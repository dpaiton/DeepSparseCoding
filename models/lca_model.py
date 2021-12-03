import numpy as np
import torch

from DeepSparseCoding.models.base_model import BaseModel
from DeepSparseCoding.modules.lca_module import LcaModule
import DeepSparseCoding.modules.losses as losses


class LcaModel(BaseModel, LcaModule):
    def setup(self, params, logger=None):
        super(LcaModel, self).setup(params, logger)
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
            update_dict = super(LcaModel, self).generate_update_dict(input_data, input_labels, batch_step)
        stat_dict = dict()
        latents = self.get_encodings(input_data)
        recon = self.get_recon_from_latents(latents)
        recon_loss = losses.half_squared_l2(input_data, recon).item()
        sparse_loss = self.params.sparse_mult * losses.l1_norm(latents).item()
        stat_dict['weight_lr'] = self.scheduler.get_last_lr()[0]
        stat_dict['loss_recon'] = recon_loss
        stat_dict['loss_sparse'] = sparse_loss
        stat_dict['loss_total'] = recon_loss + sparse_loss
        stat_dict['input_max_mean_min'] = [
                input_data.max().item(), input_data.mean().item(), input_data.min().item()]
        stat_dict['recon_max_mean_min'] = [
                recon.max().item(), recon.mean().item(), recon.min().item()]
        def count_nonzero(array, dim):
            # TODO: github issue 23907 requests torch.count_nonzero, integrated in torch 1.7
            return torch.sum(array !=0, dim=dim, dtype=torch.float)
        latent_dims = tuple([i for i in range(len(latents.shape))])
        latent_nnz = count_nonzero(latents, dim=latent_dims).item()
        stat_dict['fraction_active_all_latents'] = latent_nnz / latents.numel()
        if self.params.layer_types[0] == 'conv':
            latent_map_dims = latent_dims[2:]
            latent_map_size = np.prod(list(latents.shape[2:]))
            latent_channel_nnz = count_nonzero(latents, dim=latent_map_dims)/latent_map_size
            latent_channel_mean_nnz = torch.mean(latent_channel_nnz).item()
            stat_dict['fraction_active_latents_per_channel'] = latent_channel_mean_nnz
            num_channels = latents.shape[1]
            latent_patch_mean_nnz = torch.mean(count_nonzero(latents, dim=1)/num_channels).item()
            stat_dict['fraction_active_latents_per_patch'] = latent_patch_mean_nnz
        update_dict.update(stat_dict)
        return update_dict
