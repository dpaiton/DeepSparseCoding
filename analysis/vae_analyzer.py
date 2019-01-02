import os
import numpy as np
import tensorflow as tf
from analysis.base_analysis import Analyzer
import utils.data_processing as dp

class VaeAnalyzer(Analyzer):
  def __init__(self, params):
    super(VaeAnalyzer, self).__init__(params)
    self.var_names = [
      "weights/w_enc_mean:0",
      "weights/w_enc_std:0",
      "weights/w_dec:0",
      "inference/activity:0"]

  def run_analysis(self, images, save_info=""):
    super(VaeAnalyzer, self).run_analysis(images, save_info)
    self.evals = self.eval_analysis(images, self.var_names, save_info)
    if self.do_basis_analysis:
      self.bf_stats = self.basis_analysis(self.evals["weights/w_enc_mean:0"], save_info)
    if self.do_atas:
      self.atas, self.atcs = self.ata_analysis(images, self.evals["inference/activity:0"],
        save_info)
      self.noise_activity, self.noise_atas, self.noise_atcs = self.run_noise_analysis(save_info)
    if (self.ot_contrasts is not None
      and self.ot_orientations is not None
      and self.ot_phases is not None
      and self.do_basis_analysis):
      self.ot_grating_responses, self.co_grating_responses = self.grating_analysis(self.bf_stats,
        save_info)
    if self.do_adversaries:
      self.adversarial_images, self.adversarial_recons, mses = self.adversary_analysis(images,
        input_id=self.adversarial_input_id, target_id=self.adversarial_target_id,
        eps=self.adversarial_eps, num_steps=self.adversarial_num_steps, save_info=save_info)
      self.adversarial_input_target_mses = mses["input_target_mse"]
      self.adversarial_input_recon_mses = mses["input_recon_mses"]
      self.adversarial_input_adv_mses = mses["input_adv_mses"]
      self.adversarial_target_recon_mses = mses["target_recon_mses"]
      self.adversarial_target_adv_mses = mses["target_adv_mses"]
      self.adversarial_adv_recon_mses = mses["adv_recon_mses"]
