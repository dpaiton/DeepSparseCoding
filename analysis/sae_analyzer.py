import os
import numpy as np
import tensorflow as tf
from analysis.base_analyzer import Analyzer
import utils.data_processing as dp

class SaeAnalyzer(Analyzer):
  def __init__(self):
    Analyzer.__init__(self)
    self.var_names = [
      "layer0/w_0:0",
      "layer1/w_1:0",
      "inference/activity:0"]

  def run_analysis(self, images, labels=None, save_info=""):
    super(SaeAnalyzer, self).run_analysis(images, labels, save_info=save_info)
    if self.analysis_params.do_evals:
      self.evals = self.eval_analysis(images, self.var_names, save_info)
    if self.analysis_params.do_basis_analysis:
      self.bf_stats = self.basis_analysis(self.evals["layer0/w_0:0"], save_info)
    if self.analysis_params.do_atas:
      self.atas, self.atcs = self.ata_analysis(images, self.evals["inference/activity:0"],
        save_info)
      self.noise_activity, self.noise_atas, self.noise_atcs = self.run_noise_analysis(save_info)
    if self.analysis_params.do_orientation_analysis:
      self.ot_grating_responses, self.co_grating_responses = self.grating_analysis(self.bf_stats,
        save_info)
    if self.analysis_params.do_recon_adversaries:
      self.recon_adversary_analysis(images,
        labels=labels, batch_size=self.analysis_params.eval_batch_size,
        input_id=self.analysis_params.adversarial_input_id,
        target_method=self.analysis_params.adversarial_target_method,
        target_id=self.analysis_params.adversarial_target_id,
        save_info=save_info)
