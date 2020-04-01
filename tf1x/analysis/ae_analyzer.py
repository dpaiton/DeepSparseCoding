import os

import numpy as np
import tensorflow as tf

from DeepSparseCoding.tf1x.analysis.base_analyzer import Analyzer
import DeepSparseCoding.tf1x.utils.data_processing as dp

class AeAnalyzer(Analyzer):
  def __init__(self):
    Analyzer.__init__(self)
    # TODO: the weight var names can be pulled from train_vars in the log file now, this should
    #   be done automatically
    self.var_names = [
      "ae/layer0/fc_w_0:0", # encoding
      #"ae/layer1/fc_w_1:0", # decoding
      "ae/layer5/fc_w_5:0", # decoding
      "inference/activity:0"]

  def run_analysis(self, images, labels=None, save_info=""):
    super(AeAnalyzer, self).run_analysis(images, labels, save_info=save_info)
    if self.analysis_params.do_evals:
      self.evals = self.eval_analysis(images, self.var_names, save_info)
    if self.analysis_params.do_basis_analysis:
      self.bf_stats = self.basis_analysis(self.evals["ae/layer0/fc_w_0:0"], save_info)
    if self.analysis_params.do_atas:
      self.atas, self.atcs = self.ata_analysis(images, self.evals["inference/activity:0"],
        save_info)
      self.noise_activity, self.noise_atas, self.noise_atcs = self.run_noise_analysis(save_info)
    if self.analysis_params.do_orientation_analysis:
      if not self.analysis_params.do_basis_analysis:
        try:
          self.load_basis_stats(save_info)
        except FileNotFoundError:
          assert False, (
          "Basis analysis must have been previously run, or do_basis_analysis must be True.")
      self.ot_grating_responses, self.co_grating_responses = self.grating_analysis(self.bf_stats,
        save_info)
    if self.analysis_params.do_recon_adversaries:
      self.recon_adversary_analysis(images,
        labels=labels, batch_size=self.analysis_params.eval_batch_size,
        input_id=self.analysis_params.adversarial_input_id,
        target_method=self.analysis_params.adversarial_target_method,
        target_id=self.analysis_params.adversarial_target_id,
        save_info=save_info)
    elif self.analysis_params.do_neuron_visualization:
      self.neuron_visualization_analysis(save_info=save_info)
