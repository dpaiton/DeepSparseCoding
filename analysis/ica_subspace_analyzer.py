import numpy as np
import tensorflow as tf
from analysis.base_analyzer import Analyzer
import utils.data_processing as dp

class IcaSubspaceAnalyzer(Analyzer):
  def __init__(self):
    super(IcaSubspaceAnalyzer, self).__init__()
    self.var_names = [
      "weights/w_synth:0",
      "weights/w_analy:0",
      "inference/latent_vars:0"]

  def run_analysis(self, images, labels=None, save_info=""):
    super(IcaSubspaceAnalyzer, self).run_analysis(images, labels, save_info=save_info)
    if self.analysis_params.do_evals:
      self.evals = self.eval_analysis(images, self.var_names, save_info)
    if self.analysis_params.do_basis_analysis:
      self.bf_stats = self.basis_analysis(self.evals["weights/w_synth:0"], save_info)
    if self.analysis_params.do_atas:
      self.atas, self.atcs = self.ata_analysis(images, self.evals["inference/latent_vars:0"],
        save_info)
      self.noise_activity, self.noise_atas, self.noise_atcs = self.run_noise_analysis(save_info)
    if self.analysis_params.do_orientation_analysis:
      self.ot_grating_responses, self.co_grating_responses = self.grating_analysis(self.bf_stats,
        save_info)
