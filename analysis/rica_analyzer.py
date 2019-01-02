import os
import numpy as np
import tensorflow as tf
from analysis.base_analysis import Analyzer
import utils.data_processing as dp

class RicaAnalyzer(Analyzer):
  def __init__(self, params):
    Analyzer.__init__(self, params)
    self.var_names = [
      "weights/w:0",
      "inference/activity:0",
      "output/reconstruction:0"]

  def run_analysis(self, images, save_info=""):
    super(RicaAnalyzer, self).run_analysis(images, save_info)
    self.evals = self.eval_analysis(images, self.var_names, save_info)
    if self.do_basis_analysis:
      self.bf_stats = self.basis_analysis(self.evals["weights/w:0"], save_info)
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
