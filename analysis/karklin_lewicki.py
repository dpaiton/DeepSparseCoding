import numpy as np
import matplotlib
import utils.plot_functions as pf
from analysis.base_analysis import Analyzer

class karklin_lewicki(Analyzer):
  def __init__(self, params):
    Analyzer.__init__(self, params)

  def load_params(self, params):
    Analyzer.load_params(self, params)
    self.eval_inference = params["eval_inference"]
    self.eval_density_weights = params["eval_density_weights"]

  def plot_stats(self, stats):
    losses = {
     "batch_step":stats["batch_step"],
     "recon_loss":stats["recon_loss"],
     "sparse_loss":stats["sparse_loss"],
     "feedback_loss":stats["feedback_loss"],
     "total_loss":stats["total_loss"]}
    loss_filename = self.out_dir+"losses_v"+self.version+self.file_ext
    pf.save_losses(data=losses, labels=None, out_filename=loss_filename)



