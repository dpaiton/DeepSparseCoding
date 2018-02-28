import numpy as np
import tensorflow as tf
from analysis.base_analysis import Analyzer
import utils.data_processing as dp

class ICA(Analyzer):
  def __init__(self, params):
    Analyzer.__init__(self, params)
    self.var_names = [
      "weights/a:0",
      "inference/coefficients:0"]

  def load_params(self, params):
    super(ICA, self).load_params(params)
    self.ft_padding = params["ft_padding"]
    self.num_gauss_fits = 20
    self.gauss_thresh = 0.2

  def run_analysis(self, images, save_info=""):
    self.run_stats = self.get_log_stats()
    self.evals = self.evaluate_model(images, self.var_names)
    self.atas = self.compute_atas(self.evals["inference/coefficients:0"], images)
    self.bf_stats = dp.get_dictionary_stats(self.evals["weights/a:0"].T, padding=self.ft_padding,
      num_gauss_fits=self.num_gauss_fits, gauss_thresh=self.gauss_thresh)
    np.savez(self.analysis_out_dir+"analysis_"+save_info+".npz",
      data={"run_stats":self.run_stats, "evals":self.evals, "atas":self.atas,
      "var_names":self.var_names, "bf_stats":self.bf_stats})

  def load_analysis(self, save_info=""):
    file_loc = self.analysis_out_dir+"analysis_"+save_info+".npz"
    analysis = np.load(file_loc)["data"].item()
    self.var_names = analysis["var_names"]
    self.run_stats = analysis["run_stats"]
    self.evals = analysis["evals"]
    self.atas = analysis["atas"]
    self.bf_stats = analysis["bf_stats"]

  def compute_activations(self, images):
    with tf.Session(graph=self.model.graph) as sess:
      feed_dict = self.model.get_feed_dict(images)
      sess.run(self.model.init_op, feed_dict)
      self.model.load_weights(sess, self.cp_loc)
      activations = sess.run(self.model.u, feed_dict) 
    return activations
