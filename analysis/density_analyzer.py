import tensorflow as tf
import numpy as np
from analysis.base_analysis import Analyzer
import utils.log_parser as log_parser
import utils.plot_functions as pf

class density_analyzer(Analyzer):
  def __init__(self, params):
    Analyzer.__init__(self, params)

  def load_params(self, params):
    Analyzer.load_params(self, params)
    self.eval_inference = params["eval_inference"]

  def evaluate_model(self, images):
    var_names = [
      "weights/phi:0",
      "weights/b:0",
      "inference/u:0",
      "inference/v:0",
      "inference/activity:0",
      "output/image_estimate/reconstruction:0",
      "performance_metrics/reconstruction_quality/recon_quality:0"]
    feed_dict = self.model.get_feed_dict(images)
    evals = {}
    with tf.Session(graph=self.model.graph) as tmp_sess:
      tmp_sess.run(self.model.init_op, feed_dict)
      self.model.weight_saver.restore(tmp_sess, self.cp_loc)
      for step in range(self.model.num_steps):
        tmp_sess.run([self.model.step_inference], feed_dict)
      for name in var_names:
        tensor = self.model.graph.get_tensor_by_name(name)
        evals[name] = tmp_sess.run(tensor, feed_dict)
    return evals

  """
  plot loss values during learning
  """
  def save_log_stats(self):
    stats = log_parser.read_stats(self.log_text)
    relevant_stats = {
     "batch_step":stats["batch_step"],
     "recon_loss":stats["recon_loss"],
     "sparse_loss":stats["sparse_loss"],
     "density_loss":stats["density_loss"],
     "b_loss":stats["b_loss"],
     "total_loss":stats["total_loss"],
     "a_frac_act":stats["a_fraction_active"]}
    loss_filename = self.analysis_out_dir+"log_stats_v"+self.version+self.file_ext
    pf.plot_stats(data=relevant_stats, labels=None, save_filename=loss_filename)

  """
  Replicate figure 6 from K&L2005
  """
  def save_density_weights(self, phi, b):
   pass
