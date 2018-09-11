import numpy as np
import tensorflow as tf
from analysis.base_analysis import Analyzer
import utils.data_processing as dp

class GA_Analyzer(Analyzer):
  def __init__(self, params):
    super(GA_Analyzer, self).__init__(params)
    self.var_names = [
      "weights/w_enc:0",
      "weights/w_dec:0",
      "weights/b_enc:0",
      "weights/b_dec:0",
      "weights/w_gdn0:0",
      "weights/b_gdn0:0",
      "weights/w_igdn1:0",
      "weights/b_igdn1:0",
      "inference/activity:0"]

  def run_analysis(self, images, save_info=""):
    super(GA_Analyzer, self).run_analysis(images, save_info)
    self.evals = self.eval_analysis(images, self.var_names, save_info)
    if self.do_basis_analysis:
      self.bf_stats = self.basis_analysis(self.evals["weights/w_enc:0"], save_info)
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

  def compute_activations(self, images):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=self.model.graph) as sess:
      feed_dict = self.model.get_feed_dict(images)
      sess.run(self.model.init_op, feed_dict)
      self.model.load_weights(sess, self.cp_loc)
      activations = sess.run(self.model.gdn_output, feed_dict) 
    return activations
