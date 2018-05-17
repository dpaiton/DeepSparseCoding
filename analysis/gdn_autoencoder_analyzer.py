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
      "weights/gdn_w:0",
      "weights/gdn_b:0",
      "weights/igdn_w:0",
      "weights/igdn_b:0",
      "inference/gdn_output:0"]

  def load_params(self, params):
    super(GA_Analyzer, self).load_params(params)
    self.ft_padding = params["ft_padding"]
    self.ot_neurons = params["neuron_indices"]
    self.ot_contrasts = params["contrasts"]
    self.ot_orientations = params["orientations"]
    self.ot_phases = params["phases"]
    if "num_gauss_fits" in params.keys():
      self.num_gauss_fits = params["num_gauss_fits"]
    else:
      self.num_gauss_fits = 20
    if "gauss_thresh" in params.keys():
      self.gauss_thresh = params["gauss_thresh"]
    else:
      self.gauss_thresh = 0.2

  def run_analysis(self, images, save_info=""):
    super(GA_Analyzer, self).run_analysis(images, save_info)
    self.evals = self.evaluate_model(images, self.var_names)
    self.atas = self.compute_atas(self.evals["inference/gdn_output:0"], images)
    self.bf_stats = dp.get_dictionary_stats(self.evals["weights/w_enc:0"],
      padding=self.ft_padding, num_gauss_fits=self.num_gauss_fits, gauss_thresh=self.gauss_thresh)
    np.savez(self.analysis_out_dir+"analysis_"+save_info+".npz",
      data={"run_stats":self.run_stats, "evals":self.evals, "atas":self.atas,
      "var_names":self.var_names, "bf_stats":self.bf_stats})
    self.ot_grating_responses = self.orientation_tuning(self.bf_stats, self.ot_contrasts,
      self.ot_orientations, self.ot_phases, self.ot_neurons)
    np.savez(self.analysis_out_dir+"ot_responses_"+save_info+".npz", data=self.ot_grating_responses)
    ot_mean_activations = self.ot_grating_responses["mean_responses"]
    base_orientations = [self.ot_orientations[np.argmax(ot_mean_activations[bf_idx,-1,:])]
      for bf_idx in range(len(self.ot_grating_responses["neuron_indices"]))]
    self.co_grating_responses = self.cross_orientation_suppression(self.bf_stats,
      self.ot_contrasts, self.ot_phases, base_orientations, self.ot_orientations, self.ot_neurons)
    np.savez(self.analysis_out_dir+"co_responses_"+save_info+".npz", data=self.co_grating_responses)

  def load_analysis(self, save_info=""):
    file_loc = self.analysis_out_dir+"analysis_"+save_info+".npz"
    analysis = np.load(file_loc)["data"].item()
    self.var_names = analysis["var_names"]
    self.run_stats = analysis["run_stats"]
    self.evals = analysis["evals"]
    self.atas = analysis["atas"]
    self.bf_stats = analysis["bf_stats"]
    tuning_file_locs = [self.analysis_out_dir+"ot_responses_"+save_info+".npz",
      self.analysis_out_dir+"co_responses_"+save_info+".npz"]
    self.ot_grating_responses = np.load(tuning_file_locs[0])["data"].item()
    self.co_grating_responses = np.load(tuning_file_locs[1])["data"].item()

  def compute_activations(self, images):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=self.model.graph) as sess:
      feed_dict = self.model.get_feed_dict(images)
      sess.run(self.model.init_op, feed_dict)
      self.model.load_weights(sess, self.cp_loc)
      activations = sess.run(self.model.gdn_output, feed_dict) 
    return activations
