import os
import numpy as np
import tensorflow as tf
from analysis.base_analysis import Analyzer
import utils.data_processing as dp

class RICA_Analyzer(Analyzer):
  def __init__(self, params):
    Analyzer.__init__(self, params)
    self.var_names = [
      "weights/w:0",
      "inference/activity:0",
      "output/reconstruction:0"]

  def load_params(self, params):
    super(RICA_Analyzer, self).load_params(params)
    if "num_noise_images" in params.keys():
      self.num_noise_images = params["num_noise_images"]
    else:
      self.num_noise_images = 100

  def run_recon_analysis(self, full_image, save_info):
    wht_img, img_mean, ft_filter = dp.whiten_data(full_image,
      method="FT", lpf_cutoff=self.model_params["lpf_cutoff"])
    img_patches = dp.extract_patches(wht_img,
      out_shape=(1, self.model_params["patch_edge_size"], self.model_params["patch_edge_size"], 1),
      overlapping=False, randomize=False, var_thresh=0.0)
    img_patches, orig_shape = dp.reshape_data(img_patches, flatten=True)[:2]
    model_eval = self.evaluate_model(img_patches,
      ["inference/activity:0", "output/reconstruction:0"])
    recon_patches = model_eval["output/reconstruction:0"]
    a_vals = model_eval["inference/activity:0"]
    self.recon_frac_act = np.array(np.count_nonzero(a_vals) / float(a_vals.size))
    recon_patches = dp.reshape_data(recon_patches, flatten=False, out_shape=orig_shape)[0]
    wht_recon = dp.patches_to_image(recon_patches, full_image.shape)
    self.full_image = full_image
    self.full_recon = dp.unwhiten_data(wht_recon, img_mean, ft_filter, method="FT")
    np.savez(self.analysis_out_dir+"full_recon_"+save_info+".npz",
      data={"full_image":self.full_image, "full_recon":self.full_recon,
      "recon_frac_act":self.recon_frac_act})
    self.analysis_logger.log_info("Recon analysis is complete.")

  def run_base_analysis(self, images, save_info):
    self.evals = self.evaluate_model(images, self.var_names)
    self.atas = self.compute_atas(self.evals["inference/activity:0"], images)
    self.bf_stats = dp.get_dictionary_stats(self.evals["weights/w:0"],
      padding=self.ft_padding, num_gauss_fits=self.num_gauss_fits, gauss_thresh=self.gauss_thresh)
    np.savez(self.analysis_out_dir+"analysis_"+save_info+".npz",
      data={"run_stats":self.run_stats, "evals":self.evals, "atas":self.atas,
      "var_names":self.var_names, "bf_stats":self.bf_stats})
    self.analysis_logger.log_info("Base analysis is complete.")

  def run_grating_analysis(self, save_info):
    self.ot_grating_responses = self.orientation_tuning(self.bf_stats, self.ot_contrasts,
      self.ot_orientations, self.ot_phases, self.ot_neurons, scale=self.input_scale)
    np.savez(self.analysis_out_dir+"ot_responses_"+save_info+".npz", data=self.ot_grating_responses)
    ot_mean_activations = self.ot_grating_responses["mean_responses"]
    base_orientations = [self.ot_orientations[np.argmax(ot_mean_activations[bf_idx,-1,:])]
      for bf_idx in range(len(self.ot_grating_responses["neuron_indices"]))]
    self.co_grating_responses = self.cross_orientation_suppression(self.bf_stats,
      self.ot_contrasts, self.ot_phases, base_orientations, self.ot_orientations, self.ot_neurons,
      scale=self.input_scale)
    np.savez(self.analysis_out_dir+"co_responses_"+save_info+".npz", data=self.co_grating_responses)
    self.analysis_logger.log_info("Grating  analysis is complete.")

  def run_analysis(self, images, save_info=""):
    super(RICA_Analyzer, self).run_analysis(images, save_info)
    self.run_base_analysis(images, save_info)
    self.run_noise_analysis(save_info)
    if (self.ot_contrasts is not None
      and self.ot_orientations is not None
      and self.ot_phases is not None):
      self.run_grating_analysis(save_info)

  def load_analysis(self, save_info=""):
    file_loc = self.analysis_out_dir+"analysis_"+save_info+".npz"
    analysis = np.load(file_loc)["data"].item()
    self.var_names = analysis["var_names"]
    self.run_stats = analysis["run_stats"]
    self.evals = analysis["evals"]
    self.atas = analysis["atas"]
    self.bf_stats = analysis["bf_stats"]
    noise_file_loc = self.analysis_out_dir+"noise_responses_"+save_info+".npz"
    if os.path.exists(noise_file_loc):
      noise_analysis = np.load(noise_file_loc)["data"].item()
      self.noise_activity = noise_analysis["noise_activity"]
      self.noise_atas = noise_analysis["noise_atas"]
      self.noise_atcs = noise_analysis["noise_atcs"]
      self.num_noise_images = noise_analysis["num_noise_images"]
    tuning_file_locs = [self.analysis_out_dir+"ot_responses_"+save_info+".npz",
      self.analysis_out_dir+"co_responses_"+save_info+".npz"]
    if os.path.exists(tuning_file_locs[0]):
      self.ot_grating_responses = np.load(tuning_file_locs[0])["data"].item()
    if os.path.exists(tuning_file_locs[1]):
      self.co_grating_responses = np.load(tuning_file_locs[1])["data"].item()
    recon_file_loc = self.analysis_out_dir+"full_recon_"+save_info+".npz"
    if os.path.exists(recon_file_loc):
      recon_analysis = np.load(recon_file_loc)["data"].item()
      self.full_image = recon_analysis["full_image"]
      self.full_recon = recon_analysis["full_recon"]
      self.recon_frac_act = recon_analysis["recon_frac_act"]

  def compute_activations(self, images):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=self.model.graph) as sess:
      feed_dict = self.model.get_feed_dict(images)
      sess.run(self.model.init_op, feed_dict)
      self.model.load_weights(sess, self.cp_loc)
      activations = sess.run(self.model.a, feed_dict)
    return activations
