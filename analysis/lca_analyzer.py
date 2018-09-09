import os
import numpy as np
import tensorflow as tf
from analysis.base_analysis import Analyzer
import utils.data_processing as dp

class LCA_Analyzer(Analyzer):
  def __init__(self, params):
    super(LCA_Analyzer, self).__init__(params)
    self.var_names = [
      "weights/phi:0",
      "inference/u:0",
      "inference/activity:0",
      "output/reconstruction:0",
      "performance_metrics/reconstruction_quality/recon_quality:0"]

  def load_params(self, params):
    super(LCA_Analyzer, self).load_params(params)
    if "num_noise_images" in params.keys():
      self.num_noise_images = params["num_noise_images"]
    else:
      self.num_noise_images = 100
    if "num_inference_images" in params.keys():
      self.num_inference_images = params["num_inference_images"]
    else:
      self.num_inference_images = 1

  def run_recon_analysis(self, full_image, save_info):
    wht_img, img_mean, ft_filter = dp.whiten_data(full_image,
      method=self.model_params["whiten_method"], lpf_cutoff=self.model_params["lpf_cutoff"])
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
    self.atcs = self.compute_atcs(self.evals["inference/activity:0"], images, self.atas)
    self.bf_stats = dp.get_dictionary_stats(self.evals["weights/phi:0"], padding=self.ft_padding,
      num_gauss_fits=self.num_gauss_fits, gauss_thresh=self.gauss_thresh)
    image_indices = np.random.choice(np.arange(images.shape[0]), self.num_inference_images,
      replace=False)
    self.inference_stats = self.evaluate_inference(images[image_indices, ...])
    np.savez(self.analysis_out_dir+"analysis_"+save_info+".npz",
      data={"run_stats":self.run_stats, "evals":self.evals, "atas":self.atas, "atcs":self.atcs,
      "inference_stats":self.inference_stats, "var_names":self.var_names, "bf_stats":self.bf_stats})
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
    super(LCA_Analyzer, self).run_analysis(images, save_info)
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
    self.atcs = analysis["atcs"]
    self.inference_stats = analysis["inference_stats"]
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

  def compute_time_varied_response(self, images, steps_per_image=None):
    """
    Converge LCA inference with stimulus that has a matched time constant,
    so that the frames change during inference.
    """
    if steps_per_image is None:
      steps_per_image = self.model_params["num_steps"]
    num_imgs, num_pixels = images.shape
    num_neurons = self.model_params["num_neurons"]
    u = np.zeros((int(num_imgs*steps_per_image), num_neurons), dtype=np.float32)
    a = np.zeros((int(num_imgs*steps_per_image), num_neurons), dtype=np.float32)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=self.model.graph) as sess:
      sess.run(self.model.init_op, self.model.get_feed_dict(images[0, None, ...]))
      self.model.load_weights(sess, self.cp_loc)
      inference_idx = 1 # first step of inference is zeros
      for img_idx in range(num_imgs):
        feed_dict = self.model.get_feed_dict(images[img_idx, None, ...])
        lca_b = sess.run(self.model.compute_excitatory_current(), feed_dict)
        lca_g = sess.run(self.model.compute_inhibitory_connectivity(), feed_dict)
        for step in range(inference_idx, int((img_idx+1)*steps_per_image)):
          current_u = u[inference_idx-1, :][None, ...]
          current_a = a[inference_idx-1, :][None, ...]
          lca_u_and_ga = sess.run(self.model.step_inference(current_u, current_a, lca_b,
            lca_g, step), feed_dict)
          lca_a = sess.run(self.model.threshold_units(lca_u_and_ga[0]), feed_dict)
          u[inference_idx, :] = lca_u_and_ga[0]
          a[inference_idx, :] = lca_a
          inference_idx += 1
    return a

  def evaluate_inference(self, images, num_inference_steps=None):
    """Evaluates inference on images, produces outputs over time"""
    if num_inference_steps is None:
      num_inference_steps = self.model_params["num_steps"]
    num_imgs, num_pixels = images.shape
    num_neurons = self.model_params["num_neurons"]
    loss_funcs = self.model.get_loss_funcs()
    b = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    ga = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    u = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    a = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    psnr = np.zeros((num_imgs, num_inference_steps), dtype=np.float32)
    losses = dict(zip([str(key) for key in loss_funcs.keys()],
      [np.zeros((num_imgs, num_inference_steps), dtype=np.float32)
      for _ in range(len(loss_funcs))]))
    total_loss = np.zeros((num_imgs, num_inference_steps), dtype=np.float32)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=self.model.graph) as sess:
      sess.run(self.model.init_op, self.model.get_feed_dict(images[0, None, ...]))
      self.model.load_weights(sess, self.cp_loc)
      for img_idx in range(num_imgs):
        feed_dict = self.model.get_feed_dict(images[img_idx, None, ...])
        lca_b = sess.run(self.model.compute_excitatory_current(), feed_dict)
        lca_g = sess.run(self.model.compute_inhibitory_connectivity(), feed_dict)
        for step in range(1, num_inference_steps):
          current_u = u[img_idx, step-1, :][None, ...]
          current_a = a[img_idx, step-1, :][None, ...]
          x_ = self.model.compute_recon(current_a)
          MSE = tf.reduce_mean(tf.square(tf.subtract(self.model.x, x_)), axis=[1, 0])
          img_var = tf.nn.moments(self.model.x, axes=[1])[1]
          pSNRdB = tf.multiply(10.0, tf.log(tf.divide(tf.square(img_var), MSE)))
          loss_list = [func(current_a) for func in loss_funcs.values()]
          run_list = [self.model.step_inference(current_u, current_a, lca_b, lca_g, step),
            self.model.compute_total_loss(current_a, loss_funcs), pSNRdB]+loss_list
          run_outputs = sess.run(run_list, feed_dict)
          [lca_u_and_ga, current_total_loss, current_psnr] = run_outputs[0:3]
          current_losses = run_outputs[3:]
          lca_a = sess.run(self.model.threshold_units(lca_u_and_ga[0]), feed_dict)
          b[img_idx, step, :] = lca_b
          u[img_idx, step, :] = lca_u_and_ga[0]
          ga[img_idx, step, :] = lca_u_and_ga[1]
          a[img_idx, step, :] = lca_a
          total_loss[img_idx, step] = current_total_loss
          psnr[img_idx, step] = current_psnr
          for idx, key in enumerate(loss_funcs.keys()):
              losses[key][img_idx, step] = current_losses[idx]
      losses["total"] = total_loss
    return {"b":b, "ga":ga, "u":u, "a":a, "psnr":psnr, "losses":losses, "images":images}
