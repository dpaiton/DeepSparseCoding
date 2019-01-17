import os
import numpy as np
import tensorflow as tf
from analysis.base_analyzer import Analyzer
import utils.data_processing as dp
import utils.plot_functions as pf
import utils.entropy_functions as ef

class LcaAnalyzer(Analyzer):
  def __init__(self):
    super(LcaAnalyzer, self).__init__()
    self.var_names = [
      "weights/w:0",
      "inference/activity:0",
      "output/reconstruction:0",
      "performance_metrics/recon_quality:0"]

  def check_params(self):
    super(LcaAnalyzer, self).check_params()
    if hasattr(self.analysis_params, "do_inference"):
      if not hasattr(self.analysis_params, "num_inference_steps"):
        self.analysis_params.num_inference_steps = None
      if not hasattr(self.analysis_params, "num_inference_images"):
        self.analysis_params.num_inference_images = 1
    else:
      self.analysis_params.do_inference = False
      self.analysis_params.num_inference_images = None
      self.analysis_params.num_inference_steps = None

  def run_analysis(self, images, labels=None, save_info=""):
    super(LcaAnalyzer, self).run_analysis(images, labels, save_info=save_info)
    if self.analysis_params.do_evals:
      self.evals = self.eval_analysis(images, self.var_names, save_info)
    if self.analysis_params.do_basis_analysis:
      self.bf_stats = self.basis_analysis(self.evals["weights/w:0"], save_info)
    if self.analysis_params.do_atas:
      self.atas, self.atcs = self.ata_analysis(images[:int(self.analysis_params.num_ata_images), ...],
        self.evals["inference/activity:0"], save_info)
      self.noise_activity, self.noise_atas, self.noise_atcs = self.run_noise_analysis(save_info)
    if self.analysis_params.do_inference:
      self.inference_stats = self.inference_analysis(images, save_info,
        self.analysis_params.num_inference_images, self.analysis_params.num_inference_steps)
    if self.analysis_params.do_recon_adversaries:
      self.adversarial_images, self.adversarial_recons, mses = self.recon_adversary_analysis(images,
        labels=labels, batch_size=self.analysis_params.adversarial_batch_size,
        input_id=self.analysis_params.adversarial_input_id,
        target_method=self.analysis_params.adversarial_target_method,
        target_id=self.analysis_params.adversarial_target_id,
        step_size=self.analysis_params.adversarial_step_size,
        num_steps=self.analysis_params.adversarial_num_steps,
        save_info=save_info)
      self.adversarial_input_target_mses = mses["input_target_mse"]
      self.adversarial_input_recon_mses = mses["input_recon_mses"]
      self.adversarial_input_adv_mses = mses["input_adv_mses"]
      self.adversarial_target_recon_mses = mses["target_recon_mses"]
      self.adversarial_target_adv_mses = mses["target_adv_mses"]
      self.adversarial_adv_recon_mses = mses["adv_recon_mses"]
    if self.analysis_params.do_orientation_analysis:
      self.ot_grating_responses, self.co_grating_responses = self.grating_analysis(self.bf_stats,
        save_info)

  def load_analysis(self, save_info=""):
    super(LcaAnalyzer, self).load_analysis(save_info)
    # Inference analysis
    inference_file_loc = self.analysis_out_dir+"savefiles/inference_"+save_info+".npz"
    if os.path.exists(inference_file_loc):
      self.inference_stats = np.load(inference_file_loc)["data"].item()["inference_stats"]

  def compute_time_varied_response(self, images, steps_per_image=None):
    """
    Converge LCA inference with stimulus that has a matched time constant,
    so that the frames change during inference.
    """
    if steps_per_image is None:
      steps_per_image = self.model_params.num_steps
    num_imgs = images.shape[0]
    num_neurons = self.model_params.num_neurons
    u = np.zeros((int(num_imgs*steps_per_image), num_neurons), dtype=np.float32) # membrane potential
    a = np.zeros((int(num_imgs*steps_per_image), num_neurons), dtype=np.float32) # output activity
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=self.model.graph) as sess:
      sess.run(self.model.init_op, self.model.get_feed_dict(images[0, None, ...]))
      self.model.load_full_model(sess, self.analysis_params.cp_loc)
      inference_idx = 1 # first step of inference is zeros
      for img_idx in range(num_imgs):
        feed_dict = self.model.get_feed_dict(images[img_idx, None, ...])
        lca_b = sess.run(self.model.module.compute_excitatory_current(), feed_dict)
        lca_g = sess.run(self.model.module.compute_inhibitory_connectivity(), feed_dict)
        for step in range(inference_idx, int((img_idx+1)*steps_per_image)):
          current_u = u[inference_idx-1, :][None, ...]
          current_a = a[inference_idx-1, :][None, ...]
          lca_u_and_ga = sess.run(self.model.module.step_inference(current_u, current_a, lca_b,
            lca_g, step), feed_dict)
          lca_a = sess.run(self.model.module.threshold_units(lca_u_and_ga[0]), feed_dict)
          u[inference_idx, :] = lca_u_and_ga[0]
          a[inference_idx, :] = lca_a
          inference_idx += 1
    return a

  def inference_analysis(self, images, save_info, num_images=None, num_steps=None):
    if num_images is None:
      image_indices = np.arange(images.shape[0])
    else:
      image_indices = np.random.choice(np.arange(images.shape[0]),
        self.analysis_params.num_inference_images, replace=False)
    if num_steps is None:
      num_steps = self.model_params.num_steps # this is replicated in self.add_inference_ops_to_graph
    inference_stats = self.evaluate_inference(images[image_indices, ...], num_steps)
    np.savez(self.analysis_out_dir+"savefiles/inference_"+save_info+".npz",
      data={"inference_stats":inference_stats})
    self.analysis_logger.log_info("Inference analysis is complete.")
    return inference_stats

  def add_pre_init_ops_to_graph(self):
    super(LcaAnalyzer, self).add_pre_init_ops_to_graph()
    if self.analysis_params.do_inference:
      self.add_inference_ops_to_graph(self.analysis_params.num_inference_images,
        self.analysis_params.num_inference_steps)

  def add_inference_ops_to_graph(self, num_imgs=1, num_inference_steps=None):
    if num_inference_steps is None:
      num_inference_steps = self.model_params.num_steps # this is replicated in self.inference_analysis
    loss_funcs = self.model.module.get_loss_funcs()
    with tf.device(self.model_params.device):
      with self.model.graph.as_default():
        self.lca_b = self.model.module.compute_excitatory_current()
        self.lca_g = self.model.module.compute_inhibitory_connectivity()
        self.u_list = [self.model.module.u_zeros]
        self.a_list = [self.model.module.threshold_units(self.u_list[0])]
        self.ga_list = [self.model.module.u_zeros]
        self.psnr_list = [tf.constant(0.0, dtype=tf.float32)]
        self.loss_list = {}
        current_loss_list = [func(self.a_list[0]) for func in loss_funcs.values()]
        for index, key in enumerate(loss_funcs.keys()):
          self.loss_list[key] = [current_loss_list[index]]
        self.loss_list["total_loss"] = [self.model.module.compute_total_loss(self.a_list[0],
          loss_funcs)]
        for step in range(num_inference_steps-1):
          u, ga = self.model.module.step_inference(self.u_list[step], self.a_list[step],
            self.lca_b, self.lca_g, step)
          self.u_list.append(u)
          self.a_list.append(self.model.module.threshold_units(self.u_list[step+1]))
          self.ga_list.append(ga)
          current_loss_list = [func(self.a_list[-1]) for func in loss_funcs.values()]
          for index, key in enumerate(loss_funcs.keys()):
            self.loss_list[key].append(current_loss_list[index])
          self.loss_list["total_loss"].append(self.model.module.compute_total_loss(self.a_list[-1],
            loss_funcs))
          current_recon = self.model.module.compute_recon(self.a_list[-1])
          MSE = tf.reduce_mean(tf.square(tf.subtract(self.model.x, current_recon)))
          pixel_var = tf.nn.moments(self.model.x, axes=[1])[1]
          current_pSNRdB = tf.multiply(10.0, ef.safe_log(tf.divide(tf.square(pixel_var), MSE)))
          self.psnr_list.append(current_pSNRdB)

  def evaluate_inference(self, images, num_steps):
    num_imgs = images.shape[0]
    b = np.zeros([num_imgs, num_steps]+self.model.module.u_shape, dtype=np.float32)
    u = np.zeros([num_imgs, num_steps]+self.model.module.u_shape, dtype=np.float32)
    ga = np.zeros([num_imgs, num_steps]+self.model.module.u_shape, dtype=np.float32)
    a = np.zeros([num_imgs, num_steps]+self.model.module.u_shape, dtype=np.float32)
    psnr = np.zeros((num_imgs, num_steps), dtype=np.float32)
    sparse_mult = self.model.get_schedule(key="sparse_mult")
    losses = [{} for _ in range(num_imgs)]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=self.model.graph) as sess:
      sess.run(self.model.init_op, self.model.get_feed_dict(images[0, None, ...]))
      sess.graph.finalize() # Graph is read-only after this statement
      self.model.load_full_model(sess, self.analysis_params.cp_loc)
      for img_idx in range(num_imgs):
        self.analysis_logger.log_info("Inference analysis on image "+str(img_idx))
        feed_dict = self.model.get_feed_dict(images[img_idx, None, ...])
        run_list = [self.lca_b, self.u_list, self.a_list, self.ga_list, self.psnr_list,
          self.loss_list]
        evals = sess.run(run_list, feed_dict)
        b[img_idx, :] = evals[0]
        u[img_idx, ...] = np.stack(np.squeeze(evals[1]), axis=0)
        ga[img_idx, ...] = np.stack(np.squeeze(evals[2]), axis=0)
        a[img_idx, ...] = np.stack(np.squeeze(evals[3]), axis=0)
        psnr[img_idx, ...] = np.stack(np.squeeze(evals[4]), axis=0)
        losses[img_idx].update(evals[5])
    out_losses = dict.fromkeys(losses[0].keys())
    for key in losses[0].keys():
      out_losses[key] = np.array([losses[idx][key] for idx in range(len(losses))])
    return {"b":b, "ga":ga, "u":u, "a":a, "sparse_mult": sparse_mult, "psnr":psnr,
      "losses":out_losses, "images":images}
