import numpy as np
import tensorflow as tf
from analysis.base_analysis import Analyzer

class LCA(Analyzer):
  def __init__(self, params):
    Analyzer.__init__(self, params)

  def load_params(self, params):
    Analyzer.load_params(self, params)

  def run_analysis(self, images, save_info=""):
    self.run_stats = self.get_log_stats()
    self.var_names = [
      "weights/phi:0",
      "inference/u:0",
      "inference/activity:0",
      "output/image_estimate/reconstruction:0",
      "performance_metrics/reconstruction_quality/recon_quality:0"]
    self.evals = self.evaluate_model(images, self.var_names)
    self.atas = self.compute_atas(self.evals["inference/activity:0"], images)
    self.inference_stats = self.evaluate_inference(images[0:5])
    np.savez(self.analysis_out_dir+"analysis_"+save_info+".npz",
      data={"run_stats":self.run_stats, "evals":self.evals, "atas":self.atas,
      "inference_stats":self.inference_stats, "var_names":self.var_names})

  def load_analysis(self, save_info=""):
    file_loc = self.analysis_out_dir+"analysis_"+save_info+".npz"
    analysis = np.load(file_loc)["data"].item()
    self.var_names = analysis["var_names"]
    self.run_stats = analysis["run_stats"]
    self.evals = analysis["evals"]
    self.atas = analysis["atas"]
    self.inference_stats = analysis["inference_stats"]

  def evaluate_inference(self, images, num_inference_steps=None):
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
    with tf.Session(graph=self.model.graph) as sess:
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
          loss_list  = [func(current_a) for func in loss_funcs.values()]
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
