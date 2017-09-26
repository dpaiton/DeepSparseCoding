import tensorflow as tf
import numpy as np
from analysis.base_analysis import Analyzer

class lca(Analyzer):
  def __init__(self, params):
    Analyzer.__init__(self, params)

  def load_params(self, params):
    Analyzer.load_params(self, params)

  def run_analysis(self, images):
    self.run_stats = self.get_log_stats()
    self.evals = self.evaluate_model(images)
    self.atas = self.compute_atas(self.evals["weights/phi:0"],
      self.evals["inference/activity:0"], images)

  def evaluate_model(self, images):
    var_names = [
      "weights/phi:0",
      "inference/u:0",
      "inference/activity:0",
      "output/image_estimate/reconstruction:0",
      "performance_metrics/reconstruction_quality/recon_quality:0"]
    feed_dict = self.model.get_feed_dict(images)
    with tf.Session(graph=self.model.graph) as tmp_sess:
      tmp_sess.run(self.model.init_op, feed_dict)
      self.model.load_weights(tmp_sess, self.cp_loc)
      tensors = [self.model.graph.get_tensor_by_name(name) for name in var_names]
      eval_list = tmp_sess.run(tensors, feed_dict)
    evals = dict(zip(var_names, eval_list))
    return evals

  def compute_atas(self, weights, activities, images):
    """
    Returns activity triggered averages
    Outputs:
      atas [np.ndarray] of the same shape as 'weights' input
    Inputs:
      weights [np.ndarray] model weights of shape (num_img_pixels, num_neurons)
      activities [np.ndarray] of shape (num_imgs, num_neurons)
      images [np.ndarray] of shape (num_imgs, num_img_pixels)
    """
    num_imgs, num_neurons = activities.shape
    num_pixels = images.shape[1]
    atas = np.zeros((num_pixels, num_neurons))
    norm_activities = activities / np.max(activities, axis=0)[None, :] #max is across images
    for img_idx in range(num_imgs):
      for neuron_idx in range(num_neurons):
        if norm_activities[img_idx, neuron_idx] > 0:
          atas[:, neuron_idx] += norm_activities[img_idx, neuron_idx] * images[img_idx, :]
    avg_atas = atas / num_imgs
    return avg_atas

  def evaluate_inference(self, images, num_inference_steps=None):
    if num_inference_steps is None:
      num_inference_steps = self.model_params["num_steps"]
    num_imgs, num_pixels = images.shape
    num_neurons = self.model_params["num_neurons"]
    #out_shape = (num_imgs, num_inference_steps, num_neurons)
    b = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    ga = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    u = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    a = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    total_loss = np.zeros((num_imgs, num_inference_steps), dtype=np.float32)
    #psnr = np.zeros((num_imgs, num_inference_steps))
    #recon_loss = np.zeros((num_imgs, num_inference_steps))
    #sparse_loss = np.zeros((num_imgs, num_inference_steps))
    with tf.Session(graph=self.model.graph) as tmp_sess:
      tmp_sess.run(self.model.init_op, self.model.get_feed_dict(images[0, None, ...]))
      for img_idx in range(num_imgs):
        feed_dict = self.model.get_feed_dict(images[img_idx, None, ...])
        lca_b = tmp_sess.run(self.model.compute_excitatory_current(), feed_dict)
        lca_g = tmp_sess.run(self.model.compute_inhibitory_connectivity(), feed_dict)
        for step in range(1, num_inference_steps):
          current_u = u[img_idx, step-1, :][None, ...]
          current_a = a[img_idx, step-1, :][None, ...]
          run_list = [self.model.step_inference(current_u, current_a, lca_b, lca_g, step),
            self.model.compute_total_loss(current_a)]
          [lca_u_and_ga, current_loss] = tmp_sess.run(run_list, feed_dict)
          lca_a = tmp_sess.run(self.model.threshold_units(lca_u_and_ga[0]), feed_dict)
          b[img_idx, step, :] = lca_b
          u[img_idx, step, :] = lca_u_and_ga[0]
          ga[img_idx, step, :] = lca_u_and_ga[1]
          a[img_idx, step, :] = lca_a
          total_loss[img_idx, step] = current_loss
          #x_ = self.model.compute_recon(lca_a)
          #MSE = tf.reduce_mean(tf.pow(tf.subtract(self.model.x, x_), 2.0),
          #  axis=[1, 0], name="mean_squared_error")
          #pSNRdB = tf.multiply(10.0, tf.log(tf.divide(tf.pow(1.0,
          #   2.0), MSE)), name="recon_quality")
    return {"b":b, "ga":ga, "u":u, "a":a, "total_loss":total_loss, "images":images}
