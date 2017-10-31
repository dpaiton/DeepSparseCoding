import numpy as np
import tensorflow as tf
from analysis.lca_pca_analyzer import LCA_PCA
import utils.data_processing as dp
import utils.notebook as nb

class LCA_PCA_FB(LCA_PCA):
  def __init__(self, params):
    super(LCA_PCA_FB, self).__init__(params)

  def run_analysis(self, images, save_info=""):
    self.run_stats = self.get_log_stats()
    var_names = [
      "weights/phi:0",
      "inference/u:0",
      "inference/activity:0",
      "output/image_estimate/reconstruction:0",
      "performance_metrics/reconstruction_quality/recon_quality:0"]
    self.evals = self.evaluate_model(images, var_names)
    self.atas = self.compute_atas(self.evals["inference/activity:0"], images)
    #self.cov = self.analyze_cov(images)
    #self.evec_atas = self.compute_atas(self.cov["b"], images)
    #self.pool_atas = self.compute_atas(self.cov["pooled_act"], images)
    self.bf_stats = dp.get_dictionary_stats(self.evals["weights/phi:0"], padding=self.ft_padding,
      num_gauss_fits=self.num_gauss_fits, gauss_thresh=self.gauss_thresh)
    self.inference_stats = self.evaluate_inference(images[73:74])
    #np.savez(self.analysis_out_dir+"analysis_"+save_info+".npz",
    #  data={"run_stats":self.run_stats, "evals":self.evals, "atas":self.atas,
    #      "evec_atas":self.evec_atas, "pool_atas":self.pool_atas})
    #np.savez(self.analysis_out_dir+"act_cov_"+save_info+".npz", data=self.cov)
    #np.savez(self.analysis_out_dir+"bf_stats_"+save_info+".npz", data=self.bf_stats)
    #np.savez(self.analysis_out_dir+"inference_stats_"+save_info+".npz", data=self.inference_stats)

  def evaluate_model(self, images, var_names):
    feed_dict = self.model.get_feed_dict(images)
    with tf.Session(graph=self.model.graph) as sess:
      sess.run(self.model.init_op, feed_dict)
      self.model.load_weights(sess, self.cp_loc)
      tensors = [self.model.graph.get_tensor_by_name(name) for name in var_names]
      eval_list = sess.run(tensors, feed_dict)
    evals = dict(zip(var_names, eval_list))
    return evals

  def evaluate_inference(self, images, num_inference_steps=None):
    if num_inference_steps is None:
      num_inference_steps = self.model_params["num_steps"]
    num_imgs, num_pixels = images.shape
    num_neurons = self.model_params["num_neurons"]
    b = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    ga = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    fb = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    u = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    a = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    total_loss = np.zeros((num_imgs, num_inference_steps), dtype=np.float32)
    #psnr = np.zeros((num_imgs, num_inference_steps))
    #recon_loss = np.zeros((num_imgs, num_inference_steps))
    #sparse_loss = np.zeros((num_imgs, num_inference_steps))
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
          run_list = [self.model.step_inference(current_u, current_a, lca_b, lca_g, step),
            self.model.compute_total_loss(current_a)]
          [lca_inf_out, current_loss] = sess.run(run_list, feed_dict)
          lca_a = sess.run(self.model.threshold_units(lca_inf_out[0]), feed_dict)
          b[img_idx, step, :] = lca_b
          u[img_idx, step, :] = lca_inf_out[0]
          ga[img_idx, step, :] = lca_inf_out[1]
          fb[img_idx, step, :] = lca_inf_out[2]
          a[img_idx, step, :] = lca_a
          total_loss[img_idx, step] = current_loss
          #x_ = self.model.compute_recon(lca_a)
          #MSE = tf.reduce_mean(tf.pow(tf.subtract(self.model.x, x_), 2.0),
          #  axis=[1, 0], name="mean_squared_error")
          #pSNRdB = tf.multiply(10.0, tf.log(tf.divide(tf.pow(1.0,
          #   2.0), MSE)), name="recon_quality")
    return {"b":b, "ga":ga, "fb":fb, "u":u, "a":a, "total_loss":total_loss, "images":images}

  def analyze_cov(self, images):
    num_imgs, num_pixels = images.shape
    with tf.Session(graph=self.model.graph) as sess:
      sess.run(self.model.init_op,
        feed_dict={self.model.x:np.zeros([num_imgs, num_pixels], dtype=np.float32)})
      self.model.load_weights(sess, self.cp_loc)
      act_cov = None
      num_cov_in_avg = 0
      for cov_batch_idx  in nb.log_progress(range(0, self.cov_num_images, self.model.batch_size),
        every=1):
        input_data = images[cov_batch_idx:cov_batch_idx+self.model.batch_size, ...]
        feed_dict = self.model.get_feed_dict(input_data)
        if act_cov is None:
          act_cov = sess.run(self.model.act_cov, feed_dict)
        else:
          act_cov += sess.run(self.model.act_cov, feed_dict)
        num_cov_in_avg += 1
      act_cov /= num_cov_in_avg
      feed_dict = self.model.get_feed_dict(images)
      feed_dict[self.model.full_cov] = act_cov
      run_list = [self.model.eigen_vals, self.model.eigen_vecs, self.model.pooling_filters,
        self.model.b, self.model.pooled_activity]
      a_eigvals, a_eigvecs, pooling_filters, b, pooled_act = sess.run(run_list, feed_dict)
    return {"act_cov": act_cov, "a_eigvals": a_eigvals, "a_eigvecs":a_eigvecs,
      "pooling_filters": pooling_filters, "b":b, "pooled_act":pooled_act}
