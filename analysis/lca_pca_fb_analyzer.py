import numpy as np
import tensorflow as tf
from analysis.lca_pca_analyzer import LCA_PCA_Analyzer
import utils.data_processing as dp
import utils.notebook as nb

class LCA_PCA_FB_Analyzer(LCA_PCA_Analyzer):
  def __init__(self, params):
    super(LCA_PCA_FB_Analyzer, self).__init__(params)

  def evaluate_inference(self, images, num_inference_steps=None):
    if num_inference_steps is None:
      num_inference_steps = self.model_params.num_steps
    num_imgs, num_pixels = images.shape
    num_neurons = self.model_params.num_neurons
    num_pooling_units = self.model_params.num_pooling_units
    loss_funcs = self.model.get_loss_funcs()
    b = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    ga = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    u = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    a = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    fb = np.zeros((num_imgs, num_inference_steps, num_neurons), dtype=np.float32)
    a2 = np.zeros((num_imgs, num_inference_steps, num_pooling_units), dtype=np.float32)
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
          loss_list = [func(current_a) for func in loss_funcs.values()]
          run_list = [self.model.step_inference(current_u, current_a, lca_b, lca_g, step),
            self.model.compute_total_loss(current_a, loss_funcs), pSNRdB, self.model.a2]+loss_list
          run_outputs = sess.run(run_list, feed_dict)
          [lca_u_ga_fb, current_total_loss, current_psnr, current_a2] = run_outputs[0:4]
          current_losses = run_outputs[4:]
          lca_a = sess.run(self.model.threshold_units(lca_u_ga_fb[0]), feed_dict)
          b[img_idx, step, :] = lca_b
          u[img_idx, step, :] = lca_u_ga_fb[0]
          ga[img_idx, step, :] = lca_u_ga_fb[1]
          fb[img_idx, step, :] = lca_u_ga_fb[2]
          a[img_idx, step, :] = lca_a
          a2[img_idx, step, :] = current_a2
          total_loss[img_idx, step] = current_total_loss
          psnr[img_idx, step] = current_psnr
          for idx, key in enumerate(loss_funcs.keys()):
              losses[key][img_idx, step] = current_losses[idx]
      losses["total"] = total_loss
    return {"b":b, "ga":ga, "u":u, "a":a, "fb":fb, "a2":a2, "psnr":psnr, "losses":losses,
      "images":images}

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
