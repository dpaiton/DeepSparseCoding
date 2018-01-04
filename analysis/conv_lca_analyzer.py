import numpy as np
import tensorflow as tf
from analysis.lca_analyzer import LCA

class CONV_LCA(LCA):
  def __init__(self, params):
    super(CONV_LCA, self).__init__(params)
    self.var_names = ["weights/phi:0"]

  def run_analysis(self, images, save_info=""):
    self.run_stats = self.get_log_stats()
    self.evals = self.evaluate_model(images, self.var_names)
    image_indices = np.random.choice(np.arange(images.shape[0]), self.num_inference_images,
      replace=False)
    self.inference_stats = self.evaluate_inference(images[image_indices, ...])
    np.savez(self.analysis_out_dir+"analysis_"+save_info+".npz",
      data={"run_stats":self.run_stats, "evals":self.evals, "inference_stats":self.inference_stats,
      "var_names":self.var_names})

  def load_analysis(self, save_info=""):
    file_loc = self.analysis_out_dir+"analysis_"+save_info+".npz"
    analysis = np.load(file_loc)["data"].item()
    self.var_names = analysis["var_names"]
    self.run_stats = analysis["run_stats"]
    self.evals = analysis["evals"]
    self.inference_stats = analysis["inference_stats"]

  def evaluate_inference(self, images, num_inference_steps=None):
    if num_inference_steps is None:
      num_inference_steps = self.model_params["num_steps"]
    num_imgs = images.shape[0]
    loss_funcs = self.model.get_loss_funcs()
    u = np.zeros([num_imgs, num_inference_steps]+self.model.u_shape, dtype=np.float32)
    a = np.zeros([num_imgs, num_inference_steps]+self.model.u_shape, dtype=np.float32)
    psnr = np.zeros((num_imgs, num_inference_steps), dtype=np.float32)
    losses = dict(zip([str(key) for key in loss_funcs.keys()],
      [np.zeros((num_imgs, num_inference_steps), dtype=np.float32)
      for _ in range(len(loss_funcs))]))
    total_loss = np.zeros((num_imgs, num_inference_steps), dtype=np.float32)
    with self.model.graph.as_default():
      u_list, a_list = self.model.infer_coefficients()
    with tf.Session(graph=self.model.graph) as sess:
      sess.run(self.model.init_op, self.model.get_feed_dict(images[0, None, ...]))
      self.model.load_weights(sess, self.cp_loc)
      for img_idx in range(num_imgs):
        feed_dict = self.model.get_feed_dict(images[img_idx, None, ...])
        u_vals, a_vals = sess.run([u_list, a_list], feed_dict)
        for step, vals in enumerate(zip(u_vals, a_vals)):
          u_val, a_val = vals
          x_ = self.model.compute_recon(a_val)
          reduce_axis = list(np.arange(images.ndim)[::-1]) # images should have same ndim as x_
          MSE = tf.reduce_mean(tf.square(tf.subtract(self.model.x, x_)), axis=reduce_axis)
          img_var = tf.nn.moments(self.model.x, axes=reduce_axis[:-1])[1]
          pSNRdB = tf.multiply(10.0, tf.log(tf.divide(tf.square(img_var), MSE)))
          loss_list = [func(a_val) for func in loss_funcs.values()]
          run_list = [self.model.compute_total_loss(a_val, loss_funcs), pSNRdB]+loss_list
          run_outputs = sess.run(run_list, feed_dict)
          [current_total_loss, current_psnr] = run_outputs[0:2]
          current_losses = run_outputs[2:]
          u[img_idx, step, ...] = u_val
          a[img_idx, step, ...] = a_val
          total_loss[img_idx, step] = current_total_loss
          psnr[img_idx, step] = current_psnr
          for idx, key in enumerate(loss_funcs.keys()):
              losses[key][img_idx, step] = current_losses[idx]
              if not np.all(np.isfinite(current_losses[idx])):
                  import IPython; IPython.embed(); raise SystemExit
      losses["total"] = total_loss
    return {"u":u, "a":a, "psnr":psnr, "losses":losses, "images":images}
