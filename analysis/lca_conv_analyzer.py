import numpy as np
import tensorflow as tf
import utils.entropy_functions as ef
from analysis.lca_analyzer import LcaAnalyzer

class LcaConvAnalyzer(LcaAnalyzer):
  def __init__(self):
    super(LcaConvAnalyzer, self).__init__()
    self.var_names = ["weights/phi:0"]

  def eval_analysis(self, images, var_names, save_info):
    evals = self.evaluate_model(images, var_names)
    evals["weights/phi:0"] = evals["weights/phi:0"].reshape(self.model.num_pixels,
      self.model.num_neurons)
    np.savez(self.analysis_out_dir+"savefiles/evals_"+save_info+".npz", data={"evals":evals})
    self.analysis_logger.log_info("Image analysis is complete.")
    return evals

  def add_inference_ops_to_graph(self, num_imgs, num_inference_steps=None):
    if num_inference_steps is None:
      num_inference_steps = self.model_params.num_steps
    loss_funcs = self.model.module.get_loss_funcs()
    with tf.device(self.model_params.device):
      with self.model.graph.as_default():
        self.u_list = [self.model.module.u_zeros]
        self.a_list = [self.model.module.threshold_units(self.u_list[0])]
        self.psnr_list = [tf.constant(0.0, dtype=tf.float32)]
        self.loss_list = {}
        current_loss_list = [func(self.a_list[0]) for func in loss_funcs.values()]
        for index, key in enumerate(loss_funcs.keys()):
          self.loss_list[key] = [current_loss_list[index]]
        self.loss_list["total_loss"] = [self.model.compute_total_loss(self.a_list[0], loss_funcs)]
        for step in range(num_inference_steps-1):
          u = self.model.module.step_inference(self.u_list[step], self.a_list[step], step)
          self.u_list.append(u)
          self.a_list.append(self.model.module.threshold_units(self.u_list[step+1]))
          current_loss_list = [func(self.a_list[-1]) for func in loss_funcs.values()]
          for index, key in enumerate(loss_funcs.keys()):
            self.loss_list[key].append(current_loss_list[index])
          self.loss_list["total_loss"].append(self.model.module.compute_total_loss(self.a_list[-1],
            loss_funcs))
          current_recon = self.model.module.compute_recon(self.a_list[-1])
          MSE = tf.reduce_mean(tf.square(tf.subtract(self.model.x, current_recon)))
          reduc_dim = list(range(1, len(self.model.x.shape)))
          pixel_var = tf.nn.moments(self.model.x, axes=reduc_dim)[1]
          pSNRdB = tf.multiply(10.0, ef.safe_log(tf.divide(tf.square(pixel_var), MSE)))
          self.psnr_list.append(pSNRdB)

  def evaluate_inference(self, images, num_steps):
    num_imgs = images.shape[0]
    u = np.zeros([num_imgs, num_steps]+self.model.module.u_shape, dtype=np.float32)
    a = np.zeros([num_imgs, num_steps]+self.model.module.u_shape, dtype=np.float32)
    psnr = np.zeros((num_imgs, num_steps), dtype=np.float32)
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
        run_list = [self.u_list, self.a_list, self.psnr_list, self.loss_list]
        evals = sess.run(run_list, feed_dict)
        u[img_idx, ...] = np.squeeze(np.stack(evals[0], axis=0))
        a[img_idx, ...] = np.squeeze(np.stack(evals[1], axis=0))
        psnr[img_idx, ...] = np.stack(np.squeeze(evals[2]), axis=0)
        losses[img_idx].update(evals[3])
    out_losses = dict.fromkeys(losses[0].keys())
    for key in losses[0].keys():
      out_losses[key] = np.array([losses[idx][key] for idx in range(len(losses))])
    return {"u":u, "a":a, "psnr":psnr, "losses":out_losses, "images":images}
