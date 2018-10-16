import numpy as np
import tensorflow as tf
from analysis.lca_analyzer import LCA_Analyzer

class CONV_LCA_Analyzer(LCA_Analyzer):
  def __init__(self, params):
    super(CONV_LCA_Analyzer, self).__init__(params)
    self.var_names = ["weights/phi:0"]

  def eval_analysis(self, images, var_names, save_info):
    evals = self.evaluate_model(images, var_names)
    evals["weights/phi:0"] = evals["weights/phi:0"].reshape(self.model.num_pixels,
      self.model.num_neurons)
    np.savez(self.analysis_out_dir+"savefiles/evals_"+save_info+".npz", data={"evals":evals})
    self.analysis_logger.log_info("Image analysis is complete.")
    return evals

  def add_inference_ops_to_graph(self, num_imgs, num_inference_steps):
    loss_funcs = self.model.get_loss_funcs()
    losses = dict(zip([str(key) for key in loss_funcs.keys()],
      [np.zeros((num_imgs, num_inference_steps), dtype=np.float32)
      for _ in range(len(loss_funcs))]))
    with tf.device(self.model.device):
      with self.model.graph.as_default():
        self.u_list = [self.model.u_zeros]
        self.a_list = [self.model.threshold_units(self.u_list[0])]
        self.psnr_list = [tf.constant(0.0, dtype=tf.float32)]
        self.loss_list = {}
        current_loss_list = [func(self.a_list[0]) for func in loss_funcs.values()]
        for index, key in enumerate(loss_funcs.keys()):
          self.loss_list[key] = [current_loss_list[index]]
        self.loss_list["total_loss"] = [self.model.compute_total_loss(self.a_list[0], loss_funcs)]
        for step in range(num_inference_steps-1):
          u = self.model.step_inference(self.u_list[step], self.a_list[step], step)
          self.u_list.append(u)
          self.a_list.append(self.model.threshold_units(self.u_list[step+1]))
          loss_funcs = self.model.get_loss_funcs()
          current_loss_list = [func(self.a_list[-1]) for func in loss_funcs.values()]
          for index, key in enumerate(loss_funcs.keys()):
            self.loss_list[key].append(current_loss_list[index])
          self.loss_list["total_loss"].append(self.model.compute_total_loss(self.a_list[-1],
            loss_funcs))
          current_x_ = self.model.compute_recon(self.a_list[-1])
          MSE = tf.reduce_mean(tf.square(tf.subtract(self.model.x, current_x_)), axis=[1, 0])
          pixel_var = tf.nn.moments(self.model.x, axes=[1])[1]
          pSNRdB = tf.multiply(10.0, tf.log(tf.divide(tf.square(pixel_var), MSE)))
          self.psnr_list.append(pSNRdB)

  def evaluate_inference(self, images):
    num_imgs = images.shape[0]
    u = np.zeros([num_imgs, self.num_inference_steps]+self.model.u_shape, dtype=np.float32)
    a = np.zeros([num_imgs, self.num_inference_steps]+self.model.u_shape, dtype=np.float32)
    psnr = np.zeros((num_imgs, self.num_inference_steps), dtype=np.float32)
    losses = [{} for _ in range(num_imgs)]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=self.model.graph) as sess:
      sess.run(self.model.init_op, self.model.get_feed_dict(images[0, None, ...]))
      sess.graph.finalize() # Graph is read-only after this statement
      self.model.load_weights(sess, self.cp_loc)
      for img_idx in range(num_imgs):
        self.analysis_logger.log_info("Inference analysis on image "+str(img_idx))
        feed_dict = self.model.get_feed_dict(images[img_idx, None, ...])
        run_list = [self.u_list, self.a_list, self.psnr_list, self.loss_list]
        evals = sess.run(run_list, feed_dict)
        u[img_idx, ...] = np.stack(np.squeeze(evals[0]), axis=0)
        a[img_idx, ...] = np.stack(np.squeeze(evals[1]), axis=0)
        psnr[img_idx, ...] = np.stack(np.squeeze(evals[3]), axis=0)
        losses[img_idx].update(evals[4])
    out_losses = dict.fromkeys(losses[0].keys())
    for key in losses[0].keys():
      out_losses[key] = np.array([losses[idx][key] for idx in range(len(losses))])
    return {"u":u, "a":a, "psnr":psnr, "losses":out_losses, "images":images}
