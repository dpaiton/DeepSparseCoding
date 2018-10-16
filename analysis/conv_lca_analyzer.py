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

  def run_analysis(self, images, save_info=""):
    super(LCA_Analyzer, self).run_analysis(images, save_info)
    self.evals = self.eval_analysis(images, self.var_names, save_info)
    if self.do_basis_analysis:
      self.bf_stats = self.basis_analysis(self.evals["weights/phi:0"], save_info)
    if self.do_atas:
      self.atas, self.atcs = self.ata_analysis(images, self.evals["inference/activity:0"],
        save_info)
      self.noise_activity, self.noise_atas, self.noise_atcs = self.run_noise_analysis(save_info)
    if self.do_inference:
      self.inference_stats = self.inference_analysis(images, save_info)
    if self.do_adversaries:
      self.adversarial_losses, self.adversarial_recons, self.adversarial_images = self.adversary_analysis(images,
        input_id=0, target_id=25, eps=self.adversarial_eps, num_steps=self.adversarial_num_steps)

  def load_analysis(self, save_info=""):
    file_loc = self.analysis_out_dir+"analysis_"+save_info+".npz"
    analysis = np.load(file_loc)["data"].item()
    self.var_names = analysis["var_names"]
    self.run_stats = analysis["run_stats"]
    self.evals = analysis["evals"]
    self.inference_stats = analysis["inference_stats"]

  def evaluate_inference(self, images, num_inference_steps=None):
    """Evaluates inference on images, produces outputs over time"""
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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=self.model.graph) as sess:
      sess.run(self.model.init_op, self.model.get_feed_dict(images[0, None, ...]))
      sess.graph.finalize() # Graph is read-only after this statement
      self.model.load_weights(sess, self.cp_loc)
      for img_idx in range(num_imgs):
        self.analysis_logger.log_info("Inference analysis on image "+str(img_idx))
        feed_dict = self.model.get_feed_dict(images[img_idx, None, ...])
        for step in range(1, num_inference_steps):
          current_u = u[img_idx, step-1, ...][None, ...]
          current_a = a[img_idx, step-1, ...][None, ...]
          self.analysis_logger.log_info((
            "Inference analysis on step "+str(step)," of "+str(num_inference_steps)))
          loss_list = [func(current_a) for func in loss_funcs.values()]
          run_list = [self.model.step_inference(current_u, current_a, step),
            self.model.compute_total_loss(current_a, loss_funcs), self.model.pSNRdB]+loss_list
          run_outputs = sess.run(run_list, feed_dict)
          [lca_u, current_total_loss, current_psnr] = run_outputs[0:3]
          current_losses = run_outputs[3:]
          u[img_idx, step, ...] = lca_u
          a[img_idx, step, ...] = sess.run(self.model.threshold_units(lca_u), feed_dict)
          total_loss[img_idx, step] = current_total_loss
          psnr[img_idx, step] = current_psnr
          for idx, key in enumerate(loss_funcs.keys()):
              losses[key][img_idx, step] = current_losses[idx]
      losses["total"] = total_loss
    return {"u":u, "a":a, "psnr":psnr, "losses":losses, "images":images}
