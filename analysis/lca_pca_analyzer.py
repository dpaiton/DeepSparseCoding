import numpy as np
import tensorflow as tf
from analysis.lca_analyzer import LCA
import utils.image_processing as ip
import utils.notebook as nb

class LCA_PCA(LCA):
  def __init__(self, params):
    super(LCA_PCA, self).__init__(params)

  def load_params(self, params):
    super(LCA_PCA, self).load_params(params)
    if "rand_seed" in params.keys():
      self.rand_seed = params["rand_seed"]
      self.rand_state = np.random.RandomState(self.rand_seed)
    self.cov_num_images = params["cov_num_images"]
    self.ft_padding = params["ft_padding"]
    self.num_gauss_fits = 20
    self.gauss_thresh = 0.2

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
    self.cov = self.analyze_cov(images)
    self.evec_atas = self.compute_atas(self.cov["b"], images)
    self.pool_atas = self.compute_atas(self.cov["pooled_act"], images)
    self.bf_stats = ip.get_dictionary_stats(self.evals["weights/phi:0"], padding=self.ft_padding,
      num_gauss_fits=self.num_gauss_fits, gauss_thresh=self.gauss_thresh)
    self.inference_stats = self.evaluate_inference(images[0:1])
    np.savez(self.out_dir+"analysis_"+save_info+".npz",
      data={"run_stats":self.run_stats, "evals":self.evals, "atas":self.atas})
    np.savez(self.out_dir+"act_cov_"+save_info+".npz", data=self.cov)
    np.savez(self.out_dir+"bf_stats_"+save_info+".npz", data=self.bf_stats)
    np.savez(self.out_dir+"inference_stats_"+save_info+".npz", data=self.inference_stats)

  def load_analysis(self, save_info=""):
    file_loc = self.out_dir+"analysis_"+save_info+".npz"
    analysis = np.load(file_loc)["data"]
    self.run_stats = analysis.item().get("run_stats")
    self.evals = analysis.item().get("evals")
    self.atas = analysis.item().get("atas")
    #self.cov = np.load(self.out_dir+"act_cov_"+save_info+".npz")["data"]
    cov_items = np.load(self.out_dir+"act_cov_"+save_info+".npz")["data"]
    self.cov = {"act_cov":cov_items.item().get("act_cov"),
      "a_eigvals":cov_items.item().get("a_eigvals"),
      "a_eigvecs":cov_items.item().get("a_eigvecs"),
      "pooling_filters":cov_items.item().get("pooling_filters"),
      "b":cov_items.item().get("b"),
      "pooled_act":cov_items.item().get("pooled_act")}
    self.bf_stats = np.load(self.out_dir+"bf_stats_"+save_info+".npz")["data"]
    self.inference_stats = np.load(self.out_dir+"inference_stats_"+save_info+".npz")["data"]

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
