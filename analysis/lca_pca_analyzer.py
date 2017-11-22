import numpy as np
import tensorflow as tf
from analysis.lca_analyzer import LCA
import utils.data_processing as dp
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
    super(LCA_PCA, self).run_analysis(images, save_info)
    self.cov = self.analyze_cov(images)
    self.evec_atas = self.compute_atas(self.cov["a2"], images)
    self.pool_atas = self.compute_atas(self.cov["pooled_act"], images)
    self.bf_stats = dp.get_dictionary_stats(self.evals["weights/phi:0"], padding=self.ft_padding,
      num_gauss_fits=self.num_gauss_fits, gauss_thresh=self.gauss_thresh)
    np.savez(self.analysis_out_dir+"pca_analysis_"+save_info+".npz",
      data={"evec_atas":self.evec_atas, "pool_atas":self.pool_atas, "act_cov":self.cov,
      "bf_stats":self.bf_stats})

  def load_analysis(self, save_info=""):
    super(LCA_PCA, self).load_analysis(save_info)
    pca_file_loc = self.analysis_out_dir+"pca_analysis_"+save_info+".npz"
    pca_analysis = np.load(pca_file_loc)["data"].item()
    self.evec_atas = pca_analysis["evec_atas"]
    self.pool_atas = pca_analysis["pool_atas"]
    self.cov = pca_analysis["act_cov"]
    self.bf_stats = pca_analysis["bf_stats"]

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
        self.model.a2, self.model.pooled_activity]
      a_eigvals, a_eigvecs, pooling_filters, a2, pooled_act = sess.run(run_list, feed_dict)
    return {"act_cov": act_cov, "a_eigvals": a_eigvals, "a_eigvecs":a_eigvecs,
      "pooling_filters": pooling_filters, "a2":a2, "pooled_act":pooled_act}
