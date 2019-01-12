import os
import numpy as np
import tensorflow as tf
from data.dataset import Dataset
import utils.data_processing as dp
from analysis.lca_analyzer import LcaAnalyzer

class LcaPcaAnalyzer(LcaAnalyzer):
  def run_analysis(self, images, save_info=""):
    super(LcaPcaAnalyzer, self).run_analysis(images, save_info=save_info)
    # Need to create a dataset object for cov analysis
    image_dataset = {"test":Dataset(dp.reshape_data(images, flatten=False)[0], lbls=None)}
    image_dataset = self.model.reshape_dataset(image_dataset, self.model_params)
    cov = self.cov_analysis(image_dataset["test"][:self.analysis_params.num_LCA_PCA_cov_images],
      save_info)
    self.act_cov, self.a_eigvals, self.a_eigvecs, self.pooling_filters, self.a2, self.pooled_act = cov
    self.evec_atas = self.compute_atas(self.a2, image_dataset["test"].images)
    self.pool_atas = self.compute_atas(self.pooled_act, image_dataset["test"].images)
    np.savez(self.analysis_out_dir+"savefiles/second_layer_atas_"+save_info+".npz",
      data={"evec_atas":self.evec_atas, "pool_atas":self.pool_atas})
    self.analysis_logger.log_info("2nd layer activity  analysis is complete.")

  def load_analysis(self, save_info=""):
    super(LcaPcaAnalyzer, self).load_analysis(save_info)
    # pca filters
    pca_file_loc = self.analysis_out_dir+"pca_"+save_info+".npz"
    pca_analysis = np.load(pca_file_loc)["data"].item()
    self.act_cov = pca_analysis["act_cov"]
    self.a_eigvals = pca_analysis["a_eigvals"]
    self.a_eigvecs = pca_analysis["a_eigvecs"]
    self.pooling_filters = pca_analysis["pooling_filters"]
    self.a2 = pca_analysis["a2"]
    self.pooled_act = pca_analysis["pooled_act"]
    # activity triggered averages with 2nd layer units
    ata_file_loc = self.analysis_out_dir+"second_layers_atas_"+save_info+".npz"
    if os.path.exists(ata_file_loc):
      ata_analysis = np.load(ata_file_loc)["data"].item()
      self.evec_atas = ata_analysis["evec_atas"]
      self.pool_atas = ata_analysis["pool_atas"]

  def compute_pooled_activations(self, images, cov):
    """
    Computes the 2nd layer output code for a set of images.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=self.model.graph) as sess:
      feed_dict = self.model.get_feed_dict(images)
      feed_dict[self.model.full_cov] = cov
      sess.run(self.model.init_op, feed_dict)
      self.model.load_full_model(sess, self.analysis_params.cp_loc)
      activations = sess.run(self.model.pooled_activity, feed_dict)
    return activations

  def cov_analysis(self, dataset, save_info=""):
    with tf.Session(graph=self.model.graph) as sess:
      sess.run(self.model.init_op,
        feed_dict={self.model.x:np.zeros([self.model_params.batch_size,
        self.model_params.num_pixels], dtype=np.float32)})
      self.model.load_full_model(sess, self.analysis_params.cp_loc)
      act_cov = None
      num_cov_in_avg = 0
      while dataset.curr_epoch_idx + self.model_params.batch_size < dataset.num_examples:
        if act_cov is not None and num_cov_in_avg == 10: # prevent numerical overflow
          act_cov /= num_cov_in_avg
          num_cov_in_avg = 0
        input_data = dataset.next_batch(self.model_params.batch_size)[0]
        feed_dict = self.model.get_feed_dict(input_data)
        if act_cov is None:
          act_cov = sess.run(self.model.act_cov, feed_dict)
        else:
          act_cov = act_cov + sess.run(self.model.act_cov, feed_dict)
        num_cov_in_avg += 1
      act_cov = act_cov / num_cov_in_avg
    with tf.Session(graph=self.model.graph) as sess:
      sess.run(self.model.init_op, feed_dict={self.model.x:dataset.images})
      self.model.load_full_model(sess, self.analysis_params.analysis_params.cp_loc)
      feed_dict = self.model.get_feed_dict(dataset.images)
      feed_dict[self.model.full_cov] = act_cov
      run_list = [self.model.eigen_vals, self.model.eigen_vecs, self.model.pooling_filters,
        self.model.a2, self.model.pooled_activity]
      a_eigvals, a_eigvecs, pooling_filters, a2, pooled_act = sess.run(run_list, feed_dict)
    np.savez(self.analysis_out_dir+"savefiles/pca_"+save_info+".npz",
      data={"act_cov":act_cov, "a_eigvals":a_eigvals, "a_eigvecs":a_eigvecs,
      "pooling_filters":pooling_filters, "a2":a2, "pooled_act":pooled_act})
    self.analysis_logger.log_info("PCA analysis is complete.")
    return (act_cov, a_eigvals, a_eigvecs, pooling_filters, a2, pooled_act)
