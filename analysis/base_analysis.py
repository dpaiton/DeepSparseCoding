import os
import numpy as np
import utils.log_parser as lp
import utils.plot_functions as pf
import models.model_picker as mp
import tensorflow as tf

class Analyzer(object):
  def __init__(self, params):
    self.log_file = (params["model_dir"]+"/logfiles/"+params["model_name"]
      +"_v"+params["version"]+".log")
    self.log_text = lp.load_file(self.log_file)
    self.model_params = lp.read_params(self.log_text)
    assert self.model_params["model_type"] == params["model_type"], (
      "Model type defined in log text must match model type given in params.")
    self.model_params["rand_state"] = np.random.RandomState(
      self.model_params["rand_seed"])
    self.model_schedule = lp.read_schedule(self.log_text)
    self.load_params(params)
    self.make_dirs()
    self.load_model()
    self.model.log_params(params)

  def load_params(self, params):
    """Load analysis parameters into object"""
    # Model details
    self.model_name = params["model_name"]
    self.version = params["version"]
    self.device = params["device"]
    self.analysis_out_dir = params["model_dir"]+"/analysis/"+self.version+"/"
    if "cp_load_step" in params.keys() and params["cp_load_step"] is not None:
      self.cp_load_step = params["cp_load_step"]
      self.cp_loc = (params["model_dir"]+"/checkpoints/"+params["model_name"]
        +"_v"+params["version"]+"_full-"+str(self.cp_load_step))
    else:
      self.cp_load_step = None
      self.cp_loc = tf.train.latest_checkpoint(params["model_dir"]+"/checkpoints/")
    self.model_params["model_out_dir"] = self.analysis_out_dir
    if "data_dir" in params.keys():
      self.model_params["data_dir"] = params["data_dir"]

  def make_dirs(self):
    """Make output directories"""
    if not os.path.exists(self.analysis_out_dir):
      os.makedirs(self.analysis_out_dir)

  def load_model(self):
    """Load model object into analysis object"""
    self.model = mp.get_model(self.model_params["model_type"],
      self.model_params, self.model_schedule)

  def get_log_stats(self):
    """Wrapper function for parsing the log statistics"""
    return lp.read_stats(self.log_text)

  def evaluate_model(self, images, var_names):
    feed_dict = self.model.get_feed_dict(images)
    with tf.Session(graph=self.model.graph) as sess:
      sess.run(self.model.init_op, feed_dict)
      self.model.load_weights(sess, self.cp_loc)
      tensors = [self.model.graph.get_tensor_by_name(name) for name in var_names]
      eval_list = sess.run(tensors, feed_dict)
    evals = dict(zip(var_names, eval_list))
    return evals

  def compute_atas(self, activities, images):
    """
    Returns activity triggered averages
    Outputs:
      atas [np.ndarray] of shape (num_pixels, num_neurons)
    Inputs:
      activities [np.ndarray] of shape (num_imgs, num_neurons)
      images [np.ndarray] of shape (num_imgs, num_img_pixels)
    """
    atas = np.dot(images.T, activities)
    avg_atas = atas / images.shape[1]
    return avg_atas
