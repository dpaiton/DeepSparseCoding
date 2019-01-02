import numpy as np
import os
import tensorflow as tf
import params.param_picker as pp
import utils.plot_functions as pf
from models.lca_pca_model import LcaPcaModel

class LcaPcaFbModel(LcaPcaModel):
  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(LcaPcaFbModel, self).load_params(params)
    self.act_cov_loc = (params.out_dir+self.cp_load_name+"/analysis/"+self.cp_load_ver
      +"/act_cov_"+self.act_cov_suffix+".npz")
    assert os.path.exists(self.act_cov_loc), ("Can't find activity covariance file. "
      +"Maybe you didn't run the analysis? File location: "+self.act_cov_loc)

  def get_feed_dict(self, input_data, input_labels=None, dict_args=None):
    """
    Return dictionary containing all placeholders
    Inputs:
      input_data: data to be placed in self.x
      input_labels: label to be placed in self.y
      dict_args: optional dictionary to be appended to the automatically generated feed_dict
    """
    placeholders = [op.name
      for op
      in self.graph.get_operations()
      if ("auto_placeholders" in op.name
      and "full_covariance_matrix" not in op.name
      and "input_data" not in op.name
      and "input_label" not in op.name)]
    activity_covariance = np.load(self.act_cov_loc)["data"].item().get("act_cov")
    if input_labels is not None and hasattr(self, "y"):
      feed_dict = {self.full_cov:activity_covariance, self.x:input_data, self.y:input_labels}
    else:
      feed_dict = {self.full_cov:activity_covariance, self.x:input_data}
    for placeholder in placeholders:
      feed_dict[self.graph.get_tensor_by_name(placeholder+":0")] = (
        self.get_schedule(placeholder.split("/")[1]))
    return feed_dict

  def compute_feedback_loss(self, a_in):
    # TF produces SORTED eig vecs / vals -> indices do not match up with a values?
    #  for this inner product...
    #  eigen_vecs is square, of course, since act_cov is square (neurons x neurons)
    #  it is also still inputs x outputs
    #  vals & vecs are sorted the same, but does this sort outputs or inputs? Presumably outputs...
    #TODO:  verify eigen_vals is broadcasting properly - should be 1 x num_pooling_filters
    # how to better normalize the fb strength?
    # It seems as though TF is producing normalized eigenvalues, need to compensate for that?
    #   can denormalize by comparing sum of eig vals to trace of cov matrix?
    #     data proj onto first PC should = first eig val
    #     so multiply all eigvals by data on first PC & get unnormalized
    #     not clear that we actually need to denormalize, though
    # Some eigenvals are very close to 0 - not good for division, results in massive loss.
    # Probably need to truncate the eigenvalues - minimum is close to zero (actually zero somehow?)

    current_b = tf.matmul(a_in, self.eigen_vecs, name="eigen_act") #columns (last dim) index vecs
    #current_b = tf.matmul(a_in, self.pooling_filters, name="pooled_act")

    #masked_vals = tf.where(tf.greater(self.eigen_vals, 1e-2), self.eigen_vals,
    #  tf.multiply(1e-2, tf.ones_like(self.eigen_vals)))
    #fb_loss = tf.multiply(self.fb_mult,
    #  tf.reduce_sum(tf.square(tf.divide(current_b, masked_vals)), axis=1),
    #  name="feedback")

    #fb_loss = tf.multiply(self.fb_mult,
    #  tf.reduce_sum(tf.square(tf.divide(current_b, self.eigen_vals)), axis=1), name="feedback")

    fb_loss = tf.multiply(self.fb_mult, tf.reduce_sum(tf.square(tf.matmul(tf.matmul(current_b,
      tf.transpose(self.eigen_vecs)), self.inv_sigma)), axis=1), name="feedback")

    return fb_loss

  def compute_inference_feedback(self, a_in):
    #lca_fb_grad = tf.gradients(self.compute_feedback_loss(a_in), a_in)[0]
    # TODO: Figure out why it would be zero at all?
    #  None probably comes from compute_feedback_loss. Should be asserting that none are None
    #lca_fb = lca_fb_grad if lca_fb_grad is not None else tf.zeros_like(a_in)
    lca_fb = tf.gradients(self.compute_feedback_loss(a_in), a_in)[0]
    return lca_fb

  def step_inference(self, u_in, a_in, b, g, step):
    with tf.name_scope("update_u"+str(step)) as scope:
      lca_explain_away = tf.matmul(a_in, g, name="explaining_away")
      lca_fb = self.compute_inference_feedback(a_in)
      du = tf.identity(b - lca_explain_away - u_in - lca_fb, name="du")
      u_out = tf.add(u_in, tf.multiply(self.eta, du))
    return (u_out, lca_explain_away, lca_fb)

  def infer_coefficients(self):
   lca_b = self.compute_excitatory_current()
   lca_g = self.compute_inhibitory_connectivity()
   u_list = [self.u_zeros]
   a_list = [self.threshold_units(u_list[0])]
   for step in range(self.num_steps-1):
     inf_out = self.step_inference(u_list[step], a_list[step], lca_b, lca_g, step)
     u_list.append(inf_out[0])
     a_list.append(self.threshold_units(u_list[step+1]))
   return (u_list, a_list)

  def compute_mean_feedback_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      feedback_loss = tf.reduce_mean(self.compute_feedback_loss(a_in), axis=0,
        name="feedback_loss")
    return feedback_loss

  def get_loss_funcs(self):
    return {"recon_loss":self.compute_recon_loss, "sparse_loss":self.compute_sparse_loss,
        "feedback_loss":self.compute_mean_feedback_loss}

  def build_graph(self):
    with self.graph.as_default():
      with tf.name_scope("auto_placeholders") as scope:
        self.fb_mult = tf.placeholder(tf.float32, shape=(), name="fb_mult")
    super(LcaPcaFbModel, self).build_graph()
