import numpy as np
import json as js
import os
import tensorflow as tf
import params.param_picker as pp
import utils.plot_functions as pf
from models.lca_pca import LCA_PCA

class LCA_PCA_FB(LCA_PCA):
  def __init__(self, params, schedule):
    lca_params, lca_schedule = pp.get_params("lca")
    new_params = lca_params.copy()
    lca_pca_params, lca_pca_schedule = pp.get_params("lca_pca")
    new_params.update(lca_pca_params)
    new_params.update(params)
    super(LCA_PCA_FB, self).__init__(new_params, schedule)

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [dict] model parameters
    Modifiable Parameters:
      num_pooling_units [int] indicating the number of 2nd layer units
      act_cov_suffix [str] suffix appended to activity covariance matrix
    """
    super(LCA_PCA_FB, self).load_params(params)
    self.num_pooling_units = int(params["num_pooling_units"])
    self.act_cov_suffix = str(params["activity_covariance_suffix"])
    self.act_cov_loc = (params["out_dir"]+self.cp_load_name+"/analysis/"+self.cp_load_ver
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
      if ("placeholders" in op.name
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
        self.get_sched(placeholder.split("/")[1]))
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
      with tf.name_scope("placeholders") as scope:
        self.fb_mult = tf.placeholder(tf.float32, shape=(), name="fb_mult")
    super(LCA_PCA_FB, self).build_graph()

  def print_update(self, input_data, input_labels=None, batch_step=0):
    """
    Log train progress information
    Inputs:
      input_data: data object containing the current image batch
      activity_cov: covariance matrix of shape [num_neurons, num_neurons] for computing total loss
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    feed_dict = self.get_feed_dict(input_data, input_labels)
    current_step = np.array(self.global_step.eval())
    recon_loss = np.array(self.loss_dict["recon_loss"].eval(feed_dict))
    sparse_loss = np.array(self.loss_dict["sparse_loss"].eval(feed_dict))
    feedback_loss = np.array(self.loss_dict["feedback_loss"].eval(feed_dict))
    total_loss = np.array(self.total_loss.eval(feed_dict))
    a_vals = tf.get_default_session().run(self.a, feed_dict)
    a_vals_max = np.array(a_vals.max())
    a_frac_act = np.array(np.count_nonzero(a_vals)
      / float(self.batch_size * self.num_neurons))
    a2_vals = tf.get_default_session().run(self.a2, feed_dict)
    a2_vals_max = np.array(a2_vals.max())
    a2_frac_act = np.array(np.count_nonzero(a2_vals)
      / float(self.batch_size * self.num_neurons))
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "schedule_index":self.sched_idx,
      "recon_loss":recon_loss,
      "sparse_loss":sparse_loss,
      "feedback_loss":feedback_loss,
      "total_loss":total_loss,
      "a_max":a_vals_max,
      "a_fraction_active":a_frac_act,
      "a2_max":a2_vals_max,
      "a2_fraction_active":a2_frac_act}
    js_str = self.js_dumpstring(stat_dict)
    self.log_info("<stats>"+js_str+"</stats>")

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    feed_dict = self.get_feed_dict(input_data, input_labels)
    current_step = str(self.global_step.eval())
    recon = tf.get_default_session().run(self.x_, feed_dict)
    weights = tf.get_default_session().run(self.phi, feed_dict)
    pf.plot_data_tiled(weights.T.reshape(self.num_neurons,
      int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
      normalize=False, title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"phi_v"+self.version+"_"+current_step.zfill(5)+".png"))
    pf.plot_data_tiled(input_data.reshape((self.batch_size,
      np.int(np.sqrt(self.num_pixels)), np.int(np.sqrt(self.num_pixels)))),
      normalize=False, title="Images at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"images_"+self.version+"-"+current_step.zfill(5)+".png"))
    pf.plot_data_tiled(recon.reshape((self.batch_size,
      np.int(np.sqrt(self.num_pixels)), np.int(np.sqrt(self.num_pixels)))),
      normalize=False, title="Recons at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"recons_v"+self.version+"-"+current_step.zfill(5)+".png"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      pf.plot_data_tiled(grad.T.reshape(self.num_neurons,
        int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
        normalize=True, title="Gradient for phi at step "+current_step, vmin=None, vmax=None,
        save_filename=(self.disp_dir+"dphi_v"+self.version+"_"+current_step.zfill(5)+".pdf"))
