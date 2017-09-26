import tensorflow as tf
import params.param_picker as pp
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
    """
    super(LCA_PCA_FB, self).load_params(params)
    self.num_pooling_units = int(params["num_pooling_units"])
    self.fb_mult = int(params["fb_mult"])

  def compute_feedback_loss(self, a_in):
    current_b = tf.matmul(a_in, self.eigen_vecs)
    #TODO:  verify eigen_vals is broadcasting properly - should be 1 x num_pooling_filters
    fb = tf.reduce_sum(tf.divide(current_b, self.eigen_vals), axis=1, name="feedback")
    fb_loss = tf.reduce_mean(tf.multiply(self.fb_mult, tf.square(fb)), axis=0,
      name="feedback_loss")
    return (fb, fb_loss)

  def step_inference(self, u_in, a_in, b, g):
    with tf.name_scope("update_u") as scope:
      lca_explain_away = tf.matmul(a_in, g, name="explaining_away")
      lca_fb = tf.gradients(self.compute_feedback_loss(a_in)[1], a_in)[0]
      du = tf.identity(b - lca_explain_away - u_in - lca_fb, name="du")
      u_out = tf.add(u_in, tf.multiply(self.eta, du))
    return (u_out, lca_explain_away)

  def infer_coefficients(self):
   lca_b = self.compute_excitatory_current()
   lca_g = self.compute_inhibitory_connectivity()
   u_list = [self.u_zeros]
   a_list = [self.threshold_units(u_list[0])]
   for step in range(self.num_steps-1):
     u, _ = self.step_inference(u_list[step], a_list[step], lca_b, lca_g)
     u_list.append(u)
     a_list.append(self.threshold_units(u_list[step+1]))
   return (u_list[-1], a_list[-1])

  def compute_total_loss(self, a_in):
      with tf.name_scope("unsupervised"):
        self.recon_loss = tf.reduce_mean(0.5 *
          tf.reduce_sum(tf.pow(tf.subtract(self.x, self.compute_recon(a_in)), 2.0),
          axis=[1]), name="recon_loss")
        self.sparse_loss = self.sparse_mult * tf.reduce_mean(
          tf.reduce_sum(tf.abs(a_in), axis=[1]), name="sparse_loss")
        self.feedback_loss = self.compute_feedback_loss(a_in)[1]
        self.unsupervised_loss = (self.recon_loss + self.sparse_loss + self.feedback_loss)
      total_loss = self.unsupervised_loss
      return total_loss

  def build_graph(self):
    super(LCA_PCA_FB, self).build_graph()
    with tf.name_scope("inference") as scope:
      self.b = tf.matmul(self.a, self.eigen_vecs)

  def print_update(self, input_data, input_labels=None, batch_step=0):
    """
    Log train progress information
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    feed_dict = self.get_feed_dict(input_data, input_labels)
    current_step = np.array(self.global_step.eval()).tolist()
    recon_loss = np.array(self.recon_loss.eval(feed_dict)).tolist()
    sparse_loss = np.array(self.sparse_loss.eval(feed_dict)).tolist()
    feedback_loss = np.array(self.feedback_loss.eval(feed_dict)).tolist()
    total_loss = np.array(self.total_loss.eval(feed_dict)).tolist()
    a_vals = tf.get_default_session().run(self.a, feed_dict)
    a_vals_max = np.array(a_vals.max()).tolist()
    a_frac_act = np.array(np.count_nonzero(a_vals)
      / float(self.batch_size * self.num_neurons)).tolist()
    b_vals = tf.get_default_session().run(self.b, feed_dict)
    b_vals_max = np.array(b_vals.max()).tolist()
    b_frac_act = np.array(np.count_nonzero(b_vals)
      / float(self.batch_size * self.num_neurons)).tolist()
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "schedule_index":self.sched_idx,
      "recon_loss":recon_loss,
      "sparse_loss":sparse_loss,
      "total_loss":total_loss,
      "a_max":a_vals_max,
      "a_fraction_active":a_frac_act,
      "b_max":b_vals_max,
      "b_fraction_active":b_frac_act}
    js_str = js.dumps(stat_dict, sort_keys=True, indent=2)
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
    pf.plot_data_tiled(input_data.reshape((self.batch_size,
      np.int(np.sqrt(self.num_pixels)), np.int(np.sqrt(self.num_pixels)))),
      normalize=False, title="Images at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"images_"+self.version+"-"+current_step.zfill(5)+".pdf"))
    pf.plot_data_tiled(recon.reshape((self.batch_size,
      np.int(np.sqrt(self.num_pixels)), np.int(np.sqrt(self.num_pixels)))),
      normalize=False, title="Recons at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"recons_v"+self.version+"-"+current_step.zfill(5)+".pdf"))
