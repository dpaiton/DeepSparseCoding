import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_functions as ef
from models.base_model import Model

class SigmoidAutoencoderModel(Model):
  """
  Implementation of sparse autoencoder described in Andrew Ng's 2011 Stanford CS294A lecture notes
  Sigmoidal activation function
  Untied encoding & decoding weights
  Linear reconstructions - input images do not have 0-1 range
  """
  def __init__(self):
    super(SigmoidAutoencoderModel, self).__init__()
    self.vector_inputs = True

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [dict] model parameters
    Modifiable Parameters:
    """
    super(SigmoidAutoencoderModel, self).load_params(params)
    # Network Size
    self.num_pixels = int(np.prod(self.params.data_shape))
    self.w_enc_shape = [self.num_pixels, self.params.num_neurons]
    self.x_shape = [None, self.num_pixels]

  def activation(self, a_in, name=""):
    return tf.sigmoid(a_in, name=name)

  def compute_recon(self, a_in):
    return tf.add(tf.matmul(a_in, self.w_dec), self.b_dec, name="reconstruction")

  def compute_weight_decay_loss(self):
    with tf.name_scope("unsupervised"):
      w_enc_decay = tf.reduce_sum(tf.square(self.w_enc))
      w_dec_decay = tf.reduce_sum(tf.square(self.w_dec))
      decay_loss = tf.multiply(0.5*self.decay_mult, tf.add_n([w_enc_decay, w_dec_decay]))
    return decay_loss

  def compute_recon_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      reduc_dim = list(range(1, len(a_in.shape))) # Want to avg over batch, sum over the rest
      recon_loss = tf.reduce_mean(0.5 *
        tf.reduce_sum(tf.square(tf.subtract(self.x, self.compute_recon(a_in))),
        axis=reduc_dim), name="recon_loss")
    return recon_loss

  def compute_sparse_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      avg_act = tf.reduce_mean(a_in, axis=[0], name="batch_avg_activity")
      #p_dist = self.target_act * tf.subtract(ef.safe_log(self.target_act), ef.safe_log(avg_act), name="kl_p")
      #q_dist = (1-self.target_act) * tf.subtract(ef.safe_log(1-self.target_act), ef.safe_log(1-avg_act), name="kl_q")
      p_dist = self.target_act * tf.subtract(tf.log(self.target_act), tf.log(avg_act), name="kl_p")
      q_dist = (1-self.target_act) * tf.subtract(tf.log(1-self.target_act), tf.log(1-avg_act), name="kl_q")
      kl_divergence = tf.reduce_sum(tf.add(p_dist, q_dist), name="kld")
      sparse_loss = tf.multiply(self.sparse_mult, kl_divergence, name="sparse_loss")
    return sparse_loss

  def build_graph(self):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")
          self.sparse_mult = tf.placeholder(tf.float32, shape=(), name="sparse_mult")
          self.decay_mult = tf.placeholder(tf.float32, shape=(), name="decay_mult")
          self.target_act = tf.placeholder(tf.float32, shape=(), name="target_act")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.name_scope("weight_inits") as scope:
          w_init = tf.truncated_normal(self.w_enc_shape, mean=0.0,
            stddev=0.2, dtype=tf.float32, name="w_init")
          b_enc_init = tf.ones([1, self.params.num_neurons])*0.0001
          b_dec_init = tf.ones([1, self.num_pixels])*0.0001

        with tf.variable_scope("weights") as scope:
          self.w_enc = tf.get_variable(name="w_enc", dtype=tf.float32,
            initializer=w_init, trainable=True)
          self.w_dec = tf.get_variable(name="w_dec", dtype=tf.float32,
            initializer=tf.transpose(w_init), trainable=True)
          self.b_enc = tf.get_variable(name="b_enc", dtype=tf.float32,
            initializer=b_enc_init, trainable=True)
          self.b_dec = tf.get_variable(name="b_dec", dtype=tf.float32,
            initializer=b_dec_init, trainable=True)
          self.trainable_variables[self.w_enc.name] = self.w_enc
          self.trainable_variables[self.b_enc.name] = self.b_enc
          self.trainable_variables[self.w_dec.name] = self.w_dec
          self.trainable_variables[self.b_dec.name] = self.b_dec

        with tf.variable_scope("inference") as scope:
         self.a = self.activation(tf.add(tf.matmul(self.x, self.w_enc), self.b_enc),
           name="activity")

        with tf.name_scope("loss") as scope:
          self.loss_dict = {"recon_loss":self.compute_recon_loss(self.a),
            "sparse_loss":self.compute_sparse_loss(self.a),
            "weight_decay_loss":self.compute_weight_decay_loss()}
          self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")

        with tf.name_scope("output") as scope:
          self.reconstruction = self.compute_recon(self.a)

        with tf.name_scope("performance_metrics") as scope:
          with tf.name_scope("reconstruction_quality"):
            MSE = tf.reduce_mean(tf.square(tf.subtract(self.x, self.reconstruction)), axis=[1, 0],
              name="mean_squared_error")
            pixel_var = tf.nn.moments(self.x, axes=[1])[1]
            self.pSNRdB = tf.multiply(10.0, ef.safe_log(tf.divide(tf.square(pixel_var), MSE)),
              name="recon_quality")
    self.graph_built = True

  def get_total_loss(self):
    return self.total_loss

  def get_encodings(self):
    return self.a

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    update_dict = super(SigmoidAutoencoderModel, self).generate_update_dict(input_data,
      input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.loss_dict["recon_loss"], self.loss_dict["sparse_loss"],
      self.loss_dict["weight_decay_loss"], self.get_total_loss(), self.get_encodings(),
      self.reconstruction, self.learning_rates]
    grad_name_list = []
    for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
      eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name[1]]
      grad_name = weight_grad_var[0][1].name.split('/')[1].split(':')[0] #2nd is np.split
      grad_name_list.append(grad_name)
    out_vals =  tf.get_default_session().run(eval_list, feed_dict)
    current_step, recon_loss, sparse_loss, decay_loss, total_loss, a_vals, recon = out_vals[0:7]
    input_mean = np.mean(input_data)
    input_max = np.max(input_data)
    input_min = np.min(input_data)
    recon_mean = np.mean(recon)
    recon_max = np.max(recon)
    recon_min = np.min(recon)
    a_vals_max = np.array(a_vals.max())
    a_vals_min = np.array(a_vals.min())
    a_vals_mean = np.mean(a_vals)
    a_frac_act = np.array(np.count_nonzero(a_vals)
      / float(a_vals.size))
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "schedule_index":self.sched_idx,
      "recon_loss":recon_loss,
      "sparse_loss":sparse_loss,
      "total_loss":total_loss,
      "a_fraction_active":a_frac_act,
      "a_max_mean_min":[a_vals_max, a_vals_mean, a_vals_min],
      "x_max_mean_min":[input_max, input_mean, input_min],
      "x_hat_max_mean_min":[recon_max, recon_mean, recon_min]}
    lrs = out_vals[7]
    grads = out_vals[8:]
    for w_idx, (grad, name) in enumerate(zip(grads, grad_name_list)):
      grad_max = lrs[0][w_idx]*np.array(grad.max())
      grad_min = lrs[0][w_idx]*np.array(grad.min())
      grad_mean = lrs[0][w_idx]*np.mean(np.array(grad))
      stat_dict[name+"_lr"] = lrs[0][w_idx]
      stat_dict[name+"_grad_max_mean_min"] = [grad_max, grad_mean, grad_min]
    update_dict.update(stat_dict)
    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(SigmoidAutoencoderModel, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.w_enc, self.b_enc, self.w_dec, self.reconstruction,
      self.get_encodings()]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    w_enc, b_enc, w_dec, recon, activity = eval_out[1:]
    w_enc_norm = np.linalg.norm(w_enc, axis=0, keepdims=False)
    w_dec_norm = np.linalg.norm(w_dec, axis=1, keepdims=False)
    w_enc = dp.reshape_data(w_enc.T, flatten=False)[0]
    w_dec = dp.reshape_data(w_dec, flatten=False)[0]
    fig = pf.plot_data_tiled(w_enc, normalize=False,
      title="Encoding weights at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.params.disp_dir+"w_enc_v"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_data_tiled(w_dec, normalize=False,
      title="Decoding weights at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.params.disp_dir+"w_dec_v"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_activity_hist(b_enc, title="Encoding Bias Histogram",
      save_filename=(self.params.disp_dir+"b_enc_hist_v"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_activity_hist(activity, title="Activity Histogram",
      save_filename=(self.params.disp_dir+"act_hist_v"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_bar(w_enc_norm, num_xticks=5,
      title="w_enc l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
      save_filename=(self.params.disp_dir+"w_enc_norm_v"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_bar(w_dec_norm, num_xticks=5,
      title="w_dec l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
      save_filename=(self.params.disp_dir+"w_dec_norm_v"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
    if eval_out[0]*10 % self.params.cp_int == 0:
      fig = pf.plot_activity_hist(input_data, title="Image Histogram",
        save_filename=(self.params.disp_dir+"img_hist_"+self.params.version+"-"
        +current_step.zfill(5)+".png"))
      input_data = dp.reshape_data(input_data, flatten=False)[0]
      fig = pf.plot_data_tiled(input_data, normalize=False,
        title="Images at step "+current_step, vmin=None, vmax=None,
        save_filename=(self.params.disp_dir+"images_"+self.params.version+"-"
        +current_step.zfill(5)+".png"))
      recon = dp.reshape_data(recon, flatten=False)[0]
      fig = pf.plot_data_tiled(recon, normalize=False,
        title="Recons at step "+current_step, vmin=None, vmax=None,
        save_filename=(self.params.disp_dir+"recons_v"+self.params.version+"-"
        +current_step.zfill(5)+".png"))
      #for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      #  grad = weight_grad_var[0][0].eval(feed_dict)
      #  shape = grad.shape
      #  name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      #  grad = dp.reshape_data(grad.T, flatten=False)[0]
      #  fig = pf.plot_data_tiled(grad, normalize=True,
      #    title="Gradient for"+name+" at step "+current_step, vmin=None, vmax=None,
      #    save_filename=(self.params.disp_dir+"d"+name+"_v"+self.params.version+"-"
      #    +current_step.zfill(5)+".png"))
