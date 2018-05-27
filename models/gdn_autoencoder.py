import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_funcs as ef
from models.base_model import Model

class GDN_Autoencoder(Model):
  """
  Implementation of autoencoder described in Balle, Laparra, Simoncelli (2017)
  End-to-End Optimized Image Compression
  ## Key differences:
  #  Fully connected
  #  Single hidden layer, complete
  ## Methods ignored:
  #  add a small amount of uniform noise to input, to simulate pixel quantization
  """
  def __init__(self):
    super(GDN_Autoencoder, self).__init__()
    self.vector_inputs = True

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [dict] model parameters
    Modifiable Parameters:
    """
    super(GDN_Autoencoder, self).load_params(params)
    # Network Size
    self.data_shape = params["data_shape"]
    self.batch_size = int(params["batch_size"])
    self.num_pixels = int(np.prod(self.data_shape))
    self.num_neurons = int(params["num_neurons"])
    self.x_shape = [None, self.num_pixels]
    self.w_enc_shape = [self.num_pixels, self.num_neurons]
    self.b_enc_shape = [self.num_neurons]
    self.b_dec_shape = [self.num_pixels]
    self.gdn_w_shape = [self.num_neurons, self.num_neurons]
    self.gdn_b_shape = [1, self.num_neurons]
    # Hyper Parameters
    self.mle_step_size = float(params["mle_step_size"])
    self.num_mle_steps = int(params["num_mle_steps"])
    self.num_triangles = int(params["num_triangles"])
    self.sigmoid_beta = float(params["sigmoid_beta"])

  def sigmoid(self, a_in, beta=1):
    """Hyperbolic tangent non-linearity"""
    a_out = tf.subtract(tf.multiply(2.0, tf.divide(1.0,
      tf.add(1.0, tf.exp(tf.multiply(-beta, a_in))))), 1.0)
    return a_out

  def compute_entropies(self, a_in):
    with tf.name_scope("unsupervised"):
      a_sig = self.sigmoid(a_in, self.sigmoid_beta)
      a_probs = ef.prob_est(a_sig, self.mle_thetas, self.triangle_centers)
      a_entropies = tf.identity(ef.calc_entropy(a_probs), name="a_entropies")
    return a_entropies

  def compute_weight_decay_loss(self):
    with tf.name_scope("unsupervised"):
      decay_loss = tf.multiply(0.5*self.decay_mult,
        tf.add_n([tf.nn.l2_loss(self.w_enc), tf.nn.l2_loss(self.w_dec)]), name="weight_decay_loss")
    return decay_loss

  def compute_entropy_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      a_entropies = self.compute_entropies(a_in)
      entropy_loss = tf.multiply(self.ent_mult, tf.reduce_sum(a_entropies), name="entropy_loss")
    return entropy_loss

  def compute_recon_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      reduc_dim = list(range(1, len(a_in.shape))) # Want to avg over batch, sum over the rest
      x_ = self.compute_recon(a_in)
      recon_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(tf.subtract(self.x, x_)),
        axis=reduc_dim), name="recon_loss")
    return recon_loss

  def compute_recon(self, a_in):
    a_igdn = self.gdn(a_in, inverse=True, name="igdn")
    recon = tf.add(tf.matmul(a_igdn, self.w_dec), self.b_dec)
    return recon

  def compute_gdn_const(self, a_in, inverse, name=None):
    if inverse:
      name_prefix = "i"
    else:
      name_prefix = ""
    with tf.variable_scope(self.weight_scope, reuse=True) as scope:
      gdn_w = tf.get_variable(name=name_prefix+"gdn_w", shape=self.gdn_w_shape)
      gdn_b = tf.get_variable(name=name_prefix+"gdn_b", shape=self.gdn_b_shape)
    w_threshold = tf.where(tf.less(gdn_w, tf.constant(1e-6, dtype=tf.float32)),
      tf.multiply(1e-6, tf.ones_like(gdn_w)), gdn_w)
    b_threshold = tf.where(tf.less(gdn_b, tf.constant(1e-6, dtype=tf.float32)),
      tf.multiply(1e-6, tf.ones_like(gdn_b)), gdn_b)
    symmetric_weights = tf.multiply(0.5, tf.add(w_threshold, tf.transpose(w_threshold)))
    weighted_norm = tf.matmul(tf.square(a_in), symmetric_weights)
    GDN_const = tf.sqrt(tf.add(weighted_norm, tf.square(b_threshold)))
    return GDN_const

  def gdn(self, a_in, inverse, name=None):
    GDN_const = self.compute_gdn_const(a_in, inverse, name)
    if inverse:
      a_out = tf.multiply(a_in, GDN_const, name=name)
    else:
      a_out = tf.where(tf.less(GDN_const, tf.constant(1e-6, dtype=tf.float32)), a_in,
        tf.divide(a_in, GDN_const), name=name)
    return a_out

  def get_loss_funcs(self):
    return {"recon_loss":self.compute_recon_loss, "entropy_loss":self.compute_entropy_loss}

  def build_graph(self):
    """Build the TensorFlow graph object"""
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")
          self.triangle_centers = tf.placeholder(tf.float32, shape=[self.num_triangles],
            name="triangle_centers")
          self.ent_mult = tf.placeholder(tf.float32, shape=(), name="ent_mult")
          self.decay_mult = tf.placeholder(tf.float32, shape=(), name="decay_mult")
          self.noise_var_mult = tf.placeholder(tf.float32, shape=(), name="noise_var_mult")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("probability_estimate") as scope:
          self.mle_thetas, self.theta_init = ef.construct_thetas(self.num_neurons,
            self.num_triangles)

        with tf.name_scope("weight_inits") as scope:
          w_init = tf.truncated_normal(self.w_enc_shape, mean=0.0, stddev=0.1,
            dtype=tf.float32, name="w_init")

          gdn_w_init = tf.truncated_normal(self.gdn_w_shape, mean=0.0, stddev=0.01,
            dtype=tf.float32, name="gdn_w_init")
          #gdn_w_init = tf.truncated_normal(self.gdn_w_shape, mean=1.0, stddev=0.01,
          #  dtype=tf.float32, name="gdn_w_init")
          #gdn_w_init = tf.add(tf.eye(self.gdn_w_shape[0]), tf.truncated_normal(self.gdn_w_shape,
          #  mean=0.0, stddev=0.01), name="igdn_w_init")

          gdn_b_init = tf.truncated_normal(self.gdn_b_shape, mean=1.0, stddev=0.01,
            dtype=tf.float32, name="gdn_b_init")
          #gdn_b_init = tf.truncated_normal(self.gdn_b_shape, mean=0.0, stddev=0.1,
          #  dtype=tf.float32, name="gdn_b_init")

          #igdn_w_init = tf.add(tf.eye(self.gdn_w_shape[0]), tf.truncated_normal(self.gdn_w_shape,
          #  mean=0.0, stddev=0.01), name="igdn_w_init")
          igdn_w_init = tf.identity(tf.transpose(gdn_w_init), name="igdn_w_init")

          igdn_b_init = tf.identity(gdn_b_init, name="igdn_b_init")

          b_enc_init = tf.truncated_normal(self.b_enc_shape, mean=0.0, stddev=0.1,
            dtype=tf.float32, name="b_enc_init")
          b_dec_init = tf.truncated_normal(self.b_dec_shape, mean=0.0, stddev=0.1,
            dtype=tf.float32, name="b_dec_init")

        with tf.variable_scope("weights") as scope:
          self.weight_scope = tf.get_variable_scope()
          self.w_enc = tf.get_variable(name="w_enc", dtype=tf.float32,
            initializer=w_init, trainable=True)
          self.b_enc = tf.get_variable(name="b_enc", dtype=tf.float32,
            initializer=b_enc_init, trainable=True)
          self.gdn_w = tf.get_variable(name="gdn_w", dtype=tf.float32,
            initializer=gdn_w_init, trainable=True)
          self.gdn_b = tf.get_variable(name="gdn_b", dtype=tf.float32,
            initializer=gdn_b_init, trainable=True)
          self.igdn_w = tf.get_variable(name="igdn_w", dtype=tf.float32,
            initializer=igdn_w_init, trainable=True)
          self.igdn_b = tf.get_variable(name="igdn_b", dtype=tf.float32,
            initializer=igdn_b_init, trainable=True)
          self.w_dec = tf.get_variable(name="w_dec", dtype=tf.float32,
            initializer=tf.transpose(w_init), trainable=True)
          self.b_dec = tf.get_variable(name="b_dec", dtype=tf.float32,
            initializer=b_dec_init, trainable=True)

        with tf.variable_scope("inference") as scope:
          self.gdn_output = self.gdn(tf.add(tf.matmul(self.x, self.w_enc), self.b_enc),
            inverse=False, name="gdn_output")
          self.gdn_const = self.compute_gdn_const(self.gdn_output, inverse=False, name="gdn_const")
          self.gdn_entropies = self.compute_entropies(self.gdn_output)
          noise_var = tf.multiply(self.noise_var_mult, tf.subtract(tf.reduce_max(self.gdn_output),
            tf.reduce_min(self.gdn_output))) # 1/2 of 10% of range of gdn output
          uniform_noise = tf.random_uniform(shape=tf.stack(tf.shape(self.gdn_output)),
            minval=tf.subtract(0.0, noise_var), maxval=tf.add(0.0, noise_var))
          self.a = tf.add(uniform_noise, self.gdn_output, name="activity")

        with tf.variable_scope("probability_estimate") as scope:
          self.a_sig = self.sigmoid(self.a, self.sigmoid_beta)
          ll = ef.log_likelihood(self.a_sig, self.mle_thetas, self.triangle_centers)
          self.mle_update = [ef.mle(ll, self.mle_thetas, self.mle_step_size)
            for _ in range(self.num_mle_steps)]

        with tf.name_scope("loss") as scope:
          self.loss_dict = {"recon_loss":self.compute_recon_loss(self.a),
            "entropy_loss":self.compute_entropy_loss(self.a),
            "weight_decay_loss":self.compute_weight_decay_loss()}
          self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")

        with tf.name_scope("output") as scope:
          with tf.name_scope("image_estimate"):
            self.x_ = self.compute_recon(self.a)

        with tf.name_scope("performance_metrics") as scope:
          with tf.name_scope("reconstruction_quality"):
            MSE = tf.reduce_mean(tf.square(tf.subtract(self.x, self.x_)), axis=[1, 0],
              name="mean_squared_error")
            pixel_var = tf.nn.moments(self.x, axes=[1])[1]
            self.pSNRdB = tf.multiply(10.0, tf.log(tf.divide(tf.square(pixel_var), MSE)),
              name="recon_quality")
    self.graph_built = True

  def add_optimizers_to_graph(self):
    """
    Same as base class, except the weight scope is a member variable instead of "weights"
    """
    with self.graph.as_default():
      with tf.name_scope("optimizers") as scope:
        self.grads_and_vars = list() # [sch_idx][weight_idx]
        self.apply_grads = list() # [sch_idx][weight_idx]
        for schedule_idx, sch in enumerate(self.sched):
          sch_grads_and_vars = list() # [weight_idx]
          sch_apply_grads = list() # [weight_idx]
          for w_idx, weight in enumerate(sch["weights"]):
            learning_rates = tf.train.exponential_decay(
              learning_rate=sch["weight_lr"][w_idx],
              global_step=self.global_step,
              decay_steps=sch["decay_steps"][w_idx],
              decay_rate=sch["decay_rate"][w_idx],
              staircase=sch["staircase"][w_idx],
              name="annealing_schedule_"+weight)
            if self.optimizer == "annealed_sgd":
              optimizer = tf.train.GradientDescentOptimizer(learning_rates,
                name="grad_optimizer_"+weight)
            elif self.optimizer == "adam":
              optimizer = tf.train.AdamOptimizer(learning_rates, beta1=0.9, beta2=0.99,
                epsilon=1e-07, name="adam_optimizer_"+weight)
            elif self.optimizer == "adadelta":
              optimizer = tf.train.AdadeltaOptimizer(learning_rates, epsilon=1e-07,
                name="adadelta_optimizer_"+weight)
            with tf.variable_scope(self.weight_scope, reuse=True) as scope:
              weight_op = [tf.get_variable(weight)]
            sch_grads_and_vars.append(self.compute_weight_gradients(optimizer, weight_op))
            gstep = self.global_step if w_idx == 0 else None # Only update once
            sch_apply_grads.append(optimizer.apply_gradients(sch_grads_and_vars[w_idx],
              global_step=gstep))
          self.grads_and_vars.append(sch_grads_and_vars)
          self.apply_grads.append(sch_apply_grads)
    self.optimizers_added = True

  def print_update(self, input_data, input_labels=None, batch_step=0):
    """
    Log train progress information
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.loss_dict["recon_loss"], self.loss_dict["entropy_loss"],
      self.total_loss, self.a, self.x_]
    init_eval_length = len(eval_list)
    grad_name_list = []
    learning_rate_dict = {}
    for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
      eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name[1]]
      grad_name = weight_grad_var[0][1].name.split('/')[1].split(':')[0] #2nd is np.split
      grad_name_list.append(grad_name)
      learning_rate_dict[grad_name] = self.get_schedule("weight_lr")[w_idx]
    out_vals =  tf.get_default_session().run(eval_list, feed_dict)
    current_step, recon_loss, entropy_loss, total_loss, a_vals, recon = out_vals[:init_eval_length]
    input_mean = np.mean(input_data)
    input_max = np.max(input_data)
    input_min = np.min(input_data)
    recon_mean = np.mean(recon)
    recon_max = np.max(recon)
    recon_min = np.min(recon)
    a_vals_max = np.array(a_vals.max())
    a_vals_min = np.array(a_vals.min())
    a_frac_act = np.array(np.count_nonzero(a_vals)
      / float(a_vals.size))
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "schedule_index":self.sched_idx,
      "recon_loss":recon_loss,
      "entropy_loss":entropy_loss,
      "total_loss":total_loss,
      "a_max":a_vals_max,
      "a_min":a_vals_min,
      "a_fraction_active":a_frac_act,
      "x_max_mean_min":[input_max, input_mean, input_min],
      "x_hat_max_mean_min":[recon_max, recon_mean, recon_min]}
    grads = out_vals[init_eval_length:]
    for grad, name in zip(grads, grad_name_list):
      stat_dict[name+"_max_grad"] = learning_rate_dict[name]*np.array(grad.max())
      stat_dict[name+"_min_grad"] = learning_rate_dict[name]*np.array(grad.min())
    js_str = self.js_dumpstring(stat_dict)
    self.log_info("<stats>"+js_str+"</stats>")

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(GDN_Autoencoder, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.w_enc, self.b_enc, self.b_dec, self.w_dec,
      self.gdn_w, self.igdn_w, self.gdn_b, self.igdn_b, self.x_, self.gdn_output,
      self.gdn_entropies, self.gdn_const]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    w_enc, b_enc, b_dec, w_dec, gdn_w, igdn_w, gdn_b, igdn_b, recon, activity, entropies, gdn_constant = eval_out[1:]
    w_enc_norm = np.linalg.norm(w_enc, axis=1, keepdims=False)
    w_dec_norm = np.linalg.norm(w_dec, axis=0, keepdims=False)
    w_enc = dp.reshape_data(w_enc.T, flatten=False)[0]
    w_dec = dp.reshape_data(w_dec, flatten=False)[0]
    fig = pf.plot_weight_image(gdn_w, title="GDN Weights", figsize=(10,10),
      save_filename=self.disp_dir+"w_gdn_v"+self.version+"-"+current_step.zfill(5)+".png")
    fig = pf.plot_weight_image(igdn_w, title="Inverse GDN Weights", figsize=(10,10),
      save_filename=self.disp_dir+"w_igdn_v"+self.version+"-"+current_step.zfill(5)+".png")
    fig = pf.plot_activity_hist(gdn_b, title="GDN Bias Histogram",
      save_filename=(self.disp_dir+"b_gdn_hist_v"+self.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_activity_hist(igdn_b, title="IGDN Bias Histogram",
      save_filename=(self.disp_dir+"b_igdn_hist_v"+self.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_data_tiled(w_enc, normalize=False,
      title="Encoding weights at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"w_enc_v"+self.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_data_tiled(w_dec, normalize=False,
      title="Decoding weights at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"w_dec_v"+self.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_activity_hist(b_enc, title="Encoding Bias Histogram",
      save_filename=(self.disp_dir+"b_enc_hist_v"+self.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_activity_hist(b_dec, title="Decoding Bias Histogram",
      save_filename=(self.disp_dir+"b_dec_hist_v"+self.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_activity_hist(gdn_constant, title="GDN Constant Histogram",
      save_filename=(self.disp_dir+"gdn_const_v"+self.version+"-"
      +current_step.zfill(5)+".png"))
    entropy_sort_indices = np.argsort(entropies)[::-1] # ascending
    for fig_id, neuron_id in enumerate(entropy_sort_indices[:3]):
      fig = pf.plot_activity_hist(activity[:, neuron_id],
        title=("Actvity Histogram (pre-noise) for Neuron "+str(neuron_id)+"\nwith entropy = "
        +str(np.round(entropies[neuron_id]))),
        save_filename=(self.disp_dir+"indv_act_hist_v"+self.version+"-"
        +current_step.zfill(5)+"_"+str(fig_id).zfill(3)+".png"))
    fig = pf.plot_activity_hist(activity, title="Activity Histogram (pre-noise)",
      save_filename=(self.disp_dir+"act_hist_v"+self.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_bar(w_enc_norm, num_xticks=5,
      title="w_enc l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
      save_filename=(self.disp_dir+"w_enc_norm_v"+self.version+"-"+current_step.zfill(5)+".png"))
    fig = pf.plot_bar(w_dec_norm, num_xticks=5,
      title="w_dec l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
      save_filename=(self.disp_dir+"w_dec_norm_v"+self.version+"-"+current_step.zfill(5)+".png"))
    if eval_out[0]*10 % self.cp_int == 0:
      fig = pf.plot_activity_hist(input_data, title="Image Histogram",
        save_filename=(self.disp_dir+"img_hist_"+self.version+"-"
        +current_step.zfill(5)+".png"))
      input_data = dp.reshape_data(input_data, flatten=False)[0]
      fig = pf.plot_data_tiled(input_data, normalize=False,
        title="Images at step "+current_step, vmin=None, vmax=None,
        save_filename=(self.disp_dir+"images_"+self.version+"-"
        +current_step.zfill(5)+".png"))
      recon = dp.reshape_data(recon, flatten=False)[0]
      fig = pf.plot_data_tiled(recon, normalize=False,
        title="Recons at step "+current_step, vmin=None, vmax=None,
        save_filename=(self.disp_dir+"recons_v"+self.version+"-"+current_step.zfill(5)+".png"))
      for weight_grad_var in self.grads_and_vars[self.sched_idx]:
        grad = weight_grad_var[0][0].eval(feed_dict)
        shape = grad.shape
        name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
        grad = dp.reshape_data(grad.T, flatten=False)[0]
        fig = pf.plot_data_tiled(grad, normalize=True,
          title="Gradient for"+name+" at step "+current_step, vmin=None, vmax=None,
          save_filename=(self.disp_dir+"d"+name+"_v"+self.version+"-"+current_step.zfill(5)+".png"))
