import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_funcs as ef
from models.base_model import Model
import pdb
class VAE(Model):
  def __init__(self):
    """
    Variational Autoencoder using Mixture of Gaussians
    Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes."
    arXiv preprint arXiv:1312.6114 (2013).
    """
    super(VAE, self).__init__()
    self.vector_inputs = True

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [dict] model parameters
    Modifiable Parameters:
    """
    super(VAE, self).load_params(params)
    self.data_shape = params["data_shape"]
    # Network Size
    self.batch_size = int(params["batch_size"])
    self.num_pixels = int(np.prod(self.data_shape))
    self.num_neurons = params["num_neurons"] #is list

    self.num_encoder_layers = len(self.num_neurons)
    #Calculate encoder and decoder shapes
    prev_input_features = self.num_pixels
    self.w_enc_shape = []
    self.b_enc_shape = []
    self.b_dec_shape = []
    for l in range(self.num_encoder_layers):
      self.w_enc_shape.append([prev_input_features, self.num_neurons[l]])
      self.b_enc_shape.append([1, self.num_neurons[l]])
      self.b_dec_shape.append([1, prev_input_features])
      prev_input_features = self.num_neurons[l]

    self.x_shape = [None, self.num_pixels]

  def compute_weight_decay_loss(self):
    with tf.name_scope("unsupervised"):
      w_decay_list = [tf.reduce_sum(tf.square(w)) for w in self.w_enc_list]
      w_decay_list += [tf.reduce_sum(tf.square(w)) for w in self.w_dec_list]
      w_decay_list += [tf.reduce_sum(tf.square(self.w_enc_std))]

      decay_loss = tf.multiply(0.5*self.decay_mult,
        tf.add_n(w_decay_list))
    return decay_loss

  def compute_latent_loss(self, a_mean, a_log_std_sq):
    with tf.name_scope("latent"):
      reduc_dim = list(range(1, len(a_mean.shape))) # Want to avg over batch, sum over the rest
      latent_loss= self.kld_mult * -0.5 * tf.reduce_mean(tf.reduce_sum(1 + a_log_std_sq -
        tf.square(a_mean) - tf.exp(a_log_std_sq), reduc_dim))
    return latent_loss

  def compute_recon_loss(self, reconstruction):
    with tf.name_scope("unsupervised"):
      #Want to avg over batch, sum over the rest
      reduc_dim = list(range(1, len(reconstruction.shape)))
      recon_loss = 0.5 * tf.reduce_mean(
        tf.reduce_sum(tf.square(tf.subtract(reconstruction, self.x)),
        axis=reduc_dim), name="recon_loss")
    return recon_loss

  def compute_sparse_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      reduc_dim = list(range(1, len(a_in.shape))) # Want to avg over batch, sum over the rest
      act_l1 = tf.reduce_mean(tf.reduce_sum(tf.square(a_in), axis=reduc_dim))
      sparse_loss = self.sparse_mult * act_l1
    return sparse_loss

  def build_decoder(self, a_in):
    curr_input = a_in
    out_activations = []
    for (l, (curr_w, curr_b)) in enumerate(zip(self.w_dec_list, self.b_dec_list)):
      curr_output = tf.matmul(curr_input, curr_w) + curr_b
      if(l < self.num_encoder_layers - 1):
        curr_output = tf.nn.relu(curr_output)
      out_activations.append(curr_output)
      curr_input = curr_output
    return(out_activations)

  def compute_recon(self, a_in):
    return self.build_decoder(a_in)[-1]

  def build_graph(self):
    """Build the TensorFlow graph object"""
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")
          self.sparse_mult = tf.placeholder(tf.float32, shape=(), name="sparse_mult")
          self.decay_mult = tf.placeholder(tf.float32, shape=(), name="decay_mult")
          self.kld_mult = tf.placeholder(tf.float32, shape=(), name="kld_mult")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.name_scope("weight_inits") as scope:
          w_init = []
          b_enc_init = []
          b_dec_init = []
          for l in range(self.num_encoder_layers):
            w_shape = self.w_enc_shape[l]
            b_enc_shape = self.b_enc_shape[l]
            b_dec_shape = self.b_dec_shape[l]

            w_init.append(tf.truncated_normal(w_shape, mean=0.0,
              stddev=0.001, dtype=tf.float32, name="w_init"))
            b_enc_init.append(tf.ones(b_enc_shape)*0.0001)
            b_dec_init.append(tf.ones(b_dec_shape)*0.0001)

        with tf.variable_scope("weights") as scope:
          self.weight_scope = tf.get_variable_scope()
          self.w_enc_list = []
          self.b_enc_list = []
          self.w_dec_list = []
          self.b_dec_list = []
          #Extra weights for std to generate latent space
          self.w_enc_std = None
          self.b_enc_std = None
          for l in range(self.num_encoder_layers):
            #Encoder weights
            self.w_enc_list.append(tf.get_variable(name="w_enc_"+str(l), dtype=tf.float32,
              initializer=w_init[l], trainable=True))
            self.b_enc_list.append(tf.get_variable(name="b_enc_"+str(l), dtype=tf.float32,
              initializer=b_enc_init[l], trainable=True))

            #Decoder weights
            self.w_dec_list.append(tf.get_variable(name="w_dec_"+str(l), dtype=tf.float32,
              initializer=tf.transpose(w_init[l]), trainable=True))
            self.b_dec_list.append(tf.get_variable(name="b_dec_"+str(l), dtype=tf.float32,
              initializer=b_dec_init[l], trainable=True))

          #Std weights
          #l should be last encoder layer, i.e., layer right before latent space
          self.w_enc_std = tf.get_variable(name="w_enc_"+str(l)+"_std", dtype=tf.float32,
            initializer=w_init[l], trainable=True)
          self.b_enc_std = tf.get_variable(name="b_enc_"+str(l)+"_std", dtype=tf.float32,
            initializer=b_enc_init[l], trainable=True)

          #Reverse decoder weights to order them in order of operations
          self.w_dec_list = self.w_dec_list[::-1]
          self.b_dec_list = self.b_dec_list[::-1]

        with tf.variable_scope("inference") as scope:
          curr_input = self.x
          #Encoder
          self.encoder_activations = []
          for (l, (curr_w, curr_b)) in enumerate(zip(self.w_enc_list, self.b_enc_list)):
            curr_output = tf.matmul(curr_input, curr_w) + curr_b
            if(l == self.num_encoder_layers-1):
              self.latent_std_activation = tf.matmul(curr_input, self.w_enc_std) + self.b_enc_std
            elif(l < self.num_encoder_layers-1):
              curr_output = tf.nn.relu(curr_output)
            else:
              assert(0)
            self.encoder_activations.append(curr_output)
            curr_input = curr_output

          #Calculate latent
          #Std is in log std sq space
          noise  = tf.random_normal(tf.shape(self.latent_std_activation))
          self.a = tf.identity(self.encoder_activations[-1] +
            tf.sqrt(tf.exp(self.latent_std_activation)) * noise, name="activity")

          #TODO tf.placeholder_with_default doesn't allow for dynamic shape
          ##Placeholders for injecting activity
          #self.inject_a_flag = tf.placeholder_with_default(False,
          #  shape=(), name="inject_activation_flag")
          #self.inject_a = tf.placeholder_with_default(
          #  tf.zeros_initializer(), shape=[None, self.num_neurons[-1]],
          #  name="inject_activation")
          #curr_input = tf.where(inject_a_flag, inject_a, self.a)

          self.decoder_activations = self.build_decoder(self.a)

        with tf.name_scope("output") as scope:
          self.reconstruction = self.decoder_activations[-1]

        with tf.name_scope("loss") as scope:
          self.loss_dict = {"recon_loss":self.compute_recon_loss(self.reconstruction),
            "weight_decay_loss":self.compute_weight_decay_loss(),
            "latent_loss":self.compute_latent_loss(self.encoder_activations[-1],
              self.latent_std_activation)}
          self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")

        with tf.name_scope("performance_metrics") as scope:
          with tf.name_scope("reconstruction_quality"):
            MSE = tf.reduce_mean(tf.square(tf.subtract(self.x, self.reconstruction)), axis=[1, 0],
              name="mean_squared_error")
            pixel_var = tf.nn.moments(self.x, axes=[1])[1]
            self.pSNRdB = tf.multiply(10.0, tf.log(tf.divide(tf.square(pixel_var), MSE)),
              name="recon_quality")
    self.graph_built = True

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    update_dict = super(VAE, self).generate_update_dict(input_data,
      input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.loss_dict["recon_loss"],
      self.loss_dict["latent_loss"], self.loss_dict["weight_decay_loss"],
      self.total_loss, self.a, self.reconstruction, self.learning_rates]
    grad_name_list = []
    for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
      eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name[1]]
      grad_name = weight_grad_var[0][1].name.split('/')[1].split(':')[0] #2nd is np.split
      grad_name_list.append(grad_name)
    out_vals =  tf.get_default_session().run(eval_list, feed_dict)
    current_step, recon_loss, latent_loss, decay_loss, total_loss, a_vals, recon = out_vals[0:7]
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
      "latent_loss":latent_loss,
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
    super(VAE, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)

    eval_list = [self.global_step, self.w_enc_list, self.w_dec_list, self.w_enc_std,
      self.b_enc_list, self.b_enc_std, self.b_dec_list,
      self.encoder_activations, self.decoder_activations, self.a]

    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    w_enc, w_dec, w_enc_std, b_enc, b_enc_std, b_dec, enc_act, dec_act, a = eval_out[1:]
    recon = dec_act[-1]

    w_enc_norm = [np.linalg.norm(w, axis=0, keepdims=False) for w in w_enc]
    w_enc_std_norm = np.linalg.norm(w_enc_std, axis=0, keepdims=False)
    w_dec_norm = [np.linalg.norm(w, axis=1, keepdims=False) for w in w_dec]

    #Reshapes flat data into image
    w_enc_img = dp.reshape_data(w_enc[0].T, flatten=False)[0]
    w_dec_img = dp.reshape_data(w_dec[-1], flatten=False)[0]

    w_enc_img = dp.norm_weights(w_enc_img)
    w_dec_img = dp.norm_weights(w_dec_img)

    filename_suffix = "_v"+self.version+"_"+current_step.zfill(5)+".png"

    fig = pf.plot_data_tiled(w_enc_img, normalize=False,
      title="Encoding weights at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"w_enc" + filename_suffix))
    fig = pf.plot_data_tiled(w_dec_img, normalize=False,
      title="Decoding weights at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"w_dec" + filename_suffix))

    #TODO histogram with large bins is broken
    #fig = pf.plot_activity_hist(b_enc_mean, title="Encoding Bias Mean Histogram",
    #  save_filename=(self.disp_dir+"b_enc_mean_hist_v"+self.version+"-"
    #  +current_step.zfill(5)+".png"))
    #fig = pf.plot_activity_hist(b_enc_std, title="Encoding Bias Std Histogram",
    #  save_filename=(self.disp_dir+"b_enc_std_hist_v"+self.version+"-"
    #  +current_step.zfill(5)+".png"))
    for l in range(len(enc_act)):
      fig = pf.plot_activity_hist(enc_act[l], title="Activity Encoder " + str(l) + " Histogram",
        save_filename=(self.disp_dir+"act_enc_"+str(l)+"_hist" + filename_suffix))
      fig = pf.plot_bar(w_enc_norm[l], num_xticks=5,
        title="w_enc_"+str(l)+" l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
        save_filename=(self.disp_dir+"w_enc_"+str(l)+"_norm"+filename_suffix))

      fig = pf.plot_activity_hist(dec_act[l], title="Activity Decoder " + str(l) + " Histogram",
        save_filename=(self.disp_dir+"act_dec_"+str(l)+"_hist" + filename_suffix))
      fig = pf.plot_bar(w_dec_norm[l], num_xticks=5,
        title="w_dec_"+str(l)+" l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
        save_filename=(self.disp_dir+"w_dec_"+str(l)+"_norm"+filename_suffix))

    fig = pf.plot_activity_hist(w_enc_std, title="Activity Encoder " + str(l) + " Std Histogram",
      save_filename=(self.disp_dir+"act_enc_"+str(l)+"_std_hist" + filename_suffix))
    fig = pf.plot_bar(w_enc_std_norm, num_xticks=5,
      title="w_enc_"+str(l)+"_std l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
      save_filename=(self.disp_dir+"w_enc_"+str(l)+"_std_norm" + filename_suffix))

    if eval_out[0]*10 % self.cp_int == 0:
      #Scale image by max and min of images and/or recon
      r_max = np.max([np.max(input_data), np.max(recon)])
      r_min = np.min([np.min(input_data), np.min(recon)])

      fig = pf.plot_activity_hist(input_data, title="Image Histogram",
        save_filename=(self.disp_dir+"img_hist" + filename_suffix))
      input_data = dp.reshape_data(input_data, flatten=False)[0]
      fig = pf.plot_data_tiled(input_data, normalize=False,
        title="Images at step "+current_step, vmin=r_min, vmax=r_max,
        save_filename=(self.disp_dir+"images"+filename_suffix))
      recon = dp.reshape_data(recon, flatten=False)[0]
      fig = pf.plot_data_tiled(recon, normalize=False,
        title="Recons at step "+current_step, vmin=r_min, vmax=r_max,
        save_filename=(self.disp_dir+"recons"+filename_suffix))
