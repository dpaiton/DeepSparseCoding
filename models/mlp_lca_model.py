import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_functions as ef
from models.mlp_model import MlpModel
from modules.lca_module import LcaModule
from modules.lca_conv_module import LcaConvModule


class MlpLcaModel(MlpModel):
  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(MlpLcaModel, self).load_params(params)
    # Network Size
    self.label_shape = [None, self.params.num_classes]
    # Hyper Parameters
    self.eta = self.params.dt / self.params.tau

  def build_lca_module(self, input_node):
    if(self.params.lca_conv):
      module = LcaConvModule(input_node, self.params.num_neurons, self.sparse_mult,
        self.eta, self.params.thresh_type, self.params.rectify_a,
        self.params.num_steps, self.params.lca_patch_size_y, self.params.lca_patch_size_x,
        self.params.lca_stride_y, self.params.lca_stride_x, self.params.eps)
    else:
      module = LcaModule(input_node, self.params.num_neurons, self.sparse_mult,
        self.eta, self.params.thresh_type, self.params.rectify_a,
        self.params.num_steps, self.params.eps)
    return module

  #We overwrite get_load_vars here to get a dict from str to variable
  def get_load_vars(self):
    """Get variables for loading"""
    all_vars = tf.global_variables()
    if self.params.cp_load_var is None:
      load_v = [v for v in all_vars if v not in self.full_model_load_ignore]
    else:
      if(self.params.load_dict):
        load_v = {}
        #Can only load dictionary by itself here
        assert(len(self.params.cp_load_var) == 1)
        #Saver var_list can't take appended :0 as variable name
        idx = self.params.cp_load_var[0].rfind(":")
        assert idx != -1, ("cp_load_var name must contain \":\"")
        offset = len(self.params.cp_load_var[0]) - idx
        load_v[self.params.cp_load_var[0][:-offset]] = self.lca_module.w
      else:
        load_v = []
        for weight in self.params.cp_load_var:
          found=False
          for var in all_vars:
            if var.name == weight:
              load_v.append(var)
              found=True
              break
          if not found:
            assert False, ("Weight specified in cp_load_var "+str(weight)+" not found")
    return load_v

  #We overwrite here to allow for the saver to reshape variables as necessary
  def construct_savers(self):
    """Add savers to graph"""
    with self.graph.as_default():
      #Loop through to find variables to ignore
      all_vars = tf.global_variables()
      load_vars = [v for v in all_vars if v not in self.full_model_load_ignore]
      self.full_saver = tf.train.Saver(max_to_keep=self.params.max_cp_to_keep,
        var_list=load_vars)
      if self.params.cp_load:
        self.loader = tf.train.Saver(var_list=self.get_load_vars(), reshape=True)
    self.savers_constructed = True

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.variable_scope("auto_placeholders") as scope:
          self.label_placeholder = tf.placeholder(
            tf.float32, shape=self.label_shape, name="input_labels")
          self.sparse_mult = tf.placeholder(tf.float32, shape=(), name="sparse_mult")
          self.train_lca = tf.placeholder(tf.bool, shape=(), name="train_lca")

        with tf.variable_scope("placeholders") as scope:
          self.dropout_keep_probs = tf.placeholder(tf.float32, shape=[None],
            name="dropout_keep_probs")

        self.train_lca = tf.cast(self.train_lca, tf.float32)

        with tf.variable_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        if(not self.params.lca_conv):
          #Flatten input_node if not flat
          data_shape = input_node.get_shape().as_list()
          if len(data_shape) == 4:
            self.batch_size, self.y_size, self.x_size, self.num_data_channels = data_shape
            self.num_pixels = self.y_size * self.x_size * self.num_data_channels
            input_node = tf.reshape(input_node, [-1, self.num_pixels])
          else:
            assert False, ("FC lca must have datashape of rank 2 or 4")

        self.input_node = tf.identity(input_node, name="input_node")

        self.lca_module = self.build_lca_module(self.input_node)
        self.reconstruction = tf.identity(self.lca_module.reconstruction, name="reconstruction")
        self.activations = tf.identity(self.lca_module.a, name="activations")

        self.trainable_variables.update(self.lca_module.trainable_variables)

        if self.params.train_on_recon:
          if self.params.layer_types[0] == "conv":
            data_shape = [tf.shape(self.input_node)[0]]+self.params.full_data_shape
            mlp_input = tf.reshape(self.lca_module.reconstruction, shape=data_shape)
          elif self.params.layer_types[0] == "fc":
            mlp_input = self.lca_module.reconstruction
          else:
            assert False, ("params.layer_types must be 'fc' or 'conv'")
        else: # train on LCA latent encoding
          mlp_input = self.lca_module.a
          data_shape = mlp_input.get_shape().as_list()

        #Add pooling layer to mlp_input if set
        if(self.params.lca_max_pool):
          mlp_input = tf.nn.max_pool(mlp_input, ksize=self.params.lca_max_pool_ksize,
            strides=self.params.lca_max_pool_strides, padding="SAME")

        self.mlp_module = self.build_mlp_module(mlp_input)
        self.trainable_variables.update(self.mlp_module.trainable_variables)

        with tf.variable_scope("loss") as scope:
          #Loss switches based on train_lca flag
          self.total_loss = self.train_lca * self.lca_module.total_loss + \
            (1-self.train_lca) * self.mlp_module.total_loss

        with tf.variable_scope("norm_weights") as scope:
          self.norm_weights = tf.group(self.lca_module.norm_w, name="l2_normalization")

        self.label_est = tf.identity(self.mlp_module.label_est, name="label_est")

        with tf.variable_scope("performance_metrics") as scope:
          #LCA metrics
          MSE = tf.reduce_mean(tf.square(tf.subtract(self.input_node, self.lca_module.reconstruction)),
            axis=[1, 0], name="mean_squared_error")
          pixel_var = tf.nn.moments(self.input_node, axes=[1])[1]
          self.pSNRdB = tf.multiply(10.0, ef.safe_log(tf.divide(tf.square(pixel_var), MSE)),
            name="recon_quality")
          with tf.variable_scope("prediction_bools"):
            self.correct_prediction = tf.equal(tf.argmax(self.label_est, axis=1),
              tf.argmax(self.label_placeholder, axis=1), name="individual_accuracy")
          with tf.variable_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
              tf.float32), name="avg_accuracy")

  def get_total_loss(self):
    return self.total_loss

  def compute_recon_from_encoding(self, a_in):
    return self.lca_module.build_decoder(a_in, name="reconstruction")

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
    Log train progress information
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    update_dict = super(MlpLcaModel, self).generate_update_dict(input_data, input_labels,
      batch_step)

    feed_dict = self.get_feed_dict(input_data, input_labels)

    eval_list = [self.lca_module.loss_dict["recon_loss"],
      self.lca_module.loss_dict["sparse_loss"],
      self.lca_module.a,
      self.lca_module.reconstruction,
      self.pSNRdB, self.input_node]

    out_vals =  tf.get_default_session().run(eval_list, feed_dict)

    recon_loss, sparse_loss, lca_a_vals, recon, pSNRdB, input_node\
      = out_vals[0:6]

    train_on_adversarial = feed_dict[self.train_on_adversarial]
    if(train_on_adversarial):
      orig_img = feed_dict[self.input_placeholder]
      adv_feed_dict = feed_dict.copy()
      adv_feed_dict[self.use_adv_input] = True
      adv_img = tf.get_default_session().run(self.adv_module.get_adv_input(), adv_feed_dict)

      reduc_dims = tuple(range(1, len(orig_img.shape)))
      orig_adv_linf = np.max(np.abs(orig_img - adv_img), axis=reduc_dims)
      orig_recon_linf = np.max(np.abs(orig_img - recon), axis=reduc_dims)

      orig_adv_linf_max = np.max(orig_adv_linf)
      orig_adv_linf_mean = np.mean(orig_adv_linf)
      orig_adv_linf_min = np.min(orig_adv_linf)

      orig_recon_linf_max = np.max(orig_recon_linf)
      orig_recon_linf_mean = np.mean(orig_recon_linf)
      orig_recon_linf_min = np.min(orig_recon_linf)

    input_max = np.max(input_node)
    input_mean = np.mean(input_node)
    input_min = np.min(input_node)
    recon_max = np.max(recon)
    recon_mean = np.mean(recon)
    recon_min = np.min(recon)
    lca_a_vals_max = np.array(lca_a_vals.max())
    lca_a_vals_mean = np.array(lca_a_vals.mean())
    lca_a_vals_min = np.array(lca_a_vals.min())
    lca_a_frac_act = np.array(np.count_nonzero(lca_a_vals)
      / float(lca_a_vals.size))

    stat_dict = {
      "lca_recon_loss":recon_loss,
      "lca_sparse_loss":sparse_loss,
      "pSNRdB": np.mean(pSNRdB),
      "lca_a_fraction_active":lca_a_frac_act,
      "lca_a_max_mean_min":[lca_a_vals_max, lca_a_vals_mean, lca_a_vals_min],
      "x_max_mean_min":[input_max, input_mean, input_min],
      "x_hat_max_mean_min":[recon_max, recon_mean, recon_min]}

    if(train_on_adversarial):
      stat_dict["orig_adv_linf_max_mean_min"] = [orig_adv_linf_max,
        orig_adv_linf_mean, orig_adv_linf_min]

      stat_dict["orig_recon_linf_max_mean_min"] = [orig_recon_linf_max,
        orig_recon_linf_mean, orig_recon_linf_min]

    update_dict.update(stat_dict) #stat_dict overwrites for same keys

    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(MlpLcaModel, self).generate_plots(input_data, input_labels)

    feed_dict = self.get_feed_dict(input_data, input_labels)

    eval_list = [self.global_step, self.lca_module.w,
      self.lca_module.reconstruction, self.lca_module.a, self.input_node]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    filename_suffix = "_v"+self.params.version+"_"+current_step.zfill(5)+".png"
    weights, recon, lca_activity, input_node = eval_out[1:]

    batch_size = input_data.shape[0]
    fig = pf.plot_activity_hist(np.reshape(input_node, [batch_size, -1]), title="Image Histogram",
      save_filename=self.params.disp_dir+"img_hist"+filename_suffix)

    fig = pf.plot_activity_hist(np.reshape(recon, [batch_size, -1]), title="Recon Histogram",
      save_filename=self.params.disp_dir+"recon_hist"+filename_suffix)

    recon = dp.reshape_data(recon, flatten=False)[0]
    weights = dp.reshape_data(weights.T, flatten=False)[0] # [num_neurons, height, width]
    #Scale image by max and min of images and/or recon
    r_max = np.max([np.max(input_node), np.max(recon)])
    r_min = np.min([np.min(input_node), np.min(recon)])
    input_node = dp.reshape_data(input_node, flatten=False)[0]
    fig = pf.plot_data_tiled(input_node, normalize=False,
      title="Scaled Images at step "+current_step, vmin=r_min, vmax=r_max,
      save_filename=self.params.disp_dir+"images"+filename_suffix)
    fig = pf.plot_data_tiled(recon, normalize=False,
      title="Recons at step "+current_step, vmin=r_min, vmax=r_max,
      save_filename=self.params.disp_dir+"recons"+filename_suffix)

    num_features = lca_activity.shape[-1]
    lca_activity = np.reshape(lca_activity, [-1, num_features])
    fig = pf.plot_activity_hist(lca_activity, title="LCA Activity Histogram",
      save_filename=self.params.disp_dir+"lca_act_hist"+filename_suffix)

    if(len(weights.shape) == 4):
      weights = np.transpose(weights, (0, 2, 3, 1))
    fig = pf.plot_data_tiled(weights, normalize=False,
      title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=self.params.disp_dir+"phi"+filename_suffix)

    #Plot loss over time
    eval_list = [self.lca_module.recon_loss_list, self.lca_module.sparse_loss_list, self.lca_module.total_loss_list]
    (recon_losses, sparse_losses, total_losses) = tf.get_default_session().run(eval_list, feed_dict)
    #TODO put this in plot functions
    pf.plot_sc_losses(recon_losses, sparse_losses, total_losses,
      save_filename=self.params.disp_dir+"losses"+filename_suffix)
