import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_functions as ef
import ops.init_ops as init_ops
from models.base_model import Model
from modules.mlp_module import MlpModule
from modules.activations import lca_threshold

class MlpListaModel(Model):
  def __init__(self):
      """
      MLP trained on the reconstruction from LISTA
      """
      super(MlpListaModel, self).__init__()

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(MlpListaModel, self).load_params(params)
    # Network Size
    self.num_pixels = int(np.prod(self.params.data_shape))
    self.input_shape = [None, self.num_pixels]
    self.w_shape = [self.num_pixels, self.params.num_neurons]
    self.s_shape = [self.params.num_neurons, self.params.num_neurons]
    self.label_shape = [None, self.params.num_classes]
    # Hyper Parameters
    self.eta = self.params.dt / self.params.tau

  def get_input_shape(self):
    return self.input_shape

  def build_mlp_module(self):
    assert self.params.layer_types[0] == "fc", (
      "MLP must have FC layers to train on LISTA activity")
    module = MlpModule(self.a, self.label_placeholder, self.params.layer_types,
      self.params.output_channels, self.params.batch_norm, self.params.dropout,
      self.params.max_pool, self.params.max_pool_ksize, self.params.max_pool_strides,
      self.params.patch_size_y, self.params.patch_size_x, self.params.conv_strides,
      self.params.eps, loss_type="softmax_cross_entropy", name="MLP")
    return module

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.label_placeholder = tf.placeholder(tf.float32, shape=self.label_shape, name="input_labels")
          self.sparse_mult = tf.placeholder(tf.float32, shape=(), name="sparse_mult")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # LISTA module
        with tf.name_scope("weight_inits") as scope:
          self.w_init = tf.truncated_normal_initializer(mean=0, stddev=0.05, dtype=tf.float32)
          self.s_init = init_ops.GDNGammaInitializer(diagonal_gain=0.0, off_diagonal_gain=0.001, dtype=tf.float32)

        with tf.variable_scope("weights") as scope:
          self.w = tf.get_variable(name="w_enc", shape=self.w_shape, dtype=tf.float32,
            initializer=self.w_init, trainable=True)
          self.s = tf.get_variable(name="lateral_connectivity", shape=self.s_shape,
            dtype=tf.float32, initializer=self.s_init, trainable=True)

        with tf.name_scope("inference") as scope:
          feedforward_drive = tf.matmul(input_node, self.w, name="feedforward_drive")
          self.a_list = [lca_threshold(feedforward_drive, self.params.thresh_type,
            self.params.rectify_a, self.sparse_mult, name="a_init")]
          for layer_id in range(self.params.num_layers):
            u_in = feedforward_drive + tf.matmul(self.a_list[layer_id], self.s)
            self.a_list.append(lca_threshold(u_in, self.params.thresh_type, self.params.rectify_a,
              self.sparse_mult, name="a_init"))
          self.a = self.a_list[-1]

        # MLP module
        with tf.name_scope("mlp_module") as scope:
          self.mlp_module = self.build_mlp_module()
          self.trainable_variables.update(self.mlp_module.trainable_variables)

        with tf.name_scope("loss") as scope:
          self.total_loss =  self.mlp_module.total_loss

        self.label_est = self.mlp_module.label_est

        with tf.name_scope("performance_metrics") as scope:
          #LISTA metrics
          with tf.name_scope("prediction_bools"):
            self.correct_prediction = tf.equal(tf.argmax(self.label_est, axis=1),
              tf.argmax(self.label_placeholder, axis=1), name="individual_accuracy")
          with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
              tf.float32), name="avg_accuracy")


  def get_encodings(self):
    return self.mlp_module.layer_list[-1]

  def get_total_loss(self):
    return self.total_loss

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
    Log train progress information
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    update_dict = super(MlpListaModel, self).generate_update_dict(input_data, input_labels, batch_step)
    
    feed_dict = self.get_feed_dict(input_data, input_labels)
    current_step = np.array(self.global_step.eval())
    total_loss = np.array(self.get_total_loss().eval(feed_dict))
    logits_vals = tf.get_default_session().run(self.get_encodings(), feed_dict)
    logits_vals_max = np.array(logits_vals.max())
    logits_frac_act = np.array(np.count_nonzero(logits_vals) / float(logits_vals.size))
    accuracy = np.array(self.accuracy.eval(feed_dict))
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "number_of_batch_steps":self.params.schedule[self.sched_idx]["num_batches"],
      "schedule_index":self.sched_idx,
      "total_loss":total_loss,
      "logits_max":logits_vals_max,
      "logits_frac_active":logits_frac_act,
      "train_accuracy":accuracy}
    update_dict.update(stat_dict) #stat_dict overwrites
    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(MlpListaModel, self).generate_plots(input_data, input_labels)

    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.w, self.a, self.get_encodings()]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    weights, lista_activity, activity = eval_out[1:]
    fig = pf.plot_activity_hist(input_data, title="Image Histogram",
      save_filename=(self.params.disp_dir+"img_hist_"+self.params.version+"-"
      +current_step.zfill(5)+".png"))

    weights = dp.reshape_data(weights.T, flatten=False)[0] # [num_neurons, height, width]
    #Scale image by max and min of images and/or recon
    r_max = np.max(input_data)
    r_min = np.min(input_data)
    name_suffix = "_v"+self.params.version+"-"+current_step.zfill(5)+".png"
    input_data = dp.reshape_data(input_data, flatten=False)[0]
    fig = pf.plot_data_tiled(input_data, normalize=False,
      title="Scaled Images at step "+current_step, vmin=r_min, vmax=r_max,
      save_filename=(self.params.disp_dir+"images" + name_suffix))
    fig = pf.plot_activity_hist(lista_activity, title="LISTA Activity Histogram",
      save_filename=(self.params.disp_dir+"lista_act_hist" + name_suffix))
    fig = pf.plot_data_tiled(weights, normalize=False,
      title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.params.disp_dir+"w_lista" + name_suffix))
    fig = pf.plot_activity_hist(activity, title="Logit Histogram",
      save_filename=(self.params.disp_dir+"act_hist_"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
