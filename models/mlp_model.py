import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
from models.base_model import Model
from modules.mlp_module import MlpModule

class MlpModel(Model):
  def __init__(self):
    super(MlpModel, self).__init__()

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(MlpModel, self).load_params(params)
    self.num_pixels = int(np.prod(self.params.data_shape))
    self.x_shape = [None,] + self.params.data_shape
    self.y_shape = [None, self.params.num_classes]

  def build_graph(self):
    """
    Build an MLP TensorFlow Graph.
    """
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")
          self.y = tf.placeholder(tf.float32, shape=self.y_shape, name="input_labels")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False,
            name="global_step")

        self.mlp_module = MlpModule(self.x, self.y, self.params.layer_types,
          self.params.output_channels, self.params.batch_norm, self.params.dropout,
          self.params.max_pool, self.params.max_pool_ksize, self.params.max_pool_strides,
          self.params.patch_size_y, self.params.patch_size_x, self.params.conv_strides,
          self.params.eps, name="MLP")
        self.trainable_variables.update(self.mlp_module.trainable_variables)
        self.y_ = self.mlp_module.y_

        with tf.name_scope("performance_metrics") as scope:
          with tf.name_scope("prediction_bools"):
            self.correct_prediction = tf.equal(tf.argmax(self.y_, axis=1),
              tf.argmax(self.y, axis=1), name="individual_accuracy")
          with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
              tf.float32), name="avg_accuracy")

  def get_encodings(self):
    return self.mlp_module.layer_list[-1]

  def get_total_loss(self):
    return self.mlp_module.total_loss

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
    Generates a dictionary to be logged in the print_update function
    Inputs:
      input_data: load_MNIST data object containing the current image batch
      input_labels: load_MNIST data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    update_dict = super(MlpModel, self).generate_update_dict(input_data, input_labels, batch_step)
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
    Plot weights, gradients, etc
    Inputs: input_data and input_labels used for the session
    """
    super(MlpModel, self).generate_plots(input_data, input_labels)
    # TODO: there is a lot of variability in the MLP - which plots are general?
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.get_encodings()] # how to get first layer weights?
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    activity = eval_out[1]
    fig = pf.plot_activity_hist(activity, title="Logit Histogram",
      save_filename=(self.params.disp_dir+"act_hist_"+self.params.version+"-"
      +current_step.zfill(5)+".png"))

