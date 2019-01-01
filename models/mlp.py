import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
from models.base_model import Model
from modules.mlp import MLPModule

class MLP(Model):
  def __init__(self):
    super(MLP, self).__init__()

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(MLP, self).load_params(params)
    self.num_pixels = int(np.prod(self.params.data_shape))
    self.x_shape = [None,] + self.params.data_shape
    self.y_shape = [None, self.params.num_classes]

  def build_graph(self):
    """
    Build an MLP TensorFlow Graph.
    """
    self.graph = tf.Graph()
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")
          self.y = tf.placeholder(tf.float32, shape=self.y_shape, name="input_labels")
          if self.params.do_batch_norm:
            self.batch_norm_decay_mult = tf.placeholder(tf.float32, shape=(),
              name="batch_norm_decay_mult")
          else:
            self.batch_norm_decay_mult = None

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False,
            name="global_step")

        self.mlp = MLPModule(self.x, self.y, self.batch_norm_decay_mult,
          self.params.layer_types, self.params.output_channels, self.params.strides_y,
          self.params.strides_x, self.params.patch_size_y, self.params.patch_size_x,
          self.params.eps, name="mlp")
        self.y_ = self.mlp.y_
        self.trainable_variables.update(self.mlp.trainable_variables)

        with tf.name_scope("performance_metrics") as scope:
          with tf.name_scope("prediction_bools"):
            self.correct_prediction = tf.equal(tf.argmax(self.y_, axis=1),
              tf.argmax(self.y, axis=1), name="individual_accuracy")
          with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
              tf.float32), name="avg_accuracy")

  def get_encodings(self):
    return self.mlp.layer_list[-1]

  def get_total_loss(self):
    return self.mlp.total_loss

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
    Generates a dictionary to be logged in the print_update function
    Inputs:
      input_data: load_MNIST data object containing the current image batch
      input_labels: load_MNIST data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    update_dict = super(MLP, self).generate_update_dict(input_data, input_labels, batch_step)
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
    Inputs: input_data and input_labels used for the session
    """
    super(MLP, self).generate_plots(input_data, input_labels)
    #TODO
    #feed_dict = self.get_feed_dict(input_data, input_labels)
    #current_step = str(self.global_step.eval())
    #w1, w2 = tf.get_default_session().run([self.w1, self.w2], feed_dict)
    #w1 = dp.reshape_data(w1.T, flatten=False)[0]
    #w2 = dp.reshape_data(w2.T, flatten=False)[0]
    #pf.plot_data_tiled(w2, normalize=True,
    #  title="Classification matrix at step number "+current_step,
    #  vmin=None, vmax=None, save_filename=(self.disp_dir+"w2_v"+self.version+"-"
    #  +current_step.zfill(5)+".png"))
    #pf.plot_data_tiled(w1, normalize=True,
    #  title="Dictionary at step "+current_step, vmin=None, vmax=None,
    #  save_filename=(self.disp_dir+"w1_v"+self.version+"-"+current_step.zfill(5)+".png"))
    #for weight_grad_var in self.grads_and_vars[self.sched_idx]:
    #  grad = weight_grad_var[0][0].eval(feed_dict)
    #  shape = grad.shape
    #  name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
    #  grad = dp.reshape_data(grad.T, flatten=False)[0]
    #  if name == "w1":
    #    pf.plot_data_tiled(grad, normalize=True,
    #      title="Gradient for w1 at step "+current_step, vmin=None, vmax=None,
    #      save_filename=(self.disp_dir+"dw1_v"+self.version+"_"+current_step.zfill(5)+".png"))
    #  elif name == "w2":
    #    pf.plot_data_tiled(grad, normalize=True,
    #      title="Gradient for w2 at step "+current_step, vmin=None, vmax=None,
    #      save_filename=(self.disp_dir+"dw2_v"+self.version+"_"+current_step.zfill(5)+".png"))
