import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
from models.base_model import Model

class MLP(Model):
  def __init__(self):
    super(MLP, self).__init__()
    self.vector_inputs = True

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(MLP, self).load_params(params)
    self.vector_inputs = True
    self.num_pixels = int(np.prod(self.params.data_shape))
    self.x_shape = [None, self.num_pixels]
    self.y_shape = [None, self.params.num_classes]
    self.w1_shape = [self.num_pixels, self.params.num_hidden]
    self.w2_shape = [self.params.num_hidden, self.params.num_classes]

  def build_graph(self):
    """
    Build an MLP TensorFlow Graph.
    """
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")
          self.y = tf.placeholder(tf.float32, shape=self.y_shape, name="input_labels")

        with tf.name_scope("constants") as scope:
          ## For semi-supervised learning, loss is 0 if there is no label
          self.label_mult = tf.reduce_sum(self.y, axis=[1])

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False,
            name="global_step")

        with tf.variable_scope("weights") as scope:
          self.w1 = tf.get_variable(name="w1", dtype=tf.float32,
            initializer=tf.truncated_normal(self.w1_shape, mean=0.0,
            stddev=1.0, dtype=tf.float32, name="w1_init"), trainable=True)
          self.w2 = tf.get_variable(name="w2", dtype=tf.float32,
            initializer=tf.truncated_normal(self.w2_shape, mean=0.0,
            stddev=1.0, dtype=tf.float32, name="w2_init"), trainable=True)
          self.bias1 = tf.get_variable(name="bias1", dtype=tf.float32,
            initializer=tf.zeros([1, self.params.num_hidden], dtype=tf.float32,
            name="bias1_init"), trainable=True)
          self.bias2 = tf.get_variable(name="bias2", dtype=tf.float32,
            initializer=tf.zeros([1, self.params.num_classes], dtype=tf.float32,
            name="bias2_init"), trainable=True)
          self.trainable_variables[self.w1.name] = self.w1
          self.trainable_variables[self.w2.name] = self.w2
          self.trainable_variables[self.bias1.name] = self.bias1
          self.trainable_variables[self.bias2.name] = self.bias2

        with tf.name_scope("norm_weights") as scope:
          self.norm_w1 = self.w1.assign(tf.nn.l2_normalize(self.w1,
            dim=0, epsilon=self.params.eps, name="row_l2_norm"))
          self.norm_weights = tf.group(self.norm_w1,
            name="l2_normalization")

        with tf.name_scope("hidden_variables") as scope:
          if self.params.rectify_a:
            self.a = tf.nn.relu(tf.add(tf.matmul(self.x, self.w1),
              self.bias1), name="activity")
          else:
            self.a = tf.add(tf.matmul(self.x, self.w1), self.bias1,
              name="activity")

          if self.params.norm_a:
            self.c = tf.add(tf.matmul(tf.nn.l2_normalize(self.a,
              dim=1, epsilon=self.params.eps, name="row_l2_norm"), self.w2,
              name="classify"), self.bias2, name="c")
          else:
            self.c = tf.add(tf.matmul(self.a, self.w2, name="classify"),
              self.bias2, name="c")

        with tf.name_scope("output") as scope:
          with tf.name_scope("label_estimate"):
            self.y_ = tf.nn.softmax(self.c)

        with tf.name_scope("loss") as scope:
          with tf.name_scope("supervised"):
            with tf.name_scope("cross_entropy_loss"):
              self.cross_entropy_loss = (self.label_mult
                * -tf.reduce_sum(tf.multiply(self.y, tf.log(tf.clip_by_value(
                self.y_, self.params.eps, 1.0))), axis=[1]))
              label_count = tf.reduce_sum(self.label_mult)
              f1 = lambda: tf.reduce_sum(self.cross_entropy_loss)
              f2 = lambda: tf.reduce_sum(self.cross_entropy_loss) / label_count
              pred_fn_pairs = {
                tf.equal(label_count, tf.constant(0.0)): f1,
                tf.greater(label_count, tf.constant(0.0)): f2}
              self.mean_cross_entropy_loss = tf.case(pred_fn_pairs,
                default=f2, exclusive=True, name="mean_cross_entropy_loss")
            self.supervised_loss = self.mean_cross_entropy_loss
          self.total_loss = self.supervised_loss

        with tf.name_scope("performance_metrics") as scope:
          with tf.name_scope("prediction_bools"):
            self.correct_prediction = tf.equal(tf.argmax(self.y_, axis=1),
              tf.argmax(self.y, axis=1), name="individual_accuracy")
          with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
              tf.float32), name="avg_accuracy")
    self.graph_built = True

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
    total_loss = np.array(self.total_loss.eval(feed_dict))
    a_vals = tf.get_default_session().run(self.a, feed_dict)
    a_vals_max = np.array(a_vals.max())
    a_frac_act = np.array(np.count_nonzero(a_vals)
      / float(self.batch_size * self.params.num_hidden))
    accuracy = np.array(self.accuracy.eval(feed_dict))
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "number_of_batch_steps":self.get_schedule("num_batches"),
      "schedule_index":self.sched_idx,
      "total_loss":total_loss,
      "a_max":a_vals_max,
      "a_frac_active":a_frac_act,
      "train_accuracy":accuracy}
    update_dict.update(stat_dict) #stat_dict overwrites
    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs: input_data and input_labels used for the session
    """
    super(MLP, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    current_step = str(self.global_step.eval())
    w1, w2 = tf.get_default_session().run([self.w1, self.w2], feed_dict)
    w1 = dp.reshape_data(w1.T, flatten=False)[0]
    w2 = dp.reshape_data(w2.T, flatten=False)[0]
    pf.plot_data_tiled(w2, normalize=True,
      title="Classification matrix at step number "+current_step,
      vmin=None, vmax=None, save_filename=(self.disp_dir+"w2_v"+self.version+"-"
      +current_step.zfill(5)+".png"))
    pf.plot_data_tiled(w1, normalize=True,
      title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"w1_v"+self.version+"-"+current_step.zfill(5)+".png"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      grad = dp.reshape_data(grad.T, flatten=False)[0]
      if name == "w1":
        pf.plot_data_tiled(grad, normalize=True,
          title="Gradient for w1 at step "+current_step, vmin=None, vmax=None,
          save_filename=(self.disp_dir+"dw1_v"+self.version+"_"+current_step.zfill(5)+".png"))
      elif name == "w2":
        pf.plot_data_tiled(grad, normalize=True,
          title="Gradient for w2 at step "+current_step, vmin=None, vmax=None,
          save_filename=(self.disp_dir+"dw2_v"+self.version+"_"+current_step.zfill(5)+".png"))
