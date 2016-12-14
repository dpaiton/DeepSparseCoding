import numpy as np
import logging
import tensorflow as tf
import utils.plot_functions as pf
from models.base_model import Model

class MLP(Model):
  def __init__(self, params, schedule):
    Model.__init__(self, params, schedule)
    self.build_graph()
    Model.setup_graph(self, self.graph)

  """
  Load parameters into object
  Inputs:
   params: [dict] model parameters
  Modifiable Parameters:
    rectify_a      [bool] If set, rectify layer 1 activity
    norm_a         [bool] If set, l2 normalize layer 1 activity
    norm_weights   [bool] If set, l2 normalize weights after updates
    batch_size     [int] Number of images in a training batch
    num_pixels     [int] Number of pixels
    num_neurons    [int] Number of layer 1 elements (# basis vectors)
    num_classes    [int] Number of layer 2 elements (# categories)
    num_val        [int] Number of validation images
  """
  def load_params(self, params):
    Model.load_params(self, params)
    if "rectify_a" in params.keys():
      self.rectify_a = bool(params["rectify_a"])
    self.norm_a = bool(params["norm_a"])
    self.norm_weights = bool(params["norm_weights"])
    # Hyper-parameters
    self.batch_size = int(params["batch_size"])
    self.num_pixels = int(params["num_pixels"])
    self.num_neurons = int(params["num_neurons"])
    if "num_classes" in params.keys():
      self.num_classes = int(params["num_classes"])
    else:
      self.num_classes = 0
    if "num_val" in params.keys():
      self.num_val = int(params["num_val"])
    else:
      self.num_val = 0
    self.phi_shape = [self.num_pixels, self.num_neurons]
    self.w_shape = [self.num_classes, self.num_neurons]

  """
  Build an MLP TensorFlow Graph.
  """
  def build_graph(self):
    self.graph = tf.Graph()
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.x = tf.placeholder(tf.float32,
            shape=[self.num_pixels, None], name="input_data")
          self.y = tf.placeholder(tf.float32,
            shape=[self.num_classes, None], name="input_label")
          self.sup_mult = tf.placeholder(
            tf.float32, shape=(), name="sup_mult")

        with tf.name_scope("constants") as scope:
          self.label_mult = tf.reduce_sum(self.y, reduction_indices=[0])

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False,
            name="global_step")

        with tf.variable_scope("weights") as scope:
          self.phi = tf.get_variable(name="phi", dtype=tf.float32,
            initializer=tf.truncated_normal(self.phi_shape, mean=0.0,
            stddev=1.0, dtype=tf.float32, name="phi_init"), trainable=True)
          self.w = tf.get_variable(name="w", dtype=tf.float32,
            initializer=tf.truncated_normal(self.w_shape, mean=0.0,
            stddev=1.0, dtype=tf.float32, name="w_init"), trainable=True)
          self.bias1 = tf.get_variable(name="bias1", dtype=tf.float32,
            initializer=tf.zeros([self.num_neurons, 1], dtype=tf.float32,
            name="bias1_init"), trainable=True)
          self.bias2 = tf.get_variable(name="bias2", dtype=tf.float32,
            initializer=tf.zeros([self.num_classes, 1], dtype=tf.float32,
            name="bias2_init"), trainable=True)

        with tf.name_scope("normalize_weights") as scope:
          self.norm_phi = self.phi.assign(tf.nn.l2_normalize(self.phi,
            dim=0, epsilon=self.eps, name="row_l2_norm"))
          self.normalize_weights = tf.group(self.norm_phi,
            name="do_normalization")

        with tf.name_scope("hidden_variables") as scope:
          if self.rectify_a:
            self.a = tf.nn.relu(tf.add(tf.matmul(tf.transpose(self.phi),
              self.x), self.bias1), name="activity")
          else:
            self.a = tf.add(tf.matmul(tf.transpose(self.phi), self.x),
              self.bias1, name="activity")

          if self.norm_a:
            self.c = tf.add(tf.matmul(self.w, tf.nn.l2_normalize(self.a,
              dim=0, epsilon=self.eps, name="row_l2_norm"),
              name="classify"), self.bias2, name="c")
          else:
            self.c = tf.add(tf.matmul(self.w, self.a, name="classify"),
              self.bias2, name="c")

        with tf.name_scope("output") as scope:
          with tf.name_scope("label_estimate"):
            self.y_ = tf.transpose(tf.nn.softmax(tf.transpose(self.c)))

        with tf.name_scope("loss") as scope:
          with tf.name_scope("supervised"):
            with tf.name_scope("cross_entropy_loss"):
              self.cross_entropy_loss = (self.label_mult
                * -tf.reduce_sum(tf.mul(self.y, tf.log(tf.clip_by_value(
                self.y_, self.eps, 1.0))), reduction_indices=[0]))
              label_count = tf.reduce_sum(self.label_mult)
              self.mean_cross_entropy_loss = (self.sup_mult
                * tf.reduce_sum(self.cross_entropy_loss) /
                (label_count + self.eps))
            self.supervised_loss = self.mean_cross_entropy_loss
          self.total_loss = self.supervised_loss

        with tf.name_scope("performance_metrics") as scope:
          with tf.name_scope("prediction_bools"):
            self.correct_prediction = tf.equal(tf.argmax(self.y_, dimension=0),
              tf.argmax(self.y, dimension=0), name="individual_accuracy")
          with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
              tf.float32), name="avg_accuracy")
    self.graph_built = True

  """
  Log train progress information
  Inputs:
    input_data: load_MNIST data object containing the current image batch
    input_label: load_MNIST data object containing the current label batch
    batch_step: current batch number within the schedule
  """
  def print_update(self, input_data, input_label, batch_step):
    current_step = self.global_step.eval()
    feed_dict = self.get_feed_dict(input_data, input_label)
    a_vals = tf.get_default_session().run(self.a, feed_dict)
    logging.info("Global batch index is %g"%(current_step))
    logging.info("Finished step %g out of %g for schedule %g"%(batch_step,
      self.get_sched("num_batches"), self.sched_idx))
    logging.info("\tloss:\t%g"%(
      self.total_loss.eval(feed_dict)))
    logging.info("\tmax val of a:\t\t%g"%(a_vals.max()))
    logging.info("\tpercent active:\t\t%0.2f%%"%(
      100.0 * np.count_nonzero(a_vals)
      / float(self.num_neurons * self.batch_size)))
    logging.info("\ttrain accuracy:\t\t%g"%(self.accuracy.eval(feed_dict)))
    logging.info("\tnum labeled data:\t%g"%(
      self.num_labeled_ex.eval(feed_dict)))

  """
  Plot weights, reconstruction, and gradients
  Inputs: input_data and input_label used for the session
  """
  def generate_plots(self, input_image, input_label):
    feed_dict = self.get_feed_dict(input_image, input_label)
    current_step = str(self.global_step.eval())
    pf.save_data_tiled(
      self.w.eval().reshape(self.num_classes,
      int(np.sqrt(self.num_neurons)), int(np.sqrt(self.num_neurons))),
      normalize=True, title="Classification matrix at step number "
      +current_step, save_filename=(self.disp_dir+"w_v"+self.version+"-"
      +current_step.zfill(5)+".pdf"))
    pf.save_data_tiled(
      tf.transpose(self.phi).eval().reshape(self.num_neurons,
      int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
      normalize=True, title="Dictionary at step "+current_step,
      save_filename=(self.disp_dir+"phi_v"+self.version+"-"
      +current_step.zfill(5)+".pdf"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]
      if name == "phi":
        pf.save_data_tiled(grad.T.reshape(self.num_neurons,
          int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
          normalize=True, title="Gradient for phi at step "+current_step,
          save_filename=(self.disp_dir+"dphi_v"+self.version+"_"
          +current_step.zfill(5)+".pdf"))
      elif name == "w":
        pf.save_data_tiled(grad.reshape(self.num_classes,
          int(np.sqrt(self.num_neurons)), int(np.sqrt(self.num_neurons))),
          normalize=True, title="Gradient for w at step "+current_step,
          save_filename=(self.disp_dir+"dw_v"+self.version+"_"
          +current_step.zfill(5)+".pdf"))
