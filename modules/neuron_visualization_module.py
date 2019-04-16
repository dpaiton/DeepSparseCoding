import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import utils.data_processing as dp

class NeuronVisualizationModule(object):
  def __init__(self, data_tensor, num_steps, step_size, clip_output=True, clip_range=[0.0, 1.0],
    norm_constraint_mag=None, l2_regularize_coeff=None, variation_coeff=None, method="erhan",
    optimizer="sgd", variable_scope="neuron_visualization"):
    """
    Neuron optimal input visualization  module
    Inputs:
      data_tensor: Input images to be perturbed, of size [batch, pixels]
      num_steps: How many optimization steps to use
      step_size: How big of a perturbation to use
      clip_output: [bool] If True, clip output images to clip_range
      clip_range: [tuple or list] (min, max) values for adversarial images
      norm_constraint_mag: [float or None] if float, ensure that l2 norm of perturbation matches
        input via hard (change of variables) constraint, else do not impose hard constraint
      l2_regularize_coeff [float or None] if float, add soft l2 norm constraint on
        perturbation magnitude, with a tradeoff coefficient
      method: [str] allowed values are:
        "erhan"
      optimizer: [str] specifying "sgd" or "adam" for the optimizer
      variable_scope: [str] scope for adv module graph operators
    """
    self.data_tensor = data_tensor
    self.input_shape = self.data_tensor.get_shape().as_list()
    self.num_steps = num_steps
    self.step_size = step_size
    self.clip_output = clip_output
    self.clip_range = clip_range
    self.norm_constraint_mag = norm_constraint_mag
    self.l2_regularize_coeff = l2_regularize_coeff
    self.variation_coeff = variation_coeff
    self.method = method
    self.optimizer = optimizer.lower()
    self.ignore_load_var_list = [] # List of vars to ignore in savers/loaders
    self.variable_scope = str(variable_scope)
    self.build_init_graph()

  def build_init_graph(self):
    with tf.variable_scope(self.variable_scope) as scope:
      # These placeholders are here since they're only needed for construct adv examples
      with tf.variable_scope("placeholders") as scope:
        self.selection_vector = tf.placeholder(tf.float32, shape=(None),
          name="selection_vector")
      with tf.variable_scope("input_var"):
        vis_init = tf.identity(self.data_tensor)
        self.vis_var = tf.Variable(vis_init, dtype=tf.float32, trainable=True,
          validate_shape=False, name="vis_var")
        self.vis_var.set_shape([None,] + self.input_shape[1:])
        self.ignore_load_var_list.append(self.vis_var)
        self.reset = self.vis_var.initializer

        self.vis_pert = self.vis_var - self.data_tensor

        if(self.norm_constraint_mag is not None):
          self.vis_pert = self.norm_constraint_mag * tf.nn.l2_normalize(self.vis_pert,
            axis=list(range(1,len(self.input_shape)))) # don't normalize along batch dim

        self.vis_image = self.vis_pert + self.data_tensor

        if(self.clip_output): # Clip final visualization image to bound specified
          self.vis_image = tfc.upper_bound(tfc.lower_bound(
            self.vis_image, self.clip_range[0]), self.clip_range[1])

  def get_vis_input(self):
    return self.vis_image

  def build_visualization_ops(self, layer_activity):
    """
    Build loss & optimization ops into graph
    Inputs:
      layer_activity: tf variable for neuron layer activations
    """
    with tf.variable_scope(self.variable_scope) as scope:
      self.layer_activity = layer_activity + tf.random.normal(tf.shape(layer_activity), mean=0.001,
        stddev=0.001)
      with tf.variable_scope("loss") as scope:
        if(self.method == "erhan"):
          self.selected_activities = tf.matmul(layer_activity, self.selection_vector,
            name="selected_activities")
          self.vis_loss = -tf.reduce_sum(self.selected_activities)
        if(self.l2_regularize_coeff is not None):
          self.vis_loss = self.vis_loss + self.l2_regularize_coeff * tf.nn.l2_loss(self.vis_pert)
          #self.vis_loss += self.l2_regularize_coeff * tf.reduce_sum(tf.norm(self.vis_pert,
          #  ord="euclidean", axis=1, keepdims=False))
        if(self.variation_coeff is not None):
          vis_img_shape = (tf.shape(self.vis_pert)[0], int(np.sqrt(self.input_shape[1])),
            int(np.sqrt(self.input_shape[1])), 1)
          self.vis_loss = (self.vis_loss + self.variation_coeff
            * tf.reduce_sum(tf.image.total_variation(tf.reshape(self.vis_pert, vis_img_shape))))
        else:
          assert False, ("method " + self.method +" not recognized.\n"+
            "Options are 'erhan'.")

      with tf.variable_scope("optimizer") as scope:
        if(self.optimizer == "adam"):
          self.vis_opt = tf.train.AdamOptimizer(learning_rate=self.step_size)
          # code below ensures that adam variables are also reset when the reset op is run
          initializer_ops = [v.initializer for v in self.vis_opt.variables()]
          self.reset = tf.group(initializer_ops + [self.reset])
        elif(self.optimizer == "sgd"):
          self.vis_opt = tf.train.GradientDescentOptimizer(learning_rate=self.step_size)
        else:
          assert False, ("optimizer must be 'adam' or 'sgd', not '"+str(self.optimizer)+"'.")
        grads_and_vars = self.vis_opt.compute_gradients(self.vis_loss, var_list=[self.vis_var])
        self.vis_update_op = self.vis_opt.apply_gradients(grads_and_vars)
        # add optimizer vars to ignore list because the graph was not checkpointed with them
        self.ignore_load_var_list.extend(self.vis_opt.variables())

  def construct_optimal_stimulus(self, feed_dict, selection_vector, save_int=None):
    """
    Construct optimal stimulus
    Inputs:
      feed_dict: session feed dictionary
      selection_vector: numpy vector of length (num_neurons) to select target neurons
      save_int: integer specifying save interval; if None then don't save
    """
    feed_dict = feed_dict.copy()
    if(self.method == "erhan"):
      if selection_vector.ndim == 1:
        selection_vector = selection_vector[:, None] # must be matrix of shape [num_neurons, 1]
      feed_dict[self.selection_vector] = selection_vector
    else:
      assert False, ("method " + self.method +" not recognized.\n"+
        "Options are 'erhan'.")

    if(save_int is None): # If save_int is none, don't save at all
      save_int = self.num_steps + 1
    sess = tf.get_default_session()
    sess.run(self.reset, feed_dict=feed_dict) # Reset input to orig image
    # Always store orig image
    eval_vect = [self.vis_image, self.selected_activities, self.vis_loss]
    eval_out = sess.run(eval_vect, feed_dict)
    [orig_images, orig_activity, loss] = eval_out
    # Init vals ("adv" is input + perturbation)
    out_dict = {}
    out_dict["step"] = [0]
    out_dict["images"] = [orig_images]
    out_dict["loss"] = [loss]
    # Calculate adversarial examples
    for step in range(self.num_steps):
      sess.run(self.vis_update_op, feed_dict)
      if((step+1) % save_int == 0):
        run_list = [self.vis_image, self.vis_loss]
        [vis_images, loss] = sess.run(run_list, feed_dict)
        # +1 since this is post update
        # We do this here since we want to store the last step
        out_dict["step"].append(step+1)
        out_dict["images"].append(vis_images)
        out_dict["loss"].append(loss)
    return out_dict
