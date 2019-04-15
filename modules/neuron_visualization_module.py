import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import utils.data_processing as dp

class NeuronVisualizationModule(object):
  def __init__(self, input_shape, num_steps, step_size, clip_output=True, clip_range=[0.0, 1.0],
    vis_method="erhan", vis_optimizer="sgd", variable_scope="neuron_visualization"):
    """
    Neuron optimal input visualization  module
    Inputs:
      num_steps: How many optimization steps to use
      step_size: How big of a perturbation to use
      clip_output: [bool] If True, clip output images to clip_range
      clip_range: [tuple or list] (min, max) values for adversarial images
      vis_method: [str] allowed values are:
        "erhan"
      vis_optimizer: [str] specifying "sgd" or "adam" for the optimizer
      variable_scope: [str] scope for adv module graph operators
    """
    self.input_shape = input_shape
    self.num_steps = num_steps
    self.step_size = step_size
    self.clip_output = clip_output
    self.clip_range = clip_range
    self.vis_method = vis_method
    self.vis_optimizer = vis_optimizer.lower()
    self.ignore_load_var_list = [] # List of vars to ignore in savers/loaders
    self.variable_scope = str(variable_scope)
    self.build_init_graph()

  def build_init_graph(self):
    with tf.variable_scope(self.variable_scope) as scope:
      # These placeholders are here since they're only needed for construct adv examples
      with tf.variable_scope("placeholders") as scope:
        self.target_layer = tf.placeholder(tf.float32, shape=(), name="target_layer")
        self.selection_vector = tf.placeholder(tf.float32, shape=(None),
          name="selection_vector")

      with tf.variable_scope("input_var"):
        vis_init = tf.random.normal(shape=self.input_shape, mean=0.0, stddev=1e-4, name="vis_init")
        self.vis_var = tf.Variable(vis_init, dtype=tf.float32, trainable=True,
          validate_shape=False, name="vis_var")
        self.ignore_load_var_list.append(self.vis_var)
        self.reset = self.vis_var.initializer
        if(self.clip_output): # Clip final visualization image to bound specified
          self.vis_image = tfc.upper_bound(tfc.lower_bound(
            self.vis_var, self.clip_range[0]), self.clip_range[1])

  def get_vis_input(self):
    return self.vis_image

  def build_visualization_ops(self, neuron_activities_list)
    """
    Build loss & optimization ops into graph
    Inputs:
    neuron_activities_list: list of tf variable for neuron activations for each layer
    """
    with tf.variable_scope(self.variable_scope) as scope:
      self.neuron_activities_list = neuron_activities_list
      with tf.variable_scope("loss") as scope:
        if(self.vis_method == "erhan"):
          selected_layer = self.neuron_activities_list[self.target_layer]
          self.selected_activities = tf.matmul(selected_layer, self.selection_vector,
            name="selected_activities")
          self.vis_loss = -tf.reduce_sum(self.selected_activities)
        else:
          assert False, ("vis_method " + self.vis_method +" not recognized.\n"+
            "Options are 'erhan'.")

      with tf.variable_scope("optimizer") as scope:
        if(self.vis_optimizer == "adam"):
          self.vis_opt = tf.train.AdamOptimizer(learning_rate=self.step_size)
          # code below ensures that adam variables are also reset when the reset op is run
          initializer_ops = [v.initializer for v in self.vis_opt.variables()]
          self.reset = tf.group(initializer_ops + [self.reset])
        elif(self.vis_optimizer == "sgd"):
          self.vis_opt = tf.train.GradientDescentOptimizer(learning_rate=self.step_size)
        else:
          assert False, ("optimizer must be 'adam' or 'sgd', not '"+str(self.vis_optimizer)+"'.")
        grads_and_vars = self.vis_opt.compute_gradients(self.vis_loss, var_list=[self.vis_var])
        self.vis_update_op = self.vis_opt.apply_gradients(grads_and_vars)
        # add optimizer vars to ignore list because the graph was not checkpointed with them
        self.ignore_load_var_list.extend(self.vis_opt.variables())

  def construct_optimal_stimulus(self, feed_dict, target_layer, selection_vector, save_int=None):
    """
    Construct optimal stimulus
    Inputs:
      feed_dict: session feed dictionary
      target_layer: integer specifying the target layer to select neurons from
      selection_vector: numpy vector of length (num_neurons) to select target neurons
      save_int: integer specifying save interval; if None then don't save
    """
    feed_dict = feed_dict.copy()
    if(self.vis_method == "erhan"):
      feed_dict[self.target_layer] = target_layer
      if selection_vector.ndim == 1:
        selection_vector = selection_vector[:, None] # must be matrix of shape [num_neurons, 1]
      feed_dict[self.selection_vector] = selection_vector
    else:
      assert False, ("vis_method " + self.vis_method +" not recognized.\n"+
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
        run_list = [self.vis_images, self.vis_loss]
        [adv_images, recons, loss, perturbations] = sess.run(run_list, feed_dict)
        # +1 since this is post update
        # We do this here since we want to store the last step
        out_dict["step"].append(step+1)
        out_dict["images"].append(adv_images)
        out_dict["loss"].append(loss)
    return out_dict
