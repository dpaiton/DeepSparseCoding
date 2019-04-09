import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import utils.data_processing as dp
import pdb

class ReconAdversarialModule(object):
  def __init__(self, data_tensor, use_adv_input, num_steps, step_size, max_adv_change=None,
    clip_adv=True, clip_range=[0.0, 1.0], attack_method="kurakin_targeted",
    carlini_change_variable=False, adv_optimizer="sgd", variable_scope="recon_adversarial"):
    """
    Adversarial module
    Inputs:
      data_tensor: Input images to be perturbed, of size [batch, pixels]
      use_adv_input: Flag to (True) use adversarial ops or (False) passthrough data_tensor
      num_steps: How many adversarial steps to use
      step_size: How big of an adversarial perturbation to use
      max_adv_change: Maximum perturbation size (None to have no limit)
      clip_adv: [bool] If True, clip adversarial images to clip_range
      clip_range: [tuple or list] (min, max) values for adversarial images
      attack_method: [str] allowed values are:
        "kurakin_targeted", "carlini_targeted", "marzi_untargeted", "marzi_latent"
      carlini_change_variables: [bool] if True, follow the change of variable recommendation
        described in carlini & wagner (2017) Section V, subsection B, number 3
      adv_optimizer: [str] specifying "sgd" or "adam" for the carlini optimizer
      variable_scope: [str] scope for adv module graph operators
    """
    self.data_tensor = data_tensor
    self.use_adv_input = use_adv_input
    self.input_shape = self.data_tensor.get_shape().as_list()
    self.num_steps = num_steps
    self.step_size = step_size
    # TODO: This is an upper and lower bound, so max_change is a better name
    self.max_adv_change = max_adv_change
    self.clip_adv = clip_adv
    self.clip_range = clip_range
    self.attack_method = attack_method
    self.carlini_change_variable = carlini_change_variable
    self.adv_optimizer = adv_optimizer.lower()
    # List of vars to ignore in savers/loaders
    self.ignore_load_var_list = []
    self.variable_scope = str(variable_scope)
    self.build_init_graph()

  def build_init_graph(self):
    with tf.variable_scope(self.variable_scope) as scope:
      # These placeholders are here since they're only needed for construct adv examples
      with tf.variable_scope("placeholders") as scope:
        self.adv_target = tf.placeholder(tf.float32, shape=self.input_shape,
          name="adversarial_target_data")
        self.recon_mult = tf.placeholder(tf.float32, shape=(), name="recon_mult")
        if(self.attack_method == "marzi_untargeted"):
          self.orig_recons = tf.placeholder(tf.float32, shape=self.input_shape,
            name="orig_recons")

      with tf.variable_scope("input_var"):
        # Adversarial pertubation

        # Initialize perturbation to data tensor
        adv_init = tf.identity(self.data_tensor)
        if(self.carlini_change_variable):
          # adv_init is the inverse of the change-of-variable, so that the final pert init is 0
          adv_init = 2 * adv_init - 1 # Map adv_init to be between -1 and 1
          adv_init = 0.5 * tf.log((1 + adv_init) / (1 - adv_init)) # tanh^-1(x)

        self.adv_var = tf.Variable(adv_init, dtype=tf.float32, trainable=True,
          validate_shape=False, name="adv_var")

        self.ignore_load_var_list.append(self.adv_var)
        self.reset = self.adv_var.initializer

        # Here, adv_var has a fully dynamic shape. We reshape it to give the variable
        # a semi-dymaic shape (i.e., only batch dimension unknown)
        self.adv_var.set_shape([None,] + self.input_shape[1:])

        if(self.carlini_change_variable):
          carlini_change_pert = 0.5 * (tf.tanh(self.adv_var) + 1) - self.data_tensor
        else:
          carlini_change_pert = self.adv_var - self.data_tensor

        # Clip pertubations by maximum amount of change allowed
        if(self.max_adv_change is not None):
          self.clipped_pert = tfc.upper_bound(tfc.lower_bound(
            carlini_change_pert, -self.max_adv_change), self.max_adv_change)
        else:
          self.clipped_pert = carlini_change_pert

        self.adv_images = self.clipped_pert + self.data_tensor

        if(self.clip_adv):
          self.adv_images = tfc.upper_bound(tfc.lower_bound(
            self.adv_images, self.clip_range[0]), self.clip_range[1])

      with tf.variable_scope("input_switch"):
        self.adv_switch_input = tf.cond(self.use_adv_input,
          true_fn=lambda:self.adv_images, false_fn=lambda:self.data_tensor, strict=True)

  def get_adv_input(self):
    return self.adv_switch_input

  def build_adversarial_ops(self, recons):
    with tf.variable_scope(self.variable_scope) as scope:
      self.recons = recons
      with tf.variable_scope("loss") as scope:
        """
        TODO:
        * When computing gratings wrt weights (e.g. madry defense), you would want to avg or sum across batch, because you want the weights to defend against all of the attacks.
        * When coputing attacks for a batch, you want to compute the loss for each individual image, and optimize them independently. Otherwise, when you compute the average/sum loss then the gradients can find solutions where some of the images are successful attacks and others are not. The adv losses per image are independent of each other, so the solution for the summed loss might not be the same as the solution for each loss individually. We should do `tf.group([grad(loss(x1),x1), grad(loss(x2), x2), grad(loss(x3), x3), ...])` or we can make the loss `[batch, 1]` instead of a scalar and `tf.gradients` will take care of it? 
        * The gradient wrt e.g. x0 is not going to be affected by whether there is a sum or not?
        * update: giving the loss a batch dim does work for attacks, but we still need to build in a fix for the madry defense
        """
        #adv_recon_loss = 0.5 * tf.reduce_sum(tf.square(self.adv_target - self.recons),
        #  axis=list(range(1, len(self.adv_var.shape))))
        adv_recon_loss = 0.5 * tf.reduce_sum(tf.square(self.adv_target - self.recons))

        if(self.attack_method == "kurakin_targeted"):
          self.adv_loss = adv_recon_loss

        elif(self.attack_method == "carlini_targeted"):
          #input_pert_loss = 0.5 * tf.reduce_sum(tf.square(self.clipped_pert),
          #  axis=list(range(1, len(self.clipped_pert.shape))))
          input_pert_loss = 0.5 * tf.reduce_sum(tf.square(self.clipped_pert))

          #self.adv_loss = (1-self.recon_mult) * input_pert_loss + self.recon_mult * adv_recon_loss
          self.adv_loss =  input_pert_loss + self.recon_mult * adv_recon_loss

        elif(self.attack_method == "marzi_untargeted"):
          # optimizer uses grad descent, so we need negative loss to maximize
          # add eps to avoid 0 solution
          self.adv_loss = -tf.reduce_sum(tf.sqrt(tf.square(self.recons - self.orig_recons + 1e-12)))

        else:
          assert False, ("attack_method " + self.attack_method +" not recognized. "+
            "Options are \"kurakin_targeted\", \"carlini_targeted\", or \"marzi_untargeted\"")

      with tf.variable_scope("optimizer") as scope:
        if(self.adv_optimizer == "adam"):
          self.adv_opt = tf.train.AdamOptimizer(learning_rate=self.step_size)
          # code below ensures that adam variables are also reset when the reset op is run
          initializer_ops = [v.initializer for v in self.adv_opt.variables()]
          self.reset = tf.group(initializer_ops + [self.reset])
        elif(self.adv_optimizer == "sgd"):
          self.adv_opt = tf.train.GradientDescentOptimizer(learning_rate=self.step_size)
        else:
          assert False, ("adv_optimizer must be 'adam' or 'sgd'")

        adv_grads_and_vars = self.adv_opt.compute_gradients(self.adv_loss, var_list=[self.adv_var])
        if(self.attack_method == "kurakin_targeted"):
          adv_grads_and_vars = [(tf.sign(gv[0]), gv[1]) for gv in adv_grads_and_vars]

        self.adv_update_op = self.adv_opt.apply_gradients(adv_grads_and_vars)

        # add optimizer vars to ignore list because the graph was not checkpointed with them
        self.ignore_load_var_list.extend(self.adv_opt.variables())

  def construct_adversarial_examples(self, feed_dict,
    # Optional parameters
    recon_mult=None, # For carlini attack
    target_generation_method="specified",
    target_images=None,
    save_int=None): # Will not save if None
    feed_dict = feed_dict.copy()
    feed_dict[self.use_adv_input] = True
    if(self.attack_method == "kurakin_targeted"):
      if(target_generation_method == "random"):
        assert False, ("target_generation_method must be specified")
      else:
        feed_dict[self.adv_target] = target_images
    elif(self.attack_method == "carlini_targeted"):
      if(target_generation_method == "random"):
        assert False, ("target_generation_method must be specified")
      else:
        feed_dict[self.adv_target] = target_images
      feed_dict[self.recon_mult] = recon_mult
    # If save_int is none, don't save at all
    if(save_int is None):
      save_int = self.num_steps + 1
    # Reset input to orig image
    sess = tf.get_default_session()
    sess.run(self.reset, feed_dict=feed_dict)
    # Always store orig image
    [orig_images, recons] = sess.run([self.data_tensor, self.recons], feed_dict)
    if(self.attack_method == "marzi_untargeted"):
      feed_dict[self.orig_recons] = recons
    loss = sess.run(self.adv_loss, feed_dict)
    # Init vals ("adv" is input + perturbation)
    out_dict = {}
    out_dict["step"] = [0]
    out_dict["adv_images"] = [orig_images]
    out_dict["adv_recons"] = [recons]
    out_dict["adv_loss"] = [loss]
    # Mean Squared Error between different images of interest
    out_dict["input_recon_mses"] = [dp.mse(orig_images, recons)]
    out_dict["input_adv_mses"] = [dp.mse(orig_images, orig_images)]
    out_dict["target_recon_mses"] = [dp.mse(target_images, recons)]
    out_dict["target_adv_mses"] = [dp.mse(target_images, orig_images)]
    out_dict["adv_recon_mses"] = [dp.mse(orig_images, recons)]
    # Cosine similarity between different vectors of interest
    out_dict["target_adv_sim"] = [dp.cos_similarity(target_images, orig_images)]
    out_dict["input_adv_sim"] = [dp.cos_similarity(orig_images, orig_images)]
    out_dict["target_pert_sim"] = []
    out_dict["input_pert_sim"] = []
    # Calculate adversarial examples
    for step in range(self.num_steps):
      sess.run(self.adv_update_op, feed_dict)
      if((step+1) % save_int == 0):
        run_list = [self.adv_images, self.recons, self.adv_loss, self.clipped_pert]
        [adv_images, recons, loss, perturbations] = sess.run(run_list, feed_dict)
        # +1 since this is post update
        # We do this here since we want to store the last step
        out_dict["step"].append(step+1)
        out_dict["adv_images"].append(adv_images)
        out_dict["adv_recons"].append(recons)
        out_dict["adv_loss"].append(loss)
        out_dict["input_recon_mses"].append(dp.mse(orig_images, recons))
        out_dict["input_adv_mses"].append(dp.mse(orig_images, adv_images))
        out_dict["adv_recon_mses"].append(dp.mse(adv_images, recons))
        out_dict["input_adv_sim"].append(dp.cos_similarity(orig_images, adv_images))
        if not np.all(perturbations == 0):
          out_dict["input_pert_sim"].append(dp.cos_similarity(orig_images, perturbations))
        else:
          out_dict["input_pert_sim"].append(None)
        if target_images is not None:
          out_dict["target_recon_mses"].append(dp.mse(target_images, recons))
          out_dict["target_adv_mses"].append(dp.mse(target_images, adv_images))
          out_dict["target_adv_sim"].append(dp.cos_similarity(target_images, adv_images))
          if not np.all(perturbations == 0):
            out_dict["target_pert_sim"].append(dp.cos_similarity(target_images, perturbations))
          else:
            out_dict["target_pert_sim"].append(None)
        else:
          out_dict["target_recon_mses"].append(None)
          out_dict["target_adv_mses"].append(None)
          out_dict["target_adv_sim"].append(None)
          out_dict["target_pert_sim"].append(None)
    return out_dict
