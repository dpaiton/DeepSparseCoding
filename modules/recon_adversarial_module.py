import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import utils.data_processing as dp
import pdb

class ReconAdversarialModule(object):
  def __init__(self, data_tensor, use_adv_input, num_steps, step_size, max_step=None, clip_adv=True,
    clip_range = [0.0, 1.0], attack_method="kurakin_targeted", eps=1e-8, variable_scope="recon_adversarial"):
    """
    TODO:
    Adversarial module
    Inputs:
      data_tensor
      name
    Outputs:
      dictionary
    """

    self.data_tensor = data_tensor
    self.use_adv_input = use_adv_input

    self.input_shape = self.data_tensor.get_shape().as_list()

    self.num_steps = num_steps
    self.step_size = step_size
    self.max_step = max_step
    self.clip_adv = clip_adv
    self.clip_range = clip_range
    self.attack_method = attack_method
    self.eps = eps
    #List of vars to ignore in savers/loaders
    self.ignore_load_var_list = []

    self.variable_scope = str(variable_scope)
    self.build_init_graph()

  def build_init_graph(self):
    with tf.variable_scope(self.variable_scope) as scope:
      #These placeholders are here since they're only needed for construct adv examples
      with tf.variable_scope("placeholders") as scope:
        self.adv_target = tf.placeholder(tf.float32, shape=self.input_shape,
          name="adversarial_target_data")
        self.recon_mult = tf.placeholder(tf.float32, shape=(), name="recon_mult")

      with tf.variable_scope("input_var"):
        #Adversarial pertubation
        self.adv_var = tf.Variable(tf.zeros_like(self.data_tensor),
          dtype=tf.float32, trainable=True, validate_shape=False, name="adv_var")

        self.ignore_load_var_list.append(self.adv_var)
        #TODO:
        #NOTE: This will get overwritten in build_adversarial_ops
        self.reset = self.adv_var.initializer

        #Here, adv_var has a fully dynamic shape. We reshape it to give the variable
        #a semi-dymaic shape (i.e., only batch dimension unknown)
        self.adv_var.set_shape([None,] + self.input_shape[1:])

        #Clip pertubations by maximum amount of change allowed
        if(self.max_step is not None):
          max_pert = tfc.upper_bound(tfc.lower_bound(
            self.adv_var, -self.max_step), self.max_step)
        else:
          max_pert = self.adv_var

        self.adv_image = self.data_tensor + max_pert

        if(self.clip_adv):
          self.adv_image = tfc.upper_bound(tfc.lower_bound(
            self.adv_image, self.clip_range[0]), self.clip_range[1])

      with tf.variable_scope("input_switch"):
        ##Switch between adv_image and input placeholder
        #Option to not use switch input if use_adv_input is None
        if(self.use_adv_input is not None):
          self.adv_switch_input = tf.cond(self.use_adv_input,
            true_fn=lambda: self.adv_image, false_fn=lambda: self.data_tensor,
            strict=True)

  def get_adv_input(self):
    return self.adv_switch_input

  def build_adversarial_ops(self, recon):
    with tf.variable_scope(self.variable_scope) as scope:
      self.recon = recon
      with tf.variable_scope("loss") as scope:
        if(self.attack_method == "kurakin_targeted"):
          self.adv_loss = 0.5 * tf.reduce_sum(tf.square(self.adv_target - self.recon))
        elif(self.attack_method == "carlini_targeted"):
          adv_recon_loss = 0.5 * tf.reduce_sum(tf.square(self.adv_target - self.recon))
          input_pert_loss = 0.5 * tf.reduce_sum(tf.square(self.adv_var))
          self.adv_loss = (1-self.recon_mult) * input_pert_loss + self.recon_mult * adv_recon_loss
        else:
          assert False, ("attack_method " + self.attack_method +" not recognized. "+
            "Options are \"kurakin_targeted\" or \"carlini_targeted\"")

      with tf.variable_scope("optimizer") as scope:
        if(self.attack_method == "kurakin_targeted"):
          self.adv_grad = -tf.sign(tf.gradients(self.adv_loss, self.adv_var)[0])
          self.adv_update_op = self.adv_var.assign_add(self.step_size * self.adv_grad)
        elif(self.attack_method == "carlini_targeted"):
          self.adv_opt = tf.train.AdamOptimizer(
            learning_rate = self.step_size)
          self.adv_grad = self.adv_opt.compute_gradients(
            self.adv_loss, var_list=[self.adv_var])
          self.adv_update_op = self.adv_opt.apply_gradients(self.adv_grad)
          #Add adam vars to reset variable
          initializer_ops = [v.initializer for v in self.adv_opt.variables()]
          self.reset = tf.group(initializer_ops + [self.reset])
          #Add adam vars to list of ignore vars
          self.ignore_load_var_list.extend(self.adv_opt.variables())

  def construct_adversarial_examples(self, feed_dict,
    #Optional parameters
    recon_mult=None, #For carlini attack
    target_generation_method="specified",
    target_images=None,
    save_int=None): #Will not save if None

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
    else:
      assert(False)

    #If save_int is none, don't save at all
    if(save_int is None):
      save_int = self.num_steps + 1

    #Reset input to orig image
    sess = tf.get_default_session()
    sess.run(self.reset, feed_dict=feed_dict)
    #Always store orig image
    [orig_img, recon, loss] = sess.run([self.adv_image, self.recon, self.adv_loss],
      feed_dict)
    #Init vals
    out_dict = {}
    out_dict["step"] = [0]
    out_dict["adv_images"] = [orig_img]
    out_dict["adv_recon"] = [recon]
    out_dict["adv_losses"] = [loss]
    out_dict["input_recon_mses"] = [dp.mse(orig_img, recon)]
    out_dict["input_adv_mses"] = [dp.mse(orig_img, orig_img)]
    out_dict["target_recon_mses"] = [dp.mse(target_images, recon)]
    out_dict["target_adv_mses"] = [dp.mse(target_images, orig_img)]
    out_dict["adv_recon_mses"] = [dp.mse(orig_img, recon)]
    out_dict["target_adv_angles"] = [dp.cos_similarity(target_images, orig_img)]
    out_dict["input_adv_angles"] = [dp.cos_similarity(orig_img, orig_img)]

    #calculate adversarial examples
    for step in range(self.num_steps):
      sess.run(self.adv_update_op, feed_dict)
      if((step+1) % save_int == 0):
        [adv_img, recon, loss] = sess.run([self.adv_image, self.recon, self.adv_loss], feed_dict)
        #+1 since this is post update
        #We do this here since we want to store the last step
        out_dict["step"].append(step+1)
        out_dict["adv_images"].append(adv_img)
        out_dict["adv_recon"].append(recon)
        out_dict["adv_losses"].append(loss)

        out_dict["input_recon_mses"].append(dp.mse(orig_img, recon))
        out_dict["input_adv_mses"].append(dp.mse(orig_img, adv_img))
        out_dict["target_recon_mses"].append(dp.mse(target_images, recon))
        out_dict["target_adv_mses"].append(dp.mse(target_images, adv_img))
        out_dict["adv_recon_mses"].append(dp.mse(adv_img, recon))
        out_dict["target_adv_angles"].append(dp.cos_similarity(target_images, adv_img))
        out_dict["input_adv_angles"].append(dp.cos_similarity(orig_img, adv_img))

    return out_dict
