import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import utils.data_processing as dp
import pdb

class ClassAdversarialModule(object):
  def __init__(self, data_tensor, use_adv_input, num_classes, num_steps, step_size, max_step=None, clip_adv=True,
    clip_range = [0.0, 1.0], attack_method="kurakin_untargeted", eps=1e-8, variable_scope="class_adversarial"):
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

    self.num_classes = num_classes
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
        self.adv_target = tf.placeholder(tf.float32, shape=[None, self.num_classes],
          name="adversarial_target_data")
        self.recon_mult = tf.placeholder(tf.float32, shape=(), name="recon_mult")

      with tf.variable_scope("input_var"):
        #Adversarial pertubation
        self.adv_var = tf.Variable(tf.zeros_like(self.data_tensor),
          dtype=tf.float32, trainable=True, validate_shape=False, name="adv_var")

        self.ignore_load_var_list.append(self.adv_var)
        #TODO:
        #NOTE: This will get overwritten in build_adersarialv_ops
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
        self.adv_switch_input = tf.cond(self.use_adv_input,
          true_fn=lambda: self.adv_image, false_fn=lambda: self.data_tensor,
          strict=True)

  def get_adv_input(self):
    return self.adv_switch_input

  def build_adversarial_ops(self, label_est, label_tensor=None, model_logits=None, loss=None):
    with tf.variable_scope(self.variable_scope) as scope:
      self.label_est = label_est
      with tf.variable_scope("loss") as scope:
        if(self.attack_method == "kurakin_untargeted"):
          self.adv_loss = tf.reduce_sum(-loss, name="sum_loss")
        elif(self.attack_method == "kurakin_targeted"):
          self.adv_loss = -tf.reduce_sum(tf.multiply(self.adv_target,
            tf.log(tf.clip_by_value(self.label_est, self.eps, 1.0))))
        elif(self.attack_method == "carlini_targeted"):
          self.input_pert_loss = 0.5 * tf.reduce_sum(
            tf.square(self.adv_var), name="input_perturbed_loss")

          #Assuming adv_target is one hot
          with tf.control_dependencies([
            tf.assert_equal(tf.reduce_sum(self.adv_target, axis=-1), 1.0)]):
            self.adv_target = self.adv_target

          #Construct two boolean masks, one with only target class as true
          #and one with everything not target class
          target_mask = self.adv_target > .5
          not_target_mask = self.adv_target < .5

          #Z(x)_t
          #boolean_mask returns a flattened array, so need to reshape back
          logits_target_val = tf.boolean_mask(model_logits, target_mask)[:, None]
          #max_{i!=t} Z(x)_i
          logits_not_target_val = tf.boolean_mask(model_logits, not_target_mask)
          logits_not_target_val = tf.reshape(logits_not_target_val,
            [-1, self.num_classes-1])

          max_logits_not_target_val = tf.reduce_max(logits_not_target_val, axis=-1)

          self.target_class_loss = tf.reduce_sum(tf.nn.relu(
            max_logits_not_target_val - logits_target_val))

          self.adv_loss = self.input_pert_loss + \
            self.recon_mult * self.target_class_loss
        else:
          assert False, ("attack_method " + self.attack_method +" not recognized. "+
            "Options are \"kurakin_untargeted\", \"kurakin_targeted\", or \"carlini_targeted\"")

      with tf.variable_scope("optimizer") as scope:
        if(self.attack_method == "kurakin_untargeted" or self.attack_method == "kurakin_targeted"):
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

  def generate_random_target_labels(self, input_labels, rand_state=None):
    input_classes = np.argmax(input_labels, axis=-1)
    target_classes = input_classes.copy()
    while(np.any(target_classes == input_classes)):
      resample_idx = np.nonzero(target_classes == input_classes)
      if(rand_state is not None):
        target_classes[resample_idx] = rand_state.randint(0, self.num_classes,
          size=resample_idx[0].shape)
      else:
        target_classes[resample_idx] = np.random.randint(0, self.num_classes,
          size=resample_idx[0].shape)
    #Convert to one hot
    return dp.dense_to_one_hot(target_classes, self.num_classes)

  def construct_adversarial_examples(self, feed_dict,
    #Optional parameters
    labels=None,  #For untargeted attacks
    recon_mult=None, #For carlini attack
    rand_state=None,
    target_generation_method="random",
    target_labels=None,
    save_int=None): #Will not save if None

    feed_dict = feed_dict.copy()
    feed_dict[self.use_adv_input] = True
    if(self.attack_method == "kurakin_untargeted"):
      pass
    elif(self.attack_method == "kurakin_targeted"):
      if(target_generation_method == "random"):
        feed_dict[self.adv_target] = self.generate_random_target_labels(labels, rand_state)
      else:
        feed_dict[self.adv_target] = target_labels
    elif(self.attack_method == "carlini_targeted"):
      if(target_generation_method == "random"):
        feed_dict[self.adv_target] = self.generate_random_target_labels(labels, rand_state)
      else:
        feed_dict[self.adv_target] = target_labels
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
    [orig_img, output, loss] = sess.run([self.adv_image, self.label_est, self.adv_loss],
      feed_dict)
    #Init vals
    out_dict = {}
    out_dict["step"] = [0]
    out_dict["adv_images"] = [orig_img]
    out_dict["adv_outputs"] = [output]
    out_dict["adv_losses"] = [loss]

    reduc_dim = tuple(range(1, len(orig_img.shape)))
    out_dict["input_adv_mses"]= [np.mean((orig_img-orig_img)**2, axis=reduc_dim)]

    #calculate adversarial examples
    for step in range(self.num_steps):
      sess.run(self.adv_update_op, feed_dict)
      if((step+1) % save_int == 0):
        [adv_img, output, loss] = sess.run([self.adv_image, self.label_est, self.adv_loss], feed_dict)
        #+1 since this is post update
        #We do this here since we want to store the last step
        out_dict["step"].append(step+1)
        out_dict["adv_images"].append(adv_img)
        out_dict["adv_outputs"].append(output)
        out_dict["adv_losses"].append(loss)
        #Everything but batch dim
        out_dict["input_adv_mses"].append(np.mean((orig_img-adv_img)**2, axis=reduc_dim))

    return out_dict
