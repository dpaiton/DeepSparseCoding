import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import pdb

class ClassAdversarialModule(object):
  def __init__(self, data_tensor, use_adv_input, num_classes, num_steps, step_size, max_step=None, clip_adv=True,
    clip_range = [0.0, 1.0], attack_method="kurakin_untargeted", eps=1e-8, name="class_adversarial"):
    """
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
    self.attack_method = attack_method
    self.clip_range = clip_range
    self.eps = eps

    self.name = str(name)
    self.build_init_graph()

    #Define session to use for finding

  def build_init_graph(self):
    #These placeholders are here since they're only needed for construct adv examples
    with tf.name_scope("placeholders") as scope:
      self.adv_target = tf.placeholder(tf.float32, shape=[None, self.num_classes],
        name="adversarial_target_data")
      self.recon_mult = tf.placeholder(tf.float32, shape=(), name="recon_mult")

    with tf.name_scope("input_var"):
      #Adversarial pertubation
      self.adv_var = tf.Variable(tf.zeros_like(self.data_tensor),
        dtype=tf.float32, trainable=True, validate_shape=False)
      self.reset_adv_var = self.adv_var.initializer
      #Here, adv_var has a fully dynamic shape. We reshape it to give the variable
      #a semi-dymaic shape (i.e., only batch dimension unknown)
      self.adv_var.set_shape([None,] + self.input_shape[1:])

      #Clip pertubations by maximum amount of change allowed
      if(self.max_step is not None):
        max_pert = tfc.upper_bound(tfc.lower_bound(
          self.adv_var, -self.max_step), self.max_step)
        #max_pert = tf.clip_by_value(reshape_adv_var, -self.max_step, self.max_step)
      else:
        max_pert = self.adv_var

      self.adv_image = self.data_tensor + max_pert

      if(self.clip_adv):
        self.adv_image = tfc.upper_bound(tfc.lower_bound(
          self.adv_image, self.clip_range[0]), self.clip_range[1])
        #self.adv_image = tf.clip_by_value(self.adv_image, self.clip_range[0],
        #  self.clip_range[1])

    with tf.name_scope("input_switch"):
      ##Switch between adv_image and input placeholder
      self.adv_switch_input = tf.cond(self.use_adv_input,
        true_fn = lambda: self.adv_image, false_fn = lambda: self.data_tensor,
        strict=True)
      #tiled_bool = self.use_adv_input * tf.ones(self.input_shape)
      #self.adv_switch_input = tf.where(tiled_bool,
      #  self.adv_image, self.data_tensor)

  def get_adv_input(self):
    return self.adv_switch_input

  def build_adversarial_ops(self, label_est, label_tensor=None, model_logits=None, loss=None):
    self.label_est = label_est
    with tf.name_scope("loss") as scope:
      if(self.attack_method == "kurakin_untargeted"):
        self.adv_loss = -loss
      elif(self.attack_method == "kurakin_targeted"):
        self.adv_loss = -tf.reduce_sum(tf.multiply(self.adv_target,
          tf.log(tf.clip_by_value(label_est, self.eps, 1.0))))
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

    with tf.name_scope("optimizer") as scope:
      if(self.attack_method == "kurakin_untargeted" or self.attack_method == "kurakin_targeted"):
        self.adv_grad = -tf.sign(tf.gradients(self.adv_loss, self.adv_var)[0])
        self.adv_update_op = self.adv_var.assign_add(
          self.step_size * self.adv_grad)
      elif(self.attack_method == "carlini_untargeted"):
        adv_opt = tf.train.AdamOptimizer(
          learning_rate = self.step_size)
        self.adv_grad = adv_opt.compute_gradients(
          self.adv_loss, var_list=[self.adv_var])
        self.adv_update_op = adv_opt.apply_gradients(self.adv_grad)

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
    num_labels = target_classes.shape[0]
    out = np.zeros((num_labels, self.num_classes))
    target_label_idx = (np.arange(num_labels).astype(np.int32), target_classes)
    out[target_label_idx] = 1
    return out

  #This feed_dict must contain elements generated by gen_feed_dict in here
  def construct_adversarial_examples(self, feed_dict,
    #Optional parameters
    labels=None,  #For untargeted attacks
    recon_mult=None, #For carlini attack
    rand_state=None,
    target_generation_method="random",
    save_int=None): #Will not save if None

    feed_dict = feed_dict.copy()
    feed_dict[self.use_adv_input] = True
    if(self.attack_method == "kurakin_untargeted"):
      pass
    elif(self.attack_method == "kurakin_targeted"):
      #TODO allow for specified targets
      assert(target_generation_method == "random")
      feed_dict[self.adv_target] = self.generate_random_target_labels(labels, rand_state)
    elif(self.attack_method == "carlini_targeted"):
      feed_dict[self.adv_target] = self.generate_random_target_labels(labels, rand_state)
      feed_dict[self.recon_mult] = recon_mult
    else:
      assert(False)

    #If save_int is none, don't save at all
    if(save_int is None):
      save_int = self.num_steps + 1

    adversarial_images = []
    adversarial_output = []

    #Reset input to orig image
    sess = tf.get_default_session()
    sess.run(self.reset_adv_var, feed_dict=feed_dict)
    #Always store orig image
    [img, output] = sess.run([self.adv_image, self.label_est], feed_dict)
    adversarial_images.append(img)
    adversarial_output.append(output)
    #calculate adversarial examples
    for step in range(self.num_steps):
      sess.run(self.adv_update_op, feed_dict)
      if(step+1 % save_int == 0):
        [img, output] = sess.run([self.adv_image, self.label_est], feed_dict)
        adversarial_images.append(img)
        adversarial_output.append(output)

    return adversarial_images, adversarial_output
