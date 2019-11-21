import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
from models.base_model import Model
from modules.mlp_module import MlpModule
from modules.class_adversarial_module import ClassAdversarialModule
from modules.activations import activation_picker

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
    self.input_shape = [None,] + self.params.data_shape
    self.label_shape = [None, self.params.num_classes]
    self.mlp_act_funcs = [activation_picker(act_func_str)
      for act_func_str in self.params.mlp_activation_functions]

  def build_adv_module(self, input_node):
    #Placeholders for using adv or clean examples
    with tf.compat.v1.variable_scope("placeholders") as scope:
      #This is a swith used internally to use clean or adv examples
      self.use_adv_input = tf.compat.v1.placeholder(tf.bool, shape=(), name="use_adv_input")
    with tf.compat.v1.variable_scope("auto_placeholders") as scope:
      #This is a schedule flag to determine if we're training on adv examples
      self.train_on_adversarial = tf.compat.v1.placeholder(tf.bool, shape=(), name="train_on_adversarial")
    self.adv_module = ClassAdversarialModule(input_node, self.use_adv_input,
      self.params.num_classes, self.params.adversarial_num_steps, self.params.adversarial_step_size,
      max_step=self.params.adversarial_max_change,
      clip_adv=self.params.adversarial_clip, clip_range=self.params.adversarial_clip_range,
      attack_method=self.params.adversarial_attack_method,
      eps=self.params.eps)
    return self.adv_module.get_adv_input()

  def build_graph(self):
    input_node = self.build_input_placeholder()
    #Always build adv module
    #Uses flag to control using clean/adv examples for training
    with tf.device(self.params.device):
      with self.graph.as_default():
        input_node = self.build_adv_module(input_node)
    input_node = self.normalize_input(input_node)
    self.build_graph_from_input(input_node)
    with tf.device(self.params.device):
      with self.graph.as_default():
        #Build the rest of the ops here since we need nodes from this graph
        #Use sum loss here for untargted
        self.adv_module.build_adversarial_ops(self.label_est,
          model_logits=self.get_encodings(),
          label_gt=self.label_placeholder)
    #Add adv module ignore list to model ignore list
    self.full_model_load_ignore.extend(self.adv_module.ignore_load_var_list)

  def build_mlp_module(self, input_node):
    module = MlpModule(input_node, self.label_placeholder, self.params.mlp_layer_types,
      self.params.mlp_output_channels, self.params.batch_norm, self.mlp_dropout_keep_probs,
      self.params.max_pool, self.params.max_pool_ksize, self.params.max_pool_strides,
      self.params.mlp_patch_size, self.params.mlp_conv_strides, self.mlp_act_funcs,
      self.params.eps, lrn=self.params.lrn, loss_type="softmax_cross_entropy",
      decay_mult=self.params.mlp_decay_mult, norm_mult=self.params.mlp_norm_mult)
    return module

  def build_graph_from_input(self, input_node):
    """
    Build an MLP TensorFlow Graph.
    """
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.compat.v1.variable_scope("label_placeholders") as scope:
          self.label_placeholder = tf.compat.v1.placeholder(tf.float32, shape=self.label_shape,
            name="input_labels")
        with tf.compat.v1.variable_scope("placeholders") as scope:
          self.mlp_dropout_keep_probs = tf.compat.v1.placeholder(tf.float32, shape=[None],
            name="mlp_dropout_keep_probs")
        self.mlp_module = self.build_mlp_module(input_node)
        self.trainable_variables.update(self.mlp_module.trainable_variables)
        #TODO analysis depends on this name for label ests. Can we abstract this?
        self.label_est = tf.identity(self.mlp_module.label_est, name="label_est")
        with tf.compat.v1.variable_scope("performance_metrics") as scope:
          with tf.compat.v1.variable_scope("prediction_bools"):
            self.correct_prediction = tf.equal(tf.argmax(self.label_est, axis=1),
              tf.argmax(self.label_placeholder, axis=1), name="individual_accuracy")
          with tf.compat.v1.variable_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
              tf.float32), name="avg_accuracy")

  def get_input_shape(self):
    return self.input_shape

  def get_num_latent(self):
    # returns the size of the logit (pre softmax) layer
    return self.params.mlp_output_channels[-1]

  def get_encodings(self):
    # returns the logit (pre softmax) layer
    return self.mlp_module.layer_list[-1]

  def get_total_loss(self):
    return self.mlp_module.total_loss

  #def modify_input(self, feed_dict):
  def modify_input(self, feed_dict, train_on_adversarial):
    sess = tf.compat.v1.get_default_session()
    if train_on_adversarial:
      #TODO add in rand_state and target_generation_method here
      #Generate adversarial examples to store within internal variable
      self.adv_module.construct_adversarial_examples(feed_dict,
        labels=feed_dict[self.label_placeholder],
        recon_mult=self.params.carlini_recon_mult,
        rand_state=None, target_generation_method="random")
    else:
      sess.run(self.adv_module.reset, feed_dict)

  def get_feed_dict(self, input_data, input_labels=None, dict_args=None, is_test=False):
    feed_dict = super(MlpModel, self).get_feed_dict(input_data, input_labels, dict_args, is_test)
    if(is_test): # Turn off dropout when not training
      feed_dict[self.mlp_dropout_keep_probs] = [1.0,] * len(self.params.mlp_dropout)
    else:
      feed_dict[self.mlp_dropout_keep_probs] = self.params.mlp_dropout
    #train_on_adversarial is not built in analyzers, so only set this var if this exists
    if(hasattr(self, 'train_on_adversarial')):
      if(feed_dict[self.train_on_adversarial]):
        feed_dict[self.use_adv_input] = True
      else:
        feed_dict[self.use_adv_input] = False
    return feed_dict

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
    sess = tf.compat.v1.get_default_session()
    train_on_adversarial = feed_dict[self.train_on_adversarial]
    if(train_on_adversarial):
      adv_feed_dict = feed_dict.copy()
      adv_feed_dict[self.use_adv_input] = True
      nadv_feed_dict = feed_dict.copy()
      nadv_feed_dict[self.use_adv_input] = False
    current_step = np.array(self.global_step.eval())
    logits_vals = sess.run(self.get_encodings(), feed_dict)
    logits_vals_max = np.array(logits_vals.max())
    logits_frac_act = np.array(np.count_nonzero(logits_vals) / float(logits_vals.size))
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "number_of_batch_steps":self.params.schedule[self.sched_idx]["num_batches"],
      "schedule_index":self.sched_idx,
      "logits_max":logits_vals_max,
      "logits_frac_active":logits_frac_act}
    if(train_on_adversarial):
      adv_accuracy = np.array(self.accuracy.eval(adv_feed_dict))
      nadv_accuracy = np.array(self.accuracy.eval(nadv_feed_dict))
      adv_loss = np.array(self.get_total_loss().eval(adv_feed_dict))
      nadv_loss = np.array(self.get_total_loss().eval(nadv_feed_dict))
      stat_dict["accuracy_adv"] = adv_accuracy
      stat_dict["accuracy_nadv"] = nadv_accuracy
      stat_dict["total_loss_adv"] = adv_loss
      stat_dict["total_loss_nadv"] = nadv_loss
    else:
      accuracy = np.array(self.accuracy.eval(feed_dict))
      total_loss = np.array(self.get_total_loss().eval(feed_dict))
      stat_dict["accuracy"] = accuracy
      stat_dict["total_loss"] = total_loss
    update_dict.update(stat_dict) #stat_dict overwrites
    eval_list = []
    grad_name_list = []
    learning_rate_list = []
    for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
      eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name(1)]
      grad_name = weight_grad_var[0][1].name.split('/')[1].split(':')[0] # 2nd is np.split
      grad_name_list.append(grad_name)
      learning_rate_list.append(self.learning_rates[self.sched_idx][w_idx])
    stat_dict = {}
    out_vals =  tf.compat.v1.get_default_session().run(eval_list, feed_dict)
    out_lr = tf.compat.v1.get_default_session().run(learning_rate_list, feed_dict)
    for grad, name, lr in zip(out_vals, grad_name_list, out_lr):
      grad_max = np.array(grad.max())
      grad_min = np.array(grad.min())
      grad_mean = np.mean(np.array(grad))
      stat_dict[name+"_grad_max_mean_min"] = [grad_max, grad_mean, grad_min]
      stat_dict[name+"_learning_rate"] = lr
    update_dict.update(stat_dict) #stat_dict overwrites for same keys
    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, gradients, etc
    Inputs: input_data and input_labels used for the session
    """
    super(MlpModel, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.get_encodings(), self.mlp_module.weight_list]
    train_on_adversarial = feed_dict[self.train_on_adversarial]
    if(train_on_adversarial):
      eval_list += [self.adv_module.get_adv_input()]
    eval_out = tf.compat.v1.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    filename_suffix = "_v"+self.params.version+"_"+current_step.zfill(5)+".png"
    activity = eval_out[1]
    fig = pf.plot_activity_hist(activity, title="Logit Histogram",
      save_filename=self.params.disp_dir+"act_hist"+filename_suffix)
    #First layer weights
    mlp_weights = eval_out[2]
    w_enc = mlp_weights[0]
    if self.params.mlp_layer_types[0] == "fc":
      w_enc_norm = np.linalg.norm(w_enc, axis=0, keepdims=False)
      # Don't plot weights as images if input is not square
      w_input_sqrt = np.sqrt(w_enc.shape[0])
      if(np.floor(w_input_sqrt) == np.ceil(w_input_sqrt)):
        w_enc = dp.reshape_data(w_enc.T, flatten=False)[0] # [num_neurons, height, width]
        fig = pf.plot_data_tiled(w_enc, normalize=False,
          title="Weights at step "+current_step, vmin=None, vmax=None,
          save_filename=self.params.disp_dir+"w_enc"+filename_suffix)
    else: # conv
      w_enc = np.transpose(dp.rescale_data_to_one(w_enc.T)[0].T, axes=(3,0,1,2))
      if(w_enc.shape[-1] == 1 or w_enc.shape[-1] == 3):
        pf.plot_data_tiled(w_enc, normalize=False, title="Weights at step "+current_step,
          save_filename=self.params.disp_dir+"w_enc"+filename_suffix)
    for (w, tf_w) in zip(mlp_weights, self.mlp_module.weight_list):
      #simplify tensorflow node name to only be the last one
      w_name = tf_w.name.split("/")[-1].split(":")[0]
      num_f = w.shape[-1]
      w_reshape = np.reshape(w, [-1, num_f])
      w_norm = np.linalg.norm(w_reshape, axis=0, keepdims=False)
      fig = pf.plot_bar(w_norm, num_xticks=5,
        title=w_name+" l2 norm", xlabel="w index", ylabel="L2 Norm",
        save_filename=self.params.disp_dir+"w_norm_"+w_name+filename_suffix)
    if(train_on_adversarial):
      adv_input = eval_out[-1]
      adv_input = dp.reshape_data(adv_input, flatten=False)[0]
      fig = pf.plot_data_tiled(adv_input, normalize=False,
        title="Adv inputs at "+current_step,
        save_filename=self.params.disp_dir+"adv_input"+filename_suffix)
