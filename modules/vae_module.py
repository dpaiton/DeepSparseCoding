import numpy as np
import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict
import pdb

class VaeModule(object):
  def __init__(self, data_tensor, output_channels, sparse_mult, decay_mult, kld_mult,
    latent_act_func, noise_level=0, name="VAE"):
    """
    Variational Autoencoder module
    Inputs:
      data_tensor
      params
      output_channels: A list of channels to make, also defines number of layers
      name
    Outputs:
      dictionary
    """
    data_ndim = len(data_tensor.get_shape().as_list())
    assert data_ndim == 2, (
      "Module requires datal_tensor to have shape [batch, num_classes]")

    self.data_tensor = data_tensor

    if data_ndim == 2:
      self.batch_size, self.num_pixels = self.data_tensor.get_shape()
    else:
      assert False, ("Shouldn't get here")

    self.noise_level = noise_level
    if self.noise_level > 0:
      self.corrupt_data = 0.5 * tf.add(tf.random.normal(shape=tf.shape(data_tensor),
        mean=tf.reduce_mean(data_tensor), stddev=noise_level, dtype=tf.float32, name="data_noise"),
        data_tensor)
    else:
      self.corrupt_data = data_tensor

    self.name = str(name)
    self.output_channels = output_channels
    self.sparse_mult = sparse_mult
    self.decay_mult = decay_mult
    self.kld_mult = kld_mult
    self.latent_act_func = latent_act_func

    self.trainable_variables = TrainableVariableDict()
    self.num_encoder_layers = len(self.output_channels)

    self.w_enc_shape = []
    self.b_enc_shape = []
    self.b_dec_shape = []
    prev_input_features = int(self.num_pixels)
    for l in range(self.num_encoder_layers):
      self.w_enc_shape.append([prev_input_features, self.output_channels[l]])
      self.b_enc_shape.append([1, self.output_channels[l]])
      self.b_dec_shape.append([1, prev_input_features])
      prev_input_features = self.output_channels[l]

    self.build_graph()

  def compute_weight_decay_loss(self):
    with tf.name_scope("unsupervised"):
      w_decay_list = [tf.reduce_sum(tf.square(w)) for w in self.w_enc_list]
      w_decay_list += [tf.reduce_sum(tf.square(w)) for w in self.w_dec_list]
      w_decay_list += [tf.reduce_sum(tf.square(self.w_enc_std))]

      decay_loss = tf.multiply(0.5*self.decay_mult,
        tf.add_n(w_decay_list))
    return decay_loss

  def compute_latent_loss(self, a_mean, a_log_std_sq):
    with tf.name_scope("latent"):
      reduc_dim = list(range(1, len(a_mean.shape))) # Want to avg over batch, sum over the rest
      latent_loss= self.kld_mult * -0.5 * tf.reduce_mean(tf.reduce_sum(1 + a_log_std_sq -
        tf.square(a_mean) - tf.exp(a_log_std_sq), reduc_dim))
    return latent_loss

  def compute_recon_loss(self, reconstruction):
    with tf.name_scope("unsupervised"):
      #Want to avg over batch, sum over the rest
      reduc_dim = list(range(1, len(reconstruction.shape)))
      recon_loss = 0.5 * tf.reduce_mean(
        tf.reduce_sum(tf.square(tf.subtract(reconstruction, self.data_tensor)),
        axis=reduc_dim), name="recon_loss")
    return recon_loss

  def compute_sparse_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      reduc_dim = list(range(1, len(a_in.shape))) # Want to avg over batch, sum over the rest
      act_l1 = tf.reduce_mean(tf.reduce_sum(tf.square(a_in), axis=reduc_dim))
      sparse_loss = self.sparse_mult * act_l1
    return sparse_loss

  def build_decoder(self, a_in):
    curr_input = a_in
    out_activations = []
    for (l, (curr_w, curr_b)) in enumerate(zip(self.w_dec_list, self.b_dec_list)):
      curr_output = tf.matmul(curr_input, curr_w) + curr_b

      #Adding relu to mean output
      #TODO fix this with parameter
      if(l < self.num_encoder_layers - 1):
        curr_output = tf.nn.relu(curr_output)
      else:
        #Latent layer
        if(self.latent_act_func == "relu"):
          curr_output = tf.nn.relu(curr_output)
        elif(self.latent_act_func == "none"):
          pass
        else:
          assert False, ("latent_act_func must be \"relu\" or \"none\"")

      out_activations.append(curr_output)
      curr_input = curr_output
    return(out_activations)

  def build_graph(self):
    with tf.name_scope("weight_inits") as scope:
      w_init = []
      b_enc_init = []
      b_dec_init = []
      for l in range(self.num_encoder_layers):
        w_shape = self.w_enc_shape[l]
        b_enc_shape = self.b_enc_shape[l]
        b_dec_shape = self.b_dec_shape[l]

        w_init.append(tf.truncated_normal(w_shape, mean=0.0,
          stddev=0.001, dtype=tf.float32, name="w_init"))
        b_enc_init.append(tf.ones(b_enc_shape)*0.0001)
        b_dec_init.append(tf.ones(b_dec_shape)*0.0001)

    with tf.variable_scope("weights") as scope:
      self.weight_scope = tf.get_variable_scope()
      self.w_enc_list = []
      self.b_enc_list = []
      self.w_dec_list = []
      self.b_dec_list = []
      #Extra weights for std to generate latent space
      self.w_enc_std = None
      self.b_enc_std = None
      for l in range(self.num_encoder_layers):
        #Encoder weights
        self.w_enc_list.append(tf.get_variable(name="w_enc_"+str(l), dtype=tf.float32,
          initializer=w_init[l], trainable=True))
        self.b_enc_list.append(tf.get_variable(name="b_enc_"+str(l), dtype=tf.float32,
          initializer=b_enc_init[l], trainable=True))

        #Decoder weights
        self.w_dec_list.append(tf.get_variable(name="w_dec_"+str(l), dtype=tf.float32,
          initializer=tf.transpose(w_init[l]), trainable=True))
        self.b_dec_list.append(tf.get_variable(name="b_dec_"+str(l), dtype=tf.float32,
          initializer=b_dec_init[l], trainable=True))

        self.trainable_variables[self.w_enc_list[l].name] = self.w_enc_list[l]
        self.trainable_variables[self.b_enc_list[l].name] = self.b_enc_list[l]
        self.trainable_variables[self.w_dec_list[l].name] = self.w_dec_list[l]
        self.trainable_variables[self.b_dec_list[l].name] = self.b_dec_list[l]

      #Std weights
      #l should be last encoder layer, i.e., layer right before latent space
      self.w_enc_std = tf.get_variable(name="w_enc_"+str(l)+"_std", dtype=tf.float32,
        initializer=w_init[l], trainable=True)
      self.b_enc_std = tf.get_variable(name="b_enc_"+str(l)+"_std", dtype=tf.float32,
        initializer=b_enc_init[l], trainable=True)
      self.trainable_variables[self.w_enc_std.name] = self.w_enc_std
      self.trainable_variables[self.b_enc_std.name] = self.b_enc_std

      #Reverse decoder weights to order them in order of operations
      self.w_dec_list = self.w_dec_list[::-1]
      self.b_dec_list = self.b_dec_list[::-1]

    with tf.variable_scope("inference") as scope:
      curr_input = self.corrupt_data
      #Encoder
      self.encoder_activations = []
      for (l, (curr_w, curr_b)) in enumerate(zip(self.w_enc_list, self.b_enc_list)):
        curr_output = tf.matmul(curr_input, curr_w) + curr_b
        if(l == self.num_encoder_layers-1):
          self.latent_std_activation = tf.matmul(curr_input, self.w_enc_std) + self.b_enc_std
        elif(l < self.num_encoder_layers-1):
          curr_output = tf.nn.relu(curr_output)
        else:
          assert(0)
        self.encoder_activations.append(curr_output)
        curr_input = curr_output

      #Calculate latent
      #Std is in log std sq space
      noise  = tf.random_normal(tf.shape(self.latent_std_activation))
      self.a = tf.identity(self.encoder_activations[-1] +
        tf.sqrt(tf.exp(self.latent_std_activation)) * noise, name="activity")

      #TODO tf.placeholder_with_default doesn't allow for dynamic shape
      ##Placeholders for injecting activity
      #self.inject_a_flag = tf.placeholder_with_default(False,
      #  shape=(), name="inject_activation_flag")
      #self.inject_a = tf.placeholder_with_default(
      #  tf.zeros_initializer(), shape=[None, self.output_channels[-1]],
      #  name="inject_activation")
      #curr_input = tf.where(inject_a_flag, inject_a, self.a)

      self.decoder_activations = self.build_decoder(self.a)

    with tf.name_scope("output") as scope:
      self.reconstruction = self.decoder_activations[-1]

    with tf.name_scope("loss") as scope:
      self.loss_dict = {"recon_loss":self.compute_recon_loss(self.reconstruction),
        "weight_decay_loss":self.compute_weight_decay_loss(),
        "latent_loss":self.compute_latent_loss(self.encoder_activations[-1],
          self.latent_std_activation)}
      self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")


