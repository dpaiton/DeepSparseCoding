import numpy as np
import tensorflow as tf
import utils.entropy_functions as ef
from utils.trainable_variable_dict import TrainableVariableDict
from modules.ae_module import AeModule

class VaeModule(AeModule):
  def __init__(self, data_tensor, output_channels, sparse_mult, decay_mult, kld_mult,
    act_funcs, dropout, tie_decoder_weights, noise_level=0, recon_loss_type="mse",
    name_scope="VAE"):
    """
    Variational Autoencoder module
    Inputs:
      data_tensor
      output_channels [list of ints] A list of channels to make, also defines number of layers
      sparse_mult [float] tradeoff multiplier for latent sparsity loss
      decay_mult [float] tradeoff multiplier for weight decay loss
      kld_mult [float] tradeoff multiplier for latent variational kld loss
      act_funcs [list of functions] activation functions
      dropout [list of floats] specifies the keep probability or None
      noise_level [float] stddev of noise to be added to the input (for denoising VAE)
      recon_loss_type [str] either "mse" or the cross entropy loss used in Kingma & Welling
      name_scope [str] specifies the name_scope for the module
    Outputs:
      dictionary
    """
    self.noise_level = noise_level
    with tf.name_scope(name_scope) as scope:
      if self.noise_level > 0.0:
          self.corrupt_data = tf.add(tf.random.normal(shape=tf.shape(data_tensor),
            mean=tf.reduce_mean(data_tensor), stddev=noise_level, dtype=tf.float32, name="data_noise"),
            data_tensor)
      else:
        self.corrupt_data = data_tensor
    self.recon_loss_type = recon_loss_type
    self.sparse_mult = sparse_mult
    self.kld_mult = kld_mult
    super(VaeModule, self).__init__(data_tensor, output_channels, decay_mult, act_funcs,
      dropout, tie_decoder_weights, name_scope)

  def compute_recon_loss(self, reconstruction):
    if self.recon_loss_type == "mse":
      return super(VaeModule, self).compute_recon_loss(reconstruction)
    elif self.recon_loss_type == "crossentropy":
      reduc_dim = list(range(1, len(reconstruction.shape)))# We want to avg over batch
      recon_loss = tf.reduce_mean(-tf.reduce_sum(self.data_tensor * ef.safe_log(reconstruction) \
        + (1-self.data_tensor) * ef.safe_log(1-reconstruction), axis=reduc_dim))
      return recon_loss
    else:
      assert False, ("recon_loss_type param must be `mse` or `crossentropy`")

  def compute_latent_loss(self, a_mean, a_log_std_sq):
    with tf.name_scope("latent"):
      reduc_dim = list(range(1, len(a_mean.shape))) # Want to avg over batch, sum over the rest
      latent_loss = self.kld_mult * tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + 2.0 * a_log_std_sq
        - tf.square(a_mean) - tf.exp(2.0 * a_log_std_sq), reduc_dim))
    return latent_loss

  def compute_sparse_loss(self, a_in):
    with tf.name_scope("unsupervised"): # TODO: change to loss from sae? look into this
      reduc_dim = list(range(1, len(a_in.shape))) # Want to avg over batch, sum over the rest
      act_l1 = tf.reduce_mean(tf.reduce_sum(tf.abs(a_in), axis=reduc_dim))
      sparse_loss = self.sparse_mult * act_l1
    return sparse_loss

  def build_graph(self):
    with tf.name_scope(self.name_scope) as scope:
      with tf.name_scope("weight_inits") as scope:
        #self.w_init = tf.initializers.truncated_normal(mean=0.0, stddev=0.01, dtype=tf.float32)
        self.w_init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
        self.b_init = tf.initializers.constant(1e-8, dtype=tf.float32)

      self.u_list = [self.corrupt_data]
      self.w_list = []
      self.b_list = []
      enc_u_list, enc_w_list, enc_b_list = self.build_encoder(self.u_list[0],
        self.act_funcs[:self.num_encoder_layers], self.w_shapes[:self.num_encoder_layers])
      self.u_list += enc_u_list[:-1] # don't store the mean value in u_list
      self.w_list += enc_w_list
      self.b_list += enc_b_list

      #Std weights
      self.w_enc_std = tf.get_variable(name="w_enc_"+str(self.num_encoder_layers)+"_std",
        shape=self.w_shapes[self.num_encoder_layers-1], dtype=tf.float32,
        initializer=self.w_init, trainable=True)
      self.b_enc_std = tf.get_variable(name="b_enc_"+str(self.num_encoder_layers)+"_std",
        shape=self.w_shapes[self.num_encoder_layers-1][1], dtype=tf.float32,
        initializer=self.b_init, trainable=True)
      self.trainable_variables[self.w_enc_std.name] = self.w_enc_std
      self.trainable_variables[self.b_enc_std.name] = self.b_enc_std

      self.latent_mean_activation = enc_u_list[-1]
      self.a = self.latent_mean_activation # alias for AE model

      self.latent_std_activation = tf.add(tf.matmul(enc_u_list[-2], self.w_enc_std),
        self.b_enc_std)
      #self.latent_std_activation = 1e-8 + tf.nn.softplus(tf.matmul(enc_u_list[-2],
      #  self.w_enc_std) + self.b_enc_std) # std must be positive

      noise = tf.random_normal(tf.shape(self.latent_std_activation))

      #act = self.latent_mean_activation + self.latent_std_activation * noise
      #Calculate latent - std is in log(std**2) space
      #act = self.latent_mean_activation + tf.sqrt(tf.exp(self.latent_std_activation)) * noise
      #act = self.latent_mean_activation + tf.exp(0.5 * self.latent_std_activation) * noise
      act = self.latent_mean_activation + tf.exp(self.latent_std_activation) * noise

      act = tf.identity(act, name="activity")

      self.u_list.append(act)

      dec_u_list, dec_w_list, dec_b_list = self.build_decoder(self.u_list[-1],
        self.act_funcs[self.num_encoder_layers:], self.w_shapes[self.num_encoder_layers:])
      self.u_list += dec_u_list
      self.w_list += dec_w_list
      self.b_list += dec_b_list

      for w,b in zip(self.w_list, self.b_list):
        self.trainable_variables[w.name] = w
        self.trainable_variables[b.name] = b

      with tf.name_scope("output") as scope:
        self.reconstruction = tf.identity(self.u_list[-1], name="reconstruction")

      with tf.name_scope("loss") as scope:
        self.loss_dict = {"recon_loss":self.compute_recon_loss(self.reconstruction),
          "weight_decay_loss":self.compute_weight_decay_loss(),
          "latent_loss":self.compute_latent_loss(self.latent_mean_activation, self.latent_std_activation)}
        self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")
