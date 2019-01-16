import numpy as np
import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict
from modules.ae_module import AeModule

class VaeModule(AeModule):
  def __init__(self, data_tensor, output_channels, sparse_mult, decay_mult, kld_mult,
    act_func, noise_level=0, name="VAE"):
    """
    Variational Autoencoder module
    Inputs:
      data_tensor
      params
      output_channels: A list of channels to make, also defines number of layers
      name
    Outputs:
      dictionary
    TODO: Add optional flag to turn off ReLU for the final encoding layer (means)
    """
    self.noise_level = noise_level
    if self.noise_level > 0.0:
      self.corrupt_data = tf.add(tf.random.normal(shape=tf.shape(data_tensor),
        mean=tf.reduce_mean(data_tensor), stddev=noise_level, dtype=tf.float32, name="data_noise"),
        data_tensor)
    else:
      self.corrupt_data = data_tensor
    self.sparse_mult = sparse_mult
    self.kld_mult = kld_mult

    super(VaeModule, self).__init__(data_tensor, output_channels, decay_mult, act_func, name)

  def compute_latent_loss(self, a_mean, a_log_std_sq):
    with tf.name_scope("latent"):
      reduc_dim = list(range(1, len(a_mean.shape))) # Want to avg over batch, sum over the rest
      latent_loss= self.kld_mult * -0.5 * tf.reduce_mean(tf.reduce_sum(1 + a_log_std_sq -
        tf.square(a_mean) - tf.exp(a_log_std_sq), reduc_dim))
    return latent_loss

  def compute_sparse_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      reduc_dim = list(range(1, len(a_in.shape))) # Want to avg over batch, sum over the rest
      act_l1 = tf.reduce_mean(tf.reduce_sum(tf.square(a_in), axis=reduc_dim))
      sparse_loss = self.sparse_mult * act_l1
    return sparse_loss

  def build_graph(self):
    with tf.name_scope("weight_inits") as scope:
      self.w_init = tf.initializers.random_normal(mean=0.0, stddev=0.1, dtype=tf.float32)
      self.b_init = tf.initializers.constant(1e-4, dtype=tf.float32)

    self.u_list = [self.corrupt_data]
    self.w_list = []
    self.b_list = []
    enc_u_list, enc_w_list, enc_b_list = self.build_encoder(self.u_list[0],
      [self.act_func,]*(self.num_encoder_layers-1)+[tf.identity],
      self.w_shapes[:self.num_encoder_layers])
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
    self.latent_std_activation = tf.add(tf.matmul(self.u_list[-1], self.w_enc_std),
      self.b_enc_std)
    #Calculate latent - std is in log(std**2) space
    noise  = tf.random_normal(tf.shape(self.latent_std_activation))
    act = self.latent_mean_activation + tf.sqrt(tf.exp(self.latent_std_activation)) * noise
    #Add name to act
    act = tf.identity(act, name="activity")
    self.u_list.append(act)

    self.a = self.u_list[-1]

    dec_u_list, dec_w_list, dec_b_list = self.build_decoder(self.num_encoder_layers+1,
      self.u_list[-1], [self.act_func,]*(self.num_decoder_layers-1) + [tf.identity],
      self.w_shapes[self.num_encoder_layers:])
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
        "latent_loss":self.compute_latent_loss(self.a, self.latent_std_activation)}
      self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")
