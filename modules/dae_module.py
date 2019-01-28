import numpy as np
import tensorflow as tf
import utils.entropy_functions as ef
from utils.trainable_variable_dict import TrainableVariableDict
from modules.ae_module import AeModule
from ops.init_ops import GDNGammaInitializer 

class DaeModule(AeModule):
  def __init__(self, data_tensor, output_channels, ent_mult, decay_mult, bounds_slope, latent_min,
    latent_max, num_quant_bins, gdn_w_init_const, gdn_eps, gdn_b_init_const, act_funcs, dropout):
    """
    Divisive Autoencoder module
    Inputs:
      data_tensor
      output_channels: A list of channels to make, also defines number of layers
      ent_mult: tradeoff multiplier for latent entropy loss 
      decay_mult: tradeoff multiplier for weight decay loss
      bounds_slope: slope for out of bounds loss (two relus back to back) 
      latent_min: min value you want for latent variable (max value for left relu)
      latent_max: max value you want for latent variable (max value for right relu)
      num_quant_bins: number of bins you want for quantization
        e.g. if min is -50 and max is 50 and num_quant_bins is 100, will qauntize on integers
        formula: quant noise drawn from
        U(-(latent_max-latent_min)/(2*num_quant_bins), (latent_max-latent_min)/(2*num_quant_bins))
      gdn_w_init_const: diagonal of gdn gamma initializer 
      gdn_eps: off diagonal of gdn gamma initializer 
      gdn_b_init_const: diagonal of gdn beta initializer 
      act_funcs: activation functions
      dropout: specifies the keep probability or None
    Outputs:
      dictionary
    """
    self.ent_mult = ent_mult
    self.bounds_slope = bounds_slope
    self.latent_min = latent_min
    self.latent_max = latent_max
    self.num_quant_bins = num_quant_bins
    self.gdn_w_init_const = gdn_w_init_const
    self.gdn_eps = gdn_eps
    super(DaeModule, self).__init__(data_tensor, output_channels, decay_mult, act_funcs,
      dropout)

  def compute_entropy_loss(self, a_in):
    with tf.name_scope("latent"):
      a_entropies = self.compute_entropies(a_in)
      entropy_loss = tf.multiply(self.ent_mult, tf.reduce_sum(a_entropies), name="entropy_loss")
    return entropy_loss

  def compute_ramp_loss(self, a_in):
    reduc_dim = list(range(1,len(a_in.shape))) # Want to avg over batch
    ramp_loss = tf.reduce_mean(tf.reduce_sum(self.bounds_slope
      * (tf.nn.relu(a_in - self.latent_max)
      + tf.nn.relu(self.latent_min - a_in)), axis=reduc_dim

  def build_encoder(self, input_tensor, activation_functions, w_shapes):
    assert len(activation_functions) == len(w_shapes), (
      "activation_functions & w_shapes must be the same length")
    u_list = [input_tensor]
    w_list = []
    b_list = []
    w_gdn_list = []
    b_gdn_list = []
    for layer_id in range(len(w_shapes)):
      u_out, w, b, w_gdn, b_gdn = self.layer_maker(layer_id, u_list[layer_id],
        activation_functions[layer_id], w_shapes[layer_id])
      u_list.append(u_out)
      w_list.append(w)
      b_list.append(b)
      w_gdn_list.append(w_gdn)
      b_gdn_list.append(b_gdn)
    return u_list, w_list, b_list, w_gdn_list, b_gdn_list

  def build_graph(self):
    with tf.name_scope("weight_inits") as scope:
      self.w_init = tf.initializers.random_normal(mean=0.0, stddev = 1e-2, dtype=tf.float32)
      selt.b_init = tf.initializers.zeros(dtype=tf.float32)
    self.w_gdn_init = GDNGammaInitializer(diagonal_gain = self.gdn_w_init_const,
      off_diagonal_gain = self.gdn_eps, dtype = tf.float32)
    self.w_igdn_init = self.w_gdn_init
    b_init_const = np.sqrt(self.gdn_b_init_const + self.gdn_eps**2)
    self.b_gdn_init = tf.initializers.constant(b_init_const, dtype=tf.float32)
    self.b_igdn_init = self.b_gdn_init

    self.u_list = [self.data_tensor]
    self.w_list = []
    self.b_list = []
    self.w_gdn = []
    self.b_gdn = []
    enc_u_list, enc_w_list, enc_b_list, enc_w_gdn_list, enc_b_gdn_list =\ 
      self.build_encoder(self.u_list[0],
      self.act_funcs[:self.num_encoder_layers], self.w_shapes[:self.num_encoder_layers])
    self.u_list += enc_u_list[:-1] # don't store the mean value in u_list
    self.w_list += enc_w_list
    self.b_list += enc_b_list
    self.w_gdn_list += enc_w_gdn_list
    self.b_gdn_list += enc_b_gdn_list 

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
    self.a = self.u_list[-1]

    dec_u_list, dec_w_list, dec_b_list = self.build_decoder(self.num_encoder_layers,
      self.u_list[-1], self.act_funcs[self.num_encoder_layers:],
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
