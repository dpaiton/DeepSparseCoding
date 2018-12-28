import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_funcs as ef
import utils.get_data as get_data
import utils.mem_utils as mem_utils
from layers.non_linearities import gdn
from ops.init_ops import GDNGammaInitializer
from models.gdn_autoencoder import GDN_Autoencoder

class Conv_GDN_Decoder(GDN_Autoencoder):
  """
  Decoder for the conv_gdn_autoencoder model
  loads decoding weights from npz files, can decode latent values to images
  """
  def __init__(self):
    super(Conv_GDN_Decoder, self).__init__()
    self.vector_inputs = False

  def setup_graph(self):
    self.graph = tf.Graph()
    self.build_graph()
    self.optimizers_added = True
    self.add_initializer_to_graph()
    self.construct_savers()

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [dict] model parameters
    Modifiable Parameters:
    """
    super(GDN_Autoencoder, self).load_params(params)
    self.w_shapes = [vals
      for vals in zip(self.patch_size_y, self.patch_size_x, self.output_channels, self.input_channels)]
    self.w_igdn_shapes = [[pout,pout] for pout in self.output_channels]
    self.b_shapes = [[pout,] for pout in self.output_channels]
    self.b_igdn_shapes = [[pout,] for pout in self.output_channels]
    self.n_mem = np.prod(self.input_shape)
    self.x_shape = [None,]+self.input_shape
    self.num_layers = len(self.input_channels)

  def gdn(self, layer_id, u_in, inverse):
    """Devisive normalizeation nonlinearity"""
    with tf.variable_scope(self.weight_scope) as scope:
      w_gdn_name = "w_igdn"+str(layer_id)
      w_gdn = tf.get_variable(name=w_gdn_name,
        dtype=tf.float32, initializer=self.w_igdn_init_list[layer_id], trainable=False)
      b_gdn_name = "b_igdn"+str(layer_id)
      b_gdn = tf.get_variable(name=b_gdn_name,
        dtype=tf.float32, initializer=self.b_igdn_init_list[layer_id], trainable=False)
      self.trainable_variables[w_gdn.name] = w_gdn
      self.trainable_variables[b_gdn.name] = b_gdn
    with tf.variable_scope("gdn"+str(layer_id)) as scope:
      u_out, gdn_mult = gdn(u_in, w_gdn, b_gdn, self.gdn_w_thresh_min,
        self.gdn_b_thresh_min, self.gdn_eps, inverse=True, conv=True, name="gdn_output"+str(layer_id))
    return u_out, w_gdn, b_gdn, gdn_mult

  def layer_maker(self, layer_id, u_in, w_shape, w_stride):
    """
    Make layer that does gdn(conv(u,w)+b)
    Note: In Balle et al:
      encoder layers compute u_out = gdn(conv(u_in,w)) i.e. linear-then-nonlinear transforms
      decoder layers compute u_out = conv(igdn(u_in),w) i.e. nonlinear-then-linear transforms
      Instead, we are going to have both layers compute u_out = [i]gdn(conv(u_in,w))
      i.e. linear-then-nonlinear transforms because this is what neural network people
      and comp neuro people typically do.
    """
    with tf.variable_scope(self.weight_scope) as scope:
      w_name = "w"+str(layer_id)
      w = tf.get_variable(name=w_name, dtype=tf.float32,
        initializer=self.w_init_list[layer_id], trainable=False)
      self.trainable_variables[w.name] = w
    with tf.variable_scope(self.weight_scope) as scope:
      b_name = "b"+str(layer_id)
      b = tf.get_variable(name=b_name,
        dtype=tf.float32, initializer=self.b_init_list[layer_id], trainable=True)
      self.trainable_variables[b.name] = b
    with tf.variable_scope("hidden"+str(layer_id)) as scope:
      height_const = 0 if u_in.get_shape()[1] % w_stride == 0 else 1
      out_height = (u_in.get_shape()[1] * w_stride) - height_const
      width_const = 0 if u_in.get_shape()[2] % w_stride == 0 else 1
      out_width = (u_in.get_shape()[2] * w_stride) - width_const
      out_shape = tf.stack([self.batch_size, # Batch
        out_height, # Height
        out_width, # Width
        tf.constant(w_shape[2], dtype=tf.int32)]) # Channels
      conv_out = tf.add(tf.nn.conv2d_transpose(u_in, w, out_shape,
        strides = [1, w_stride, w_stride, 1], padding="SAME"), b,
        name="conv_out"+str(layer_id))
      if layer_id == self.num_layers-1: # we don't want to use GDN on the last layer (recon)
        gdn_out = conv_out
        w_gdn = None
        b_gdn = None
        gdn_mult = None
      else:
        gdn_out, w_gdn, b_gdn, gdn_mult = self.gdn(layer_id, conv_out, inverse=True)
    return gdn_out, w, b, w_gdn, b_gdn, conv_out, gdn_mult

  def build_graph(self):
    """Build the TensorFlow graph object"""
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")

        with tf.name_scope("placeholders") as scope:
          self.w_init_list = [tf.placeholder(tf.float32, shape=w_shape, name="w"+str(w_id))
            for w_id, w_shape in enumerate(self.w_shapes)]
          self.b_init_list = [tf.placeholder(tf.float32, shape=b_shape, name="b"+str(b_id))
            for b_id, b_shape in enumerate(self.b_shapes)]
          self.w_igdn_init_list = [tf.placeholder(tf.float32, shape=w_igdn_shape, name="w_igdn"+str(w_igdn_id))
            for w_igdn_id, w_igdn_shape in enumerate(self.w_igdn_shapes)]
          self.b_igdn_init_list = [tf.placeholder(tf.float32, shape=b_igdn_shape, name="b_idgn"+str(b_igdn_id))
            for b_igdn_id, b_igdn_shape in enumerate(self.b_igdn_shapes)]

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("weights") as scope:
          self.weight_scope = tf.get_variable_scope()

        self.u_list = [self.x]
        self.conv_list = []
        self.w_list = []
        self.w_gdn_list = []
        self.gdn_mult_list = []
        self.b_list = []
        self.b_gdn_list = []
        for layer_id in range(self.num_layers):
          u_out, w, b, w_gdn, b_gdn, conv_out, gdn_mult = self.layer_maker(layer_id,
            self.u_list[layer_id], self.w_shapes[layer_id], self.w_strides[layer_id])
          self.u_list.append(u_out)
          self.conv_list.append(conv_out)
          self.w_list.append(w)
          self.w_gdn_list.append(w_gdn)
          self.gdn_mult_list.append(gdn_mult)
          self.b_list.append(b)
          self.b_gdn_list.append(b_gdn)

        with tf.name_scope("summaries") as scope:
          tf.summary.image("reconstruction",self.u_list[-1])
          [tf.summary.histogram("u"+str(idx),u) for idx,u in enumerate(self.u_list) if u is not None]
          [tf.summary.histogram("w"+str(idx),w) for idx,w in enumerate(self.w_list) if w is not None]
          [tf.summary.histogram("w_gdn"+str(idx),w) for idx,w in enumerate(self.w_gdn_list) if w is not None]
          [tf.summary.histogram("b"+str(idx),b) for idx,b in enumerate(self.b_list) if b is not None]
          [tf.summary.histogram("b_gdn"+str(idx),u) for idx,u in enumerate(self.b_gdn_list) if u is not None]
          self.merged_summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.save_dir, self.graph)
    self.graph_built = True
