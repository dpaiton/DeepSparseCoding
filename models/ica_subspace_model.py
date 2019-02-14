import numpy as np 
import tensorflow as tf 
import utils.plot_functions as pf 
import utils.data_processing as dp 
from models.base_model import Model 

class IcaSubspaceModel(Model):

  def __init__(self):
    super(IcaSubspaceModel, self).__init__()
    self.vector_inputs = True

  def load_params(self, params):
    super(IcaSubspaceModel, self).load_params(params)
    self.num_neurons = self.params.num_neurons
    self.input_shape = [None, self.params.num_pixels]
    # self.w_synth_shape = [self.num_neurons, self.params.num_pixels]
    # self.w_analy_shape = [self.params.num_pixels, self.num_neurons]
    
    # new params for subspace ica
    self.group_size = 4
    self.num_groups = 4
    self.w_synth_grp_shape = [self.group_size, self.params.num_pixels]

  def build_graph(self):
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.input_placeholder = tf.placeholder(tf.float32, shape=self.input_shape, name="input_data")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("weights") as scope:
          self.w_synth = []
          self.w_analy = []
          for g in range(num_groups):
            Q, R = np.linalg.qr(np.random.standard_normal(self.w_analy_shape))
            Q = Q.astype(np.float32)

            w_synth_grp_name = "w_synth_{}".format(g)
            w_analy_grp_name = "w_analy_{}".format(g)
            
            grp_patch = tf.get_variable(name=grp_name, dtype=tf.float32, initializer=Q, trainable=True)
            inv_grp_patch = tf.matrix_inverse(grp_patch, name=w_analy_grp_name)
            
            w_synth.append(grp_path)
            w_analy.append(inv_grp_patch)


        with tf.name_scope("inference") as scope:
          self.s = None # some vector (num_groups, group_size, 1)

        with tf.name_scope("output") as scope:
          with tf.name_scope("image_estimate"):
            reshape_s = tf.reshape(self.s, (self.num_groups*self.group_size, 1))
            reshape_A = tf.reshape(self.w_analy, (self.num_groups*self.group_size, self.num_neurons))
            self.recon = tf.reduce_sum(tf.matmul(reshape_a, self), axis=0, name="reconstruction")
    self.graph_built = True

  def compute_weight_gradient(self, optimizer, weight_op=None):
    return None
    

            
