import numpy as np
import tensorflow as tf

import DeepSparseCoding.tf1x.utils.plot_functions as pf
import DeepSparseCoding.tf1x.utils.data_processing as dp
from  DeepSparseCoding.tf1x.models.base_model import Model


class IcaSubspaceModel(Model):
    def __init__(self):
      super(IcaSubspaceModel, self).__init__()
      self.vector_inputs = True

    def load_params(self, params):
      super(IcaSubspaceModel, self).load_params(params)
      self.input_shape = [None, self.params.num_pixels]
      self.params.num_neurons_per_group = self.params.num_neurons // self.params.num_groups
      self.w_synth_shape = [self.params.num_pixels, self.params.num_neurons]
      self.w_analy_shape = [self.params.num_neurons, self.params.num_pixels]
      self.R = self.construct_index_matrix().astype(np.float32)
      neuron_ids = np.arange(self.params.num_neurons, dtype=np.int32)
      self.group_ids = [neuron_ids[start:start+self.params.num_neurons_per_group]
        for start in np.arange(0, self.params.num_neurons, self.params.num_neurons_per_group,
        dtype=np.int32)]

    def construct_index_matrix(self):
      R = np.zeros(shape=(self.params.num_neurons, self.params.num_groups))
      c = 0
      for r in range(self.params.num_neurons):
        if r != 0 and r % self.params.num_neurons_per_group == 0:
          c += 1
        R[r, c] = 1.0
      return R

    def get_input_shape(self):
      return self.input_shape

    def init_weight(self, method="uniform"):
      if method == "uniform":
        return np.random.uniform(low=-1.0, high=1.0, size=self.w_analy_shape)
      elif method == "gaussian":
        return np.linalg.qr(np.random.standard_normal(self.w_analy_shape), mode='complete')[0]
      elif method == "identity":
        return np.ones(shape=self.w_analy_shape)

    def build_graph_from_input(self, input_node):
      with tf.device(self.params.device):
        with self.graph.as_default():
          with tf.compat.v1.variable_scope("weights") as scope:
            Q = self.init_weight("gaussian")
            self.w_analy = tf.compat.v1.get_variable(name="w_analy",
              dtype=tf.float32,
              initializer=Q.astype(np.float32),
              trainable=True)
            self.w_synth = tf.transpose(a=self.w_analy, name="w_synth")
            self.trainable_variables[self.w_analy.name] = self.w_analy
          with tf.compat.v1.variable_scope("inference") as scope:
              self.s = tf.matmul(input_node, tf.transpose(a=self.w_analy), name="latent_vars")
          with tf.compat.v1.variable_scope("output") as scope:
              self.recon = tf.matmul(self.s, self.w_analy, name="recon")
          with tf.compat.v1.variable_scope("orthonormalize") as scope:
              self.orthonorm_weights = tf.compat.v1.assign(self.w_analy, self.orthonorm_weights(self.w_analy))
      self.graph_built = True

    def get_encodings(self):
        return self.s

    def get_group_acts(self, act):
      """
      group_amplitudes returns each neuron's group index:
        sigma_i = ||a_{j in i}||_2
      Inputs:
        act shape is [num_batch, num_neurons]
          could be u or a inputs
      Outputs:
        group_act shape is [num_batch, num_groups]
      """
      new_shape = tf.stack([tf.shape(input=self.data_tensor)[0]]+[self.num_groups,
        self.num_neurons_per_group])
      act_resh = tf.reshape(act, shape=new_shape)
      group_act = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(act_resh), axis=2, keepdims=False), name=name)
      return group_act

    def orthonorm_weights(self, w):
      m = tf.matmul(w, w, adjoint_b=True)
      s, u, v = tf.linalg.svd(m)
      new_s = tf.linalg.diag(tf.pow(s, -0.5))
      rot = tf.matmul(tf.matmul(u, new_s), v, adjoint_b=True)
      return tf.matmul(tf.math.real(rot), w)

    def compute_weight_gradients(self, optimizer, weight_op=None):
      W = weight_op
      grads = []
      for img_i in range(self.params.batch_size):
        I = tf.slice(self.input_placeholder, [img_i, 0], [1, self.params.num_pixels])
        I = tf.transpose(a=I)
        one_w_grad = self.compute_weight_gradient_per_input(I)
        self.one_grad = one_w_grad
        grads.append(one_w_grad)
      gradient = tf.stack(grads)
      avg_grad = tf.math.reduce_mean(input_tensor=gradient, axis=0)
      self.w_grad = avg_grad # for monitoring
      return [(-avg_grad, weight_op)]

    def compute_weight_gradient_per_input(self, I):
      Wt_I = tf.matmul(tf.transpose(a=self.w_analy), I)
      Wt_I_sq = tf.math.pow(Wt_I, 2)
      pre_nonlinear_term = tf.matmul(tf.transpose(a=Wt_I_sq), self.R)
      post_nonlinear_term = -0.5 * tf.math.pow(pre_nonlinear_term, -0.5)
      nonlinear_term = tf.matmul(post_nonlinear_term, tf.transpose(a=self.R))
      repeat_I = tf.tile(I, [1, self.params.num_neurons])
      return  repeat_I * tf.transpose(a=Wt_I) * nonlinear_term

    def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
      update_dict = super(IcaSubspaceModel, self).generate_update_dict(input_data, input_labels, batch_step)
      feed_dict = self.get_feed_dict(input_data, input_labels)
      eval_list  = [self.global_step, self.s, self.recon, self.w_analy, self.w_grad]
      out_vals = tf.compat.v1.get_default_session().run(eval_list, feed_dict)
      stat_dict = {
        "global_step": out_vals[0],
        "latent_vars": out_vals[1],
        "recon": out_vals[2],
        "w_analy": out_vals[3],
        "w_grad": out_vals[4],
        }
      update_dict.update(stat_dict) # stat_dict vals overwrite
      return update_dict

    def generate_plots(self, input_data, input_labels=None):
      super(IcaSubspaceModel, self).generate_plots(input_data, input_labels)
      ## ADD FUCNITONS
      feed_dict = self.get_feed_dict(input_data, input_labels)
      eval_list = [self.global_step, self.w_synth, self.w_analy, self.s, self.w_grad]
      eval_out = tf.compat.v1.get_default_session().run(eval_list, feed_dict)
      # step
      curr_step = str(eval_out[0])
      w_shape = [self.params.num_neurons, self.params.patch_edge_size, self.params.patch_edge_size, 1]
      w_synth_eval = eval_out[1].reshape(w_shape)
      w_analy_eval = eval_out[2].reshape(w_shape)
      w_grad_eval = eval_out[4].reshape(w_shape)
      latent_vars = eval_out[3]
      pf.plot_weights(w_synth_eval, title="w_synth at step {}".format(curr_step), figsize=(16, 16),
        save_filename="{}/v{}_w_synth_eval_{}.png".format(self.params.disp_dir,
        self.params.version, curr_step))
