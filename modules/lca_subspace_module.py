import numpy as np
import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict
from modules.lca_module import LcaModule

class LcaSubspaceModule(LcaModule):
  def __init__(self, data_tensor, num_neurons, sparse_mult, step_size,
    num_steps, num_groups, group_orth_mult, eps, variable_scope="lca_subspace"):

    self.num_groups = num_groups
    self.group_orth_mult = group_orth_mult

    #Add new params here
    assert num_neurons % self.num_groups == 0, (
      "Num groups must divide evenly into num neurons.")
    self.num_neurons_per_group = int(num_neurons / self.num_groups)
    # group_ids is a list of len == self.num_groups, where each element in the list is a list of
    #    neuron indices assigned to that group
    neuron_ids = np.arange(num_neurons, dtype=np.int32)
    self.group_ids = [neuron_ids[start:start+self.num_neurons_per_group]
      for start in np.arange(0, num_neurons, self.num_neurons_per_group, dtype=np.int32)]
    # group_assignemnts is an array of shape [self.num_neurons] that conatins a group assignment
    #    for each neuron index
    self.group_assignments = np.zeros(num_neurons, dtype=np.int32)
    for group_index, group_member_indices in enumerate(self.group_ids):
      for neuron_id in group_member_indices:
        self.group_assignments[neuron_id] = group_index

    super(LcaSubspaceModule, self).__init__(data_tensor, num_neurons, sparse_mult, step_size,
        None, None, num_steps, eps, variable_scope)

  def reshape_groups_per_neuron(self, sigmas, name=None):
    """
    Reshape sigmas from [num_batch, num_groups] to [num_batch, num_neurons]
    Each neuron index is assigned the group amplitude for its corresponding group
    """
    out_sigmas = tf.gather(sigmas, self.group_assignments, axis=-1)
    return out_sigmas

  def group_amplitudes(self, a_in, name=None):
    """
    group_amplitudes returns each neuron's group index:
      sigma_i = ||a_{j in i}||_2
    Inputs:
      a_in shape is [num_batch, num_neurons]
        could be u or a inputs
    Outputs:
      sigmas shape is [num_batch, num_groups]
    """
    new_shape = tf.stack([tf.shape(self.data_tensor)[0]]+[self.num_groups,
      self.num_neurons_per_group])
    a_resh = tf.reshape(a_in, shape=new_shape)
    sigmas = tf.sqrt(tf.reduce_sum(tf.square(a_resh), axis=2, keepdims=False), name=name)
    return sigmas

  def group_directions(self, a_in, sigmas, name=None):
    """
    group_directions returns each neurons direction normalized by the group amplitude:
      z_i = a_i / sigmas_i
    a_in shape [num_batch, num_neurons]
    sigms shape [num_batch, num_neurons]
    directions shape [num_batch, num_neurons]
    """
    directions = tf.where(tf.greater(sigmas, 0.0), tf.divide(a_in, sigmas, name=name),
      tf.zeros_like(sigmas))
    return directions

  def threshold_units(self, u_in):
    """
    All units in a group are above threshold if their group amplitude is above lambda
    """
    u_amplitudes = self.reshape_groups_per_neuron(self.group_amplitudes(u_in)) # [num_batch, num_neurons]
    u_directions = self.group_directions(u_in, u_amplitudes) # [num_batch, num_neurons]
    a_out = tf.where(tf.greater(u_amplitudes, self.sparse_mult),
      tf.multiply(tf.subtract(u_amplitudes, self.sparse_mult), u_directions),
      self.u_zeros)
    return a_out

  def compute_sparse_loss(self, a_in):
    with tf.variable_scope("unsupervised"):
      sigmas = self.group_amplitudes(a_in) # [num_batch, num_groups]
      sparse_loss = self.sparse_mult * tf.reduce_mean(tf.reduce_sum(sigmas, axis=1),
        name="group_sparse_loss")
    return sparse_loss

  def compute_group_orthogonalization_loss(self):
    with tf.variable_scope("unsupervised"):
      # For each group
        # assemble matrix of W = [num_pixels, num_neurons_in_group]
        # compute E =  ( W^T * W ) - I
        # loss = mean(loss_mult * E)
      group_weights = tf.reshape(self.w,
        shape=[self.num_pixels, self.num_groups, self.num_neurons_per_group], name="group_weights")
      w_orth_list = [
        tf.reduce_sum(tf.abs(tf.matmul(tf.transpose(group_weights[:, group_idx, :]),
        group_weights[:, group_idx, :]) - tf.eye(num_rows=self.num_neurons_per_group)))
        for group_idx in range(self.num_groups)]
      group_orthogonalization_loss = tf.multiply(self.group_orth_mult, tf.add_n(w_orth_list),
        name="group_orth_loss")
    return group_orthogonalization_loss

  def get_loss_funcs(self):
    #TODO: Should be able to rewrite this to have a tuple as its value where the second
    # entry in the tuple is the argument it needs (e.g. self.a or self.recon) or at least a label
    # to choose from a preset list of args
    return {"recon_loss":self.compute_recon_loss, "sparse_loss":self.compute_sparse_loss,
      "orthogonalization_loss":self.compute_group_orthogonalization_loss}

  def build_graph(self):
    super(LcaSubspaceModule, self).build_graph()
    with tf.variable_scope(self.variable_scope) as scope:
      with tf.variable_scope(self.inference_scope):
        self.group_activity = tf.identity(self.group_amplitudes(self.a),
          name="group_activity")
        self.group_angles = tf.identity(self.group_directions(self.a,
          self.reshape_groups_per_neuron(self.group_activity)), name="group_directions")
      with tf.variable_scope(self.weight_scope):
        self.group_weights = tf.reshape(self.w,
          shape=[self.num_pixels, self.num_groups, self.num_neurons_per_group],
          name="group_weights")
      with tf.variable_scope("loss") as scope:
        self.loss_dict = {"recon_loss":self.compute_recon_loss(self.reconstruction),
          "sparse_loss":self.compute_sparse_loss(self.a),
          "orthogonalization_loss":self.compute_group_orthogonalization_loss()}
        self.total_loss = tf.add_n([val for val in self.loss_dict.values()], name="total_loss")
