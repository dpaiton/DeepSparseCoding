import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
from models.lca import LCA
from modules.lca_subspace import LCASubspaceModule

class LCA_SUBSPACE(LCA):
  """
  LCA model with group sparsity constraints
  """
  def __init__(self):
    super(LCA_SUBSPACE, self).__init__()
    self.vector_inputs = True

  def build_graph(self):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.group_orth_mult = tf.placeholder(tf.float32, shape=(), name="group_orth_mult")

    super(LCA_SUBSPACE, self).build_graph()

  def build_module(self):
    module = LCASubspaceModule(self.x, self.params.num_neurons, self.sparse_mult,
      self.eta, self.params.num_steps, self.params.num_groups, self.group_orth_mult,
      self.params.eps, name="lca_subspace")
    return module

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(LCA_SUBSPACE, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.module.w]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    weights = np.reshape(eval_out[1].T, [self.params.num_neurons, int(np.sqrt(self.num_pixels)),
      int(np.sqrt(self.num_pixels))])
    fig = pf.plot_group_weights(np.squeeze(weights), self.module.group_ids,
      title="Dictionary at step "+current_step, figsize=(18,18),
      save_filename=(self.params.disp_dir+"group_phi_v"+self.params.version+"-"+
        current_step.zfill(6)+".png"))
