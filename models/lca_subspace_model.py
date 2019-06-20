import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
from models.lca_model import LcaModel
from modules.lca_subspace_module import LcaSubspaceModule

class LcaSubspaceModel(LcaModel):
  """
  LCA model with group sparsity constraints
  """
  def __init__(self):
    super(LcaSubspaceModel, self).__init__()
    self.vector_inputs = True

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.variable_scope("auto_placeholders") as scope:
          self.group_orth_mult = tf.placeholder(tf.float32, shape=(), name="group_orth_mult")
    super(LcaSubspaceModel, self).build_graph_from_input(input_node)

  def build_module(self, input_node):
    module = LcaSubspaceModule(input_node, self.params.num_neurons, self.sparse_mult,
      self.eta, self.params.num_steps, self.params.num_groups, self.group_orth_mult,
      self.params.eps)
    return module

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    update_dict = super(LcaSubspaceModel, self).generate_update_dict(input_data, input_labels,
    batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.module.loss_dict["orthogonalization_loss"]]
    orth_loss =  tf.get_default_session().run(eval_list, feed_dict)[0]
    update_dict.update({"orthogonalization_loss":orth_loss})
    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(LcaSubspaceModel, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.module.w]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    weights = np.reshape(eval_out[1].T, [self.params.num_neurons,
      int(np.sqrt(self.params.num_pixels)), int(np.sqrt(self.params.num_pixels))])
    filename_suffix = "_v"+self.params.version+"_"+current_step.zfill(5)+".png"
    fig = pf.plot_group_weights(np.squeeze(weights), self.module.group_ids,
      title="Dictionary at step "+current_step, figsize=(18,18),
      save_filename=self.params.disp_dir+"group_phi_v"+filename_suffix)
