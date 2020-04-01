import tensorflow as tf

import DeepSparseCoding.utils.plot_functions as pf
import DeepSparseCoding.utils.data_processing as dp
import DeepSparseCoding.utils.entropy_functions as ef
from DeepSparseCoding.models.mlp_lca_model import MlpLcaModel
from DeepSparseCoding.modules.lca_subspace_module import LcaSubspaceModule
from DeepSparseCoding.modules.activations import activation_picker


class MlpLcaSubspaceModel(MlpLcaModel):
  def build_lca_module(self, input_node):
    if(self.params.lca_conv):
      assert False, ("Conv LCA Subspace is not supported.")
    else:
      module = LcaSubspaceModule(input_node, self.params.num_neurons, self.sparse_mult,
        self.eta, self.params.num_steps, self.params.num_groups, self.group_orth_mult,
        self.params.eps)
    return module

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.compat.v1.variable_scope("auto_placeholders") as scope:
          self.group_orth_mult = tf.compat.v1.placeholder(tf.float32, shape=(), name="group_orth_mult")
    super(MlpLcaSubspaceModel, self).build_graph_from_input(input_node)

  def get_encodings(self):
    #return tf.reshape(self.lca_module.group_activity, shape=[None, self.lca_module.num_groups])
    #return self.lca_module.group_activity.set_shape([None, self.lca_module.num_groups])
    return self.lca_module.group_activity

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
    Log train progress information
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    update_dict = super(MlpLcaSubspaceModel, self).generate_update_dict(input_data, input_labels,
      batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.lca_module.loss_dict["orthogonalization_loss"]]
    out_vals =  tf.compat.v1.get_default_session().run(eval_list, feed_dict)[0]
    update_dict.update({"lca_orthogonalization_loss":out_vals,})
    return update_dict
