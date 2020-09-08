import numpy as np
import tensorflow as tf

import DeepSparseCoding.tf1x.utils.plot_functions as pf
import DeepSparseCoding.tf1x.utils.data_processing as dp
from DeepSparseCoding.tf1x.models.lca_model import LcaModel
from DeepSparseCoding.tf1x.modules.lca_conv_module import LcaConvModule

class LcaConvModel(LcaModel):
  """
  Convolutional LCA model
  Inference is defined within the graph
  """
  def __init__(self):
    super(LcaConvModel, self).__init__()

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [dict] model parameters
    Modifiable Parameters:
      stride_x
      stride_y
      patch_size_y
      patch_size_x
    """
    super(LcaConvModel, self).load_params(params)
    if len(self.params.data_shape) == 2:
      self.params.data_shape += [1]
    self.input_shape = [None,] + self.params.data_shape

  def build_module(self, input_node):
    module = LcaConvModule(input_node, self.params.num_neurons, self.sparse_mult,
      self.eta, self.params.thresh_type, self.params.rectify_a,
      self.params.num_steps, self.params.patch_size_y, self.params.patch_size_x,
      self.params.stride_y, self.params.stride_x, self.params.eps)
    return module

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    if input_data.shape[-1] == 3:
      cmap = None
    elif input_data.shape[-1] == 1:
      cmap = "Greys_r"
    else:
      assert False, ("Input_data.shape[-1] should indicate color channel, and should be 1 or 3")
    feed_dict = self.get_feed_dict(input_data, input_labels)
    weights, recon, activity = tf.compat.v1.get_default_session().run(
      [self.module.w, self.module.reconstruction, self.get_encodings()], feed_dict)

    recon = dp.rescale_data_to_one(recon)[0]
    weights = np.transpose(dp.rescale_data_to_one(weights.T)[0].T, axes=(3,0,1,2))
    current_step = str(self.global_step.eval())
    filename_suffix = "_v"+self.params.version+"_"+current_step.zfill(5)+".png"
    input_data = dp.rescale_data_to_one(input_data)[0]

    num_features = activity.shape[-1]
    activity = np.reshape(activity, [-1, num_features])
    fig = pf.plot_activity_hist(activity, title="LCA Activity Histogram",
      save_filename=self.params.disp_dir+"lca_act_hist"+filename_suffix)

    pf.plot_data_tiled(input_data[0,...], normalize=False,
      title="Images at step "+current_step, vmin=None, vmax=None, cmap=cmap,
      save_filename=self.params.disp_dir+"images"+filename_suffix)
    pf.plot_data_tiled(recon[0,...], normalize=False,
      title="Recons at step "+current_step, vmin=None, vmax=None, cmap=cmap,
      save_filename=self.params.disp_dir+"recons"+filename_suffix)
    pf.plot_data_tiled(weights, normalize=False, title="Dictionary at step "+current_step,
      vmin=np.min(weights), vmax=np.max(weights), cmap=cmap,
      save_filename=self.params.disp_dir+"phi"+filename_suffix)
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      #TODO this function is breaking due to the range of gradients
      #pf.plot_data_tiled(np.transpose(grad, axes=(3,0,1,2)), normalize=True,
      #  title="Gradient for phi at step "+current_step, vmin=None, vmax=None, cmap=cmap,
      #  save_filename=self.params.disp_dir+"dphi"+filename_suffix)
