import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
from models.lca_model import LcaModel
from modules.lca_conv_module import LcaConvModule
import pdb

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

  def reshape_input(self, input_node):
    data_shape = input_node.get_shape().as_list()
    if len(data_shape) != 4:
      ("LCA_conv requires datal_tensor to have shape " + \
         "[batch, num_pixels_y, num_pixels_x, num_input_features]")
    return input_node

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
    eval_list = [self.global_step, self.module.w, self.module.reconstruction, self.get_encodings(), self.input_node]

    current_step, weights, recon, activity, input_node = tf.get_default_session().run(
      eval_list, feed_dict)
    current_step = str(current_step)

    weights = np.transpose(weights, axes=(3,0,1,2))
    filename_suffix = "_v"+self.params.version+"_"+current_step.zfill(5)+".png"

    num_features = activity.shape[-1]
    activity = np.reshape(activity, [-1, num_features])
    fig = pf.plot_activity_hist(activity, title="LCA Activity Histogram",
      save_filename=self.params.disp_dir+"lca_act_hist"+filename_suffix)

    #Scale image by max and min of images and/or recon
    r_max = np.max([np.max(input_node), np.max(recon)])
    r_min = np.min([np.min(input_node), np.min(recon)])

    pf.plot_data_tiled(input_node, normalize=False,
      title="Scaled Images at step "+current_step, vmin=r_min, vmax=r_max, cmap=cmap,
      save_filename=self.params.disp_dir+"images"+filename_suffix)

    pf.plot_data_tiled(recon, normalize=False,
      title="Recons at step "+current_step, vmin=r_min, vmax=r_max, cmap=cmap,
      save_filename=self.params.disp_dir+"recons"+filename_suffix)

    pf.plot_data_tiled(weights, normalize=True,
      title="Dictionary at step "+current_step, cmap=cmap,
      save_filename=self.params.disp_dir+"phi"+filename_suffix)

    #Plot loss over time
    eval_list = [self.module.recon_loss_list, self.module.sparse_loss_list, self.module.total_loss_list]
    (recon_losses, sparse_losses, total_losses) = tf.get_default_session().run(eval_list, feed_dict)
    #TODO put this in plot functions
    pf.plot_sc_losses(recon_losses, sparse_losses, total_losses,
      save_filename=self.params.disp_dir+"losses"+filename_suffix)
