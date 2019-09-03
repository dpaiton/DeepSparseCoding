import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
from models.mp_model import MpModel
from modules.mp_conv_module import MpConvModule
import pdb

class MpConvModel(MpModel):
  """
  Convolutional Matching Pursuit model
  """
  def __init__(self):
    super(MpConvModel, self).__init__()

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(MpConvModel, self).load_params(params)
    if len(self.params.data_shape) == 2:
      self.params.data_shape += [1]
    self.input_shape = [None,] + self.params.data_shape

  def build_module(self, input_node):
    module = MpConvModule(input_node, self.params.num_neurons, self.num_k,
      self.params.patch_size_y, self.params.patch_size_x,
      self.params.stride_y, self.params.stride_x,
      self.params.eps)
    return module

  def reshape_input(self, input_node):
    data_shape = input_node.get_shape().as_list()
    if len(data_shape) != 4:
      ("MP_conv requires datal_tensor to have shape " + \
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
    fig = pf.plot_activity_hist(activity, title="Activity Histogram",
      save_filename=self.params.disp_dir+"act_hist"+filename_suffix)

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
