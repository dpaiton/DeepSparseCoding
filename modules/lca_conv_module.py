import numpy as np
import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict
from modules.lca_module import LcaModule
import pdb

class LcaConvModule(LcaModule):
  def __init__(self, data_tensor, num_neurons, sparse_mult, step_size, thresh_type,
    rectify_a, num_steps, patch_size_y, patch_size_x,
    stride_y, stride_x, eps, name="LCA_Conv"):

    #Set these before calling init
    self.patch_size_y = patch_size_y
    self.patch_size_x = patch_size_x
    self.stride_y = stride_y
    self.stride_x = stride_x

    super(LcaConvModule, self).__init__(data_tensor, num_neurons, sparse_mult, step_size,
        thresh_type, rectify_a, num_steps, eps, name)

  def calc_shapes(self):
    assert (self.num_pixels_y % self.stride_y == 0), (
      "Stride y must divide evenly into input shape")
    assert (self.num_pixels_x % self.stride_x == 0), (
      "Stride x must divide evenly into input shape")

    self.u_y = int(int(self.num_pixels_y)/self.stride_y)
    self.u_x = int(int(self.num_pixels_x)/self.stride_x)
    self.w_shape = [self.patch_size_y, self.patch_size_x,
      int(self.num_input_features), int(self.num_neurons)]
    self.u_shape = [self.u_y, self.u_x, int(self.num_neurons)]

  def check_data(self):
    data_ndim = len(self.data_tensor.get_shape().as_list())
    assert data_ndim == 4, \
      ("Module requires datal_tensor to have shape " + \
         "[batch, num_pixels_y, num_pixels_x, num_input_features]")
    self.batch_size, self.num_pixels_y, self.num_pixels_x, self.num_input_features = \
      self.data_tensor.get_shape()

  def compute_recon(self, a_in):
    x_ = tf.nn.conv2d_transpose(a_in, self.w, tf.shape(self.data_tensor),
      [1, self.stride_y, self.stride_x, 1], padding="SAME", name="reconstruction")
    return x_

  def step_inference(self, u_in, a_in, step):
    with tf.name_scope("update_u"+str(step)) as scope:
      recon_error = self.data_tensor - self.compute_recon(a_in)
      error_injection = tf.nn.conv2d(recon_error, self.w, [1, self.stride_y,
        self.stride_x, 1], padding="SAME", use_cudnn_on_gpu=True, name="forward_injection")
      du = tf.subtract(tf.add(error_injection, a_in), u_in, name="du")
      u_out = tf.add(u_in, tf.multiply(self.step_size, du))
    return u_out

  def infer_coefficients(self):
   u_list = [self.u_zeros]
   a_list = [self.threshold_units(u_list[0])]
   for step in range(self.num_steps-1):
     u = self.step_inference(u_list[step], a_list[step], step+1)
     u_list.append(u)
     a_list.append(self.threshold_units(u))
   return (u_list, a_list)
