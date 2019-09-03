import numpy as np
import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict
from modules.mp_module import MpModule
import pdb

class MpConvModule(MpModule):
  def __init__(self, data_tensor, num_neurons, num_k,
    patch_size_y, patch_size_x, stride_y, stride_x, eps,
    variable_scope="mp_conv"):

    #Set these before calling init
    self.patch_size_y = patch_size_y
    self.patch_size_x = patch_size_x
    self.stride_y = stride_y
    self.stride_x = stride_x

    super(MpConvModule, self).__init__(data_tensor, num_neurons, num_k, eps,
      variable_scope)

  def check_data(self):
    data_ndim = len(self.data_tensor.get_shape().as_list())
    assert data_ndim == 4, \
      ("Module requires datal_tensor to have shape " + \
         "[batch, num_pixels_y, num_pixels_x, num_input_features]")
    self.batch_size, self.num_pixels_y, self.num_pixels_x, self.num_input_features = \
      self.data_tensor.get_shape().as_list()
    self.num_pixels = self.num_pixels_y * self.num_pixels_x * self.num_input_features

  def calc_shapes(self):
    assert (self.num_pixels_y % self.stride_y == 0), (
      "Stride y must divide evenly into input shape")
    assert (self.num_pixels_x % self.stride_x == 0), (
      "Stride x must divide evenly into input shape")

    self.w_shape = [self.num_pixels_y, self.num_pixels_x,
      self.num_input_features, self.num_neurons]

    self.a_y = int(self.num_pixels_y/self.stride_y)
    self.a_x = int(self.num_pixels_x/self.stride_x)
    self.a_shape = [self.a_y, self.a_x, self.num_neurons]

  def build_decoder(self, input_tensor, name=None):
    x_ = tf.nn.conv2d_transpose(input_tensor, self.w, tf.shape(self.data_tensor),
      [1, self.stride_y, self.stride_x, 1], padding="SAME", name=name)
    return x_

  def step_inference(self, a_in, step):
    #TODO this isn't very efficient since we're doing 2 conv steps here
    with tf.variable_scope("update_a"+str(step)) as scope:
      recon = self.build_decoder(a_in)
      error = self.data_tensor - recon
      ff_act = tf.nn.conv2d(error, self.w, strides=[1, self.stride_y, self.stride_x, 1],
        padding="SAME")
      ff_act_flat = tf.reshape(ff_act, [-1, self.num_pixels])
      max_idx = tf.argmax(ff_act_flat, axis=-1)
      max_act = tf.reduce_max(ff_act_flat, axis=-1)
      new_a_flat = tf.one_hot(max_idx, depth=self.num_pixels) * max_act[..., tf.newaxis]
      new_a = tf.reshape(ff_act, [-1,] + self.a_shape)
      a_out = a_in + new_a

    return a_out
