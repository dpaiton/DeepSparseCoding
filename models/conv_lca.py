import numpy as np
import logging
import json as js
import tensorflow as tf
import utils.plot_functions as pf
from models.lca import LCA

class CONV_LCA(LCA):
  def __init__(self, params, schedule):
    super(CONV_LCA, self).__init__(params, schedule)

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [dict] model parameters
    Modifiable Parameters:
      input_shape
      stride_x
      stride_y
      patch_size_y
      patch_size_x
    """
    super(CONV_LCA, self).load_params(params)
    self.input_shape = params["input_shape"]
    self.stride_x = int(params["stride_x"])
    self.stride_y = int(params["stride_y"])
    self.patch_size_y = int(params["patch_size_y"])
    self.patch_size_x = int(params["patch_size_x"])
    if len(self.input_shape) == 2:
      self.input_shape += [1]
    self.phi_shape = [int(self.patch_size_y), int(self.patch_size_x),
      int(self.input_shape[2]), int(self.num_neurons)]
    assert (self.input_shape[0] % self.stride_x == 0), (
      "Stride x must divide evenly into input shape")
    assert (self.input_shape[1] % self.stride_y == 0), (
      "Stride y must divide evenly into input shape")
    self.u_x = int(self.input_shape[0]/self.stride_x)
    self.u_y = int(self.input_shape[1]/self.stride_y)
    self.u_shape = [int(self.u_y), int(self.u_x), int(self.num_neurons)]
    self.x_shape = [None,]+ self.input_shape

  def step_inference(self, u_in, a_in, step):
    with tf.name_scope("update_u"+str(step)) as scope:
      recon_grad = tf.gradients(self.compute_recon_loss(a_in), a_in)[0]
      du = tf.subtract(tf.add(-recon_grad, a_in), u_in, name="du")
      u_out = tf.add(u_in, tf.multiply(self.eta, du))
    return u_out

  def infer_coefficients(self):
   u_list = [self.u_zeros]
   a_list = [self.threshold_units(u_list[0])]
   for step in range(self.num_steps-1):
     u = self.step_inference(u_list[step], a_list[step], step+1)
     u_list.append(u)
     a_list.append(self.threshold_units(u))
   return (u_list, a_list)

  def compute_recon(self, a_in):
    x_ = tf.nn.conv2d_transpose(a_in, self.phi, tf.shape(self.x),
      [1, self.stride_y, self.stride_x, 1], padding="SAME", name="reconstruction")
    return x_

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    feed_dict = self.get_feed_dict(input_data, input_labels)
    weights, recon = tf.get_default_session().run([self.phi, self.x_], feed_dict)
    current_step = str(self.global_step.eval())
    pf.plot_data_tiled(input_data[0,...].reshape((np.int(np.sqrt(self.num_pixels)),
      np.int(np.sqrt(self.num_pixels)))),
      normalize=False, title="Images at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"images_"+self.version+"-"
      +current_step.zfill(5)+".png"))
    pf.plot_data_tiled(recon[0,...].reshape((
      np.int(np.sqrt(self.num_pixels)),
      np.int(np.sqrt(self.num_pixels)))),
      normalize=False, title="Recons at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"recons_v"+self.version+"-"+current_step.zfill(5)+".png"))
    pf.plot_data_tiled(np.transpose(weights, axes=(3,0,1,2)),
      normalize=False, title="Dictionary at step "+current_step,
      vmin=np.min(weights), vmax=np.max(weights),
      save_filename=(self.disp_dir+"phi_v"+self.version+"_"+current_step.zfill(5)+".pdf"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      pf.plot_data_tiled(np.transpose(grad, axes=(3,0,1,2)), normalize=True,
        title="Gradient for phi at step "+current_step, vmin=None, vmax=None,
        save_filename=(self.disp_dir+"dphi_v"+self.version+"_"+current_step.zfill(5)+".pdf"))
