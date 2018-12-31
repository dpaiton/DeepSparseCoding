import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
from models.lca import LCA

class CONV_LCA(LCA):
  """
  Convolutional LCA model
  Inference is defined within the graph
  """
  def __init__(self):
    super(CONV_LCA, self).__init__()

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
    super(CONV_LCA, self).load_params(params)
    if len(self.data_shape) == 2:
      self.data_shape += [1]
    assert (self.data_shape[0] % self.params.stride_x == 0), (
      "Stride x must divide evenly into input shape")
    assert (self.data_shape[1] % self.params.stride_y == 0), (
      "Stride y must divide evenly into input shape")
    self.u_x = int(self.data_shape[0]/self.params.stride_x)
    self.u_y = int(self.data_shape[1]/self.params.stride_y)
    self.num_pixels = int(self.params.patch_size_y * self.params.patch_size_x * self.data_shape[2])
    self.phi_shape = [self.params.patch_size_y, self.params.patch_size_x,
      int(self.data_shape[2]), int(self.params.num_neurons)]
    self.u_shape = [self.u_y, self.u_x, int(self.params.num_neurons)]
    self.x_shape = [None,] + self.data_shape

  def compute_recon(self, a_in):
    x_ = tf.nn.conv2d_transpose(a_in, self.phi, tf.shape(self.x),
      [1, self.params.stride_y, self.params.stride_x, 1], padding="SAME", name="reconstruction")
    return x_

  def step_inference(self, u_in, a_in, step):
    with tf.name_scope("update_u"+str(step)) as scope:
      # Can use reconstruction loss gradient for conv sparse coding
      #recon_grad = tf.gradients(self.compute_recon_loss(a_in), a_in)[0]
      #du = tf.subtract(tf.add(-recon_grad, a_in), u_in, name="du")
      recon_error = self.x - self.compute_recon(a_in)
      error_injection = tf.nn.conv2d(recon_error, self.phi, [1, self.params.stride_y,
        self.params.stride_x, 1], padding="SAME", use_cudnn_on_gpu=True, name="forward_injection")
      du = tf.subtract(tf.add(error_injection, a_in), u_in, name="du")
      u_out = tf.add(u_in, tf.multiply(self.eta, du))
    return u_out

  def infer_coefficients(self):
   u_list = [self.u_zeros]
   a_list = [self.threshold_units(u_list[0])]
   for step in range(self.params.num_steps-1):
     u = self.step_inference(u_list[step], a_list[step], step+1)
     u_list.append(u)
     a_list.append(self.threshold_units(u))
   return (u_list, a_list)

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
    weights, recon = tf.get_default_session().run([self.phi, self.x_], feed_dict)
    recon = dp.rescale_data_to_one(recon)[0]
    weights = np.transpose(dp.rescale_data_to_one(weights.T)[0].T, axes=(3,0,1,2))
    current_step = str(self.global_step.eval())
    input_data = dp.rescale_data_to_one(input_data)[0]
    pf.plot_data_tiled(input_data[0,...], normalize=False,
      title="Images at step "+current_step, vmin=None, vmax=None, cmap=cmap,
      save_filename=(self.disp_dir+"images_"+self.params.version+"-"+current_step.zfill(5)+".png"))
    pf.plot_data_tiled(recon[0,...], normalize=False,
      title="Recons at step "+current_step, vmin=None, vmax=None, cmap=cmap,
      save_filename=(self.disp_dir+"recons_v"+self.params.version+"-"+current_step.zfill(5)+".png"))
    pf.plot_data_tiled(weights, normalize=False, title="Dictionary at step "+current_step,
      vmin=np.min(weights), vmax=np.max(weights), cmap=cmap,
      save_filename=(self.disp_dir+"phi_v"+self.params.version+"-"+current_step.zfill(5)+".png"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      pf.plot_data_tiled(np.transpose(grad, axes=(3,0,1,2)), normalize=True,
        title="Gradient for phi at step "+current_step, vmin=None, vmax=None, cmap=cmap,
        save_filename=(self.disp_dir+"dphi_v"+self.params.version+"-"+current_step.zfill(5)+".png"))
