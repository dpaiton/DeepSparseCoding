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
    self.vector_inputs = False

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
    self.stride_x = int(params["stride_x"])
    self.stride_y = int(params["stride_y"])
    self.patch_size_y = int(params["patch_size_y"])
    self.patch_size_x = int(params["patch_size_x"])
    if len(self.data_shape) == 2:
      self.data_shape += [1]
    self.phi_shape = [int(self.patch_size_y), int(self.patch_size_x),
      int(self.data_shape[2]), int(self.num_neurons)]
    assert (self.data_shape[0] % self.stride_x == 0), (
      "Stride x must divide evenly into input shape")
    assert (self.data_shape[1] % self.stride_y == 0), (
      "Stride y must divide evenly into input shape")
    self.u_x = int(self.data_shape[0]/self.stride_x)
    self.u_y = int(self.data_shape[1]/self.stride_y)
    self.u_shape = [int(self.u_y), int(self.u_x), int(self.num_neurons)]
    self.x_shape = [None,]+ self.data_shape

  def preprocess_dataset(self, dataset, params=None):
    """
    Perform preprocessing on the dataset images 
    Inputs:
      dataset [dict] returned from data/data_picker
    Parameters are set using the model parameter dictionary.
    Possible parameters  are:
      center_data: subtract mean from data
      norm_data: divide data by the maximum
      whiten_data: default method is using the Fourier amplitude spectrium ("FT")
        change default with whiten_method param
      standardize_data: subtract mean and divide by the standard deviation
      contrast_normalize: divide by gaussian blurred surround pixels
      extract_patches: break up data into patches
        see utils/data_processing/exract_patches() for docs
    """
    if params is None:
      assert self.params_loaded, (
        "You must either provide parameters or load the model params before preprocessing.")
      params = self.params
    for key in dataset.keys():
      if "whiten_data" in params.keys() and params["whiten_data"]:
        if "whiten_method" in params.keys():
          if params["whiten_method"] == "FT": # other methods require patching first
            dataset[key].images, dataset[key].data_mean, dataset[key].w_filter = \
              dp.whiten_data(dataset[key].images, method=params["whiten_method"])
      if "extract_patches" in params.keys() and params["extract_patches"]:
        assert all(key in params.keys()
          for key in ["num_patches", "patch_edge_size", "overlapping_patches",
          "randomize_patches"]), ("Insufficient params for patches.")
        out_shape = (int(params["num_patches"]), int(params["patch_edge_size"]),
          int(params["patch_edge_size"]), dataset[key].num_channels)
        dataset[key].num_examples = out_shape[0]
        dataset[key].reset_counters()
        if "patch_variance_threshold" in params.keys():
          dataset[key].images = dp.extract_patches(dataset[key].images, out_shape,
            params["overlapping_patches"], params["randomize_patches"],
            params["patch_variance_threshold"], dataset[key].rand_state)
        else:
          dataset[key].images = dp.extract_patches(dataset[key].images, out_shape,
            params["overlapping_patches"], params["randomize_patches"],
            var_thresh=0, rand_state=dataset[key].rand_state)
        dataset[key].shape = dataset[key].images.shape
        dataset[key].num_rows = dataset[key].shape[1]
        dataset[key].num_cols = dataset[key].shape[2]
        dataset[key].num_channels = dataset[key].shape[3]
        dataset[key].num_pixels = np.prod(dataset[key].shape[1:])
      if "whiten_data" in params.keys() and params["whiten_data"]:
        if "whiten_method" in params.keys() and params["whiten_method"] != "FT":
          dataset[key].images, dataset[key].data_mean, dataset[key].w_filter = \
            dp.whiten_data(dataset[key].images, method=params["whiten_method"])
      if "norm_data_to_one" in params.keys() and params["norm_data"]:
        dataset[key].images = dp.rescale_data_to_one(dataset[key].images)
      if "standardize_data" in params.keys() and params["standardize_data"]:
        dataset[key].images, dataset[key].data_mean, dataset[key].data_std = \
          dp.standardize_data(dataset[key].images)
        self.data_mean = dataset[key].data_mean
        self.data_std = dataset[key].data_std
    return dataset

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
    if input_data.shape[-1] == 3:
      cmap = None
    elif input_data.shape[-1] == 1:
      cmap = "Greys_r"
    else:
      assert False, ("Input_data.shape[-1] should indicate color channel, and should be 1 or 3")
    feed_dict = self.get_feed_dict(input_data, input_labels)
    weights, recon = tf.get_default_session().run([self.phi, self.x_], feed_dict)
    recon = dp.rescale_data_to_one(recon)
    weights = np.transpose(dp.rescale_data_to_one(weights.T).T, axes=(3,0,1,2))
    current_step = str(self.global_step.eval())
    input_data = dp.rescale_data_to_one(input_data)
    pf.plot_data_tiled(input_data[0,...], normalize=False,
      title="Images at step "+current_step, vmin=None, vmax=None, cmap=cmap,
      save_filename=(self.disp_dir+"images_"+self.version+"-"+current_step.zfill(5)+".png"))
    pf.plot_data_tiled(recon[0,...], normalize=False,
      title="Recons at step "+current_step, vmin=None, vmax=None, cmap=cmap,
      save_filename=(self.disp_dir+"recons_v"+self.version+"-"+current_step.zfill(5)+".png"))
    pf.plot_data_tiled(weights, normalize=False, title="Dictionary at step "+current_step,
      vmin=np.min(weights), vmax=np.max(weights), cmap=cmap,
      save_filename=(self.disp_dir+"phi_v"+self.version+"_"+current_step.zfill(5)+".png"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      pf.plot_data_tiled(np.transpose(grad, axes=(3,0,1,2)), normalize=True,
        title="Gradient for phi at step "+current_step, vmin=None, vmax=None, cmap=cmap,
        save_filename=(self.disp_dir+"dphi_v"+self.version+"_"+current_step.zfill(5)+".png"))
