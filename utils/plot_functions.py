import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

"""
Normalize data
Outputs:
  data normalized so that when plotted 0 will be midlevel grey
Args:
  data: np.ndarray
"""
def normalize_data(data):
  if np.max(np.abs(data)) > 0:
    norm_data = (data / np.max(np.abs(data))).squeeze()
  else:
    norm_data = data.squeeze()
  return norm_data

"""
Pad data with ones for visualization
Outputs:
  padded version of input
Args:
  data: np.ndarray
"""
def pad_data(data):
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = (((0, n ** 2 - data.shape[0]),
    (1, 1), (1, 1)) # add some space between filters
    + ((0, 0),) * (data.ndim - 3)) # don't pad last dimension (if there is one)
  padded_data = np.pad(data, padding, mode="constant", constant_values=1)
  # tile the filters into an image
  padded_data = padded_data.reshape((n, n) + padded_data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, padded_data.ndim + 1)))
  padded_data = padded_data.reshape((n * padded_data.shape[1], n * padded_data.shape[3]) + padded_data.shape[4:])
  return padded_data

"""
Save figure for input data as a tiled image
Outputs:
  fig: index for figure call
  sub_axis: index for subplot call
  axis_image: index for imshow call
Inpus:
  data: [np.ndarray] of shape:
    (height, width) - single image
    (n, height, width) - n images tiled with dim (sqrt(n), sqrt(n))
  normalize: [bool] indicating whether the data should be streched (normalized)
    This is recommended for dictionary plotting.
  title: [str] for title of figure
  save_filename: [str] holding output directory for writing,
    figures will not display with GUI if set
"""
def save_data_tiled(data, normalize=False, title="", save_filename="",
  vmin=-1.0, vmax=1.0):
  if normalize:
    data = normalize_data(data)
  if len(data.shape) >= 3:
    data = pad_data(data)
  fig, sub_axis = plt.subplots(1)
  axis_image = sub_axis.imshow(data, cmap="Greys_r", interpolation="nearest")
  axis_image.set_clim(vmin=vmin, vmax=vmax)
  cbar = fig.colorbar(axis_image)
  sub_axis.tick_params(
   axis="both",
   bottom="off",
   top="off",
   left="off",
   right="off")
  sub_axis.get_xaxis().set_visible(False)
  sub_axis.get_yaxis().set_visible(False)
  sub_axis.set_title(title)
  if save_filename == "":
    save_filename = "./output.ps"
  fig.savefig(save_filename, transparent=True, bbox_inches="tight", pad_inches=0.01)
  plt.close(fig)

"""
Generate time-series plots of stats specified by keys
Outputs:
  fig: [int] corresponding to the figure number
Inputs:
  data: [dict] containing data to be plotted. len of all values should be equal
        data must have the key "batch_step"
  labels: [list of str] optional list of labels, should be same len as
          data.keys(). If nothing is given, data.keys() will be used as labels
  out_filename: [str] containing the complete output filename.
"""
def save_losses(data, labels=None, out_filename='./Fig.pdf'):
  data_keys = list(data.keys())
  data_keys.remove("batch_step")
  if labels is None:
    labels = data_keys
  num_keys = len(data_keys)
  fig, sub_axes = plt.subplots(num_keys)
  axis_image = [None]*num_keys
  for key_idx, key in enumerate(data_keys):
    axis_image[key_idx] = sub_axes[key_idx].plot(data["batch_step"], data[key])
    if key_idx < len(data_keys)-1:
      sub_axes[key_idx].get_xaxis().set_ticklabels([])
    sub_axes[key_idx].locator_params(axis="y", nbins=5)
    sub_axes[key_idx].set_ylabel(labels[key_idx])
    ylabel_xpos = -0.1
    sub_axes[key_idx].yaxis.set_label_coords(ylabel_xpos, 0.5)
  sub_axes[-1].set_xlabel("Batch Number")
  fig.suptitle("Average Losses per Batch", y=1.0, x=0.5)
  fig.savefig(out_filename, transparent=True)
  plt.close(fig)
