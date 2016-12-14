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
  norm_data = data.squeeze()
  if np.max(np.abs(data)) > 0:
    norm_data = (data / np.max(np.abs(data))).squeeze()
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
  axis_image = sub_axis.imshow(data, cmap="Greys", interpolation="nearest")
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
