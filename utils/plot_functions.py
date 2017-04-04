import os
import numpy as np
import matplotlib.pyplot as plt
import utils.image_processing as ip

"""
Generate a bar graph of data
Args:
  data: [np.ndarray] of shape (N,)
  xticklabels: [list of N str] indicating the labels for the xticks
  save_filename: [str] indicating where the file should be saved
  xlabel: [str] indicating the x-axis label
  ylabel: [str] indicating the y-axis label
  title: [str] indicating the plot title
TODO: set num_xticks
"""
def save_bar(data, num_xticks=5, title="", save_filename="./bar_fig.pdf",
  xlabel="", ylabel=""):
  fig, ax = plt.subplots(1)
  bar = ax.bar(np.arange(len(data)), data)
  #xticklabels = [str(int(val)) for val in np.arange(len(data))]
  #xticks = ax.get_xticks()
  #ax.set_xticklabels(xticklabels)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  fig.suptitle(title, y=1.0, x=0.5)
  fig.savefig(save_filename, transparent=True)
  plt.close(fig)

"""
Histogram activity matrix
Outputs:
  fig: [int] figure number
  sub_axis: index for subplot call
Inputs:
  data [np.ndarray] data matrix, can have shapes:
    1D tensor [data_points]
    2D tensor [batch, data_points] - will plot avg hist, summing over batch
    3D tensor [batch, time_point, data_points] - will plot avg hist over time
  title: [str] for title of figure
  save_filename: [str] holding output directory for writing,
"""
def save_activity_hist(data, num_bins=100, title="",
  save_filename="./hist.pdf"):
  num_dim = data.ndim
  if num_dim > 1:
    data = np.mean(data, axis=0)
  (fig, ax) = plt.subplots(1)
  vals, bins, patches = ax.hist(data, histtype="barstacked", stacked=True)
  ax.set_xlabel('Activity')
  ax.set_ylabel('Count')
  fig.suptitle(title, y=1.0, x=0.5)
  fig.tight_layout()
  fig.savefig(save_filename)
  plt.close(fig)


"""
Plot phase averaged power spectrum for a set of images
Outputs:
  fig: [int] figure number
  sub_axis: index for subplot call
Inputs:
  data: [np.ndarray] 1D data to be plotted
  title: [str] for title of figure
  save_filename: [str] holding output directory for writing,
    figures will not display with GUI if set
"""
def save_phase_avg_power_spec(data, title="", save_filename="./pow_spec.pdf"):
  (fig, ax) = plt.subplots(1)
  ax.loglog(range(data[data>1].shape[0]), data[data>1])
  fig.suptitle(title, y=1.0, x=0.5)
  fig.savefig(save_filename)
  plt.close(fig)


"""
Pad data with ones for visualization
Outputs:
  padded version of input
Inputs:
  data: np.ndarray
"""
def pad_data(data):
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = (((0, n ** 2 - data.shape[0]),
    (1, 1), (1, 1)) # add some space between filters
    + ((0, 0),) * (data.ndim - 3)) # don't pad last dimension (if there is one)
  padded_data = np.pad(data, padding, mode="constant", constant_values=1)
  # tile the filters into an image
  padded_data = padded_data.reshape((
    (n, n) + padded_data.shape[1:])).transpose((
    (0, 2, 1, 3) + tuple(range(4, padded_data.ndim + 1))))
  padded_data = padded_data.reshape((n * padded_data.shape[1],
    n * padded_data.shape[3]) + padded_data.shape[4:])
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
    (n, height, width) - n images tiled with dim (height, width)
    (n, height, width, features) - n images tiled with dim
      (height, width, features); features could be color
  normalize: [bool] indicating whether the data should be streched (normalized)
    This is recommended for dictionary plotting.
  title: [str] for title of figure
  save_filename: [str] holding output directory for writing,
    figures will not display with GUI if set
  vmin, vmax: [int] the min and max of the color range
"""
def save_data_tiled(data, normalize=False, title="", save_filename="",
  vmin=None, vmax=None):
  if vmin is None:
    vmin = np.min(data)
  if vmax is None:
    vmax = np.max(data)
  if normalize:
    data = ip.normalize_data_with_max(data)
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
