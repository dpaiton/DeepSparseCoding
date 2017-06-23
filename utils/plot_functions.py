import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
from mpl_toolkits import axes_grid1
import utils.image_processing as ip

"""
Plot the top correlated bases for basis functions indexed in bf_indices
Inputs:
  a_cov [np.ndarray]
  weights [np.ndarray] of shape [num_inputs, num_outputs]
  bf_indices [list] of basis functions indices
  num_top_cov_bases [int] number of top correlated basis functions to plot
"""
def plot_top_bases(a_cov, weights, bf_indices, num_top_cov_bases):
  num_bases = len(bf_indices)
  fig = plt.figure(figsize=(num_top_cov_bases+2, num_bases))
  gs = gsp.GridSpec(num_bases, num_top_cov_bases+2, hspace=0.6)
  for x_id in range(num_bases):
    primary_bf_idx = bf_indices[x_id]
    a_cov_row = a_cov[primary_bf_idx, :]
    sorted_cov_indices = np.argsort(a_cov[primary_bf_idx, :])[-2::-1]
    primary_bf = np.squeeze(ip.reshape_data(weights.T[primary_bf_idx,...],
      flatten=False)[0])
    ax = plt.subplot(gs[x_id,0])
    ax.imshow(primary_bf, cmap="Greys_r", interpolation="nearest")
    ax.tick_params(axis="both", bottom="off", top="off",
      left="off", right="off")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    [i.set_linewidth(3.0) for i in ax.spines.values()]
    strengths = []
    for y_id, bf_idx in enumerate(sorted_cov_indices[:num_top_cov_bases]):
      bf = np.squeeze(ip.reshape_data(weights.T[bf_idx,...],
        flatten=False)[0])
      ax = plt.subplot(gs[x_id, y_id+1])
      ax.imshow(bf, cmap="Greys_r", interpolation="nearest")
      ax.tick_params(axis="both", bottom="off", top="off",
        left="off", right="off")
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      strengths.append(a_cov[primary_bf_idx, bf_idx])
    ax = plt.subplot(gs[x_id, -1])
    ax.plot(strengths)
    ax.set_xticklabels([])
    ylims = ax.get_ylim()
    ax.set_yticks([0, ylims[1]])
    ax.xaxis.set_ticks(np.arange(0, num_top_cov_bases, 1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", bottom="off", top="off", left="off", right="off")
    ax.tick_params(axis="x", direction="in")
    for idx, tick in enumerate(ax.yaxis.get_majorticklabels()):
      if idx == 0:
        tick.set_verticalalignment("bottom")
      else:
        tick.set_verticalalignment("top")
  plt.subplot(gs[0,0]).set_title("rand bf");
  plt.subplot(gs[0,1]).set_title("strongest corr --> weakest corr", horizontalalignment="left");
  plt.show()

"""
Plot summary statistics for given dictionary and pooling weights
Inputs:
  plot_func [function] function that receives inputs (axis, bf_stats, bf_idx, colorVal)
    function should add some plot to axis
    currently must be pf.plot_func() or pf.plot_ellipse()
  weights [np.ndarray] of shape [num_inputs, num_outputs]
  pooling_filters [np.ndarray] of shape [num_outputs, num_clusters]
  num_connected_weights [int] number of connected weights to plot
    must be equal to or less than num_outputs
  num_pooling_filters [int] number of filters to plot
TODO:
  Change plot_func to use *kwargs to be more general
"""
def plot_pooling_func(plot_func, weights, pooling_filters, num_connected_weights, num_pooling_filters):
  [num_inputs, num_outputs] = weights.shape
  assert num_pooling_filters <= num_outputs, (
    "num_pooling_filters must be less than or equal to weights.shape[1]")
  bf_stats = ip.get_dictionary_stats(weights)
  cmap = plt.get_cmap('bwr')
  cNorm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
  scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
  num_plts_y = np.int32(np.ceil(np.sqrt(num_pooling_filters)))
  num_plts_x = np.int32(np.floor(np.sqrt(num_pooling_filters)))+1 # +cbar row
  fig, sub_axes = plt.subplots(num_plts_y, num_plts_x, figsize=(24,24))
  filter_idx_list = np.random.choice(np.arange(pooling_filters.shape[0],
    dtype=np.int32), size=num_pooling_filters, replace=False)
  filter_total = 0
  # TODO: rewrite with np.ndindex
  for y_id in range(num_plts_y):
    for x_id in range(num_plts_x):
      if (filter_total < num_pooling_filters and x_id != num_plts_x-1):
        filter_idx = filter_idx_list[filter_total]
        example_filter = pooling_filters[filter_idx, :]
        top_indices = np.argsort(np.abs(example_filter))[::-1]
        filter_norm = np.max(np.abs(example_filter))
        for bf_idx in top_indices[:num_connected_weights]:
          connection_strength = example_filter[bf_idx]/filter_norm
          colorVal = scalarMap.to_rgba(connection_strength)
          plot_func(sub_axes[y_id, x_id], bf_stats, bf_idx, colorVal)
        sub_axes[y_id, x_id].set_xlim(0, 15)
        sub_axes[y_id, x_id].set_ylim(0, 15)
        sub_axes[y_id, x_id].set_aspect("equal")
        filter_total += 1
      else:
        sub_axes[y_id, x_id].spines["right"].set_color("none")
        sub_axes[y_id, x_id].spines["top"].set_color("none")
        sub_axes[y_id, x_id].spines["left"].set_color("none")
        sub_axes[y_id, x_id].spines["bottom"].set_color("none")
      sub_axes[y_id, x_id].invert_yaxis()
      sub_axes[y_id, x_id].set_yticklabels([])
      sub_axes[y_id, x_id].set_xticklabels([])
      sub_axes[y_id, x_id].set_aspect("equal")
      sub_axes[y_id, x_id].tick_params(axis="both", bottom="off", top="off",
        left="off", right="off")
  scalarMap._A = []
  cbar = fig.colorbar(scalarMap, ax=list(sub_axes[:, -1]), ticks=[-1, 0, 1])
  plt.show()

"""
Plot line summary of pooled basis functions
Inputs:
  axis
  bf_stats
  bf_idx
  colorVal
"""
def plot_lines(axis, bf_stats, bf_idx, colorVal):
  y, x = bf_stats["orientations"][bf_idx]
  y_cen, x_cen = bf_stats["envelope_centers"][bf_idx]
  length = 3#bf_stats["lengths"][bf_idx][0]
  x_points = [(x_cen-x*length/2), (x_cen+x*length/2)]
  y_points = [(y_cen-y*length/2), (y_cen+y*length/2)]
  axis.plot(x_points, y_points, color=colorVal)

"""
Plot ellipse summary of pooled basis functions
Inputs:
  axis
  bf_stats
  bf_idx
  colorVal
"""
def plot_ellipse(axis, bf_stats, bf_idx, colorVal):
  y_cen, x_cen = bf_stats["envelope_centers"][bf_idx]
  (y_ang, x_ang) = bf_stats["orientations"][bf_idx]
  angle = np.rad2deg(np.arctan2(y_ang, x_ang))
  e = matplotlib.patches.Ellipse(xy=[x_cen, y_cen], width=0.8,
    height=0.3, angle=angle, color=colorVal, alpha=1.0, fill=True)
  axis.add_artist(e)
  e.set_clip_box(axis.bbox)

"""
Plot outputs of the ip.get_dictionary_stats()
Inputs:
  bf_stats [dict] output of ip.get_dictionary_stats()
  num_bf [int] number of basis functions to plot
"""
def plot_bf_stats(bf_stats, num_bf=1):
    fig, sub_axes = plt.subplots(num_bf, 6, figsize=(36,36))
    for bf_idx in range(int(num_bf)):
      bf = bf_stats["basis_functions"][bf_idx]
      env = bf_stats["envelopes"][bf_idx]
      filt = bf_stats["filters"][bf_idx]
      fourier = bf_stats["fourier_maps"][bf_idx]
      y, x = bf_stats["orientations"][bf_idx]
      y_cen, x_cen = bf_stats["envelope_centers"][bf_idx]
      length = bf_stats["lengths"][bf_idx] 
      line_img = bf_stats["line_images"][bf_idx]
      blob_img = bf_stats["blob_images"][bf_idx]
      fy_cen, fx_cen = bf_stats["fourier_centers"][bf_idx]
      # Basis function in pixel space
      sub_axes[bf_idx, 0].imshow(bf, cmap="Greys_r", interpolation="Nearest")
      sub_axes[bf_idx, 0].tick_params(axis="both", bottom="off", top="off",
        left="off", right="off")
      sub_axes[bf_idx, 0].get_xaxis().set_visible(False)
      sub_axes[bf_idx, 0].get_yaxis().set_visible(False)
      # Hilbert envelope
      sub_axes[bf_idx, 1].imshow(env, cmap="Greys_r", interpolation="Nearest")
      sub_axes[bf_idx, 1].tick_params(axis="both", bottom="off", top="off",
        left="off", right="off")
      sub_axes[bf_idx, 1].get_xaxis().set_visible(False)
      sub_axes[bf_idx, 1].get_yaxis().set_visible(False)
      # Hilbert filter
      sub_axes[bf_idx, 2].imshow(filt, cmap="Greys_r", interpolation="Nearest")
      sub_axes[bf_idx, 2].tick_params(axis="both", bottom="off", top="off",
        left="off", right="off")
      sub_axes[bf_idx, 2].get_xaxis().set_visible(False)
      sub_axes[bf_idx, 2].get_yaxis().set_visible(False)
      # Fourier transform of basis function
      sub_axes[bf_idx, 3].imshow(fourier, cmap="Greys_r", interpolation="Nearest")
      sub_axes[bf_idx, 3].tick_params(axis="both", top="off", right="off",
        bottom="off", left="off")
      sub_axes[bf_idx, 3].spines["left"].set_position("center")
      sub_axes[bf_idx, 3].spines["left"].set_color("black")
      sub_axes[bf_idx, 3].spines["left"].set_linewidth(2.5)
      sub_axes[bf_idx, 3].spines["bottom"].set_position("center")
      sub_axes[bf_idx, 3].spines["bottom"].set_color("black")
      sub_axes[bf_idx, 3].spines["bottom"].set_linewidth(2.5)
      sub_axes[bf_idx, 3].spines["top"].set_color("none")
      sub_axes[bf_idx, 3].spines["right"].set_color("none")
      sub_axes[bf_idx, 3].set_yticklabels([])
      sub_axes[bf_idx, 3].set_xticklabels([])
      sub_axes[bf_idx, 3].set_ylim([0, fourier.shape[0]-1])
      sub_axes[bf_idx, 3].set_xlim([0, fourier.shape[1]-1])
      # Pixel line summary of basis function
      line_img[y_cen, x_cen] += 1
      sub_axes[bf_idx, 4].imshow(line_img, interpolation="Nearest")
      sub_axes[bf_idx, 4].tick_params(axis="both", bottom="off", top="off",
        left="off", right="off")
      sub_axes[bf_idx, 4].get_xaxis().set_visible(False)
      sub_axes[bf_idx, 4].get_yaxis().set_visible(False)
      # Filtered Hilbert envelope 
      blob_img[np.where(blob_img>0)] = 1
      blob_img[y_cen, x_cen] += 1
      sub_axes[bf_idx, 5].imshow(blob_img, interpolation="Nearest")
      sub_axes[bf_idx, 5].tick_params(axis="both", bottom="off", top="off",
        left="off", right="off")
      sub_axes[bf_idx, 5].get_xaxis().set_visible(False)
      sub_axes[bf_idx, 5].get_yaxis().set_visible(False)
    sub_axes[0,0].set_title("bf", fontsize=32)
    sub_axes[0,1].set_title("envelope", fontsize=32)
    sub_axes[0,2].set_title("filter", fontsize=32)
    sub_axes[0,3].set_title("fourier map", fontsize=32)
    sub_axes[0,4].set_title("envelope summary line", fontsize=22)
    sub_axes[0,5].set_title("envelope summary blob", fontsize=22)
    plt.show()

"""
Plot results from performing Hilbert amplitude processing on weight matrix
Inputs:
  weights: [np.ndarray] with shape [num_inputs, num_outputs]
    num_inputs must have even square root.
"""
def plot_hilbert_analysis(weights):
  Envelope, bff_filt, Hil_filter, bff = ip.hilbertize(weights)
  num_inputs, num_outputs = weights.shape
  assert np.sqrt(num_inputs) == np.floor(np.sqrt(num_inputs)), (
    "weights.shape[0] must have an even square root.")
  patch_edge_size = int(np.sqrt(num_inputs))
  N = np.int32(np.sqrt(bff_filt.shape[1]))           
  fig, sub_axes = plt.subplots(3, 1, figsize=(64,64))  
  plot_data = pad_data(weights.T.reshape((num_outputs, patch_edge_size,
    patch_edge_size)))
  bf_axis_image = sub_axes[0].imshow(plot_data, cmap="Greys_r",
    interpolation="nearest")
  sub_axes[0].tick_params(axis="both", bottom="off", top="off", left="off",
    right="off")
  sub_axes[0].get_xaxis().set_visible(False)
  sub_axes[0].get_yaxis().set_visible(False)
  sub_axes[0].set_title("Basis Functions", fontsize=32)  
  plot_data = pad_data(np.abs(Envelope).reshape((num_outputs,
    patch_edge_size, patch_edge_size)))
  hil_axis_image = sub_axes[1].imshow(plot_data, cmap="Greys_r",
    interpolation="nearest")
  sub_axes[1].tick_params(axis="both", bottom="off", top="off", left="off",
    right="off")
  sub_axes[1].get_xaxis().set_visible(False)
  sub_axes[1].get_yaxis().set_visible(False)
  sub_axes[1].set_title("Analytic Signal Amplitude Envelope", fontsize=32)
  resh_Zf = np.abs(bff_filt).reshape((num_outputs, N, N))                             
  output_z = np.zeros(resh_Zf.shape)                                              
  for i in range(num_outputs):                                                    
    output_z[i,...] = resh_Zf[i,...] / np.max(resh_Zf[i,...])                     
  plot_data = pad_data(output_z)
  hil_axis_image = sub_axes[2].imshow(plot_data, cmap="Greys_r",
    interpolation="nearest")
  sub_axes[2].tick_params(axis="both", bottom="off", top="off", left="off",
    right="off")
  sub_axes[2].get_xaxis().set_visible(False)
  sub_axes[2].get_yaxis().set_visible(False)
  sub_axes[2].set_title("Fourier Amplitude Spectrum", fontsize=32)
  plt.show()

"""
Add a vertical color bar to an image plot.
Inputs:
  im: [AxisImage] object returned from matplotlib.plt.imshow()
  aspect: [int] aspect ratio of the colorbar
  pad_fraction: [float] how much space to place between colorbar & plot
  **kwargs: [dict] other keyword arguments that would be passed to im.axes.figure.colorbar()
"""
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
  divider = axes_grid1.make_axes_locatable(im.axes)
  width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
  pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
  current_ax = plt.gca()
  cax = divider.append_axes("right", size=width, pad=pad)
  plt.sca(current_ax)
  return im.axes.figure.colorbar(im, cax=cax, **kwargs)

"""
Generate a bar graph of data
Inputs:
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
Inputs:
  data [np.ndarray] data matrix, can have shapes:
    1D tensor [data_points]
    2D tensor [batch, data_points] - will plot avg hist, summing over batch
    3D tensor [batch, time_point, data_points] - will plot avg hist over time
  title: [str] for title of figure
  save_filename: [str] holding output directory for writing,
"""
def save_activity_hist(data, num_bins="auto", title="",
  save_filename="./hist.pdf"):
  num_dim = data.ndim
  if num_dim > 1:
    data = np.mean(data, axis=0)
  (fig, ax) = plt.subplots(1)
  vals, bins, patches = ax.hist(data, bins=num_bins, histtype="barstacked",
    stacked=True)
  ax.set_xlabel('Activity')
  ax.set_ylabel('Count')
  fig.suptitle(title, y=1.0, x=0.5)
  fig.tight_layout()
  fig.savefig(save_filename)
  plt.close(fig)


"""
Plot phase averaged power spectrum for a set of images
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
Save figure for input data as a tiled image
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
  if normalize:
    data = ip.normalize_data_with_max(data)
    vmin = -1.0
    vmax = 1.0
  if vmin is None:
    vmin = np.min(data)
  if vmax is None:
    vmax = np.max(data)
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
Inputs:
  data: [dict] containing data to be plotted. len of all values should be equal
        data must have the key "batch_step"
  labels: [list of str] optional list of labels, should be same len as
          data.keys(). If nothing is given, data.keys() will be used as labels
  save_filename: [str] containing the complete output filename.
"""
def save_stats(data, labels=None, save_filename="./Fig.pdf"):
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
  fig.suptitle("Stats per Batch", y=1.0, x=0.5)
  fig.savefig(save_filename, transparent=True)
  plt.close(fig)

"""
Pad data with ones for visualization
Outputs:
  padded version of input
Inputs:
  data: np.ndarray
  pad_values: [int] specifying what value will be used for padding
"""
def pad_data(data, pad_values=1):
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = (((0, n ** 2 - data.shape[0]),
    (1, 1), (1, 1)) # add some space between filters
    + ((0, 0),) * (data.ndim - 3)) # don't pad last dimension (if there is one)
  padded_data = np.pad(data, padding, mode="constant",
    constant_values=pad_values)
  # tile the filters into an image
  padded_data = padded_data.reshape((
    (n, n) + padded_data.shape[1:])).transpose((
    (0, 2, 1, 3) + tuple(range(4, padded_data.ndim + 1))))
  padded_data = padded_data.reshape((n * padded_data.shape[1],
    n * padded_data.shape[3]) + padded_data.shape[4:])
  return padded_data
