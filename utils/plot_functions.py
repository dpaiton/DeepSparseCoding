import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
from mpl_toolkits import axes_grid1
import utils.image_processing as ip

def plot_pooling_summaries(bf_stats, pooling_filters, num_connected_weights, num_pooling_filters, lines=False):
  """
  Plot 2nd layer (fully-connected) weights in terms of connection strengths to 1st layer weights
  Inputs:
    bf_stats [dict] output of ip.get_dictionary_stats() which was run on the 1st layer weights
    pooling_filters [np.ndarray] 2nd layer weights, of shape [num_1st_layer_neurons, num_2nd_layer_neurons]
    num_connected_weights [int] How many 1st layer weight summaries to include for a given 2nd layer neuron
    num_pooling_filters [int] How many 2nd layer neurons to plot
    lines [bool] if True, 1st layer weight summaries will appear as lines instead of ellipses
  """
  num_inputs = bf_stats["num_inputs"]
  num_outputs = bf_stats["num_outputs"]
  patch_edge_size = np.int32(np.sqrt(num_inputs))
  assert num_pooling_filters <= num_outputs, (
    "num_pooling_filters must be less than or equal to bf_stats['num_outputs']")
  cmap = plt.get_cmap('bwr')
  cNorm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
  scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
  num_plts_y = np.int32(np.ceil(np.sqrt(num_pooling_filters)))
  num_plts_x = np.int32(np.floor(np.sqrt(num_pooling_filters)))+1 # +cbar row
  fig, sub_ax = plt.subplots(num_plts_y, num_plts_x, figsize=(15,15))
  filter_idx_list = np.random.choice(np.arange(pooling_filters.shape[0],
    dtype=np.int32), size=num_pooling_filters, replace=False)
  filter_total = 0
  for plot_id in  np.ndindex((num_plts_y, num_plts_x)):
    (y_id, x_id) = plot_id
    if (filter_total < num_pooling_filters and x_id != num_plts_x-1):
      filter_idx = filter_idx_list[filter_total]
      example_filter = pooling_filters[filter_idx, :]
      top_indices = np.argsort(np.abs(example_filter))[::-1]
      filter_norm = np.max(np.abs(example_filter))
      SFs = np.asarray([np.sqrt(fcent[0]**2 + fcent[1]**2)
        for fcent in bf_stats["fourier_centers"]], dtype=np.float32)
      sf_norm = np.max(SFs)
      # Plot weakest of the top connected filters first because of occlusion
      for bf_idx in top_indices[:num_connected_weights][::-1]:
        connection_strength = example_filter[bf_idx]/filter_norm
        colorVal = scalarMap.to_rgba(connection_strength)
        center = bf_stats["gauss_centers"][bf_idx]
        evals, evecs = bf_stats["orientations"][bf_idx]
        ## TODO: Add Fourier info
        #fourier_center = bf_stats["fourier_centers"][bf_idx]
        #spatial_freq = np.sqrt(fourier_center[0]**2+fourier_center[1]**2)/sf_norm
        #angle = np.rad2deg(np.arctan2(*bf_stats["fourier_centers"][bf_idx]))
        #alpha = spatial_freq
        plot_ellipse(sub_ax[plot_id], center, evals, evecs[:,0], colorVal, alpha=1.0, lines=lines)
      sub_ax[plot_id].set_xlim(0, patch_edge_size-1)
      sub_ax[plot_id].set_ylim(0, patch_edge_size-1)
      sub_ax[plot_id].set_aspect("equal")
      filter_total += 1
    else:
      sub_ax[plot_id].spines["right"].set_color("none")
      sub_ax[plot_id].spines["top"].set_color("none")
      sub_ax[plot_id].spines["left"].set_color("none")
      sub_ax[plot_id].spines["bottom"].set_color("none")
    sub_ax[plot_id].invert_yaxis()
    sub_ax[plot_id].set_yticklabels([])
    sub_ax[plot_id].set_xticklabels([])
    sub_ax[plot_id].set_aspect("equal")
    sub_ax[plot_id].tick_params(axis="both", bottom="off", top="off",
      left="off", right="off")
  scalarMap._A = []
  cbar = fig.colorbar(scalarMap, ax=list(sub_ax[:, -1]), ticks=[-1, 0, 1])
  plt.show()

def plot_top_bases(a_cov, weights, bf_indices, num_top_cov_bases):
  """
  Plot the top correlated bases for basis functions indexed in bf_indices
  Inputs:
    a_cov [np.ndarray]
    weights [np.ndarray] of shape [num_inputs, num_outputs]
    bf_indices [list] of basis functions indices
    num_top_cov_bases [int] number of top correlated basis functions to plot
  """
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

def plot_ellipse_summaries(bf_stats, num_bf=4, lines=False):
  """
  Plot basis functions with summary ellipses drawn over them
  Inputs:
    bf_stats [dict] output of ip.get_dictionary_stats()
    num_bf [int] number of basis functions to plot (must be >=4)
    lines [bool] If true, will plot lines instead of ellipses
  """
  tot_num_bf = len(bf_stats["basis_functions"])
  bf_range = np.random.choice([i for i in range(tot_num_bf)],
    num_bf, replace=False)
  num_plots_y = int(np.ceil(np.sqrt(num_bf)))
  num_plots_x = int(np.floor(np.sqrt(num_bf)))
  filter_idx = 0
  fig, sub_ax = plt.subplots(num_plots_y, num_plots_x, figsize=(17, 17))
  for plot_id in  np.ndindex((num_plots_y, num_plots_x)):
    if filter_idx < tot_num_bf:
      bf = bf_stats["basis_functions"][filter_idx]
      sub_ax[plot_id].imshow(bf, interpolation="Nearest", cmap="Greys_r")
      center = bf_stats["gauss_centers"][filter_idx]
      evals, evecs = bf_stats["orientations"][filter_idx]
      alpha = 1.0
      colorVal = "r"
      plot_ellipse(sub_ax[plot_id], center, evals, evecs[:,0], colorVal, alpha, lines)
      sub_ax[plot_id].tick_params(axis="both", bottom="off", top="off",
        left="off", right="off")
      sub_ax[plot_id].get_xaxis().set_visible(False)
      sub_ax[plot_id].get_yaxis().set_visible(False)
      filter_idx += 1
    sub_ax[plot_id].spines["right"].set_color("none")
    sub_ax[plot_id].spines["top"].set_color("none")
    sub_ax[plot_id].spines["left"].set_color("none")
    sub_ax[plot_id].spines["bottom"].set_color("none")
    sub_ax[plot_id].tick_params(axis="both", bottom="off", top="off", left="off", right="off")
    sub_ax[plot_id].get_xaxis().set_visible(False)
    sub_ax[plot_id].get_yaxis().set_visible(False)
    sub_ax[plot_id].set_aspect("equal")
  plt.show()

def plot_ellipse(axis, center, shape, orientation, colorVal='auto', alpha=1.0, lines=False):
  """
  Add an ellipse to given axis
  Inputs:
    axis [matplotlib.axes._subplots.AxesSubplot] axis on which ellipse should be drawn
    center [tuple or list] specifying [y, x] center coordinates
    shape [tuple or list] specifying [width, height] shape of ellipse
    orientation [tuple or list] specifying [y_len, x_len] for triangle specifying angle of ellipse
    colorVal [matplotlib color spec] specifying the color of the edge & face of the ellipse
    alpha [float] specifying the transparency of the ellipse
    lines [bool] if true, output will be a line, where the secondary axis of the ellipse is collapsed
  """
  y_cen, x_cen = center
  width, height = shape
  y_ang, x_ang = orientation
  if colorVal == "b":
    angle = -np.rad2deg(np.arctan2(y_ang, x_ang))
  else:
    angle = np.rad2deg(np.arctan2(y_ang, x_ang))
  width, height = shape
  if lines:
    min_length = 0.1
    if width < height:
      width = min_length
    elif width > height:
      height = min_length
  e = matplotlib.patches.Ellipse(xy=[x_cen, y_cen], width=width,
    height=height, angle=angle, edgecolor=colorVal, facecolor=colorVal,
    alpha=alpha, fill=True)
  axis.add_artist(e)
  e.set_clip_box(axis.bbox)

def plot_bf_stats(bf_stats, num_bf=2):
  """
  Plot outputs of the ip.get_dictionary_stats()
  Inputs:
    bf_stats [dict] output of ip.get_dictionary_stats()
    num_bf [int] number of basis functions to plot
  """
  tot_num_bf = len(bf_stats["basis_functions"])
  bf_idx_list = np.random.choice(tot_num_bf, num_bf, replace=False)
  fig, sub_ax = plt.subplots(num_bf, 6, figsize=(15,15))
  for plot_id in range(int(num_bf)):
    bf_idx = bf_idx_list[plot_id]
    # Basis function in pixel space
    bf = bf_stats["basis_functions"][bf_idx]
    sub_ax[plot_id, 0].imshow(bf, cmap="Greys_r", interpolation="Nearest")
    sub_ax[plot_id, 0].tick_params(axis="both", bottom="off", top="off",
      left="off", right="off")
    sub_ax[plot_id, 0].get_xaxis().set_visible(False)
    sub_ax[plot_id, 0].get_yaxis().set_visible(False)
    # Hilbert envelope
    env = bf_stats["envelopes"][bf_idx]
    sub_ax[plot_id, 1].imshow(env, cmap="Greys_r", interpolation="Nearest")
    sub_ax[plot_id, 1].tick_params(axis="both", bottom="off", top="off",
      left="off", right="off")
    sub_ax[plot_id, 1].get_xaxis().set_visible(False)
    sub_ax[plot_id, 1].get_yaxis().set_visible(False)
    # Hilbert filter
    filt = bf_stats["filters"][bf_idx]
    sub_ax[plot_id, 2].imshow(filt, cmap="Greys_r", interpolation="Nearest")
    sub_ax[plot_id, 2].tick_params(axis="both", bottom="off", top="off",
      left="off", right="off")
    sub_ax[plot_id, 2].get_xaxis().set_visible(False)
    sub_ax[plot_id, 2].get_yaxis().set_visible(False)
    # Fourier transform of basis function
    fourier = bf_stats["fourier_maps"][bf_idx]
    sub_ax[plot_id, 3].imshow(fourier, cmap="Greys_r", interpolation="Nearest")
    sub_ax[plot_id, 3].tick_params(axis="both", top="off", right="off",
      bottom="off", left="off")
    sub_ax[plot_id, 3].spines["left"].set_position("center")
    sub_ax[plot_id, 3].spines["left"].set_color("black")
    sub_ax[plot_id, 3].spines["left"].set_linewidth(2.5)
    sub_ax[plot_id, 3].spines["bottom"].set_position("center")
    sub_ax[plot_id, 3].spines["bottom"].set_color("black")
    sub_ax[plot_id, 3].spines["bottom"].set_linewidth(2.5)
    sub_ax[plot_id, 3].spines["top"].set_color("none")
    sub_ax[plot_id, 3].spines["right"].set_color("none")
    sub_ax[plot_id, 3].set_yticklabels([])
    sub_ax[plot_id, 3].set_xticklabels([])
    sub_ax[plot_id, 3].set_ylim([0, fourier.shape[0]-1])
    sub_ax[plot_id, 3].set_xlim([0, fourier.shape[1]-1])
    # Summary ellipse
    sub_ax[plot_id, 4].imshow(bf, interpolation="Nearest", cmap="Greys_r")
    center = bf_stats["gauss_centers"][bf_idx]
    evals, evecs = bf_stats["orientations"][bf_idx]
    alpha = 1.0
    colorVal = "r"
    plot_ellipse(sub_ax[plot_id, 4], center, evals, evecs[:,0], colorVal, alpha)
    sub_ax[plot_id, 4].tick_params(axis="both", bottom="off", top="off",
      left="off", right="off")
    sub_ax[plot_id, 4].get_xaxis().set_visible(False)
    sub_ax[plot_id, 4].get_yaxis().set_visible(False)
    sub_ax[plot_id, 4].set_aspect("equal")
    # Fourier summary stats
    sub_ax[plot_id, 5].imshow(bf, interpolation="Nearest", cmap="Greys_r")
    center = bf_stats["gauss_centers"][bf_idx]
    evals, evecs = bf_stats["orientations"][bf_idx]
    orientation = bf_stats["fourier_centers"][bf_idx]
    alpha = 1.0
    colorVal = "b"
    plot_ellipse(sub_ax[plot_id, 5], center, evals, orientation, colorVal, alpha)
    sub_ax[plot_id, 5].tick_params(axis="both", bottom="off", top="off",
      left="off", right="off")
    sub_ax[plot_id, 5].get_xaxis().set_visible(False)
    sub_ax[plot_id, 5].get_yaxis().set_visible(False)
  sub_ax[0,0].set_title("bf", fontsize=12)
  sub_ax[0,1].set_title("envelope", fontsize=12)
  sub_ax[0,2].set_title("filter", fontsize=12)
  sub_ax[0,3].set_title("Fourier map", fontsize=12)
  sub_ax[0,4].set_title("spatial ellipse", fontsize=10)
  sub_ax[0,5].set_title("Fourier ellipse", fontsize=10)
  plt.show()

def plot_hilbert_analysis(weights, padding=None):
  """
  Plot results from performing Hilbert amplitude processing on weight matrix
  Inputs:
    weights: [np.ndarray] with shape [num_inputs, num_outputs]
      num_inputs must have even square root.
  """
  Envelope, bff_filt, Hil_filter, bff = ip.hilbertize(weights, padding)
  num_inputs, num_outputs = weights.shape
  assert np.sqrt(num_inputs) == np.floor(np.sqrt(num_inputs)), (
    "weights.shape[0] must have an even square root.")
  patch_edge_size = int(np.sqrt(num_inputs))
  N = np.int32(np.sqrt(bff_filt.shape[1]))           
  fig, sub_ax = plt.subplots(3, 1, figsize=(64,64))  
  plot_data = pad_data(weights.T.reshape((num_outputs, patch_edge_size,
    patch_edge_size)))
  bf_axis_image = sub_ax[0].imshow(plot_data, cmap="Greys_r",
    interpolation="nearest")
  sub_ax[0].tick_params(axis="both", bottom="off", top="off", left="off",
    right="off")
  sub_ax[0].get_xaxis().set_visible(False)
  sub_ax[0].get_yaxis().set_visible(False)
  sub_ax[0].set_title("Basis Functions", fontsize=32)  
  plot_data = pad_data(np.abs(Envelope).reshape((num_outputs,
    patch_edge_size, patch_edge_size)))
  hil_axis_image = sub_ax[1].imshow(plot_data, cmap="Greys_r",
    interpolation="nearest")
  sub_ax[1].tick_params(axis="both", bottom="off", top="off", left="off",
    right="off")
  sub_ax[1].get_xaxis().set_visible(False)
  sub_ax[1].get_yaxis().set_visible(False)
  sub_ax[1].set_title("Analytic Signal Amplitude Envelope", fontsize=32)
  resh_Zf = np.abs(bff_filt).reshape((num_outputs, N, N))                             
  output_z = np.zeros(resh_Zf.shape)                                              
  for i in range(num_outputs):                                                    
    output_z[i,...] = resh_Zf[i,...] / np.max(resh_Zf[i,...])                     
  plot_data = pad_data(output_z)
  hil_axis_image = sub_ax[2].imshow(plot_data, cmap="Greys_r",
    interpolation="nearest")
  sub_ax[2].tick_params(axis="both", bottom="off", top="off", left="off",
    right="off")
  sub_ax[2].get_xaxis().set_visible(False)
  sub_ax[2].get_yaxis().set_visible(False)
  sub_ax[2].set_title("Fourier Amplitude Spectrum", fontsize=32)
  plt.show()

def plot_cov_matrix(cov_matrix, num_cov_images=""):
  """
  Plot covariance matrix as an image
  Inputs:
    cov_matrix [np.ndarray] covariance matrix
    num_cov_images [str] indicating the number of images that were used for the plot
      this string will be placed in the title
  """
  fig, ax = plt.subplots(1, figsize=(10,10))
  im = ax.imshow(cov_matrix, cmap="Greys_r", interpolation="nearest")
  im.set_clim(vmin=0, vmax=np.max(cov_matrix))
  ax.set_title("Covariance matrix computed over "+str(num_cov_images)+" image patches", fontsize=16)
  add_colorbar(im)
  plt.show()

def plot_eigenvalues(evals, ylim=[0,1000], xlim=None):
  """
  Plot the input eigenvalues
  Inputs:
    evals [np.ndarray] 
    ylim [2-D list] specifying the [min,max] of the y-axis
    xlim [2-D list] specifying the [min,max] of the x-axis
  """
  if xlim is None:
    xlim = [0, evals.shape[0]]
  fig, ax = plt.subplots(1, figsize=(10,10))
  ax.semilogy(evals)
  ax.set_xlim(xlim[0], xlim[1]) # Ignore first eigenvalue
  ax.set_ylim(ylim[0], ylim[1])
  ax.set_yscale("log")
  ax.set_title("Sorted eigenvalues of covariance matrix", fontsize=16)
  plt.show()

def plot_gaussian_contours(bf_stats, num_plots):
  """
  Plot basis functions with contour lines for Gaussian fits 
  Inputs:
    bf_stats [dict] output from ip.get_dictionary_stats()
    num_plots [int] indicating the number of random BFs to plot
  """
  num_bf = bf_stats["num_outputs"]
  bf_range = np.random.choice([i for i in range(num_bf)], num_plots)
  num_plots_y = int(np.ceil(np.sqrt(num_plots)))
  num_plots_x = int(np.floor(np.sqrt(num_plots)))
  fig, sub_ax = plt.subplots(num_plots_y, num_plots_x, figsize=(10,10))
  filter_total = 0
  for plot_id in  np.ndindex((num_plots_y, num_plots_x)):
    if filter_total < num_plots:
      bf_idx = bf_range[filter_total]
      envelope = bf_stats["envelopes"][bf_idx]
      center = bf_stats["envelope_centers"][bf_idx]
      (gauss_fit, grid) = bf_stats["gauss_fits"][bf_idx]
      contour_levels = 3
      sub_ax[plot_id].imshow(envelope, cmap="Greys_r", extent=(0, 16, 16, 0))
      sub_ax[plot_id].contour(grid[1], grid[0], gauss_fit, contour_levels, colors='b')
      sub_ax[plot_id].plot(center[1], center[0], "ro")
      sub_ax[plot_id].set_title("bf:"+str(bf_idx), fontsize=10)
      filter_total += 1
    sub_ax[plot_id].spines["right"].set_color("none")
    sub_ax[plot_id].spines["top"].set_color("none")
    sub_ax[plot_id].spines["left"].set_color("none")
    sub_ax[plot_id].spines["bottom"].set_color("none")
    sub_ax[plot_id].tick_params(axis="both", bottom="off", top="off", left="off", right="off")
    sub_ax[plot_id].get_xaxis().set_visible(False)
    sub_ax[plot_id].get_yaxis().set_visible(False)
    sub_ax[plot_id].set_aspect("equal")
  plt.show()

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
  """
  Add a vertical color bar to an image plot.
  Inputs:
    im: [AxisImage] object returned from matplotlib.plt.imshow()
    aspect: [int] aspect ratio of the colorbar
    pad_fraction: [float] how much space to place between colorbar & plot
    **kwargs: [dict] other keyword arguments that would be passed to im.axes.figure.colorbar()
  """
  divider = axes_grid1.make_axes_locatable(im.axes)
  width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
  pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
  current_ax = plt.gca()
  cax = divider.append_axes("right", size=width, pad=pad)
  plt.sca(current_ax)
  return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def save_bar(data, num_xticks=5, title="", save_filename="./bar_fig.pdf",
  xlabel="", ylabel=""):
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

def save_activity_hist(data, num_bins="auto", title="", save_filename="./hist.pdf"):
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

def save_phase_avg_power_spec(data, title="", save_filename="./pow_spec.pdf"):
  """
  Plot phase averaged power spectrum for a set of images
  Inputs:
    data: [np.ndarray] 1D data to be plotted
    title: [str] for title of figure
    save_filename: [str] holding output directory for writing,
      figures will not display with GUI if set
  """
  (fig, ax) = plt.subplots(1)
  ax.loglog(range(data[data>1].shape[0]), data[data>1])
  fig.suptitle(title, y=1.0, x=0.5)
  fig.savefig(save_filename)
  plt.close(fig)

def save_data_tiled(data, normalize=False, title="", save_filename="",
  vmin=None, vmax=None):
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

def save_stats(data, labels=None, save_filename="./Fig.pdf"):
  """
  Generate time-series plots of stats specified by keys
  Inputs:
    data: [dict] containing data to be plotted. len of all values should be equal
          data must have the key "batch_step"
    labels: [list of str] optional list of labels, should be same len as
            data.keys(). If nothing is given, data.keys() will be used as labels
    save_filename: [str] containing the complete output filename.
  """
  data_keys = list(data.keys())
  data_keys.remove("batch_step")
  if labels is None:
    labels = data_keys
  num_keys = len(data_keys)
  fig, sub_ax = plt.subplots(num_keys)
  axis_image = [None]*num_keys
  for key_idx, key in enumerate(data_keys):
    axis_image[key_idx] = sub_ax[key_idx].plot(data["batch_step"], data[key])
    if key_idx < len(data_keys)-1:
      sub_ax[key_idx].get_xaxis().set_ticklabels([])
    sub_ax[key_idx].locator_params(axis="y", nbins=5)
    sub_ax[key_idx].set_ylabel(labels[key_idx])
    ylabel_xpos = -0.1
    sub_ax[key_idx].yaxis.set_label_coords(ylabel_xpos, 0.5)
  sub_ax[-1].set_xlabel("Batch Number")
  fig.suptitle("Stats per Batch", y=1.0, x=0.5)
  fig.savefig(save_filename, transparent=True)
  plt.close(fig)

def pad_data(data, pad_values=1):
  """
  Pad data with ones for visualization
  Outputs:
    padded version of input
  Inputs:
    data: np.ndarray
    pad_values: [int] specifying what value will be used for padding
  """
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
