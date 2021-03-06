import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from mpl_toolkits import axes_grid1

from DeepSparseCoding.tf1x.utils import data_processing as dp

def plot_ellipse(axis, center, shape, angle, color_val="auto", alpha=1.0, lines=False,
  fill_ellipse=False):
  """
  Add an ellipse to given axis
  Inputs:
    axis [matplotlib.axes._subplots.AxesSubplot] axis on which ellipse should be drawn
    center [tuple or list] specifying [y, x] center coordinates
    shape [tuple or list] specifying [width, height] shape of ellipse
    angle [float] specifying angle of ellipse
    color_val [matplotlib color spec] specifying the color of the edge & face of the ellipse
    alpha [float] specifying the transparency of the ellipse
    lines [bool] if true, output will be a line, where the secondary axis of the ellipse
      is collapsed
    fill_ellipse [bool] if true and lines is false then a filled ellipse will be plotted
  Outputs:
    ellipse [matplotlib.patches.ellipse] ellipse object
  """
  if fill_ellipse:
    face_color_val = "none" if color_val=="auto" else color_val
  else:
    face_color_val = "none"
  y_cen, x_cen = center
  width, height = shape
  if lines:
    min_length = 0.1
    if width < height:
      width = min_length
    elif width > height:
      height = min_length
  ellipse = matplotlib.patches.Ellipse(xy=[x_cen, y_cen], width=width,
    height=height, angle=angle, edgecolor=color_val, facecolor=face_color_val,
    alpha=alpha, fill=True)
  axis.add_artist(ellipse)
  ellipse.set_clip_box(axis.bbox)
  return ellipse

def plot_ellipse_summaries(bf_stats, num_bf=-1, lines=False, rand_bf=False):
  """
  Plot basis functions with summary ellipses drawn over them
  Inputs:
    bf_stats [dict] output of dp.get_dictionary_stats()
    num_bf [int] number of basis functions to plot (<=0 is all; >total is all)
    lines [bool] If true, will plot lines instead of ellipses
    rand_bf [bool] If true, will choose a random set of basis functions
  """
  tot_num_bf = len(bf_stats["basis_functions"])
  if num_bf <= 0 or num_bf > tot_num_bf:
    num_bf = tot_num_bf
  SFs = np.asarray([np.sqrt(fcent[0]**2 + fcent[1]**2)
    for fcent in bf_stats["fourier_centers"]], dtype=np.float32)
  sf_sort_indices = np.argsort(SFs)
  if rand_bf:
    bf_range = np.random.choice([i for i in range(tot_num_bf)], num_bf, replace=False)
  num_plots_y = int(np.ceil(np.sqrt(num_bf)))
  num_plots_x = int(np.ceil(np.sqrt(num_bf)))
  gs = gridspec.GridSpec(num_plots_y, num_plots_x)
  fig = plt.figure(figsize=(17,17))
  filter_idx = 0
  for plot_id in  np.ndindex((num_plots_y, num_plots_x)):
    ax = clear_axis(fig.add_subplot(gs[plot_id]))
    if filter_idx < tot_num_bf and filter_idx < num_bf:
      if rand_bf:
        bf_idx = bf_range[filter_idx]
      else:
        bf_idx = filter_idx
      bf = bf_stats["basis_functions"][bf_idx]
      ax.imshow(bf, interpolation="Nearest", cmap="Greys_r")
      ax.set_title(str(bf_idx), fontsize="8")
      center = bf_stats["gauss_centers"][bf_idx]
      evals, evecs = bf_stats["gauss_orientations"][bf_idx]
      orientations = bf_stats["fourier_centers"][bf_idx]
      angle = np.rad2deg(np.pi/2 + np.arctan2(*orientations))
      alpha = 1.0
      ellipse = plot_ellipse(ax, center, evals, angle, color_val="b", alpha=alpha, lines=lines)
      filter_idx += 1
    ax.set_aspect("equal")
  plt.show()
  return fig

def plot_pooling_summaries(bf_stats, pooling_filters, num_pooling_filters,
                           num_connected_weights=None, lines=False, figsize=None):
  """
  Plot 2nd layer (fully-connected) weights in terms of connection strengths to 1st layer weights
  Inputs:
    bf_stats [dict] output of dp.get_dictionary_stats() which was run on the 1st layer weights
    pooling_filters [np.ndarray] 2nd layer weights
      should be shape [num_1st_layer_neurons, num_2nd_layer_neurons]
    num_pooling_filters [int] How many 2nd layer neurons to plot
    num_connected_weights [int] How many 1st layer weight summaries to include
      for a given 2nd layer neuron
    lines [bool] if True, 1st layer weight summaries will appear as lines instead of ellipses
  """
  num_inputs = bf_stats["num_inputs"]
  num_outputs = bf_stats["num_outputs"]
  tot_pooling_filters = pooling_filters.shape[1]
  patch_edge_size = np.int32(np.sqrt(num_inputs))
  filter_idx_list = np.arange(num_pooling_filters, dtype=np.int32)
  assert num_pooling_filters <= num_outputs, (
    "num_pooling_filters must be less than or equal to bf_stats['num_outputs']")
  if num_connected_weights is None:
    num_connected_weights = num_inputs
  cmap = plt.get_cmap('bwr')
  cNorm = matplotlib.colors.SymLogNorm(linthresh=0.03, linscale=0.01, vmin=-1.0, vmax=1.0)
  scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
  num_plots_y = np.int32(np.ceil(np.sqrt(num_pooling_filters)))
  num_plots_x = np.int32(np.ceil(np.sqrt(num_pooling_filters)))+1 # +cbar col
  gs_widths = [1 for _ in range(num_plots_x-1)]+[0.3]
  gs = gridspec.GridSpec(num_plots_y, num_plots_x, width_ratios=gs_widths)
  if figsize is None:
    fig = plt.figure()
  else:
    fig = plt.figure(figsize=(17,17))
  filter_total = 0
  for plot_id in  np.ndindex((num_plots_y, num_plots_x-1)):
    (y_id, x_id) = plot_id
    ax = fig.add_subplot(gs[plot_id])
    if (filter_total < num_pooling_filters and x_id != num_plots_x-1):
      ax = clear_axis(ax, spines="k")
      filter_idx = filter_idx_list[filter_total]
      example_filter = pooling_filters[:, filter_idx]
      top_indices = np.argsort(np.abs(example_filter))[::-1] #descending
      filter_norm = np.max(np.abs(example_filter))
      SFs = np.asarray([np.sqrt(fcent[0]**2 + fcent[1]**2)
        for fcent in bf_stats["fourier_centers"]], dtype=np.float32)
      # Plot weakest of the top connected filters first because of occlusion
      for bf_idx in top_indices[:num_connected_weights][::-1]:
        connection_strength = example_filter[bf_idx]/filter_norm
        color_val = scalarMap.to_rgba(connection_strength)
        center = bf_stats["gauss_centers"][bf_idx]
        evals, evecs = bf_stats["gauss_orientations"][bf_idx]
        angle = np.rad2deg(np.pi/2 + bf_stats["ellipse_orientations"][bf_idx])
        alpha = 0.5
        ellipse = plot_ellipse(ax, center, evals, angle, color_val, alpha=alpha, lines=lines)
      ax.set_xlim(0, patch_edge_size-1)
      ax.set_ylim(patch_edge_size-1, 0)
      filter_total += 1
    else:
      ax = clear_axis(ax, spines="none")
    ax.set_aspect("equal")
  scalarMap._A = []
  ax = clear_axis(fig.add_subplot(gs[0, -1]))
  cbar = fig.colorbar(scalarMap, ax=ax, ticks=[-1, 0, 1])
  cbar.ax.set_yticklabels(["-1", "0", "1"])
  for label in cbar.ax.yaxis.get_ticklabels():
    label.set_weight("bold")
    label.set_fontsize(14)
  plt.show()
  return fig

def plot_pooling_centers(bf_stats, pooling_filters, num_pooling_filters, num_connected_weights=None,
  filter_indices=None, spot_size=10, figsize=None):
  """
  Plot 2nd layer (fully-connected) weights in terms of spatial/frequency centers of
    1st layer weights
  Inputs:
    bf_stats [dict] Output of dp.get_dictionary_stats() which was run on the 1st layer weights
    pooling_filters [np.ndarray] 2nd layer weights
      should be shape [num_1st_layer_neurons, num_2nd_layer_neurons]
    num_pooling_filters [int] How many 2nd layer neurons to plot
    num_connected_weights [int] How many 1st layer neurons to plot
    spot_size [int] How big to make the points
    filter_indices [list] indices to plot from pooling_filters. len should equal num_pooling_filters
      set to None for default, which is a random selection
    figsize [tuple] Containing the (width, height) of the figure, in inches.
      Set to None for default figure size
  """
  num_filters_y = int(np.ceil(np.sqrt(num_pooling_filters)))
  num_filters_x = int(np.ceil(np.sqrt(num_pooling_filters)))
  tot_pooling_filters = pooling_filters.shape[1]
  if filter_indices is None:
    filter_indices = np.random.choice(tot_pooling_filters, num_pooling_filters, replace=False)
  else:
    assert len(filter_indices) == num_pooling_filters, (
      "len(filter_indices) must equal num_pooling_filters")
  if num_connected_weights is None:
    num_connected_weights = bf_stats["num_inputs"]
  cmap = plt.get_cmap(bgr_colormap())# Could also use "nipy_spectral", "coolwarm", "bwr"
  cNorm = matplotlib.colors.SymLogNorm(linthresh=0.03, linscale=0.01, vmin=-1.0, vmax=1.0)
  scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
  x_p_cent = [x for (y,x) in bf_stats["gauss_centers"]]# Get raw points
  y_p_cent = [y for (y,x) in bf_stats["gauss_centers"]]
  x_f_cent = [x for (y,x) in bf_stats["fourier_centers"]]
  y_f_cent = [y for (y,x) in bf_stats["fourier_centers"]]
  max_sf = np.max(np.abs(x_f_cent+y_f_cent))
  pair_w_gap = 0.01
  group_w_gap = 0.03
  h_gap = 0.03
  plt_w = (num_filters_x/num_pooling_filters)
  plt_h = plt_w
  if figsize is None:
    fig = plt.figure()
  else:
    fig = plt.figure(figsize=figsize) #figsize is (w,h)
  axes = []
  filter_id = 0
  for plot_id in np.ndindex((num_filters_y, num_filters_x)):
    if all(pid == 0 for pid in plot_id):
      axes.append(clear_axis(fig.add_axes([0, plt_h+h_gap, 2*plt_w, plt_h])))
      scalarMap._A = []
      cbar = fig.colorbar(scalarMap, ax=axes[-1], ticks=[-1, 0, 1], aspect=10, location="bottom")
      cbar.ax.set_xticklabels(["-1", "0", "1"])
      cbar.ax.xaxis.set_ticks_position("top")
      cbar.ax.xaxis.set_label_position("top")
      for label in cbar.ax.xaxis.get_ticklabels():
        label.set_weight("bold")
        label.set_fontsize(10+figsize[0])
    if (filter_id < num_pooling_filters):
      example_filter = pooling_filters[:, filter_indices[filter_id]]
      top_indices = np.argsort(np.abs(example_filter))[::-1] #descending
      selected_indices = top_indices[:num_connected_weights][::-1] #select top, plot weakest first
      filter_norm = np.max(np.abs(example_filter))
      connection_colors = [scalarMap.to_rgba(example_filter[bf_idx]/filter_norm)
        for bf_idx in range(bf_stats["num_outputs"])]
      if num_connected_weights < top_indices.size:
        black_indices = top_indices[num_connected_weights:][::-1]
        xp = [x_p_cent[i] for i in black_indices]+[x_p_cent[i] for i in selected_indices]
        yp = [y_p_cent[i] for i in black_indices]+[y_p_cent[i] for i in selected_indices]
        xf = [x_f_cent[i] for i in black_indices]+[x_f_cent[i] for i in selected_indices]
        yf = [y_f_cent[i] for i in black_indices]+[y_f_cent[i] for i in selected_indices]
        c = [(0.1,0.1,0.1,1.0) for i in black_indices]+[connection_colors[i] for i in selected_indices]
      else:
        xp = [x_p_cent[i] for i in selected_indices]
        yp = [y_p_cent[i] for i in selected_indices]
        xf = [x_f_cent[i] for i in selected_indices]
        yf = [y_f_cent[i] for i in selected_indices]
        c = [connection_colors[i] for i in selected_indices]
      (y_id, x_id) = plot_id
      if x_id == 0:
        ax_l = 0
        ax_b = - y_id * (plt_h+h_gap)
      else:
        bbox = axes[-1].get_position().get_points()[0]#bbox is [[x0,y0],[x1,y1]]
        prev_l = bbox[0]
        prev_b = bbox[1]
        ax_l = prev_l + plt_w + group_w_gap
        ax_b = prev_b
      ax_w = plt_w
      ax_h = plt_h
      axes.append(clear_axis(fig.add_axes([ax_l, ax_b, ax_w, ax_h])))
      axes[-1].invert_yaxis()
      axes[-1].scatter(xp, yp, c=c, s=spot_size, alpha=0.8)
      axes[-1].set_xlim(0, bf_stats["patch_edge_size"]-1)
      axes[-1].set_ylim(bf_stats["patch_edge_size"]-1, 0)
      axes[-1].set_aspect("equal")
      axes[-1].set_facecolor("w")
      axes.append(clear_axis(fig.add_axes([ax_l+ax_w+pair_w_gap, ax_b, ax_w, ax_h])))
      axes[-1].scatter(xf, yf, c=c, s=spot_size, alpha=0.8)
      axes[-1].set_xlim([-max_sf, max_sf])
      axes[-1].set_ylim([-max_sf, max_sf])
      axes[-1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
      axes[-1].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
      axes[-1].set_aspect("equal")
      axes[-1].set_facecolor("w")
      #histogram - note: axis widths/heights are not setup for a third plot
      #axes.append(fig.add_axes([ax_l+ax_w+pair_w_gap, ax_b, ax_w, ax_h]))
      #axes[-1].set_yticklabels([])
      #axes[-1].tick_params(axis="y", bottom="off", top="off", left="off", right="off")
      #axes[-1].hist([example_filter[bf_idx]/filter_norm for bf_idx in range(bf_stats["num_outputs"])])
      filter_id += 1
  plt.show()
  return fig

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
  gs = gridspec.GridSpec(num_bases, num_top_cov_bases+2, hspace=0.6)
  for x_id in range(num_bases):
    primary_bf_idx = bf_indices[x_id]
    sorted_cov_indices = np.argsort(a_cov[primary_bf_idx, :])[-2::-1]
    primary_bf = np.squeeze(dp.reshape_data(weights.T[primary_bf_idx,...], flatten=False)[0])
    ax = plt.subplot(gs[x_id,0])
    ax.imshow(primary_bf, cmap="Greys_r", interpolation="nearest")
    ax.tick_params(axis="both", bottom="off", top="off",
      left="off", right="off")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    [i.set_linewidth(3.0) for i in ax.spines.values()]
    strengths = []
    for y_id, bf_idx in enumerate(sorted_cov_indices[:num_top_cov_bases]):
      bf = np.squeeze(dp.reshape_data(weights.T[bf_idx,...],
        flatten=False)[0])
      ax = plt.subplot(gs[x_id, y_id+1])
      ax.imshow(bf, cmap="Greys_r", interpolation="nearest")
      ax.tick_params(axis="both", bottom="off", top="off", left="off", right="off")
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
  plt.subplot(gs[0,0]).set_title("rand bf", horizontalalignment="center", fontsize=18);
  plt.subplot(gs[0,1]).set_title("stronger correlation --$>$ weaker correlation",
    horizontalalignment="left", fontsize=18);
  plt.subplot(gs[0,-1]).set_title("activity covariance", horizontalalignment="center", fontsize=18)
  plt.show()
  return fig

def plot_bf_stats(bf_stats, num_bf=2):
  """
  Plot outputs of the dp.get_dictionary_stats()
  Inputs:
    bf_stats [dict] output of dp.get_dictionary_stats()
    num_bf [int] number of basis functions to plot
  """
  tot_num_bf = len(bf_stats["basis_functions"])
  bf_idx_list = np.random.choice(tot_num_bf, num_bf, replace=False)
  fig, sub_ax = plt.subplots(num_bf, 5, figsize=(15,15))
  for plot_id in range(int(num_bf)):
    bf_idx = bf_idx_list[plot_id]
    # Basis function in pixel space
    bf = bf_stats["basis_functions"][bf_idx]
    sub_ax[plot_id, 0].imshow(bf, cmap="Greys_r", interpolation="Nearest")
    sub_ax[plot_id, 0] = clear_axis(sub_ax[plot_id, 0], spines="k")
    sub_ax[plot_id, 0].set_title(str(bf_idx), fontsize="8")
    # Hilbert envelope
    env = bf_stats["envelopes"][bf_idx]
    sub_ax[plot_id, 1].imshow(env, cmap="Greys_r", interpolation="Nearest")
    sub_ax[plot_id, 1] = clear_axis(sub_ax[plot_id, 1], spines="k")
    # Fourier transform of basis function
    fourier = bf_stats["fourier_maps"][bf_idx]
    sub_ax[plot_id, 2].imshow(fourier, cmap="Greys_r", interpolation="Nearest")
    sub_ax[plot_id, 2] = clear_axis(sub_ax[plot_id, 2], spines="k")
    sub_ax[plot_id, 2].spines["left"].set_position("center")
    sub_ax[plot_id, 2].spines["left"].set_linewidth(2.5)
    sub_ax[plot_id, 2].spines["bottom"].set_position("center")
    sub_ax[plot_id, 2].spines["bottom"].set_linewidth(2.5)
    sub_ax[plot_id, 2].spines["top"].set_color("none")
    sub_ax[plot_id, 2].spines["right"].set_color("none")
    sub_ax[plot_id, 2].set_ylim([fourier.shape[0]-1, 0])
    sub_ax[plot_id, 2].set_xlim([0, fourier.shape[1]-1])
    # Fourier summary stats
    sub_ax[plot_id, 3].imshow(bf, interpolation="Nearest", cmap="Greys_r")
    center = bf_stats["gauss_centers"][bf_idx]
    evals, evecs = bf_stats["gauss_orientations"][bf_idx]
    orientation = bf_stats["fourier_centers"][bf_idx]
    angle = np.rad2deg(np.pi/2 + np.arctan2(*orientation))
    alpha = 1.0
    color_val = "b"
    ellipse = plot_ellipse(sub_ax[plot_id, 3], center, evals, angle, color_val, alpha)
    sub_ax[plot_id, 3] = clear_axis(sub_ax[plot_id, 3], spines="k")
    sub_ax[plot_id, 4].imshow(bf, interpolation="Nearest", cmap="Greys_r")
    sub_ax[plot_id, 4] = clear_axis(sub_ax[plot_id, 4], spines="k")
    ellipse = plot_ellipse(sub_ax[plot_id, 4], center, evals, angle, color_val, alpha, lines=True)
  sub_ax[0,0].set_title("Basis function", fontsize=12)
  sub_ax[0,1].set_title("Envelope", fontsize=12)
  sub_ax[0,2].set_title("Fourier map", fontsize=12)
  sub_ax[0,3].set_title("Summary ellipse", fontsize=12)
  sub_ax[0,4].set_title("Summary line", fontsize=12)
  plt.tight_layout()
  plt.show()
  return fig

def plot_loc_freq_summary(bf_stats, spotsize=10, figsize=(15, 5), fontsize=16):
  plt.rc('text', usetex=True)
  fig = plt.figure(figsize=figsize)
  gs = fig.add_gridspec(1, 3, wspace=0.3)
  ax = fig.add_subplot(gs[0])
  x_pos = [x for (y,x) in bf_stats["gauss_centers"]]
  y_pos = [y for (y,x) in bf_stats["gauss_centers"]]
  ax.scatter(x_pos, y_pos, color='k', s=spotsize)
  ax.set_xlim([0, bf_stats["patch_edge_size"]-1])
  ax.set_ylim([bf_stats["patch_edge_size"]-1, 0])
  ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
  ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
  ax.set_aspect("equal")
  ax.set_ylabel("Pixels", fontsize=fontsize)
  ax.set_xlabel("Pixels", fontsize=fontsize)
  ax.set_title("Centers", fontsize=fontsize, pad=32)
  ax = fig.add_subplot(gs[1])
  x_sf = [x for (y,x) in bf_stats["fourier_centers"]]
  y_sf = [y for (y,x) in bf_stats["fourier_centers"]]
  max_sf = np.max(np.abs(x_sf+y_sf))
  ax.scatter(x_sf, y_sf, color='k', s=spotsize)
  ax.set_xlim([-max_sf, max_sf])
  ax.set_ylim([-max_sf, max_sf])
  ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
  ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
  ax.set_aspect("equal")
  ax.set_ylabel("Cycles / Patch", fontsize=fontsize)
  ax.set_xlabel("Cycles / Patch", fontsize=fontsize)
  ax.set_title("Spatial Frequencies", fontsize=fontsize, pad=32)
  num_bins = 360
  orientations = [np.pi + orientation
    for orientation in [np.arctan2(*fyx[::-1]) for fyx in bf_stats["fourier_centers"]]]
  bins = np.linspace(0, 2*np.pi, num_bins)
  count, bin_edges = np.histogram(orientations, bins)
  count = count / np.max(count)
  bin_left, bin_right = bin_edges[:-1], bin_edges[1:]
  bin_centers = bin_left + (bin_right - bin_left)/2
  ax = fig.add_subplot(gs[2], polar=True)
  ax.plot(bin_centers, count, linewidth=3, color='k')
  ax.set_yticks([])
  ax.set_thetamin(0)
  ax.set_thetamax(2*np.pi)
  ax.set_xticks([0, np.pi/4, 2*np.pi/4, 3*np.pi/4, 4*np.pi/4,
    5*np.pi/4, 6*np.pi/4, 7*np.pi/4, 2*np.pi])
  ax.set_xticklabels([r"0", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$",
    r"$\frac{3\pi}{4}$", r"$\pi$", r"$\frac{5\pi}{4}$", r"$\frac{3\pi}{2}$",
    r"$\frac{7\pi}{4}$"], fontsize=fontsize)
  ax.set_title("Orientaitons", fontsize=fontsize, pad=23)
  plt.show()
  return fig

def plot_hilbert_analysis(weights, padding=None):
  """
  Plot results from performing Hilbert amplitude processing on weight matrix
  Inputs:
    weights: [np.ndarray] with shape [num_inputs, num_outputs]
      num_inputs must have even square root.
  """
  Envelope, bff_filt, Hil_filter, bff = dp.hilbert_amplitude(weights, padding)
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
  sub_ax[0].set_title("Basis Functions", fontsize=20)
  plot_data = pad_data(np.abs(Envelope).reshape((num_outputs,
    patch_edge_size, patch_edge_size)))
  hil_axis_image = sub_ax[1].imshow(plot_data, cmap="Greys_r",
    interpolation="nearest")
  sub_ax[1].tick_params(axis="both", bottom="off", top="off", left="off",
    right="off")
  sub_ax[1].get_xaxis().set_visible(False)
  sub_ax[1].get_yaxis().set_visible(False)
  sub_ax[1].set_title("Analytic Signal Amplitude Envelope", fontsize=20)
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
  sub_ax[2].set_title("Fourier Amplitude Spectrum", fontsize=20)
  plt.show()
  return fig

def plot_image(image, vmin=None, vmax=None, title="", save_filename=None):
  """
  Plot single image
  Inputs:
    image [np.ndarray] 2-D image
    title [str] indicating the title for the figure
  """
  if vmin is None:
    vmin = np.min(image)
  if vmax is None:
    vmax = np.max(image)
  fig, ax = plt.subplots(1, figsize=(10,10))
  ax = clear_axis(ax)
  im = ax.imshow(image, cmap="Greys_r", vmin=vmin, vmax=vmax, interpolation="nearest")
  ax.set_title(title, fontsize=20)
  if save_filename is not None:
      fig.savefig(save_filename)
      plt.close(fig)
      return None
  plt.show()
  return fig

def plot_matrix(matrix, title="", save_filename=None):
  """
  Plot covariance matrix as an image
  Inputs:
    matrix [np.ndarray] covariance matrix
    title [str] indicating the title for the figure
  """
  fig, ax = plt.subplots(1, figsize=(10,10))
  im = ax.imshow(matrix, cmap="Greys_r", interpolation="nearest")
  im.set_clim(vmin=np.min(matrix), vmax=np.max(matrix))
  ax.set_title(title, fontsize=20)
  add_colorbar_to_im(im)
  if save_filename is not None:
      fig.savefig(save_filename)
      plt.close(fig)
      return None
  plt.show()
  return fig

def plot_eigenvalues(evals, ylim=None, xlim=None):
  """
  Plot the input eigenvalues
  Inputs:
    evals [np.ndarray]
    ylim [2-D list] specifying the [min,max] of the y-axis
    xlim [2-D list] specifying the [min,max] of the x-axis
  """
  if xlim is None:
    xlim = [0, evals.shape[0]]
  if ylim is None:
    ylim = [np.min(evals), np.max(evals)]
  fig, ax = plt.subplots(1, figsize=(10,10))
  ax.semilogy(evals)
  ax.set_xlim(xlim[0], xlim[1]) # Ignore first eigenvalue
  ax.set_ylim(ylim[0], ylim[1])
  ax.set_yscale("log")
  ax.set_title("Sorted eigenvalues of covariance matrix", fontsize=20)
  plt.show()
  return fig

def plot_gaussian_contours(bf_stats, num_plots):
  """
  Plot basis functions with contour lines for Gaussian fits
  Inputs:
    bf_stats [dict] output from dp.get_dictionary_stats()
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
  return fig

def plot_bar(data, num_xticks=5, title="", xlabel="", ylabel="", save_filename=None):
  """
  Generate a bar graph of data
  Inputs:
    data: [np.ndarray] of shape (N,)
    xticklabels: [list of N str] indicating the labels for the xticks
    save_filename: [str] indicating where the file should be saved
      if None, don't save the file
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
  if save_filename is not None:
    fig.savefig(save_filename, transparent=True)
    plt.close(fig)
    return None
  plt.show()
  return fig

def plot_contrast_orientation_tuning(bf_indices, contrasts, orientations, activations, figsize=(32,32)):
  """
  Generate contrast orientation tuning curves. Every subplot will have curves for each contrast.
  Inputs:
    bf_indices: [list or array] of neuron indices to use
      all indices should be less than activations.shape[0]
    contrasts: [list or array] of contrasts to use
    orientations: [list or array] of orientations to use
  """
  orientations = np.asarray(orientations)*(180/np.pi) #convert to degrees for plotting
  num_bfs = np.asarray(bf_indices).size
  cmap = plt.get_cmap('Greys')
  cNorm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
  scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
  fig = plt.figure(figsize=figsize)
  num_plots_y = np.int32(np.ceil(np.sqrt(num_bfs)))+1
  num_plots_x = np.int32(np.ceil(np.sqrt(num_bfs)))
  gs_widths = [1.0,]*num_plots_x
  gs_heights = [1.0,]*num_plots_y
  gs = gridspec.GridSpec(num_plots_y, num_plots_x, wspace=0.5, hspace=0.7,
    width_ratios=gs_widths, height_ratios=gs_heights)
  bf_idx = 0
  for plot_id in np.ndindex((num_plots_y, num_plots_x)):
    (y_id, x_id) = plot_id
    if y_id == 0 and x_id == 0:
      ax = fig.add_subplot(gs[plot_id])
      #ax.set_ylabel("Activation", fontsize=16)
      #ax.set_xlabel("Orientation", fontsize=16)
      ax00 = ax
    else:
      ax = fig.add_subplot(gs[plot_id])#, sharey=ax00)
    if bf_idx < num_bfs:
      for co_idx, contrast in enumerate(contrasts):
        co_idx = -1
        contrast = contrasts[co_idx]
        activity = activations[bf_indices[bf_idx], co_idx, :]
        color_val = scalarMap.to_rgba(contrast)
        ax.plot(orientations, activity, linewidth=1, color=color_val)
        ax.scatter(orientations, activity, s=4, c=[color_val])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2g'))
        ax.set_yticks([0, np.max(activity)])
        ax.set_xticks([0, 90, 180])
      bf_idx += 1
    else:
      ax = clear_axis(ax, spines="none")
  plt.show()
  return fig

def plot_masked_orientation_tuning(bf_indices, mask_orientations, base_responses, test_responses):
  """
  Generate orientation tuning curves for superimposed masks.
  Maximum contrast (index -1) will be selected for the base and mask
  Inputs:
    bf_indices: [list or array] of neuron indices to use
      all indices should be less than base_responsees.shape[0] and test_responses.shape[0]
    mask_orientations: [list or array] of mask orientation values
    base_responses: [list or array] of responses to base stimulus at optimal orientation
      should be shape [num_neurons, num_base_contrasts, num_mask_contrasts, num_orientations]
    test_responses: [list or array] of responses to the base+mask stimulus
      should be shape [num_neurons, num_base_contrasts, num_mask_contrasts, num_orientations]
  """
  mask_orientations = np.asarray(mask_orientations) * (180/np.pi)
  num_bfs = np.asarray(bf_indices).size
  num_orientations = mask_orientations.size
  cmap = plt.get_cmap('Greys')
  cNorm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
  scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
  fig = plt.figure(figsize=(32,32))
  num_plots_y = np.int32(np.ceil(np.sqrt(num_bfs)))+1
  num_plots_x = np.int32(np.ceil(np.sqrt(num_bfs)))
  gs_widths = [1.0,]*num_plots_x
  gs_heights = [1.0,]*num_plots_y
  gs = gridspec.GridSpec(num_plots_y, num_plots_x, wspace=0.5, hspace=0.7,
    width_ratios=gs_widths, height_ratios=gs_heights)
  bf_idx = 0
  for plot_id in np.ndindex((num_plots_y, num_plots_x)):
    (y_id, x_id) = plot_id
    if y_id == 0 and x_id == 0:
      ax = fig.add_subplot(gs[plot_id])
      #ax.set_ylabel("Normalized Activation", fontsize=16)
      #ax.set_xlabel("Mask Orientation", fontsize=16)
      #ax.set_ylim([0.0, np.max(co_test_mean_responses)])
      ax00 = ax
    else:
      ax = fig.add_subplot(gs[plot_id])#, sharey=ax00)
    if bf_idx < num_bfs:
      bco_idx = -1; co_idx = -1 # we want highest contrasts used for this experiment
      base_activity = base_responses[bf_indices[bf_idx], bco_idx]
      test_activity  = test_responses[bf_indices[bf_idx], bco_idx, co_idx, :]
      color_val = scalarMap.to_rgba(1.0) # One could alternatively set this to the contrast value
      ax.plot(mask_orientations, [base_activity,]*num_orientations, linestyle="--",
        linewidth=1, color=color_val)
      ax.plot(mask_orientations, test_activity, linestyle="-", linewidth=1, color=color_val)
      ax.scatter(mask_orientations, test_activity, s=4, c=color_val)
      ax.set_yticks([0, np.max(test_activity)])
      ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2g'))
      ax.set_xticks([0, 90, 180])
      bf_idx += 1
    else:
      ax = clear_axis(ax, spines="none")
  plt.show()
  return fig

def plot_plaid_contrast_tuning(bf_indices, base_contrasts, mask_contrasts, base_orientations,
  mask_orientations, test_responses):
  """
  Plot responses to orthogonal plaid stimulus at different base and mask contrasts
  Inputs:
    bf_indices: [list or array] of neuron indices to use
      all indices should be less than test_responsees.shape[0]
    base_contrasts: [list or array] of base contrasts.
    mask_contrasts: [list or array] of mask contrasts.
      each plot will have one line per mask_contrast
    base_orientations: [list or array] of optimal base orientations for all neurons
      should be a 1-D array with size = test_responses.shape[0]
    mask_orientations: [list or array] of mask orientation values
      function will compute the plaid response for orthogonal orientations
    test_responses: [list or array] of responses to the base+mask stimulus
      should be shape [num_neurons, num_base_contrasts, num_mask_contrasts, num_orientations]
  """
  bf_indices = np.asarray(bf_indices)
  mask_orientations = np.asarray(mask_orientations)
  mask_contrasts = np.asarray(mask_contrasts)
  num_bfs = bf_indices.size
  num_orientations = mask_orientations.size
  num_contrasts = mask_contrasts.size
  # index of value in mask_orientations that is closest to orthogonal to base_orientations[bf_idx]
  orthogonal_orientations = [base_orientations[bf_indices[bf_idx]]-(np.pi/2)
    for bf_idx in range(num_bfs)]
  orthogonal_orientations = np.asarray([val + np.pi if val < 0 else val
    for val in orthogonal_orientations])
  mask_or_idx = [np.argmin(orthogonal_orientations[bf_idx] - mask_orientations)
    for bf_idx in range(num_bfs)]
  cmap = plt.get_cmap('Greys')
  cNorm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
  scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
  num_plots_y = np.int32(np.ceil(np.sqrt(num_bfs)))+1
  num_plots_x = np.int32(np.ceil(np.sqrt(num_bfs)))
  gs_widths = [1.0,]*num_plots_x
  gs_heights = [1.0,]*num_plots_y
  gs = gridspec.GridSpec(num_plots_y, num_plots_x, wspace=0.5, hspace=0.7,
    width_ratios=gs_widths, height_ratios=gs_heights)
  fig = plt.figure(figsize=(32,32)) #TODO: Adjust fig size according to num plots
  bf_idx = 0
  for plot_id in np.ndindex((num_plots_y, num_plots_x)):
    (y_id, x_id) = plot_id
    if y_id == 0 and x_id == 0:
      ax = fig.add_subplot(gs[plot_id])
      #ax.set_ylabel("Normalized Activation", fontsize=16)
      #ax.set_xlabel("Base Contrast", fontsize=16)
      #ax.set_ylim([0.0, 1.0])
      ax00 = ax
    else:
      ax = fig.add_subplot(gs[plot_id], sharey=ax00)
    if bf_idx < num_bfs:
      for co_idx, mask_contrast in enumerate(mask_contrasts):
        # vary base contrast for fixed mask contrast & orthogonal mask
        activity  = test_responses[bf_indices[bf_idx], :, co_idx, mask_or_idx[bf_idx]]
        color_val = scalarMap.to_rgba(mask_contrast)
        ax.plot(base_contrasts, activity, linestyle="-", color=color_val)
        ax.scatter(base_contrasts, activity, s=4, c=color_val, label=str(mask_contrast))
      ax.set_xticks([base_contrasts[0], base_contrasts[-1]])
      bf_idx += 1
    else:
      ax = clear_axis(ax, spines="none")
  plt.show()
  return fig

def plot_activity_hist(data, num_bins="auto", title="", save_filename=None):
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
  if np.min(data) != np.max(data):
    ax.set_xlim([np.min(data), np.max(data)])
  ax.set_xlabel('Value')
  ax.set_ylabel('Count')
  fig.suptitle(title, y=1.0, x=0.5)
  fig.tight_layout()
  if save_filename is not None:
    fig.savefig(save_filename)
    plt.close(fig)
    return None
  plt.show()
  return fig

def plot_phase_avg_power_spec(data, title="", save_filename=None):
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
  if save_filename is not None:
      fig.savefig(save_filename)
      plt.close(fig)
      return None
  plt.show()
  return fig

def plot_group_weights(weights, group_ids, title="", figsize=None,  save_filename=None):
  """
    weights: [np.ndarray] of shape [num_neurons, num_input_y, num_input_x]
    group_ids: [list of lists] containing ids for each group [[,]*neurons_per_group,]*num_groups
  """
  num_neurons = weights.shape[0]
  for weight_id in range(num_neurons):
    weights[weight_id,...] = weights[weight_id,...] - weights[weight_id,...].mean()
    weights[weight_id,...] = weights[weight_id,...] / (weights[weight_id,...].max()-weights[weight_id,...].min())
  vmin = np.min(weights)
  vmax = np.max(weights)
  indices = [idx for id_list in group_ids for idx in id_list]
  num_groups = len(group_ids)
  num_groups_x = int(np.floor(np.sqrt(num_groups)))
  num_groups_y = int(np.ceil(np.sqrt(num_groups)))
  num_neurons_per_group = len(group_ids[0])
  num_neurons_x = int(np.floor(np.sqrt(num_neurons_per_group)))
  num_neurons_y = int(np.ceil(np.sqrt(num_neurons_per_group)))
  outer_spacing = 0.20
  inner_spacing = 0.1
  fig = plt.figure(figsize=figsize)
  gs1 = gridspec.GridSpec(num_groups_y, num_groups_x,
    hspace=outer_spacing*num_groups_y/(num_groups_x+num_groups_y),
    wspace=outer_spacing*num_groups_x/(num_groups_x+num_groups_y))
  neuron_index = 0
  for group_plot_id in np.ndindex((num_groups_y, num_groups_x)):
    gs_inner = gridspec.GridSpecFromSubplotSpec(num_neurons_y, num_neurons_x, gs1[group_plot_id],
      hspace=inner_spacing*num_neurons_y/(num_neurons_x+num_neurons_y),
      wspace=inner_spacing*num_neurons_x/(num_neurons_x+num_neurons_y))
    for inner_plot_id in np.ndindex((num_neurons_y, num_neurons_x)):
      ax = clear_axis(fig.add_subplot(gs_inner[inner_plot_id]))
      ax.set_aspect("equal")
      if neuron_index < num_neurons:
        ax.imshow(weights[indices[neuron_index], ...], cmap="Greys_r", vmin=vmin, vmax=vmax)
        neuron_index += 1
  fig.suptitle(title, y=0.9, x=0.5, fontsize=20)
  if save_filename is not None:
    fig.savefig(save_filename)
    plt.close(fig)
    return None
  plt.show()
  return fig

def plot_weights(weights, title="", figsize=None, save_filename=None):
  """
    weights: [np.ndarray] of shape [num_outputs, num_input_y, num_input_x]
    The matrices are renormalized before plotting.
  """
  weights = dp.norm_weights(weights)
  vmin = np.min(weights)
  vmax = np.max(weights)
  num_plots = weights.shape[0]
  num_plots_y = int(np.ceil(np.sqrt(num_plots))+1)
  num_plots_x = int(np.floor(np.sqrt(num_plots)))
  fig, sub_ax = plt.subplots(num_plots_y, num_plots_x, figsize=figsize)
  filter_total = 0
  for plot_id in  np.ndindex((num_plots_y, num_plots_x)):
    if filter_total < num_plots:
      sub_ax[plot_id].imshow(np.squeeze(weights[filter_total, ...]), vmin=vmin, vmax=vmax, cmap="Greys_r")
      filter_total += 1
    clear_axis(sub_ax[plot_id])
    sub_ax[plot_id].set_aspect("equal")
  fig.suptitle(title, y=0.95, x=0.5, fontsize=20)
  if save_filename is not None:
      fig.savefig(save_filename)
      plt.close(fig)
      return None
  plt.show()
  return fig

def plot_data_tiled(data, normalize=False, title="", vmin=None, vmax=None, cmap="Greys_r",
  save_filename=None):
  """
  Save figure for input data as a tiled image
  Inpus:
    data: [np.ndarray] of shape:
      (height, width, features) - single image
      (n, height, width, features) - n images
    normalize: [bool] indicating whether the data should be streched (normalized)
      This is recommended for dictionary plotting.
    title: [str] for title of figure
    vmin, vmax: [int] the min and max of the color range
    cmap: [str] indicating cmap, or None for imshow default
    save_filename: [str] holding output directory for writing,
      figures will not display with GUI if set
  """
  data = dp.reshape_data(data, flatten=False)[0]
  if normalize:
    data = dp.normalize_data_with_max(data)[0]
    vmin = -1.0
    vmax = 1.0
  if vmin is None:
    vmin = np.min(data)
  if vmax is None:
    vmax = np.max(data)
  if data.ndim == 3:
    data = np.squeeze(data)
  elif data.ndim == 4:
    data = np.squeeze(pad_data(data))
    #If rgb, need to rescale from 0 .. 1
    if(data.shape[-1] == 3):
      data = (data - data.min())/(data.max() - data.min())
  else:
    assert False, ("input data must have ndim==3 or 4")
  fig, sub_axis = plt.subplots(1, figsize=(24, 24))
  axis_image = sub_axis.imshow(np.squeeze(data), cmap=cmap, interpolation="nearest")
  axis_image.set_clim(vmin=vmin, vmax=vmax)
  if data.shape[-1] == 1:
    cbar = add_colorbar_to_im(axis_image)
  sub_axis = clear_axis(sub_axis, spines="k")
  sub_axis.set_title(title, fontsize=20)
  if save_filename is not None:
    if save_filename == "":
      save_filename = "./output.png"
    fig.savefig(save_filename, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return None
  plt.show()
  return fig

def plot_stats(data, keys=None, labels=None, start_index=0, figsize=None, save_filename=None):
  """
  Generate time-series plots of stats specified by keys
  Inputs:
    data: [dict] containing data to be plotted. len of all values should be equal
      data must have the key "batch_step"
    keys: [list of str] optional list of keys to plot, each should exist in data.keys()
      If nothing is given, data.keys() will be used
    labels: [list of str] optional list of labels, should be the same length as keys input
      If nothing is given, data.keys() will be used
    save_filename: [str] containing the complete output filename.
  """
  if keys is None:
    keys = list(data.keys())
  else:
    assert all([key in data.keys() for key in keys]), (
      "All input keys must exist as keys in the data dictionary")
  assert len(keys) > 0, "Keys must be None or have length > 0."
  if "batch_step" in keys:
    keys.remove("batch_step")
  if "schedule_index" in keys:
    keys.remove("schedule_index")
  if "global_batch_index" in keys:
    keys.remove("global_batch_index")
  if labels is None:
    labels = keys
  else:
    assert len(labels) == len(keys), (
      "The number of labels must match the number of keys")
  num_keys = len(keys)
  gs = gridspec.GridSpec(num_keys, 1, hspace=0.5)
  fig = plt.figure(figsize=figsize)
  axis_image = [None]*num_keys
  for key_idx, key in enumerate(keys):
    x_dat = data["batch_step"][start_index:]
    y_dat = data[key][start_index:]
    ax = fig.add_subplot(gs[key_idx])
    axis_image[key_idx] = ax.plot(x_dat, y_dat)
    if key_idx < len(keys)-1:
      ax.get_xaxis().set_ticklabels([])
    ax.locator_params(axis="y", nbins=5)
    ax.set_ylabel("\n".join(re.split("_", labels[key_idx])))
    ax.set_yticks([np.minimum(0.0, np.min(y_dat)), np.maximum(0.0, np.max(y_dat))])
    ylabel_xpos = -0.15
    ax.yaxis.set_label_coords(ylabel_xpos, 0.5)
  ax.set_xlabel("Batch Number")
  fig.suptitle("Stats per Batch", y=1.0, x=0.5)
  if save_filename is not None:
      fig.savefig(save_filename, transparent=True)
      plt.close(fig)
      return None
  plt.show()
  return fig

def plot_inference_stats(data, title="", save_filename=None):
  """
  Plot loss values during LCA inference
  Inputs:
    data: [dict] that must contain the "losses"
      this can be created by using the LCA analyzer objects
  TODO: Add a 4th plot that shows the % change in active coefficients (inactive-to-active + active-to-inactive)
        e.g. in bottom left of figure 7 in rozell et al 2008 LCA paper
  """
  labels = [key.title() for key in data["losses"].keys()]
  losses = [val for val in data["losses"].values()]
  num_im, num_steps = losses[0].shape
  means = [None,]*len(labels)
  sems = [None,]*len(labels)
  for loss_id, loss in enumerate(losses):
    means[loss_id] = np.mean(loss, axis=0) # mean across num_imgs
    sems[loss_id] = np.std(loss, axis=0) / np.sqrt(num_im)
  num_plots_y = np.int32(np.ceil(np.sqrt(len(labels))))+1
  num_plots_x = np.int32(np.ceil(np.sqrt(len(labels))))
  gs = gridspec.GridSpec(num_plots_y, num_plots_x)
  fig = plt.figure(figsize=(12,12))
  loss_id = 0
  for plot_id in np.ndindex((num_plots_y, num_plots_x)):
    (y_id, x_id) = plot_id
    ax = fig.add_subplot(gs[plot_id])
    if loss_id < len(labels):
      time_steps = np.arange(num_steps)
      ax.plot(time_steps, means[loss_id], "k-")
      ax.fill_between(time_steps, means[loss_id]-sems[loss_id],
        means[loss_id]+sems[loss_id], alpha=0.2)
      ax.set_ylabel(labels[loss_id].replace('_', ' '), fontsize=16)
      ax.set_xlim([1, np.max(time_steps)])
      ax.set_xticks([1, int(np.floor(np.max(time_steps)/2)), np.max(time_steps)])
      ax.set_xlabel("Time Step", fontsize=16)
      ax.tick_params("both", labelsize=14)
      loss_id += 1
    else:
      ax = clear_axis(ax, spines="none")
  fig.tight_layout()
  fig.suptitle(title, y=1.03, x=0.5, fontsize=20)
  if save_filename is not None:
    fig.savefig(save_filename, transparent=True, bbox_inches="tight", pad=0.1)
    plt.close(fig)
    return None
  plt.show()
  return fig

def plot_inference_traces(data, activation_threshold, img_idx=None, act_indicator_threshold=None):
  """
  Plot of model neurons' inputs over time
  Args:
    data: [dict] with each trace, with keys [b, u, a, ga, images]
      Dictionary is created by analyze_lca.evaluate_inference()
    img_idx: [int] which image in data["images"] to run analysis on
    act_indicator_threshold: [float] sets the threshold for when a neuron is marked as "recently active"
      Recently active neurons are those that became active towards the end of the inference process
      Recency is computed as any time step that is greater than num_inference_steps * act_indicator_threshold
      Recently active neurons are indicated by a dotted magenta border
      This input must be between 0.0 and 1.0
  """
  plt.rc('text', usetex=True)
  (num_images, num_time_steps, num_neurons) = data["b"].shape
  sqrt_nn = int(np.sqrt(num_neurons))
  if img_idx is None:
    img_idx = np.random.choice(num_images)
  global_max_val = float(np.max(np.abs([data["b"][img_idx,...],
    data["u"][img_idx,...], data["ga"][img_idx,...], data["a"][img_idx,...],
    np.ones_like(data["b"][img_idx,...])*activation_threshold])))
  fig, sub_axes = plt.subplots(sqrt_nn+2, sqrt_nn+1, figsize=(20, 20))
  fig.subplots_adjust(hspace=0.20, wspace=0.20)
  for (axis_idx, axis) in enumerate(fig.axes): # one axis per neuron
    if axis_idx < num_neurons:
      t = np.arange(data["b"].shape[1])
      b = data["b"][img_idx, :, axis_idx]
      u = data["u"][img_idx, :, axis_idx]
      ga = data["ga"][img_idx, :, axis_idx]
      a = data["a"][img_idx, :, axis_idx]
      l1, = axis.plot(t, b, linewidth=0.25, color="g", label="b")
      l2, = axis.plot(t, u, linewidth=0.25, color="b", label="u")
      l3, = axis.plot(t, ga, linewidth=0.25, color="r", label="Ga")
      l4, = axis.plot(t, [0 for _ in t], linewidth=0.25, color="k", linestyle="-",
        label="zero")
      l5, = axis.plot(t, [activation_threshold for _ in t], linewidth=0.25, color="k",
        linestyle=":", dashes=(1,1), label=r"$\lambda$")
      if "fb" in data.keys():
        fb = data["fb"][img_idx,:,axis_idx]
        l6, = axis.plot(t, fb, linewidth=0.25, color="darkorange", label="fb")
      max_val = np.max(np.abs([b, ga, u, a]))
      scale_ratio = max_val / global_max_val
      transFigure = fig.transFigure.inverted()
      axis_height = axis.get_window_extent().transformed(transFigure).height
      line_length = axis_height * scale_ratio
      x_offset = 0.003
      axis_origin = transFigure.transform(axis.transAxes.transform([0,0]))
      coord1 = [axis_origin[0] - x_offset, axis_origin[1]]
      coord2 = [coord1[0], coord1[1] + line_length]
      line = matplotlib.lines.Line2D((coord1[0], coord2[0]), (coord1[1],
        coord2[1]), transform=fig.transFigure, color="0.3")
      fig.lines.append(line)
      if (a[-1] > 0):
        clear_axis(axis, spines="magenta")
        if act_indicator_threshold is not None:
          assert act_indicator_threshold > 0.0 and act_indicator_threshold < 1.0, (
            "act_indicator_threshold must be between 0.0 and 1.0")
          thresh_index = int(num_time_steps * act_indicator_threshold)
          if np.all([a[idx] == 0 for idx in range(0, thresh_index)]): # neuron has recently become active
             for ax_loc in ["top", "bottom", "left", "right"]:
              axis.spines[ax_loc].set_linestyle((1, (1, 3))) #length, spacing (on, off)
      else:
        clear_axis(axis, spines="black")
        if act_indicator_threshold is not None:
          thresh_index = int(num_time_steps * act_indicator_threshold)
          if np.any([a[idx] > 0 for idx in range(thresh_index, num_time_steps)]): # neuron has recently become inactive
             for ax_loc in ["top", "bottom", "left", "right"]:
              axis.spines[ax_loc].set_linestyle((1, (1, 3))) #length, spacing (on, off)
    else:
      clear_axis(axis)
  num_pixels = np.size(data["images"][img_idx])
  image = data["images"][img_idx,...].reshape(int(np.sqrt(num_pixels)), int(np.sqrt(num_pixels)))
  sub_axes[sqrt_nn+1, 0].imshow(image, cmap="Greys", interpolation="nearest")
  for plot_col in range(sqrt_nn):
    clear_axis(sub_axes[sqrt_nn+1, plot_col])
  fig.suptitle("LCA Activity", y=0.9, fontsize=18)
  handles, labels = sub_axes[0,0].get_legend_handles_labels()
  legend = sub_axes[sqrt_nn+1, 1].legend(handles, labels, fontsize=12, ncol=3,
    borderaxespad=0., bbox_to_anchor=[0, 0], fancybox=True, loc="upper left")
  for line in legend.get_lines():
    line.set_linewidth(3)
  plt.show()
  return fig

def plot_weight_image(weights, colorbar_aspect=50, title="", figsize=None, save_filename=None):
  fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
  ax = ax.item()
  im = ax.imshow(weights, vmin=np.min(weights), vmax=np.max(weights), cmap="Greys_r")
  ax.set_title(title)
  clear_axis(ax)
  add_colorbar_to_im(im, aspect=colorbar_aspect)
  if save_filename is not None:
    fig.savefig(save_filename, transparent=True)
    plt.close(fig)
    return None
  plt.show()
  return fig

def plot_weight_angle_heatmap(weight_angles, angle_min=0, angle_max=180, title="", figsize=None, save_filename=None):
  vmin = angle_min
  vmax = angle_max
  cmap = plt.get_cmap('viridis')
  cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
  scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
  scalarMap._A = []
  fig, ax = plt.subplots(1, figsize=figsize)
  im = ax.imshow(weight_angles, vmin=vmin, vmax=vmax)
  ax.set_title(title, fontsize=18)
  cbar = add_colorbar_to_im(im, aspect=20, pad_fraction=0.5, labelsize=16, ticks=[vmin, vmax])
  cbar.ax.set_yticklabels(["{:.0f}".format(vmin), "{:.0f}".format(vmax)])
  if save_filename is not None:
    fig.savefig(save_filename, transparent=True)
    plt.close(fig)
    return None
  plt.show()
  return fig

def plot_weight_angle_histogram(weight_angles, num_bins=50, angle_min=0, angle_max=180,
  y_max=None, figsize=None, save_filename=None):
  bins = np.linspace(angle_min, angle_max, num_bins)
  hist, bin_edges = np.histogram(weight_angles.flatten(), bins)
  if y_max is None:
    y_max = np.max(hist)
  bin_left, bin_right = bin_edges[:-1], bin_edges[1:]
  bin_centers = bin_left + (bin_right - bin_left)/2
  fig, ax = plt.subplots(1, figsize=figsize)
  ax.bar(bin_centers, hist, width=2.0, log=True, align="center")
  ax.set_xticks(bin_left, minor=True)
  ax.set_xticks(bin_left[::4], minor=False)
  ax.xaxis.set_major_formatter(FormatStrFormatter("%0.0f"))
  ax.tick_params("both", labelsize=16)
  ax.set_xlim([angle_min, angle_max])
  ax.set_xticks([angle_min, int(np.floor(angle_max/4)), int(2*np.floor(angle_max/4)),
    int(3*np.floor(angle_max/4)), angle_max])
  ax.set_ylim([1, y_max])
  ax.set_title("Neuron Angle Histogram", fontsize=18)
  ax.set_xlabel("Angle (Degrees)", fontsize=18)
  ax.set_ylabel("Log Count", fontsize=18)
  if save_filename is not None:
    fig.savefig(save_filename)
    plt.close(fig)
    return None
  plt.show()
  return fig

def plot_weight_nearest_neighbor_histogram(weight_angles, num_bins=25, angle_min=0, angle_max=90,
  y_max=None, figsize=None, save_filename=None):
  nn_angles = np.zeros(weight_angles.shape[0])
  for neuron_id in range(weight_angles.shape[0]):
    neighbors = np.delete(weight_angles[neuron_id,:], neuron_id)
    nn_angles[neuron_id] = np.min(neighbors[neighbors>=0])
  bins = np.linspace(angle_min, angle_max, num_bins)
  hist, bin_edges = np.histogram(nn_angles.flatten(), bins)
  if y_max is None:
    y_max = np.max(hist)
  bin_left, bin_right = bin_edges[:-1], bin_edges[1:]
  bin_centers = bin_left + (bin_right - bin_left)/2
  fig, ax = plt.subplots(1, figsize=figsize)
  ax.bar(bin_centers, hist, width=1.0, log=True, align="center")
  ax.set_xticks(bin_left, minor=True)
  ax.set_xticks(bin_left[::4], minor=False)
  ax.xaxis.set_major_formatter(FormatStrFormatter("%0.0f"))
  ax.tick_params("both", labelsize=16)
  ax.set_xlim([angle_min, angle_max])
  ax.set_xticks([angle_min, int(np.floor(angle_max/4)), int(2*np.floor(angle_max/4)),
    int(3*np.floor(angle_max/4)), angle_max])
  ax.set_ylim([1, y_max])
  ax.set_title("Neuron Nearest Neighbor Angle", fontsize=18)
  ax.set_xlabel("Angle (Degrees)", fontsize=18)
  ax.set_ylabel("Log Count", fontsize=18)
  if save_filename is not None:
    fig.savefig(save_filename)
    plt.close(fig)
    return None
  plt.show()
  return fig

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

def bgr_colormap():
  """
  In cdict, the first column is interpolated between 0.0 & 1.0 - this indicates the value to be plotted
  the second column specifies how interpolation should be done from below
  the third column specifies how interpolation should be done from above
  if the second column does not equal the third, then there will be a break in the colors
  """
  darkness = 0.85 #0 is black, 1 is white
  cdict = {'red':   ((0.0, 0.0, 0.0),
                    (0.5, darkness, darkness),
                    (1.0, 1.0, 1.0)),
           'green': ((0.0, 0.0, 0.0),
                    (0.5, darkness, darkness),
                    (1.0, 0.0, 0.0)),
           'blue':  ((0.0, 1.0, 1.0),
                    (0.5, darkness, darkness),
                    (1.0, 0.0, 0.0))
      }
  return LinearSegmentedColormap("bgr", cdict)

def add_colorbar_to_ax(handle, ax, aspect=20, pad_fraction=0.5, labelsize=16, **kwargs):
  """
  Add a vertical color bar to an image plot.
  Inputs:
    ax: TODO
    aspect: [int] aspect ratio of the colorbar
    pad_fraction: [float] how much space to place between colorbar & plot
    labelsize: [float] font size of the colorbar labels
    **kwargs: [dict] other keyword arguments that would be passed to axes.figure.colorbar()
  """
  divider = axes_grid1.make_axes_locatable(ax)
  width = axes_grid1.axes_size.AxesY(ax, aspect=1./aspect)
  pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
  current_ax = plt.gca()
  cax = divider.append_axes("right", size=width, pad=pad)
  plt.sca(current_ax)
  cbar = ax.figure.colorbar(handle, cax=cax, **kwargs)
  cbar.ax.tick_params(labelsize=labelsize)
  return cbar

def add_colorbar_to_im(im, aspect=20, pad_fraction=0.5, labelsize=16, **kwargs):
  """
  Add a vertical color bar to an image plot.
  Inputs:
    im: [AxisImage] object returned from matplotlib.plt.imshow()
    aspect: [int] aspect ratio of the colorbar
    pad_fraction: [float] how much space to place between colorbar & plot
    labelsize: [float] font size of the colorbar labels
    **kwargs: [dict] other keyword arguments that would be passed to im.axes.figure.colorbar()
  """
  divider = axes_grid1.make_axes_locatable(im.axes)
  width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
  pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
  current_ax = plt.gca()
  cax = divider.append_axes("right", size=width, pad=pad)
  plt.sca(current_ax)
  cbar = im.axes.figure.colorbar(im, cax=cax, **kwargs)
  cbar.ax.tick_params(labelsize=labelsize)
  return cbar

def clear_axes(ax, spines="none"):
  """
  TODO: Loop over ax (regardless of shape) and call clear_axis on each element
  """
  raise NotImplementedError

def clear_axis(ax, spines="none"):
  for ax_loc in ["top", "bottom", "left", "right"]:
    ax.spines[ax_loc].set_color(spines)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.tick_params(axis="both", bottom=False, top=False, left=False, right=False)
  return ax
