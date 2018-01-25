import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits import axes_grid1
import utils.data_processing as dp

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
                           num_connected_weights, lines=False, figsize=None):
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
        orientations = bf_stats["fourier_centers"][bf_idx]
        angle = np.rad2deg(np.pi/2 + np.arctan2(*orientations))
        alpha = 0.5#todo:spatial_freq for filled ellipses?
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

def plot_pooling_centers(bf_stats, pooling_filters, num_pooling_filters,
                         spot_size=10, figsize=None):
  """
  Plot 2nd layer (fully-connected) weights in terms of spatial/frequency centers of
    1st layer weights
  Inputs:
    bf_stats [dict] Output of dp.get_dictionary_stats() which was run on the 1st layer weights
    pooling_filters [np.ndarray] 2nd layer weights
      should be shape [num_1st_layer_neurons, num_2nd_layer_neurons]
    num_pooling_filters [int] How many 2nd layer neurons to plot
    figsize [tuple] Containing the (width, height) of the figure, in inches
    spot_size [int] How big to make the points
  """
  num_filters_y = int(np.ceil(np.sqrt(num_pooling_filters)))
  num_filters_x = int(np.ceil(np.sqrt(num_pooling_filters)))
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
      cbar.ax.xaxis.set_ticks_position('top')
      cbar.ax.xaxis.set_label_position('top')
      for label in cbar.ax.xaxis.get_ticklabels():
        label.set_weight("bold")
        label.set_fontsize(10+figsize[0])
    if (filter_id < num_pooling_filters):
      example_filter = pooling_filters[:, filter_id]
      filter_norm = np.max(np.abs(example_filter))
      connection_colors = [scalarMap.to_rgba(example_filter[bf_idx]/filter_norm)
        for bf_idx in range(bf_stats["num_outputs"])]
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
      #spatial
      axes.append(clear_axis(fig.add_axes([ax_l, ax_b, ax_w, ax_h])))
      axes[-1].invert_yaxis()
      axes[-1].scatter(x_p_cent, y_p_cent, c=connection_colors, s=spot_size, alpha=0.8)
      axes[-1].set_xlim(0, bf_stats["patch_edge_size"]-1)
      axes[-1].set_ylim(bf_stats["patch_edge_size"]-1, 0)
      axes[-1].set_aspect("equal")
      axes[-1].set_facecolor("k")
      axes.append(clear_axis(fig.add_axes([ax_l+ax_w+pair_w_gap, ax_b, ax_w, ax_h])))
      axes[-1].scatter(x_f_cent, y_f_cent, c=connection_colors, s=spot_size, alpha=0.8)
      axes[-1].set_xlim([-max_sf, max_sf])
      axes[-1].set_ylim([-max_sf, max_sf])
      axes[-1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
      axes[-1].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
      axes[-1].set_aspect("equal")
      axes[-1].set_facecolor("k")
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
    a_cov_row = a_cov[primary_bf_idx, :]
    sorted_cov_indices = np.argsort(a_cov[primary_bf_idx, :])[-2::-1]
    primary_bf = np.squeeze(dp.reshape_data(weights.T[primary_bf_idx,...],
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
    sub_ax[plot_id, 2].set_ylim([0, fourier.shape[0]-1])
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

def plot_loc_freq_summary(bf_stats):
  fig, sub_ax = plt.subplots(1, 2, figsize=(10,5))
  x_pos = [x for (y,x) in bf_stats["gauss_centers"]]
  y_pos = [y for (y,x) in bf_stats["gauss_centers"]]
  sub_ax[0].scatter(x_pos, y_pos, color='k', s=10)
  sub_ax[0].set_xlim([0, bf_stats["patch_edge_size"]-1])
  sub_ax[0].set_ylim([bf_stats["patch_edge_size"]-1, 0])
  sub_ax[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
  sub_ax[0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
  sub_ax[0].set_aspect("equal")
  sub_ax[0].set_ylabel("Pixels")
  sub_ax[0].set_xlabel("Pixels")
  sub_ax[0].set_title("Basis Function Centers", fontsize=12)
  x_sf = [x for (y,x) in bf_stats["fourier_centers"]]
  y_sf = [y for (y,x) in bf_stats["fourier_centers"]]
  max_sf = np.max(np.abs(x_sf+y_sf))
  sub_ax[1].scatter(x_sf, y_sf, color='k', s=10)
  sub_ax[1].set_xlim([-max_sf, max_sf])
  sub_ax[1].set_ylim([-max_sf, max_sf])
  sub_ax[1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
  sub_ax[1].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
  sub_ax[1].set_aspect("equal")
  sub_ax[1].set_ylabel("Cycles / Patch")
  sub_ax[1].set_xlabel("Cycles / Patch")
  sub_ax[1].set_title("Basis Function Spatial Frequencies", fontsize=12)
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

def plot_matrix(matrix, title=""):
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
  plt.show()
  return fig

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
  ax.set_xlabel('Activity')
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

def plot_weights(weights, title="", save_filename=None):
  num_plots = weights.shape[0]
  num_plots_y = int(np.ceil(np.sqrt(num_plots))+1)
  num_plots_x = int(np.floor(np.sqrt(num_plots)))
  fig, sub_ax = plt.subplots(num_plots_y, num_plots_x, figsize=(18,18))
  filter_total = 0
  for plot_id in  np.ndindex((num_plots_y, num_plots_x)):
    if filter_total < num_plots:
      sub_ax[plot_id].imshow(weights[filter_total, ...], cmap="Greys_r")
      filter_total += 1
    clear_axis(sub_ax[plot_id])
    sub_ax[plot_id].set_aspect("equal")
  fig.suptitle(title, y=1.0, x=0.5, fontsize=20)
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

def plot_stats(data, keys=None, labels=None, save_filename=None):
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
  if "batch_step" in keys:
    keys.remove("batch_step")
  if labels is None:
    labels = keys
  else:
    assert len(labels) == len(keys), (
      "The number of labels must match the number of keys")
  num_keys = len(keys)
  fig, sub_ax = plt.subplots(num_keys)
  axis_image = [None]*num_keys
  for key_idx, key in enumerate(keys):
    axis_image[key_idx] = sub_ax[key_idx].plot(data["batch_step"], data[key])
    if key_idx < len(keys)-1:
      sub_ax[key_idx].get_xaxis().set_ticklabels([])
    sub_ax[key_idx].locator_params(axis="y", nbins=5)
    sub_ax[key_idx].set_ylabel(labels[key_idx])
    ylabel_xpos = -0.15
    sub_ax[key_idx].yaxis.set_label_coords(ylabel_xpos, 0.5)
  sub_ax[-1].set_xlabel("Batch Number")
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
  """
  labels = [key for key in data["losses"].keys()]
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
  fig = plt.figure(figsize=(10,10))
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
      ax.set_xlabel("time step", fontsize=16)
      ax.tick_params("both", labelsize=14)
      loss_id += 1
    else:
      ax = clear_axis(ax, spines="none")
  fig.tight_layout()
  fig.suptitle(title, y=1.03, x=0.5, fontsize=20)
  if save_filename is not None:
    fig.savefig(save_filename, transparent=True)
    plt.close(fig)
    return None
  plt.show()
  return fig

def plot_inference_traces(data, activation_threshold, img_idx=0):
  """
  Plot of model neurons' inputs over time
  Args:
    data: [dict] with each trace, with keys [b, u, a, ga, images]
      Dictionary is created by analyze_lca.evaluate_inference()
    img_idx: [int] which image in data["images"] to run analysis on
  """
  plt.rc('text', usetex=True)
  (num_images, num_time_steps, num_neurons) = data["b"].shape
  sqrt_nn = int(np.sqrt(num_neurons))
  global_max_val = float(np.max(np.abs([data["b"][img_idx,...],
    data["u"][img_idx,...], data["ga"][img_idx,...], data["a"][img_idx,...]])))
  fig, sub_axes = plt.subplots(sqrt_nn+2, sqrt_nn+1, figsize=(20, 20))
  fig.subplots_adjust(hspace=0.20, wspace=0.20)
  for (axis_idx, axis) in enumerate(fig.axes): # one axis per neuron
    if axis_idx < num_neurons:
      t = np.arange(data["b"].shape[1])
      b = data["b"][img_idx,:,axis_idx]
      u = data["u"][img_idx,:,axis_idx]
      ga = data["ga"][img_idx,:,axis_idx]
      a = data["a"][img_idx,:,axis_idx]
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
      else:
        clear_axis(axis, spines="black")
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

def add_colorbar_to_im(im, aspect=20, pad_fraction=0.5, **kwargs):
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
  cbar = im.axes.figure.colorbar(im, cax=cax, **kwargs)
  cbar.ax.tick_params(labelsize=16)
  return cbar

def clear_axis(ax, spines="none"):
  ax.spines["right"].set_color(spines)
  ax.spines["top"].set_color(spines)
  ax.spines["left"].set_color(spines)
  ax.spines["bottom"].set_color(spines)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.tick_params(axis="both", bottom="off", top="off", left="off", right="off")
  return ax
