"""
Some of my own plotting functionality
"""
import bisect
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib import colors as mcolors
from matplotlib import cm as mcolormap
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

def plot_basis_functions(tensor_3d, plot_title="", freq_dq_inds=[],
                         env_dq_inds=[],
                         reference_interval=[0., 1.],
                         renormalize=True, single_img=False):
  """
  Plot each of the basis functions side by side

  Parameters
  ----------
  tensor_3d : ndarray
      An N x H x W matrix where N is the number of basis functions, H is the
      height of each basis function and W is the width
  plot_title : str, optional
      The title of the plot
  freq_dq_inds : array-like, optional
      The indices of basis functions we will disqualify from the analysis on the
      basis of them not having good spatial frequency fits. (Usually they are
      very low frequency and not particularly oriented). Default []
  env_dq_inds : array-like, optional
      The indices of basis functions we will disqualify from the analysis on the
      basis of them having gabor envelopes which are very small - these barely
      appear in the filter, often are in the corners, and don't look
      particularly oriented. Default []
  reference_interval : array-like, optional
      If present, displayed images will have black correspond to
      reference_interval[0] and white correspond to reference_interval[1].
      Default interval is [0., 1.]
  renormalize : bool, optional
      If present, renormalize each basis function to the reference interval
  single_img : bool, optional
      If true, just make a single composite image with all the bf vectors
      separated by some margin in X and Y. A lot faster than many different
      suplots but you lose control and the ability to label things individually.
      Default False.

  Returns
  -------
  bf_figs : list
      A list containing pyplot figures. Can be saved separately, or whatever
      from the calling function
  """
  max_bf_val = np.max(tensor_3d)
  min_bf_val = np.min(tensor_3d)
  bf_height = tensor_3d.shape[1]
  bf_width = tensor_3d.shape[2]

  max_bfs_per_fig = 400
  assert np.sqrt(max_bfs_per_fig) % 1 == 0, 'please pick a square number'
  num_bfs = tensor_3d.shape[0]
  num_bf_figs = int(np.ceil(num_bfs / max_bfs_per_fig))
  # this determines how many basis_funcs are aranged in a square grid within
  # any given figure
  if num_bf_figs > 1:
    bfs_per_fig = max_bfs_per_fig
  else:
    squares = [x**2 for x in range(1, int(np.sqrt(max_bfs_per_fig))+1)]
    bfs_per_fig = squares[bisect.bisect_left(squares, num_bfs)]
  plot_sidelength = int(np.sqrt(bfs_per_fig))

  if single_img:
    h_margin = 2
    w_margin = 2
    full_img_height = (bf_height * plot_sidelength +
                       (plot_sidelength - 1) * h_margin)
    full_img_width = (bf_width * plot_sidelength +
                      (plot_sidelength - 1) * w_margin)

  bf_idx = 0
  bf_figs = []
  for in_bf_fig_idx in range(num_bf_figs):
    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(plot_title + ', fig {} of {}'.format(
                 in_bf_fig_idx+1, num_bf_figs), fontsize=15)
    if single_img:
      composite_img = (reference_interval[1] *
                       np.ones((full_img_height, full_img_width)))
    else:
      # each bf vector gets its own subplot
      subplot_grid = gridspec.GridSpec(plot_sidelength, plot_sidelength,
                                       wspace=0.15, hspace=0.15)
    fig_bf_idx = bf_idx % bfs_per_fig
    while fig_bf_idx < bfs_per_fig and bf_idx < num_bfs:

      if single_img:
        if renormalize:
          this_bf = tensor_3d[bf_idx, :, :]
          this_bf = this_bf - np.min(this_bf)
          this_bf = this_bf / np.max(this_bf)  # now in [0, 1]
          this_bf = this_bf * (reference_interval[1] -
                               reference_interval[0])
          this_bf = this_bf + reference_interval[0]
        else:
          this_bf = np.copy(tensor_3d[bf_idx, :, :])
        if freq_dq_inds != [] or env_dq_inds != []:
          raise KeyError('Cannot highlight individual filters in single_img ' +
                         'plotting mode')

        # okay, now actually plot the bfs in this figure
        row_idx = fig_bf_idx // plot_sidelength
        col_idx = fig_bf_idx % plot_sidelength
        pxr1 = row_idx * (bf_height + h_margin)
        pxr2 = pxr1 + bf_height
        pxc1 = col_idx * (bf_width + w_margin)
        pxc2 = pxc1 + bf_width
        composite_img[pxr1:pxr2, pxc1:pxc2] = this_bf

        fig_bf_idx += 1
        bf_idx += 1

      else:
        if bf_idx % 100 == 0:
          print("plotted ", bf_idx, " of ", num_bfs, " weights")
        ax = plt.Subplot(fig, subplot_grid[fig_bf_idx])
        if renormalize:
          ax.imshow(tensor_3d[bf_idx, :, :],
                    cmap='Greys_r', interpolation='nearest')
        else:
          ax.imshow(tensor_3d[bf_idx, :, :], vmin=reference_interval[0],
                    vmax=reference_interval[1],
                    cmap='Greys_r', interpolation='nearest')
        if bf_idx in freq_dq_inds:
          if bf_idx in env_dq_inds:
            #^ disqualified by both criteria
            ax.spines["right"].set_color('m')
            ax.spines["top"].set_color('m')
            ax.spines["left"].set_color('m')
            ax.spines["bottom"].set_color('m')
          else:
            ax.spines["right"].set_color('r')
            ax.spines["top"].set_color('r')
            ax.spines["left"].set_color('r')
            ax.spines["bottom"].set_color('r')
          ax.spines["right"].set_linewidth(2)
          ax.spines["top"].set_linewidth(2)
          ax.spines["left"].set_linewidth(2)
          ax.spines["bottom"].set_linewidth(2)
          ax.set_xticks([])
          ax.set_yticks([])
        if bf_idx in env_dq_inds:
          if bf_idx not in freq_dq_inds:  # will already be colored magenta
            ax.spines["right"].set_color('b')
            ax.spines["top"].set_color('b')
            ax.spines["left"].set_color('b')
            ax.spines["bottom"].set_color('b')
          ax.spines["right"].set_linewidth(2)
          ax.spines["top"].set_linewidth(2)
          ax.spines["left"].set_linewidth(2)
          ax.spines["bottom"].set_linewidth(2)
          ax.set_xticks([])
          ax.set_yticks([])
        if (bf_idx not in freq_dq_inds) and (bf_idx not in env_dq_inds):
          ax.axis('off')

        fig.add_subplot(ax)
        fig_bf_idx += 1
        bf_idx += 1

    if single_img:
      plt.imshow(composite_img, cmap='Greys_r', vmin=reference_interval[0],
                 vmax=reference_interval[1], interpolation='nearest')
      plt.axis('off')

    bf_figs.append(fig)

  return bf_figs


def plot_ot_curves(ot_activations, contrasts_dict, metrics,
                   plot_title='', small_plots=False):
  """
  Plots the orientation tuning curves along with summary metrics

  Can plot tuning curves for multiple different contrasts. Will also display
  metrics in the upper right-hand corner that try to summarize the
  orientation selectivity of the curve

  Parameters
  ----------
  ot_activations : ndarray
      The big array of size B x C x O where B is the number of basis functions
      that we are analyzing, C is the number of contrasts used, and O is
      the number of orientations at which responses were measured.
  contrasts_dict : dictionary
      Used to select which of the contrasts to actually plot the curves for and
      at what grey value to display them. Keys are indeces into the second
      dimension of ot_activations and values are the associated grey values in
      [0., 1.] to be used in the plot.
  metrics : dictionary
      The dictionary of metrics that are values in the returned dictionary
      from spencers_stuff/metrics.compute_ot_metrics(). See docstring in
      this function for more details
  plot_title : str
      The (optional) title of the plot
  """
  tab10colors = plt.get_cmap('tab10').colors
  orientations = (180 * np.arange(ot_activations.shape[2])
                  / ot_activations.shape[2]) - 90 # relative to preferred
  cmap = plt.get_cmap('Greys')
  cNorm = mcolors.Normalize(vmin=0.0, vmax=1.0)
  scalarMap = mcolormap.ScalarMappable(norm=cNorm, cmap=cmap)
  metric_ordering = list(metrics.keys())
  #^ always display the metrics in the same order on the plots

  if small_plots:
    max_bfs_per_fig = 400
  else:
    max_bfs_per_fig = 25
  assert np.sqrt(max_bfs_per_fig) % 1 == 0, 'please pick a square number'
  num_bfs = ot_activations.shape[0]
  num_bf_figs = int(np.ceil(num_bfs / max_bfs_per_fig))
  # this determines how many ot curves are aranged in a square grid within
  # any given figure
  if num_bf_figs > 1:
    bfs_per_fig = max_bfs_per_fig
  else:
    squares = [x**2 for x in range(1, int(np.sqrt(max_bfs_per_fig))+1)]
    bfs_per_fig = squares[bisect.bisect_left(squares, num_bfs)]
  plot_sidelength = int(np.sqrt(bfs_per_fig))

  bf_idx = 0
  bf_figs = []
  for in_bf_fig_idx in range(num_bf_figs):
    fig = plt.figure(figsize=(32, 32))
    plt.suptitle(plot_title + ', fig {} of {}'.format(
                 in_bf_fig_idx+1, num_bf_figs), fontsize=20)
    subplot_grid = gridspec.GridSpec(plot_sidelength, plot_sidelength,
                                     wspace=0.4 if small_plots else 0.2,
                                     hspace=0.4)
    fig_bf_idx = bf_idx % bfs_per_fig
    while fig_bf_idx < bfs_per_fig and bf_idx < num_bfs:
      if bf_idx % 100 == 0:
        print("plotted ", bf_idx, " of ", num_bfs, " ot curves")
      ax = plt.Subplot(fig, subplot_grid[fig_bf_idx])
      # plot fwhm lines first, then lay the curves over them
      ax.axhline(y=metrics['full width half maximum'][bf_idx][2],
                 color=tab10colors[0], linestyle='--', alpha=0.4,
                 linewidth=1 if small_plots else 2)
      ax.axvline(x=metrics['full width half maximum'][bf_idx][0],
                 color=tab10colors[2], linestyle='--', alpha=0.6,
                 linewidth=1 if small_plots else 2)
      ax.axvline(x=metrics['full width half maximum'][bf_idx][1],
                 color=tab10colors[2], linestyle='--', alpha=0.6,
                 linewidth=1 if small_plots else 2)
      for c_idx in contrasts_dict:
        curve = ot_activations[bf_idx, c_idx, :]
        centered_curve = center_curve(curve)
        color_val = scalarMap.to_rgba(contrasts_dict[c_idx])
        ax.plot(orientations, centered_curve, linewidth=1 if small_plots else 3,
                color=color_val)
        ax.scatter(orientations, centered_curve, s=4 if small_plots else 40,
                   c=color_val)
      ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
      ax.set_yticks([np.min(ot_activations[bf_idx, :, :]),
                     np.max(ot_activations[bf_idx, :, :])])
      ax.tick_params(axis='both', which='major',
                     labelsize=5 if small_plots else 15)
      ax.set_xticks([-90, 0, 90])
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
      # this next section adds a text display of the metrics in the upper right
      # hand corner of the plot
      current_text_vpos = 0.97 if small_plots else 0.95
      for metric_label in metric_ordering:
        if metric_label == 'full width half maximum':
          label_prefix = 'FWHM: ' if small_plots else ''
          label_color = tab10colors[2]
          value = (metrics[metric_label][bf_idx][1] -
                   metrics[metric_label][bf_idx][0])  # right_hm - left_hm
        elif metric_label == 'circular variance':
          label_prefix = 'CV: ' if small_plots else ''
          label_color = tab10colors[3]
          value = metrics[metric_label][bf_idx][2]
        elif metric_label == 'orientation selectivity index':
          label_prefix = 'OSI: ' if small_plots else ''
          label_color = tab10colors[4]
          value = metrics[metric_label][bf_idx]
        else:
          raise KeyError('Unrecognized metric type: ' + metric_label)
        ax.text(0.02 if small_plots else 0.05,
                current_text_vpos, label_prefix + '{:.2f}'.format(value),
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, color=label_color,
                fontsize=5 if small_plots else 18)
        current_text_vpos -= 0.1 if small_plots else 0.2

      fig.add_subplot(ax)
      fig_bf_idx += 1
      bf_idx += 1

    bf_figs.append(fig)

  return bf_figs



def plot_circular_variance(cv_data, plot_title=''):
  orientations = (np.pi * np.arange(len(cv_data))
                  / len(cv_data)) - (np.pi/2) # relative to preferred

  max_bfs_per_fig = 400
  assert np.sqrt(max_bfs_per_fig) % 1 == 0, 'please pick a square number'
  num_bfs = len(cv_data)
  num_bf_figs = int(np.ceil(num_bfs / max_bfs_per_fig))
  # this determines how many ot curves are aranged in a square grid within
  # any given figure
  if num_bf_figs > 1:
    bfs_per_fig = max_bfs_per_fig
  else:
    squares = [x**2 for x in range(1, int(np.sqrt(max_bfs_per_fig))+1)]
    bfs_per_fig = squares[bisect.bisect_left(squares, num_bfs)]
  plot_sidelength = int(np.sqrt(bfs_per_fig))

  bf_idx = 0
  bf_figs = []
  for in_bf_fig_idx in range(num_bf_figs):
    fig = plt.figure(figsize=(32, 32))
    plt.suptitle(plot_title + ', fig {} of {}'.format(
                 in_bf_fig_idx+1, num_bf_figs), fontsize=20)
    subplot_grid = gridspec.GridSpec(plot_sidelength, plot_sidelength,
                                     wspace=0.4, hspace=0.4)
    fig_bf_idx = bf_idx % bfs_per_fig
    while fig_bf_idx < bfs_per_fig and bf_idx < num_bfs:
      if bf_idx % 100 == 0:
        print("plotted ", bf_idx, " of ", num_bfs, " circular variance plots")
      # print("sum vector: ", np.real(cv_data[bf_idx][1]), np.imag(cv_data[bf_idx][1]))
      ax = plt.Subplot(fig, subplot_grid[fig_bf_idx])
      ax.plot(np.real(cv_data[bf_idx][0]), np.imag(cv_data[bf_idx][0]),
              c='g', linewidth=0.5)
      ax.scatter(np.real(cv_data[bf_idx][0]), np.imag(cv_data[bf_idx][0]),
                 c='g', s=4)
      ax.quiver(np.real(cv_data[bf_idx][1]), np.imag(cv_data[bf_idx][1]),
                angles='xy', scale_units='xy', scale=1.0, color='b',
                width=0.01)
      # ax.quiver(0.5, 0.5, color='b')
      ax.axvline(x=0.0, color='k', linestyle='--', alpha=0.6, linewidth=0.3)
      ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.6, linewidth=0.3)
      ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2g'))
      xaxis_size = max(np.max(np.real(cv_data[bf_idx][0])), 1.0)
      yaxis_size = max(np.max(np.imag(cv_data[bf_idx][0])), 1.0)
      ax.set_yticks([-1. * yaxis_size, yaxis_size])
      ax.set_xticks([-1. * xaxis_size, xaxis_size])
      # put the circular variance index in the upper left
      ax.text(0.02, 0.97, 'CV: {:.2f}'.format(cv_data[bf_idx][2]),
              horizontalalignment='left', verticalalignment='top',
              transform=ax.transAxes, color='b', fontsize=10)
      fig.add_subplot(ax)
      fig_bf_idx += 1
      bf_idx += 1

    bf_figs.append(fig)

  return bf_figs


def center_curve(tuning_curve):
    """
    Centers a curve about its preferred orientation
    """
    return np.roll(tuning_curve,
                   (len(tuning_curve) // 2) - np.argmax(tuning_curve))


# Just something I like for visualizing the empirical density
def compare_empirical_density(datasets, labels, num_hist_bins, lines=True):
  """
  Returns plot that shows probability density estimates for SCALAR RVs

  Parameters
  ----------
  datasets : list(ndarray) or ndarray
      The scalar valued datasets to plot. If a list, then each element is a
      different dataset
  labels : list(str) or str
      Labels for each of the dsets in datasets
  num_hist_bins : int
      The number of histogram bins to use
  lines : bool, optional
      If true, plot the binned counts using a line rather than bars. This
      can make it a lot easier to compare multiple datasets at once but
      can look kind of jagged if there aren't many samples
  """
  fig, ax = plt.subplots(1, 1)
  tab10colors = plt.get_cmap('tab10').colors

  if type(datasets) not in [list, tuple]:
    # assume it's a single dataset, passed in as an ndarray
    counts, histogram_bin_edges = np.histogram(datasets, num_hist_bins)
    empirical_density = counts / np.sum(counts)
    histogram_bin_centers = (histogram_bin_edges[:-1] +
                             histogram_bin_edges[1:]) / 2

    if lines:
      ax.plot(histogram_bin_centers, empirical_density,
              color=tab10colors[0], linewidth=1.5)
    else:
      ax.bar(histogram_bin_centers, empirical_density, align='center',
             color=tab10colors[0],
             width=histogram_bin_centers[1]-histogram_bin_centers[0])
    ax.legend([labels])

  else:
    assert len(datasets) <= 10, 'Choose a larger colormap'

    dataset_min = []
    dataset_max = []
    for dset_idx in range(len(datasets)):
      dataset_min.append(np.min(datasets[dset_idx]))
      dataset_max.append(np.max(datasets[dset_idx]))
    histogram_min = min(dataset_min)
    histogram_max = max(dataset_max)
    histogram_bin_edges = np.linspace(histogram_min, histogram_max,
                                      num_hist_bins + 1)
    histogram_bin_centers = (histogram_bin_edges[:-1] +
                             histogram_bin_edges[1:]) / 2

    for dset_idx in range(len(datasets)):
      counts, _ = np.histogram(datasets[dset_idx], histogram_bin_edges)
      empirical_density = counts / np.sum(counts)
      if lines:
        ax.plot(histogram_bin_centers, empirical_density,
                color=tab10colors[dset_idx], linewidth=1.5)
      else:
        ax.bar(histogram_bin_centers, empirical_density, align='center',
               color=tab10colors[dset_idx],
               width=histogram_bin_centers[1]-histogram_bin_centers[0],
               alpha=0.4)
    ax.legend(labels, fontsize=15)

    for dset_idx in range(len(datasets)):
    # # add some summary statistics
      ax.axvline(x=np.mean(datasets[dset_idx]), color=tab10colors[dset_idx],
                           linestyle='-.', alpha=0.5)
    #   # ax.axvline(x=np.median(datasets[dset_idx]), color=tab10colors[dset_idx],
    #   #                        linestyle='--', alpha=0.5)
    #   labels.append(labels[dset_idx] + ' mean')
      # labels.append(labels[dset_idx] + ' median')
    # labels.insert(dset_idx + (3*dset_idx) + 1, labels[dset_idx] + ' mean')
    # labels.insert(dset_idx + (3*dset_idx) + 2, labels[dset_idx] + ' median')

  fig.suptitle('Empirical density estimate based on simple histogram',
               fontsize=20)
  ax.set_ylabel('Estimated probability', fontsize=15)
  ax.set_xlabel('Scalar value', fontsize=15)
  return fig
