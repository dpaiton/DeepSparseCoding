import re
import numpy as np
from skimage.measure import compare_psnr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.font_manager
import tensorflow as tf
from data.dataset import Dataset
import data.data_selector as ds
import utils.data_processing as dp
import utils.plot_functions as pf
import analysis.analysis_picker as ap

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_goup_iso_contours(analyzer_list, neuron_indices, orth_indices, num_levels, x_range, y_range, show_contours=True,
                           figsize=None, dpi=100, fontsize=12):
  num_models = len(analyzer_list)
  num_plots_y = np.int32(np.ceil(np.sqrt(num_models)))
  num_plots_x = np.int32(np.ceil(np.sqrt(num_models)))
  gs0 = gridspec.GridSpec(num_plots_y, num_plots_x, wspace=-0.1)
  vmin = np.min([np.min(analyzer.comp_activations) for analyzer in analyzer_list])
  vmax = np.max([np.max(analyzer.comp_activations) for analyzer in analyzer_list])
  levels = np.linspace(vmin, vmax, num_levels)
  cmap = plt.get_cmap("cividis")
  cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
  scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
  fig = plt.figure(figsize=figsize, dpi=dpi)
  contour_handles = []
  curve_axes = []
  analyzer_index = 0
  for plot_id in np.ndindex((num_plots_y, num_plots_x)):
    (y_id, x_id) = plot_id
    if type(neuron_indices) == list:
      analyzer_neuron_index = neuron_indices[analyzer_index]
    else:
      analyzer_neuron_index = neuron_indices
    if type(orth_indices) == list:
      analyzer_orth_index = orth_indices[analyzer_index]
    else:
      analyzer_orth_index = orth_indices
    analyzer = analyzer_list[analyzer_index]
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 1, gs0[plot_id])
    curve_axes.append(pf.clear_axis(fig.add_subplot(inner_gs[0])))
    curve_axes[-1].set_title(analyzer.analysis_params.display_name, fontsize=fontsize)
    # plot colored mesh points
    norm_activity = analyzer.comp_activations[analyzer_neuron_index, analyzer_orth_index, ...]
    x_mesh, y_mesh = np.meshgrid(analyzer.comp_contour_dataset["x_pts"], analyzer.comp_contour_dataset["y_pts"])
    if show_contours:
      contsf = curve_axes[-1].contourf(x_mesh, y_mesh, norm_activity,
        levels=levels, vmin=vmin, vmax=vmax, alpha=1.0, antialiased=True, cmap=cmap)
    else:
      contsf = curve_axes[-1].scatter(x_mesh, y_mesh,
        vmin=vmin, vmax=vmax, cmap=cmap, marker="s", alpha=1.0, c=norm_activity, s=30.0)
    contour_handles.append(contsf)
    # plot target neuron arrow & label
    proj_target = analyzer.comp_contour_dataset["proj_target_neuron"][analyzer_neuron_index][analyzer_orth_index]
    target_vector_x = proj_target[0].item()
    target_vector_y = proj_target[1].item()
    curve_axes[-1].arrow(0, 0, target_vector_x, target_vector_y,
      width=0.00, head_width=0.15, head_length=0.15, fc='k', ec='k',
      linestyle='-', linewidth=3)
    tenth_range_shift = ((max(x_range) - min(x_range))/10) # For shifting labels
    text_handle = curve_axes[-1].text(target_vector_x+(tenth_range_shift*0.3), target_vector_y+(tenth_range_shift*0.7),
      r"$\Phi_{k}$", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')
    # plot comparison neuron arrow & label 
    proj_comparison = analyzer.comp_contour_dataset["proj_comparison_neuron"][analyzer_neuron_index][analyzer_orth_index]
    comparison_vector_x = proj_comparison[0].item()
    comparison_vector_y = proj_comparison[1].item()
    curve_axes[-1].arrow(0, 0, comparison_vector_x, comparison_vector_y,
      width=0.00, head_width=0.15, head_length=0.15, fc='k', ec='k',
      linestyle="-", linewidth=3)
    text_handle = curve_axes[-1].text(comparison_vector_x+(tenth_range_shift*0.3), comparison_vector_y+(tenth_range_shift*0.7),
      r"$\Phi_{j}$", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')
    # Plot all other comparison neurons TODO: add flag to optionally do this
    #for proj_alt in analyzer.comp_contour_dataset["proj_comparison_neuron"][analyzer_neuron_index]:
    #  if not np.all(proj_alt == proj_comparison):
    #    curve_axes[-1].arrow(0, 0, proj_alt[0].item(), proj_alt[1].item(),
    #      width=0.00, head_width=0.15, head_length=0.15, fc='w', ec='w',
    #      linestyle="dashed", linewidth=1.0, alpha=0.9)
    # Plot orthogonal vector Nu
    proj_orth = analyzer.comp_contour_dataset["proj_orth_vect"][analyzer_neuron_index][analyzer_orth_index]
    orth_vector_x = proj_orth[0].item()
    orth_vector_y = proj_orth[1].item()
    curve_axes[-1].arrow(0, 0, orth_vector_x, orth_vector_y,
      width=0.00, head_width=0.10, head_length=0.10, fc='k', ec='k',
      linestyle="-", linewidth=3)
    text_handle = curve_axes[-1].text(orth_vector_x+(tenth_range_shift*0.3), orth_vector_y+(tenth_range_shift*0.7),
      r"$\nu$", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')
    # Plot axes
    curve_axes[-1].set_aspect("equal")
    curve_axes[-1].plot(x_range, [0,0], color='k')
    curve_axes[-1].plot([0,0], y_range, color='k')
    # Include basis function image - note, need to change number of plots for inner_gs for this code
    #gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, inner_gs[1], wspace=0.0, hspace=0.5)#-0.55)
    #target_vect_ax = pf.clear_axis(fig.add_subplot(gs2[0]))
    #target_vect_ax.imshow(analyzer.bf_stats["basis_functions"][analyzer.target_neuron_ids[0]], cmap="Greys_r")
    #target_vect_ax.set_title("Primary\nBasis Function", color='r', fontsize=16)
    #comparison_vect_ax = pf.clear_axis(fig.add_subplot(gs2[1]))
    #comparison_vect_ax.imshow(analyzer.bf_stats["basis_functions"][analyzer.comparison_neuron_ids[0][0]], cmap="Greys_r")
    #comparison_vect_ax.set_title("Comparison\nBasis Function", color='k', fontsize=16)
    analyzer_index += 1
  # Add colorbar
  scalarMap._A = []
  cbar_ax = inset_axes(curve_axes[1],
    width="5%",
    height="100%",
    loc='lower left',
    bbox_to_anchor=(1.05, 0., 1, 1),
    bbox_transform=curve_axes[1].transAxes,
    borderpad=0,
    )
  cbar = fig.colorbar(scalarMap, cax=cbar_ax, ticks=[vmin, vmax])
  cbar.ax.tick_params(labelleft=False, labelright=True, left=False, right=True, labelsize=fontsize)
  cbar.ax.set_yticklabels(["{:.0f}".format(vmin), "{:.0f}".format(vmax)])
  plt.show()
  return fig, contour_handles

def plot_bf_curvature(analyzer, target_neuron_index=0, line_alpha=0.5, figsize=None, dpi=100, fontsize=12):
  fig = plt.figure(figsize=figsize, dpi=dpi)
  gs0 = gridspec.GridSpec(1, 2, hspace=0.0, wspace=0.5)
  axes = [fig.add_subplot(gs0[idx]) for idx in range(2)]
  for orth_index in range(analyzer.num_comparison_vectors):#analyzer.num_neurons-1):
    axes[0].plot(analyzer.sliced_datapoints[:, 1], analyzer.comp_sliced_activity[target_neuron_index][orth_index],
      color='b', alpha=line_alpha)
  num_rand_orth = np.minimum(analyzer.num_comparison_vectors, analyzer.num_pixels-1)
  for orth_index in range(num_rand_orth):
    axes[1].plot(analyzer.sliced_datapoints[:, 1], analyzer.rand_sliced_activity[target_neuron_index][orth_index],
      color='b', alpha=line_alpha)
  for ax, title in zip(axes, ["Basis Projection", "Random Projection"]):
    ax.set_title(title, y=1.03, fontsize=fontsize)
    ax.set_ylabel("Normalized Activation", fontsize=fontsize)
    ax.set_xlabel("Distance from Basis Function", fontsize=fontsize)
    ax.grid(True)
    ax.set_ylim([0.0, 1.0])
    x_vals = analyzer.sliced_datapoints[:,1]
    ax.set_xlim([np.min(x_vals), np.max(x_vals)])
    for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(14) 
    for tick in ax.yaxis.get_major_ticks():
      tick.label.set_fontsize(14) 
    ax.set_aspect((np.max(x_vals)-np.min(x_vals)))
    ax.tick_params(labelsize=14)
  fig.suptitle("Normalized Responses to Orthogonal Inputs\n"+analyzer.analysis_params.display_name,
    y=0.63, x=0.5, fontsize=fontsize)
  plt.show()
  return fig

def plot_fit_curvature(analyzer, target_neuron_index=0, line_alpha=0.5, figsize=None, dpi=100, fontsize=12):
  fig = plt.figure(figsize=figsize, dpi=dpi)
  gs0 = gridspec.GridSpec(1, 2, hspace=0.0, wspace=0.5)
  axes = [fig.add_subplot(gs0[idx]) for idx in range(2)]
  for orth_index in range(analyzer.num_comparison_vectors):#analyzer.num_neurons-1):
    axes[0].plot(analyzer.sliced_datapoints[:,1], analyzer.comp_fits[target_neuron_index][orth_index],
      color='r', alpha=line_alpha)
  num_rand_orth = np.minimum(analyzer.num_comparison_vectors, analyzer.num_pixels-1)
  for orth_index in range(num_rand_orth):
    axes[1].plot(analyzer.sliced_datapoints[:,1], analyzer.rand_fits[target_neuron_index][orth_index],
      color='r', alpha=line_alpha)
  for ax, title in zip(axes, ["Basis Projection", "Random Projection"]):
    ax.set_title(title, y=1.03, fontsize=fontsize)
    ax.set_ylabel("Normalized Activation", fontsize=fontsize)
    ax.set_xlabel("Distance from Basis Function", fontsize=fontsize)
    ax.grid(True)
    ax.set_ylim([0.0, 1.0])
    x_vals = analyzer.sliced_datapoints[:,1]
    ax.set_xlim([np.min(x_vals), np.max(x_vals)])
    for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(14) 
    for tick in ax.yaxis.get_major_ticks():
      tick.label.set_fontsize(14) 
    ax.set_aspect((np.max(x_vals)-np.min(x_vals)))
    ax.tick_params(labelsize=14)
  fig.suptitle("Polynomial Fit to Orthogonal Inputs\n"+analyzer.analysis_params.display_name+"\n",
    y=0.63, x=0.5, fontsize=fontsize)
  plt.show()
  return fig

def plot_curvature_histograms(activity, contour_pts, contour_angle, contour_text_loc, hist_list, label_list, color_list,
                              bin_centers, title, xlabel, figsize=None, dpi=100, fontsize=12):
  fig = plt.figure(figsize=figsize, dpi=dpi)
  num_y_plots = 4
  num_x_plots = 4
  gs0 = gridspec.GridSpec(num_y_plots, num_x_plots, wspace=0.5)

  x, y = contour_pts
  x_mesh, y_mesh = np.meshgrid(*contour_pts)
  curve_ax = fig.add_subplot(gs0[:, 0:2], projection='3d')
  curve_ax.set_zlim(0,1)
  curve_ax.set_xlim3d(5,200)
  x_ticks = curve_ax.get_xticks().tolist()
  x_ticks = np.round(np.linspace(0.6, 4.4, 11), 1).astype(str)
  a_x = [" "]*len(x_ticks)
  a_x[1] = x_ticks[1]
  a_x[-3] = x_ticks[-2]
  curve_ax.set_xticklabels(a_x, size=fontsize)
  y_ticks = curve_ax.get_yticks().tolist()
  y_ticks = np.round(np.linspace(-10, 10, 11), 1).astype(str)
  a_y = [" "]*len(y_ticks)
  a_y[1] = y_ticks[1]
  a_y[-2] = y_ticks[-2]
  curve_ax.set_yticklabels(a_y, size=fontsize)
  curve_ax.set_zticklabels([])
  curve_ax.zaxis.set_rotate_label(False)
  curve_ax.set_zlabel("Normalized Activity", rotation=90, size=fontsize)
  curve_ax.scatter(x_mesh, y_mesh, activity, color="#A9A9A9",s=0.05)
  loc0, loc1, loc2 = contour_text_loc[0]
  curve_ax.text(loc0, loc1, loc2, "Iso-\nresponse", color='black', size=fontsize)
  iso_line_offset = 165
  curve_ax.plot(np.zeros_like(x)+iso_line_offset, y, activity[:, iso_line_offset], color="black", lw=5)
  v = Arrow3D([-200/3., -200/3.], [200/2., 200/2.+200/16.], 
              [0, 0.0], mutation_scale=10, 
              lw=2, arrowstyle="-|>", color="r", linestyle="dashed")
  curve_ax.add_artist(v)
  curve_ax.text(-270/3., 300/3.0, 0.0, "v", color='red', size=fontsize)
  phi_k = Arrow3D([-200/3., 0.], [200/2., 200/2.], 
              [0, 0.0], mutation_scale=10, 
              lw=2, arrowstyle="-|>", color="r", linestyle = "dashed")
  curve_ax.add_artist(phi_k)
  curve_ax.text(-175/3., 270/3.0, 0.0, r"${\phi}_{k}$", color='red', size=fontsize)
  loc0, loc1, loc2 = contour_text_loc[1]
  curve_ax.text(loc0, loc1, loc2, "Response\nAttenuation", color='black', size=fontsize)
  lines = np.array([0.2, 0.203, 0.197]) - 0.1
  for i in lines:
      curve_ax.contour3D(x_mesh, y_mesh, activity, [i], colors="black")
  curve_ax.view_init(30, contour_angle)
  
  sub_num_y_plots = 4#len(hist_list) #2
  sub_num_x_plots = 1#len(hist_list[0]) #2
  hist_gs = gridspec.GridSpecFromSubplotSpec(sub_num_y_plots, sub_num_x_plots, gs0[:, 2:], hspace=0.40, wspace=0.15)
  all_lists = zip(hist_list, label_list, color_list, bin_centers, title, xlabel)
  orig_ax = fig.add_subplot(hist_gs[0])
  axes = []
  axis_index = 0
  for sub_plt_x in range(0, sub_num_x_plots):
    for sub_plt_y in range(0, sub_num_y_plots):
      if (sub_plt_x, sub_plt_y) == (0,0):
        axes.append(orig_ax)
      else:
        #axes.append(fig.add_subplot(hist_gs[sub_plt_y, sub_plt_x], sharey=orig_ax))
        axes.append(fig.add_subplot(hist_gs[axis_index], sharey=orig_ax))
      axis_index += 1
  axis_index = 0
  for axis_x, (sub_hist, sub_label, sub_color, sub_bins, sub_title, sub_xlabel) in enumerate(all_lists):
    handles = []
    labels = []
    max_val = 0
    for axis_y, (axis_hists, axis_labels, axis_colors) in enumerate(zip(sub_hist, sub_label, sub_color)):
      axes[axis_index].set_xticks(sub_bins, minor=True)
      axes[axis_index].set_xticks(sub_bins[::int(len(sub_bins)/5)], minor=False)
      axes[axis_index].set_xlabel(sub_xlabel, fontsize=fontsize)
      axes[axis_index].xaxis.set_major_formatter(FormatStrFormatter("%0.3f"))
      for hist, label, color in zip(axis_hists, axis_labels, axis_colors):
        axes[axis_index].plot(sub_bins, hist, color=color, linestyle="-", drawstyle="steps-mid", label=label)
        axes[axis_index].set_yscale('log')
        if np.max(hist) > max_val:
          max_val = np.max(hist)
      axes[axis_index].axvline(0.0, color='k', linestyle='dashed', linewidth=1)
      for tick in axes[axis_index].xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
      for tick in axes[axis_index].yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
      axes[axis_index].set_ylabel("Normalized\nCount", fontsize=fontsize)
      ax_handles, ax_labels = axes[axis_index].get_legend_handles_labels()
      handles += ax_handles
      labels += ax_labels
      if axis_x == 0 and axis_y == 1:#sub_num_y_plots-1:
        legend = axes[axis_index].legend(handles=handles, labels=labels, fontsize=fontsize, loc="upper right",
          borderaxespad=0.5, borderpad=0., ncol=2)
        legend.get_frame().set_linewidth(0.0)
        for text, color in zip(legend.get_texts(), [color for sublist in sub_color for color in sublist]):
          text.set_color(color)
        for item in legend.legendHandles:
          item.set_visible(False)
      axis_index += 1
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
        contrast = 1.0#contrasts[co_idx]
        activity = activations[bf_indices[bf_idx], co_idx, :]
        color_val = scalarMap.to_rgba(contrast)
        ax.plot(orientations, activity, linewidth=1, color=color_val)
        ax.scatter(orientations, activity, s=4, c=[color_val])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2g'))
        ax.set_yticks([0, np.max(activity)])
        ax.set_xticks([0, 90, 180])
      bf_idx += 1
    else:
      ax = pf.clear_axis(ax, spines="none")
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
  num_plots_y = int(np.floor(np.sqrt(num_plots)))
  num_plots_x = int(np.ceil(np.sqrt(num_plots)))
  fig, sub_ax = plt.subplots(num_plots_y, num_plots_x, figsize=figsize)
  filter_total = 0
  for plot_id in  np.ndindex((num_plots_y, num_plots_x)):
    if filter_total < num_plots:
      sub_ax[plot_id].imshow(np.squeeze(weights[filter_total, ...]), vmin=vmin, vmax=vmax, cmap="Greys_r")
      filter_total += 1
    pf.clear_axis(sub_ax[plot_id])
    sub_ax[plot_id].set_aspect("equal")
  fig.suptitle(title, y=0.95, x=0.5, fontsize=20)
  if save_filename is not None:
      fig.savefig(save_filename)
      plt.close(fig)
      return None
  plt.show()
  return fig

def center_curve(tuning_curve):
  """
  Centers a curve about its preferred orientation
  """
  return np.roll(tuning_curve, (len(tuning_curve) // 2) - np.argmax(tuning_curve))

def compute_fwhm(centered_ot_curve, corresponding_angles_deg):
  """
  Calculates the full width at half maximum of the tuning curve

  Result is expressed in degrees to make it a little more intuitive. The curve
  is often NOT symmetric about the maximum value so we don't do any fitting and
  we return the FULL width

  Parameters
  ----------
  centered_ot_curve : ndarray
      A 1d array of floats giving the value of the ot curve, at an orientation
      relative to the *preferred orientation* which is given by the angles in
      corresponding_angles_deg. This has the maximum orientation in the
      center of the array which is nicer for visualization.
  corresponding_angles_deg : ndarray
      The orientations relative to preferred orientation that correspond to
      the values in centered_ot_curve

  Returns
  -------
  half_max_left : float
      The position of the intercept to the left of the max
  half_max_right : float
      The position of the intercept to the right of the max
  half_max_value : float
      Mainly for plotting purposes, the actual curve value that corresponds
      to the left and right points
  """
  max_idx = np.argmax(centered_ot_curve)
  min_idx = np.argmin(centered_ot_curve)
  max_val = centered_ot_curve[max_idx]
  min_val = centered_ot_curve[min_idx]
  midpoint = (max_val / 2) + (min_val / 2)
  # find the left hand point
  idx = max_idx
  while centered_ot_curve[idx] > midpoint:
    idx -= 1
    if idx == -1:
      # the width is *at least* 90 degrees
      half_max_left = -90.
      break
  if idx > -1:
    # we'll linearly interpolate between the two straddling points
    # if (x2, y2) is the coordinate of the point below the half-max and
    # (x1, y1) is the point above the half-max, then we can solve for x3, the
    # x-position of the point that corresponds to the half-max on the line
    # that connects (x1, y1) and (x2, y2)
    half_max_left = (((midpoint - centered_ot_curve[idx])
      * (corresponding_angles_deg[idx+1] - corresponding_angles_deg[idx])
      / (centered_ot_curve[idx+1] - centered_ot_curve[idx]))
      + corresponding_angles_deg[idx])
  # find the right hand point
  idx = max_idx
  while centered_ot_curve[idx] > midpoint:
    idx += 1
    if idx == len(centered_ot_curve):
      # the width is *at least* 90
      half_max_right = 90.
      break
  if idx < len(centered_ot_curve):
    # we'll linearly interpolate between the two straddling points again
    half_max_right = (((midpoint - centered_ot_curve[idx-1])
      * (corresponding_angles_deg[idx] - corresponding_angles_deg[idx-1])
      / (centered_ot_curve[idx] - centered_ot_curve[idx-1]))
      + corresponding_angles_deg[idx-1])
  return half_max_left, half_max_right, midpoint

def compute_circ_var(centered_ot_curve, corresponding_angles_rad):
  """
  From
  DL Ringach, RM Shapley, MJ Hawken (2002) - Orientation Selectivity in Macaque V1:
  Diversity and Laminar Dependence
  
  Computes the circular variance of a tuning curve and returns vals for plotting

  This is a scale-invariant measure of how 'oriented' a curve is in some
  global sense. It wraps reponses around the unit circle and then sums their
  vectors, resulting in an average vector, the magnitude of which indicates
  the strength of the tuning. Circular variance is an index of 'orientedness'
  that falls in the interval [0.0, 1.0], with 0.0 indicating a delta function
  and 1.0 indicating a completely flat tuning curve.

  Parameters
  ----------
  centered_ot_curve : ndarray
      A 1d array of floats giving the value of the ot curve, at an orientation
      relative to the *preferred orientation* which is given by the angles in
      corresponding_angles_rad. This has the maximum orientation in the
      center of the array which is nicer for visualization.
  corresponding_angles_rad : ndarray
      The orientations relative to preferred orientation that correspond to
      the values in centered_ot_curve

  Returns
  -------
  numerator_sum_components : ndarray
      The complex values the are produced from r * np.exp(j*2*theta). These
      are the elements that get summed up in the numerator
  direction_vector : complex64 or complex128
      This is the vector that points in the direction of *aggregate* tuning.
      its magnitude is upper bounded by 1.0 which is the case when only one
      orientation has a nonzero value. We can plot it to get an idea of how
      tuned a curve is
  circular_variance : float
      This is 1 minus the magnitude of the direction vector. It represents and
      index of 'global selectivity'
  """
  # in the original definition, angles are [0, 2*np.pi] so the factor of 2
  # in the exponential wraps the phase twice around the complex circle,
  # placing responses that correspond to angles pi degrees apart
  # onto the same place. We know there's a redudancy in our responses at pi
  # offsets so our responses get wrapped around the unit circle once.
  numerator_sum_components = (centered_ot_curve
    * np.exp(1j * 2 * corresponding_angles_rad))
  direction_vector = (np.sum(numerator_sum_components)
    / np.sum(centered_ot_curve))
  circular_variance = 1 - np.abs(direction_vector)
  return (numerator_sum_components, direction_vector, circular_variance)

def compute_osi(centered_ot_curve):
  """
  Compute the Orientation Selectivity Index.

  This is the most coarse but popular measure of selectivity. It really
  doesn't tell you much. It just measures the maximum response relative to
  the minimum response.

  Parameters
  ----------
  centered_ot_curve : ndarray
      A 1d array of floats giving the value of the ot curve, at an orientation
      relative to the *preferred orientation*

  Returns
  -------
  osi : float
      This is (a_max - a_orth) / (a_max + a_orth) where a_max is the maximum
      response across orientations when orientation responses are
      *averages* over phase. a_orth is the orientation which is orthogonal to
      the orientation which produces a_max.
  """
  max_val = np.max(centered_ot_curve)
  # Assume that orthogonal orientation is at either end of the curve modulo 1
  # bin (if we had like an even number of orientation values)
  orth_val = centered_ot_curve[0]
  osi = (max_val - orth_val) / (max_val + orth_val)
  return osi

def plot_circular_variance(cv_data, max_bfs_per_fig=400, title="", save_filename=None):
  assert np.sqrt(max_bfs_per_fig) % 1 == 0, "Pick a square number for max_bfs_per_fig"
  orientations = (np.pi * np.arange(len(cv_data))
    / len(cv_data)) - (np.pi/2) # relative to preferred
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
    plt.suptitle(title + ', fig {} of {}'.format(
      in_bf_fig_idx+1, num_bf_figs), fontsize=20)
    subplot_grid = gridspec.GridSpec(plot_sidelength, plot_sidelength,
      wspace=0.4, hspace=0.4)
    fig_bf_idx = bf_idx % bfs_per_fig
    while fig_bf_idx < bfs_per_fig and bf_idx < num_bfs:
      #if bf_idx % 100 == 0:
      #  print("plotted ", bf_idx, " of ", num_bfs, " circular variance plots")
      ## print("sum vector: ", np.real(cv_data[bf_idx][1]), np.imag(cv_data[bf_idx][1]))
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
    if save_filename is not None:
      filename_split = os.path.split(save_filename)
      save_filename = filename_split[0]+str(in_bf_fig_idx).zfill(2)+"_"+filename_split[1]
      fig.savefig(save_filename)
      plt.close(fig)
      bf_figs.append(None)
    else:
      bf_figs.append(fig)
  if save_filename is None:
    plt.show()
  return bf_figs

def plot_circular_variance_histogram(variances_list, label_list, num_bins=50, y_max=None,
                                     fontsize=18, figsize=None, save_filename=None):
  variance_min = np.min([np.min(var) for var in variances_list])#0.0
  variance_max = np.max([np.max(var) for var in variances_list])#1.0
  bins = np.linspace(variance_min, variance_max, num_bins)
  bar_width = np.diff(bins).min()
  fig, ax = plt.subplots(1, figsize=figsize)
  hist_list = []
  handles = []
  for variances, label in zip(variances_list, label_list):
    hist, bin_edges = np.histogram(variances.flatten(), bins)
    #hist = hist / np.max(hist)
    hist_list.append(hist)
    bin_left, bin_right = bin_edges[:-1], bin_edges[1:]
    bin_centers = bin_left + (bin_right - bin_left)/2
    handles.append(ax.bar(bin_centers, hist, width=bar_width, log=True, align="center", alpha=0.5, label=label))
  ax.set_xticks(bin_left, minor=True)
  ax.set_xticks(bin_left[::4], minor=False)
  ax.xaxis.set_major_formatter(FormatStrFormatter("%0.0f"))
  ax.tick_params("both", labelsize=16)
  ax.set_xlim([variance_min, variance_max])
  ax.set_xticks([variance_min, variance_max])
  ax.set_xticklabels(["More selective", "Less selective"])
  ticks = ax.xaxis.get_major_ticks()
  ticks[0].label1.set_horizontalalignment("left")
  ticks[1].label1.set_horizontalalignment("right")
  if y_max is None:
    # Round up to the nearest power of 10
    y_max = 10**(np.ceil(np.log10(np.max([np.max(hist) for hist in hist_list]))))
  ax.set_ylim([1, y_max])
  ax.set_title("Circular Variance Histogram", fontsize=fontsize)
  ax.set_xlabel("Selectivity", fontsize=fontsize)
  ax.set_ylabel("Log Count", fontsize=fontsize)
  legend = ax.legend(handles, label_list, fontsize=fontsize, #ncol=len(label_list),
    borderaxespad=0., bbox_to_anchor=[0.98, 0.98], fancybox=True, loc="upper right")
  if save_filename is not None:
    fig.savefig(save_filename)
    plt.close(fig)
    return None
  plt.show()
  return fig

def plot_circ_variance_histogram(analyzer_list, circ_var_list, color_list, label_list, num_bins, density, width_ratios,
                                 height_ratios, fontsize, figsize, dpi):
  fig = plt.figure(figsize=figsize, dpi=dpi)
  gs0 = gridspec.GridSpec(1, 3, width_ratios=width_ratios)
  axes = []
  
  gs_hist = gridspec.GridSpecFromSubplotSpec(4, 1, gs0[0], height_ratios=height_ratios)
  axes.append(fig.add_subplot(gs_hist[1:3, 0]))
  variance_min = 0.0
  variance_max = 1.0
  bins = np.linspace(variance_min, variance_max, num_bins)
  bar_width = np.diff(bins).min()
  hist_list = []
  for variances, label, color in zip(circ_var_list, label_list, color_list):
    hist, bin_edges = np.histogram(variances.flatten(), bins, density=density)
    hist_list.append(hist)
    bin_left, bin_right = bin_edges[:-1], bin_edges[1:]
    bin_centers = bin_left + (bin_right - bin_left)/2
    axes[-1].plot(bin_centers, hist, linestyle="-", drawstyle="steps-mid", color=color, label=label)
  axes[-1].set_xticks(bin_left, minor=True)
  axes[-1].set_xticks(bin_left[::4], minor=False)
  axes[-1].xaxis.set_major_formatter(FormatStrFormatter("%0.0f"))
  axes[-1].tick_params("both", labelsize=fontsize)
  axes[-1].set_xlim([variance_min, variance_max])
  axes[-1].set_xticks([variance_min, variance_max])
  axes[-1].set_xticklabels(["More\nselective", "Less\nselective"])
  ticks = axes[-1].xaxis.get_major_ticks()
  ticks[0].label1.set_horizontalalignment("left")
  ticks[1].label1.set_horizontalalignment("right")
  y_max = np.max([np.max(hist) for hist in hist_list])
  axes[-1].set_ylim([0, y_max+1])
  axes[-1].set_title("Circular Variance", fontsize=fontsize)
  if density:
    axes[-1].set_ylabel("Density", fontsize=fontsize)
  else:
    axes[-1].set_ylabel("Count", fontsize=fontsize)
  handles, labels = axes[-1].get_legend_handles_labels()
  legend = axes[-1].legend(handles=handles, labels=labels, fontsize=fontsize,
    borderaxespad=0., framealpha=0.0, loc="upper right")
  legend.get_frame().set_linewidth(0.0)
  for text, color in zip(legend.get_texts(), color_list):
    text.set_color(color)
  for item in legend.legendHandles:
    item.set_visible(False)
  gs_weights = gridspec.GridSpecFromSubplotSpec(len(analyzer_list), 1, gs0[1], hspace=-0.6)
  for gs_idx, analyzer in enumerate(analyzer_list):
    weights = np.stack(analyzer.bf_stats["basis_functions"], axis=0)[analyzer.bf_indices, ...]
    weights = dp.norm_weights(weights)
    vmin = np.min(weights)
    vmax = np.max(weights)
    num_plots = weights.shape[0]
    num_plots_y = int(np.ceil(np.sqrt(num_plots)))
    num_plots_x = int(np.ceil(np.sqrt(num_plots)))
    gs_weights_inner = gridspec.GridSpecFromSubplotSpec(num_plots_y, num_plots_x, gs_weights[gs_idx],
      hspace=-0.85)
    bf_idx = 0
    for plot_id in  np.ndindex((num_plots_y, num_plots_x)):
      if bf_idx < num_plots:
        axes.append(fig.add_subplot(gs_weights_inner[plot_id]))
        axes[-1].imshow(np.squeeze(weights[bf_idx, ...]), vmin=vmin, vmax=vmax, cmap="Greys_r")
        bf_idx += 1
      pf.clear_axis(axes[-1])
  gs_tuning = gridspec.GridSpecFromSubplotSpec(len(analyzer_list), 1, gs0[2], hspace=-0.6)
  for analyzer_idx, analyzer in enumerate(analyzer_list):
    contrasts = analyzer.ot_grating_responses["contrasts"]
    orientations = analyzer.ot_grating_responses["orientations"]
    activations = analyzer.ot_grating_responses["mean_responses"]
    activations = activations / np.max(activations[analyzer.bf_indices, -1, ...])
    orientations = np.asarray(orientations)*(180/np.pi) #convert to degrees for plotting
    orientations = orientations / np.max(orientations)
    num_plots = len(analyzer.bf_indices)
    num_plots_y = int(np.ceil(np.sqrt(num_plots)))
    num_plots_x = int(np.ceil(np.sqrt(num_plots)))
    gs_tuning_inner = gridspec.GridSpecFromSubplotSpec(num_plots_y, num_plots_x, gs_tuning[analyzer_idx],
        hspace=-0.85)
    bf_idx = 0
    for plot_id in np.ndindex((num_plots_y, num_plots_x)):
      if bf_idx < num_plots:
        if bf_idx == 0:
          axes.append(fig.add_subplot(gs_tuning_inner[plot_id]))
          ax_orig_id = len(axes)-1
        else:
          axes.append(fig.add_subplot(gs_tuning_inner[plot_id], sharey=axes[ax_orig_id], sharex=axes[ax_orig_id]))
        contrast_idx = -1
        activity = activations[analyzer.bf_indices[bf_idx], contrast_idx, :]
        axes[-1].plot(orientations, activity, linewidth=0.5, color='k')
        axes[-1].scatter(orientations, activity, s=0.1, c='k')
        axes[-1].set_aspect('equal', adjustable='box')
        axes[-1].set_yticks([])
        axes[-1].set_xticks([])
        bf_idx += 1
      (y_id, x_id) = plot_id
      if y_id == 0 and x_id == 0:
        plt.text(x=0.1, y=1.4, s=analyzer.analysis_params.display_name, horizontalalignment='center',
          verticalalignment='center', transform=axes[-1].transAxes, fontsize=fontsize)
  #gs_circvar = gridspec.GridSpecFromSubplotSpec(len(analyzer_list), 1, gs0[3])#, hspace=-0.5)
  #for analyzer_index, analyzer in enumerate(analyzer_list):
  #  cv_data = [val for index, val in enumerate(analyzer.metrics_list["circ_var"]) if index in bf_indices]
  #  orientations = (np.pi * np.arange(len(cv_data)) / len(cv_data)) - (np.pi/2) # relative to preferred
  #  num_bfs = len(cv_data)
  #  num_plots_y = np.int32(np.ceil(np.sqrt(num_bfs)))+1
  #  num_plots_x = np.int32(np.ceil(np.sqrt(num_bfs)))
  #  gs_circvar_inner = gridspec.GridSpecFromSubplotSpec(num_plots_y, num_plots_x, gs_circvar[analyzer_index],
  #    wspace=0.4, hspace=0.4)
  #  bf_idx = 0
  #  for plot_id in np.ndindex((num_plots_y, num_plots_x)):
  #    (y_id, x_id) = plot_id
  #    if y_id == 0 and x_id == 0:
  #      axes.append(fig.add_subplot(gs_circvar_inner[plot_id]))
  #      ax00 = axes[-1]
  #    else:
  #      axes.append(fig.add_subplot(gs_circvar_inner[plot_id]))
  #    if bf_idx < num_bfs:
  #      axes[-1].plot(np.real(cv_data[bf_idx][0]), np.imag(cv_data[bf_idx][0]), c='g', linewidth=0.5)
  #      #axes[-1].scatter(np.real(cv_data[bf_idx][0]), np.imag(cv_data[bf_idx][0]), c='g', s=4)
  #      #axes[-1].quiver(np.real(cv_data[bf_idx][1]), np.imag(cv_data[bf_idx][1]),
  #      #          angles='xy', scale_units='xy', scale=1.0, color='b', width=0.01)
  #      #axes[-1].quiver(0.5, 0.5, color='b')
  #      axes[-1].yaxis.set_major_formatter(FormatStrFormatter('%0.2g'))
  #      xaxis_size = max(np.max(np.real(cv_data[bf_idx][0])), 1.0)
  #      yaxis_size = max(np.max(np.imag(cv_data[bf_idx][0])), 1.0)
  #      axes[-1].set_yticks([])#[-1. * yaxis_size, yaxis_size])
  #      axes[-1].set_xticks([])#[-1. * xaxis_size, xaxis_size])
  #      # put the circular variance index in the upper left
  #      #axes[-1].text(0.02, 0.97, '{:.2f}'.format(cv_data[bf_idx][2]),
  #      #        horizontalalignment='left', verticalalignment='top',
  #      #        transform=axes[-1].transAxes, color='b', fontsize=10)
  #      bf_idx += 1
  #    else:
  #      pf.clear_axis(axes[-1])
  plt.show()
  return fig