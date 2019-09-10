import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from skimage.measure import compare_psnr
import tensorflow as tf
from data.dataset import Dataset
import data.data_selector as ds
import utils.data_processing as dp
import utils.plot_functions as pf
import analysis.analysis_picker as ap

def plot_goup_iso_contours(analyzer_list, neuron_indices, orth_indices, num_levels, x_range, y_range, show_contours=True, figsize=None, dpi=100, fontsize=12):
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

def plot_curvature_histograms(hist_list, label_list, color_list, bin_centers, label_loc, title, xlabel, figsize=None, dpi=100, fontsize=12):
  fig = plt.figure(figsize=figsize, dpi=dpi)
  num_y_plots = len(hist_list)
  gs0 = gridspec.GridSpec(num_y_plots, 1, hspace=0.05)
  axes = []
  axes.append(fig.add_subplot(gs0[0]))
  for plt_idx in range(1, num_y_plots):
    axes.append(fig.add_subplot(gs0[plt_idx], sharey=axes[0]))
  handles = []
  labels = []
  max_val = 0
  for axis_index, (axis_hists, axis_colors, axis_labels) in enumerate(zip(hist_list, color_list, label_list)):
    for hist, color, label in zip(axis_hists, axis_colors, axis_labels):
      axes[axis_index].plot(bin_centers, hist, color=color, linestyle="-", drawstyle="steps-mid", label=label)
      axes[axis_index].set_yscale('log')
      if np.max(hist) > max_val:
        max_val = np.max(hist)
    axes[axis_index].axvline(0.0, color='k', linestyle='dashed', linewidth=1)
    for tick in axes[axis_index].xaxis.get_major_ticks():
      tick.label.set_fontsize(fontsize) 
    for tick in axes[axis_index].yaxis.get_major_ticks():
      tick.label.set_fontsize(fontsize) 
    axes[axis_index].set_ylabel("Normalized Count", fontsize=fontsize)
    ax_handles, ax_labels = axes[axis_index].get_legend_handles_labels()
    handles += ax_handles
    labels += ax_labels

  #axes[0].set_ylim([0.0, max_val+1])
  #for axis_index in range(num_y_plots):
  #  ticks = range(0, int(max_val), int(max_val/4))
  #  axes[axis_index].set_yticks(ticks, minor=False)

  axes[0].set_xticks(bin_centers, minor=True)
  axes[0].set_xticks([], minor=False)
  axes[-1].set_xticks(bin_centers, minor=True)
  axes[-1].set_xticks(bin_centers[::int(len(bin_centers)/5)], minor=False)
  axes[-1].xaxis.set_major_formatter(FormatStrFormatter("%0.3f"))

  axes[0].set_title(title, fontsize=fontsize)
  axes[-1].set_xlabel(xlabel, fontsize=fontsize)

  legend = axes[1].legend(handles=handles, labels=labels, fontsize=fontsize,
    borderaxespad=0., bbox_to_anchor=label_loc, framealpha=0.0)
  legend.get_frame().set_linewidth(0.0)
  for text, color in zip(legend.get_texts(), [color for sublist in color_list for color in sublist]):
    text.set_color(color)
  for item in legend.legendHandles:
    item.set_visible(False)
  plt.show()
  return fig