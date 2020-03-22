import numpy as np
from skimage import measure
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as plticker
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.font_manager

from DeepSparseCoding.data.dataset import Dataset
import DeepSparseCoding.data.data_selector as ds
import DeepSparseCoding.utils.data_processing as dp
import DeepSparseCoding.utils.plot_functions as pf
import DeepSparseCoding.analysis.analysis_picker as ap

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def set_size(width, fraction=1, subplot=[1, 1]):
    """ Set aesthetic figure dimensions to avoid scaling in latex.
    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches

    Usage: figsize = set_size(text_width, fraction=1, subplot=[1, 1])
    Code obtained from: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """
    fig_width_pt = width * fraction # Width of figure
    inches_per_pt = 1 / 72.27 # Convert from pt to inches
    golden_ratio = (5**.5 - 1) / 2 # Golden ratio to set aesthetic figure height
    fig_width_in = fig_width_pt * inches_per_pt # Figure width in inches
    fig_height_in = fig_width_in * golden_ratio * (subplot[0] / subplot[1]) # Figure height in inches
    fig_dim = (fig_width_in, fig_height_in) # Final figure dimensions
    return fig_dim

def plot_group_iso_contours(analyzer_list, neuron_indices, orth_indices, num_levels, x_range, y_range, show_contours=True,
                           text_width=200, width_fraction=1.0, dpi=100):
  arrow_width = 0.0
  arrow_linewidth = 1
  arrow_headsize = 0.15
  arrow_head_length = 0.15
  arrow_head_width = 0.15
  gs0_hspace = 0.5
  gs0_wspace = -0.6
  phi_k_text_x_offset = 0.6 / width_fraction
  phi_k_text_y_offset = -1.2 / width_fraction
  phi_j_text_x_offset = 0.9 / width_fraction
  phi_j_text_y_offset = 0.3 / width_fraction
  nu_text_x_offset = -0.56 / width_fraction
  nu_text_y_offset = 0.3 / width_fraction
  num_models = len(analyzer_list)
  num_plots_y = np.int32(np.ceil(np.sqrt(num_models)))
  num_plots_x = np.int32(np.ceil(np.sqrt(num_models)))
  gs0 = gridspec.GridSpec(num_plots_y, num_plots_x, wspace=gs0_wspace, hspace=gs0_hspace)
  vmin = np.min([np.min(analyzer.comp_activations) for analyzer in analyzer_list])
  vmax = np.max([np.max(analyzer.comp_activations) for analyzer in analyzer_list])
  levels = np.linspace(vmin, vmax, num_levels)
  cmap = plt.get_cmap("cividis")
  cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
  scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
  fig = plt.figure(figsize=set_size(text_width, width_fraction, [num_plots_y, num_plots_x]), dpi=dpi)
  contour_handles = []
  curve_axes = []
  analyzer_index = 0
  for plot_id in np.ndindex((num_plots_y, num_plots_x)):
    if analyzer_index < num_models:
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
      curve_axes[-1].set_title(analyzer.analysis_params.display_name)
      # plot colored mesh points
      norm_activity = analyzer.comp_activations[analyzer_neuron_index, analyzer_orth_index, ...]
      x_mesh, y_mesh = np.meshgrid(analyzer.comp_contour_dataset["x_pts"],
        analyzer.comp_contour_dataset["y_pts"])
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
        width=arrow_width, head_width=arrow_head_width, head_length=arrow_head_length,
        fc='k', ec='k', linestyle='-', linewidth=arrow_linewidth)
      tenth_range_shift = ((max(x_range) - min(x_range))/10) # For shifting labels
      text_handle = curve_axes[-1].text(
        target_vector_x+(tenth_range_shift*phi_k_text_x_offset),
        target_vector_y+(tenth_range_shift*phi_k_text_y_offset),
        r"$\Phi_{k}$", horizontalalignment='center', verticalalignment='center')
      # plot comparison neuron arrow & label
      proj_comparison = analyzer.comp_contour_dataset["proj_comparison_neuron"][analyzer_neuron_index][analyzer_orth_index]
      comparison_vector_x = proj_comparison[0].item()
      comparison_vector_y = proj_comparison[1].item()
      curve_axes[-1].arrow(0, 0, comparison_vector_x, comparison_vector_y,
        width=arrow_width, head_width=arrow_head_width, head_length=arrow_head_length,
        fc='k', ec='k', linestyle="-", linewidth=arrow_linewidth)
      text_handle = curve_axes[-1].text(
        comparison_vector_x+(tenth_range_shift*phi_j_text_x_offset),
        comparison_vector_y+(tenth_range_shift*phi_j_text_y_offset),
        r"$\Phi_{j}$", horizontalalignment='center', verticalalignment='center')
      # Plot all other comparison neurons TODO: add flag to optionally do this
      #for proj_alt in analyzer.comp_contour_dataset["proj_comparison_neuron"][analyzer_neuron_index]:
      #  if not np.all(proj_alt == proj_comparison):
      #    curve_axes[-1].arrow(0, 0, proj_alt[0].item(), proj_alt[1].item(),
      #      width=arrow_width, head_width=arrow_head_width, head_length=arrow_head_length,
      #      fc='w', ec='w', linestyle="dashed", linewidth=1.0, alpha=0.9)
      # Plot orthogonal vector Nu
      proj_orth = analyzer.comp_contour_dataset["proj_orth_vect"][analyzer_neuron_index][analyzer_orth_index]
      orth_vector_x = proj_orth[0].item()
      orth_vector_y = proj_orth[1].item()
      curve_axes[-1].arrow(0, 0, orth_vector_x, orth_vector_y,
        width=arrow_width, head_width=arrow_head_width, head_length=arrow_head_length,
        fc='k', ec='k', linestyle="-", linewidth=arrow_linewidth)
      text_handle = curve_axes[-1].text(
        orth_vector_x+(tenth_range_shift*nu_text_x_offset),
        orth_vector_y+(tenth_range_shift*nu_text_y_offset),
        r"$\nu$", horizontalalignment='center', verticalalignment='center')
      # Plot axes
      curve_axes[-1].set_aspect("equal")
      curve_axes[-1].plot(x_range, [0,0], color='k', linewidth=arrow_linewidth/2)
      curve_axes[-1].plot([0,0], y_range, color='k', linewidth=arrow_linewidth/2)
      # Include basis function image - note, need to change number of plots for inner_gs for this code
      #gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, inner_gs[1], wspace=0.0, hspace=0.5)#-0.55)
      #target_vect_ax = pf.clear_axis(fig.add_subplot(gs2[0]))
      #target_vect_ax.imshow(analyzer.bf_stats["basis_functions"][analyzer.target_neuron_ids[0]],
      #  cmap="Greys_r")
      #target_vect_ax.set_title("Primary\nBasis Function", color='r')
      #comparison_vect_ax = pf.clear_axis(fig.add_subplot(gs2[1]))
      #comparison_vect_ax.imshow(analyzer.bf_stats["basis_functions"][analyzer.comparison_neuron_ids[0][0]],
      #  cmap="Greys_r")
      #comparison_vect_ax.set_title("Comparison\nBasis Function", color='k')
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
  cbar.ax.tick_params(labelleft=False, labelright=True, left=False, right=True)
  cbar.ax.set_yticklabels(["{:.0f}".format(vmin), "{:.0f}".format(vmax)])
  plt.show()
  return fig, contour_handles

def compute_curvature_fits(analyzer_list, target_act):
  for analyzer in analyzer_list:
    analyzer.iso_comp_curvatures = []
    analyzer.iso_rand_curvatures = []
    activations_and_curvatures = ((analyzer.iso_comp_activations, analyzer.iso_comp_curvatures),
      (analyzer.iso_rand_activations, analyzer.iso_rand_curvatures))
    for activations, curvatures in activations_and_curvatures:
      (num_neurons, num_planes, num_points_y, num_points_x) = activations.shape
      analyzer.num_neurons = num_neurons
      for neuron_id in range(analyzer.num_neurons):
        sub_curvatures = []
        for plane_id in range(num_planes):
          activity = activations[neuron_id, plane_id, ...]
          ## mirror top half of activations to bottom half to only measure curvature in the upper quadrant
          num_y, num_x = activity.shape 
          activity[:int(num_y/2), :] = activity[int(num_y/2):, :][::-1,:]
          ## compute curvature
          contours = measure.find_contours(activity, target_act)[0]
          x_vals = contours[:,1]
          y_vals = contours[:,0]
          coeffs = np.polynomial.polynomial.polyfit(y_vals, x_vals, deg=2)
          sub_curvatures.append(coeffs[-1])
        curvatures.append(sub_curvatures)
    comp_x_pts = analyzer.attn_comp_contour_dataset["x_pts"]
    rand_x_pts = analyzer.attn_rand_contour_dataset["x_pts"]
    assert(np.all(comp_x_pts == rand_x_pts)) # This makes sure we don't need to recompute proj_datapoints for each case
    num_x_imgs = len(comp_x_pts)
    x_target = comp_x_pts[num_x_imgs-1] # find a location to take a slice
    proj_datapoints = analyzer.attn_comp_contour_dataset["proj_datapoints"]
    slice_indices = np.where(proj_datapoints[:, 0] == x_target)[0]
    analyzer.sliced_datapoints = proj_datapoints[slice_indices, :][:, :] # slice grid
    analyzer.attn_comp_curvatures = []
    analyzer.attn_comp_fits = []
    analyzer.attn_comp_sliced_activity = []
    analyzer.attn_rand_curvatures = []
    analyzer.attn_rand_fits = []
    analyzer.attn_rand_sliced_activity = []
    for neuron_index in range(analyzer.attn_num_target_neurons):
      sub_comp_curvatures = []
      sub_comp_fits = []
      sub_comp_sliced_activity = []
      sub_comp_delta_activity = []
      sub_rand_curvatures = []
      sub_rand_fits = []
      sub_rand_sliced_activity = []
      for orth_index in range(analyzer.attn_num_comparison_vectors):
        comp_activity = analyzer.attn_comp_activations[neuron_index, orth_index, ...].reshape([-1])
        sub_comp_sliced_activity.append(comp_activity[slice_indices][:])
        coeff = np.polynomial.polynomial.polyfit(analyzer.sliced_datapoints[:, 1],
          sub_comp_sliced_activity[-1], deg=2) # [c0, c1, c2], where p = c0 + c1x + c2x^2
        sub_comp_curvatures.append(-coeff[2]) # multiply by -1 so that positive coeff is "more" curvature
        sub_comp_fits.append(np.polynomial.polynomial.polyval(analyzer.sliced_datapoints[:, 1], coeff))
      num_rand_vectors = np.minimum(analyzer.bf_stats["num_inputs"]-1, analyzer.attn_num_comparison_vectors)
      for orth_index in range(num_rand_vectors):
        rand_activity = analyzer.attn_rand_activations[neuron_index, orth_index, ...].reshape([-1])
        sub_rand_sliced_activity.append(rand_activity[slice_indices][:])
        coeff = np.polynomial.polynomial.polyfit(analyzer.sliced_datapoints[:, 1],
          sub_rand_sliced_activity[-1], deg=2)
        sub_rand_curvatures.append(-coeff[2])
        sub_rand_fits.append(np.polynomial.polynomial.polyval(analyzer.sliced_datapoints[:, 1], coeff))
      analyzer.attn_comp_curvatures.append(sub_comp_curvatures)
      analyzer.attn_comp_fits.append(sub_comp_fits)
      analyzer.attn_comp_sliced_activity.append(sub_comp_sliced_activity)
      analyzer.attn_rand_curvatures.append(sub_rand_curvatures)
      analyzer.attn_rand_fits.append(sub_rand_fits)
      analyzer.attn_rand_sliced_activity.append(sub_rand_sliced_activity)
        
def get_bins(all_curvatures, num_bins=50):
  max_curvature = np.amax(all_curvatures)
  min_curvature = np.amin(all_curvatures)
  bin_width = (max_curvature - min_curvature) / (num_bins-1) # subtract 1 to leave room for the zero bin
  bin_centers = [0.0]
  while min(bin_centers) > min_curvature:
    bin_centers.append(bin_centers[-1]-bin_width)
  bin_centers = bin_centers[::-1]
  while max(bin_centers) < max_curvature:
    bin_centers.append(bin_centers[-1]+bin_width)
  bin_lefts = bin_centers - (bin_width / 2)
  bin_rights = bin_centers + (bin_width / 2)
  bins = np.append(bin_lefts, bin_rights[-1])
  return bins

def compute_curvature_hists(analyzer_list, num_bins):
  # uniform bins for both iso-curvature plots and both attenuation-curvature plots
  iso_all_curvatures = []
  for analyzer in analyzer_list:
    for neuron_index in range(analyzer.num_neurons):
      iso_all_curvatures += analyzer.iso_comp_curvatures[neuron_index]
      iso_all_curvatures += analyzer.iso_rand_curvatures[neuron_index]
  iso_bins = get_bins(iso_all_curvatures, num_bins)
  attn_all_curvatures = []
  for analyzer in analyzer_list:
    for neuron_index in range(analyzer.attn_num_target_neurons):
      attn_all_curvatures += analyzer.attn_comp_curvatures[neuron_index]
      attn_all_curvatures += analyzer.attn_rand_curvatures[neuron_index]
  attn_bins = get_bins(attn_all_curvatures, num_bins)
  for analyzer in analyzer_list:
    # Iso-response histogram
    flat_comp_curvatures = [item for sub_list in analyzer.iso_comp_curvatures for item in sub_list]
    comp_hist, analyzer.iso_bin_edges = np.histogram(flat_comp_curvatures, iso_bins, density=False)
    analyzer.iso_comp_hist = comp_hist / len(flat_comp_curvatures)
    flat_rand_curvatures = [item for sub_list in analyzer.iso_rand_curvatures for item in sub_list]
    rand_hist, _ = np.histogram(flat_rand_curvatures, iso_bins, density=False)
    analyzer.iso_rand_hist = rand_hist / len(flat_rand_curvatures)
    # Response attenuation histogram
    flat_comp_curvatures = [item for sub_list in analyzer.attn_comp_curvatures for item in sub_list]
    comp_hist, analyzer.attn_bin_edges = np.histogram(flat_comp_curvatures, attn_bins, density=False)
    analyzer.attn_comp_hist = comp_hist / len(flat_comp_curvatures)
    flat_rand_curvatures = [item for sub_list in analyzer.attn_rand_curvatures for item in sub_list]
    rand_hist, _ = np.histogram(flat_rand_curvatures, attn_bins, density=False)
    analyzer.attn_rand_hist = rand_hist / len(flat_rand_curvatures)

def plot_curvature_histograms(activity, contour_pts, contour_angle, contour_text_loc, hist_list, label_list,
                              color_list, mesh_color, bin_centers, title, xlabel, curve_lims,
                              text_width=200, width_ratio=1.0, dpi=100):
  gs0_wspace = 0.5
  hspace_hist = 0.7
  wspace_hist = 0.08
  view_elevation = 30
  iso_response_line_thickness = 2
  respone_attenuation_line_thickness = 2
  num_y_plots = 2
  num_x_plots = 1
  fig = plt.figure(figsize=set_size(text_width, width_ratio, [num_y_plots, num_x_plots]), dpi=dpi)
  gs_base = gridspec.GridSpec(num_y_plots, num_x_plots, wspace=gs0_wspace)
  curve_ax = fig.add_subplot(gs_base[0], projection='3d')
  x_mesh, y_mesh = np.meshgrid(*contour_pts)
  curve_ax.set_zlim(0, 1)
  curve_ax.set_xlim3d(5, 200)
  curve_ax.grid(b=True, zorder=0)
  x_ticks = curve_ax.get_xticks().tolist()
  x_ticks = np.round(np.linspace(curve_lims["x"][0], curve_lims["x"][1],
    len(x_ticks)), 1).astype(str)
  a_x = [" "]*len(x_ticks)
  a_x[1] = x_ticks[1]
  a_x[-1] = x_ticks[-1]
  curve_ax.set_xticklabels(a_x)
  y_ticks = curve_ax.get_yticks().tolist()
  y_ticks = np.round(np.linspace(curve_lims["y"][0], curve_lims["y"][1],
    len(y_ticks)), 1).astype(str)
  a_y = [" "]*len(y_ticks)
  a_y[1] = y_ticks[1]
  a_y[-1] = y_ticks[-1]
  curve_ax.set_yticklabels(a_y)
  curve_ax.set_zticklabels([])
  curve_ax.zaxis.set_rotate_label(False)
  curve_ax.set_zlabel("Normalized\nActivity", rotation=95, labelpad=-12., position=(-10., 0.))
  #curve_ax.scatter(x_mesh, y_mesh, activity, color=mesh_color, s=0.01)
  curve_ax.plot_wireframe(x_mesh, y_mesh, activity, rcount=100, ccount=100, color=mesh_color, zorder=1,
    linestyles="dotted", linewidths=0.5, alpha=1.0)
  # Plane vector visualizations
  v = Arrow3D([-200/3., -200/3.], [200/2., 200/2.+200/16.], 
              [0, 0.0], mutation_scale=10, 
              lw=1, arrowstyle="-|>", color="red", linestyle="dashed")
  curve_ax.add_artist(v)
  curve_ax.text(-300/3., 280/3.0, 0.0, r"$\nu$", color="red")
  phi_k = Arrow3D([-200/3., 0.], [200/2., 200/2.], 
              [0, 0.0], mutation_scale=10, 
              lw=1, arrowstyle="-|>", color="red", linestyle = "dashed")
  curve_ax.add_artist(phi_k)
  curve_ax.text(-175/3., 250/3.0, 0.0, r"${\phi}_{k}$", color="red")
  # Iso-response curve
  loc0, loc1, loc2 = contour_text_loc[0]
  curve_ax.text(loc0, loc1, loc2, "Iso-\nresponse", color="black", weight="bold", zorder=10)
  iso_line_offset = 165
  x, y = contour_pts
  curve_ax.plot(np.zeros_like(x)+iso_line_offset, y, activity[:, iso_line_offset],
    color="black", lw=2, zorder=2)
  # Response attenuation curve
  loc0, loc1, loc2 = contour_text_loc[1]
  curve_ax.text(loc0, loc1, loc2, "Response\nAttenuation", color="black", weight="bold", zorder=10)
  lines = np.array([0.2, 0.203, 0.197]) - 0.1
  for i in lines:
      curve_ax.contour3D(x_mesh, y_mesh, activity, [i], colors="black", linewidths=2, zorder=2)
  # Additional settings
  curve_ax.view_init(view_elevation, contour_angle)
  scaling = np.array([getattr(curve_ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
  curve_ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3) # make sure it has a square aspect
  
  num_hist_y_plots = 2
  num_hist_x_plots = 2
  gs_hist = gridspec.GridSpecFromSubplotSpec(num_hist_y_plots, num_hist_x_plots, gs_base[1],
    hspace=hspace_hist, wspace=wspace_hist)
  all_x_lists = zip(hist_list, label_list, color_list, bin_centers, title)
  orig_ax = fig.add_subplot(gs_hist[0,0])
  axes = []
  for sub_plt_y in range(0, num_hist_y_plots):
    axes.append([])
    for sub_plt_x in range(0, num_hist_x_plots):
      if (sub_plt_x, sub_plt_y) == (0,0):
        axes[sub_plt_y].append(orig_ax)
      else:
        axes[sub_plt_y].append(fig.add_subplot(gs_hist[sub_plt_y, sub_plt_x], sharey=orig_ax))
  for axis_x, (sub_hist, sub_label, sub_color, sub_bins, sub_title) in enumerate(all_x_lists):
    max_hist_val = 0.001
    min_hist_val = 100
    all_y_lists = zip(sub_hist, sub_label, sub_color, xlabel)
    for axis_y, (axis_hists, axis_labels, axis_colors, sub_xlabel) in enumerate(all_y_lists):
      axes[axis_y][axis_x].spines["top"].set_visible(False)
      axes[axis_y][axis_x].spines["right"].set_visible(False)
      axes[axis_y][axis_x].set_xticks(sub_bins, minor=True)
      axes[axis_y][axis_x].set_xticks(sub_bins[::int(len(sub_bins)/4)], minor=False)
      axes[axis_y][axis_x].xaxis.set_major_formatter(plticker.FormatStrFormatter("%0.3f"))
      for hist, label, color in zip(axis_hists, axis_labels, axis_colors):
        axes[axis_y][axis_x].plot(sub_bins, hist, color=color, linestyle="-", drawstyle="steps-mid", label=label)
        axes[axis_y][axis_x].set_yscale('log')
        if np.max(hist) > max_hist_val:
          max_hist_val = np.max(hist)
        if np.min(hist) < min_hist_val:
          min_hist_val = np.min(hist)
      axes[axis_y][axis_x].axvline(0.0, color="black", linestyle="dashed", linewidth=1)
      if axis_y == 0:
        axes[axis_y][axis_x].set_title(sub_title)
      axes[axis_y][axis_x].set_xlabel(sub_xlabel)
      if axis_x == 0:
        axes[axis_y][axis_x].set_ylabel("Relative\nFrequency")
        ax_handles, ax_labels = axes[axis_y][axis_x].get_legend_handles_labels()
        legend = axes[axis_y][axis_x].legend(handles=ax_handles, labels=ax_labels, loc="upper right",
          ncol=3, borderaxespad=0., borderpad=0., handlelength=0., columnspacing=-0.5,
          labelspacing=0., bbox_to_anchor=(0.95, 0.95))
        legend.get_frame().set_linewidth(0.0)
        for text, color in zip(legend.get_texts(), axis_colors):
          text.set_color(color)
        for item in legend.legendHandles:
          item.set_visible(False)
      if axis_x == 1:
        axes[axis_y][axis_x].tick_params(axis="y", labelleft=False)
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
      #ax.set_ylabel("Activation")#, fontsize=16)
      #ax.set_xlabel("Orientation")#, fontsize=16)
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
        ax.yaxis.set_major_formatter(plticker.FormatStrFormatter('%0.2g'))
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
  fig.suptitle(title, y=0.95, x=0.5)#, fontsize=20)
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

def plot_circ_variance_histogram(analyzer_list, circ_var_list, color_list, label_list, num_bins, density, width_ratios,
                                 height_ratios, text_width=200, width_ratio=1.0, dpi=100):
  num_y_plots = 5
  num_x_plots = 3
  fig = plt.figure(figsize=set_size(text_width, width_ratio, [num_y_plots, num_x_plots]), dpi=dpi)
  gs0 = gridspec.GridSpec(num_y_plots, num_x_plots, width_ratios=width_ratios)
  axes = []
  gs_hist = gridspec.GridSpecFromSubplotSpec(4, 1, gs0[:, 0], height_ratios=height_ratios)
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
  axes[-1].spines["top"].set_visible(False)
  axes[-1].spines["right"].set_visible(False)
  axes[-1].set_xticks(bin_left, minor=True)
  axes[-1].set_xticks(bin_left[::4], minor=False)
  axes[-1].xaxis.set_major_formatter(plticker.FormatStrFormatter("%0.0f"))
  axes[-1].set_xlim([variance_min, variance_max])
  axes[-1].set_xticks([variance_min, variance_max])
  axes[-1].set_xticklabels(["More\nselective", "Less\nselective"])
  ticks = axes[-1].xaxis.get_major_ticks()
  ticks[0].label1.set_horizontalalignment("left")
  ticks[1].label1.set_horizontalalignment("right")
  y_max = np.max([np.max(hist) for hist in hist_list])
  axes[-1].set_ylim([0, y_max+1])
  axes[-1].set_title("Circular Variance")
  if density:
    axes[-1].set_ylabel("Density")
  else:
    axes[-1].set_ylabel("Count")
  handles, labels = axes[-1].get_legend_handles_labels()
  legend = axes[-1].legend(handles=handles, labels=labels,
    borderaxespad=0., framealpha=0.0, loc="upper right",
    bbox_to_anchor=(1.0, 0.8)) #bbox sets (x, y) of legend, relative to loc
  legend.get_frame().set_linewidth(0.0)
  for text, color in zip(legend.get_texts(), color_list):
    text.set_color(color)
  for item in legend.legendHandles:
    item.set_visible(False)
  gs_weights = gridspec.GridSpecFromSubplotSpec(len(analyzer_list), 1, gs0[:, 1], hspace=-0.6)
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
  gs_tuning = gridspec.GridSpecFromSubplotSpec(len(analyzer_list), 1, gs0[:, 2], hspace=-0.6)
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
        plt.text(x=0.1, y=1.45, s=analyzer.analysis_params.display_name, horizontalalignment='center',
          verticalalignment='center', transform=axes[-1].transAxes)
  plt.show()
  return fig
