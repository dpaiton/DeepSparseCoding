import numpy as np
import proplot as plot


def clear_axes(axs, spines='none'):
    """
    Calls clear_axis iteratively for each axis in axs
    """
    for ax in axs:
        clear_axis(ax, spines)
    return axs


def clear_axis(ax, spines='none'):
    """
    Clear spines & tick labels from proplot axis object
    Args:
        ax [proplot ax object, or matplotlib axis object]
        spines [str] any matplotlib color
    Returns:
        ax
    """
    for ax_loc in ['top', 'bottom', 'left', 'right']:
        ax.spines[ax_loc].set_color(spines)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.tick_params(axis='both', bottom=False, top=False, left=False, right=False)
    return ax


def plot_stats(data, x_key, x_label=None, y_keys=None, y_labels=None, start_index=0, save_filename=None):
    """
    Generate time-series plots of stats specified by keys
    Args:
        data: [dict] containing data to be plotted. len of all values should be equal
            data must include x_key as a valid key
        x_key: [str] key for the x-axis (time varying component)
        x_label: [str] corresponding label for the x_key, if not provided then the key is used
        y_keys: [list of str] optional list of keys to plot, each should exist in data.keys()
            If nothing is given, data.keys() will be used
        y_labels: [list of str] optional list of labels, should be the same length as keys input
            If nothing is given, y_keys will be used
        start_index: [int] time offset for plotting data - default=0 will plot all data
        save_filename: [str] containing the complete output filename.
    Returns:
        fig: matplotlib figure handle
    """
    assert x_key in list(data.keys()), ('x_key=%s must be in data.keys()'%x_key)
    if x_label is None:
        x_label = x_key
    if y_keys is None:
        y_keys = list(data.keys())
        if 'epoch' in y_keys:
            y_keys.remove('epoch')
        if 'batch_step' in y_keys:
            y_keys.remove('batch_step')
    else:
        assert all([y_key in list(data.keys()) for y_key in y_keys])
    if y_labels is None:
        y_labels = [' '.join(y_key.split('_')) for y_key in y_keys]
    else:
        assert len(y_labels) == len(y_keys), (
            'The number of y_labels must match the number of y_keys')
    num_y_keys = len(y_keys)
    num_plots_y = int(np.ceil(np.sqrt(num_y_keys)))
    num_plots_x = int(np.ceil(np.sqrt(num_y_keys)))
    fig, axes = plot.subplots(nrows=num_plots_y, ncols=num_plots_x, sharex=False, sharey=False)
    key_idx = 0
    for plot_id in np.ndindex((num_plots_y, num_plots_x)):
        if key_idx < num_y_keys:
            x_dat = data[x_key][start_index:]
            y_dat = data[y_keys[key_idx]][start_index:]
            if len(x_dat) == len(y_dat):
                ax = axes[plot_id]
                ax.plot(x_dat, y_dat)
                ax.format(
                    yticks = [np.minimum(0.0, np.min(y_dat)), np.maximum(0.0, np.max(y_dat))],
                    ylabel = y_labels[key_idx])
                key_idx += 1
            else:
                ax = clear_axis(axes[plot_id])
                print('utils/plot_functions.py: WARNING: x and y for key %s must have same first dimensions but are %g and %g'%(
                    y_keys[key_idx], len(x_dat), len(y_dat)))
        else:
            ax = clear_axis(axes[plot_id])
    axes.format(
        xlabel = x_label,
        suptitle = 'Stats per Batch')
    if save_filename is not None:
        fig.savefig(save_filename, transparent=True)
        plot.close(fig)
        return None
    plot.show()
    return fig
