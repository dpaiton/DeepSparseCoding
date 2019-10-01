import matplotlib.cm as cmx
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np

from .. import shared


def make_contour_panel(activation_contours, contour_vals,
                       attack, attack_activations,
                       grads, grad_coords,
                       weight_vectors=None, skip_x_axis_grads=True,
                       cmap=None, ax=None):
    if ax is None:
        f, ax = plt.subplots(figsize=(8, 12))
    x_lims = [0, 0.27]
    y_lims = [-0.1, 0.15]

    if cmap is None:
        cmap = shared.contour_cmap

    add_contours(activation_contours, contour_vals, cmap, ax)

    if weight_vectors is not None:
        [plot_dictionary_element(weight_vector, ax=ax,
                                 arrow_kwargs={"width": 5e-3,
                                               "facecolor": "k",
                                               "zorder": 10})
         for weight_vector in weight_vectors]

    add_grads(grads, grad_coords, ax, skip_x_axis_grads)

    add_attack(attack, attack_activations, contour_vals, cmap, ax)

    ax.axis("equal")
    ax.set_ylim(*y_lims)
    ax.set_xlim(*x_lims)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def add_attack(attack, attack_activations, contour_vals, cmap, ax):
    lc = colorline(
        x=attack.T[0, :],
        y=attack.T[1, :],
        z=attack_activations,
        linewidth=6,
        cmap=cmap,
        ax=ax,
        norm=plt.Normalize(min(contour_vals), max(contour_vals)))

    ax.scatter(*attack.T[:, 0], color="black",
               edgecolor=cmap(-np.inf),
               zorder=100, s=256, lw=4)

    ax.scatter(*attack.T[:, -1], color='black',
               edgecolor=cmap(np.inf),
               zorder=100, s=256, lw=3, marker=">")

    return lc


def add_contours(activation_contours, contour_vals, cmap, ax):
    cNorm = plt.Normalize(vmin=min(contour_vals),
                          vmax=max(contour_vals))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    for activation_contour, val in zip(activation_contours, contour_vals):
        plot_kwargs = {"c": scalarMap.to_rgba(val), "linewidth": 6}
        add_contour(activation_contour, ax=ax, plot_kwargs=plot_kwargs)


def add_contour(contour, ax=None, plot_kwargs=None):
    if ax is None:
        f, ax = plt.subplots(figsize=(8, 8))
    if plot_kwargs is None:
        plot_kwargs = {"linewidth": 2}

    ax.plot(*contour.T, **plot_kwargs, zorder=11)


def add_grads(grads, grad_coords, ax, skip_x_axis_grads=True):
    for grad, (x, y) in zip(grads, grad_coords):
        if x < 0.1:
            continue
        if skip_x_axis_grads:
            if (np.abs(y) < 0.02):
                continue
        if grad[0] <= 0:
            continue
        ax.arrow(x, y, *grad / 150, width=1e-3,
                 color="gray", alpha=1., zorder=8.)


def plot_dictionary_element(dict_elem, ax=None, arrow_kwargs=None):
    if arrow_kwargs is None:
        arrow_kwargs = {"facecolor": "k"}
    if ax is None:
        f, ax = plt.subplots()
    ax.arrow(0, 0, *dict_elem,
             **arrow_kwargs)

# colorline recipe from matplotlib


def colorline(x, y, z=None,
              cmap=plt.get_cmap('copper'),
              norm=plt.Normalize(0.0, 1.0),
              colors=None,
              linewidth=3, alpha=1.0,
              ax=None):
    """
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):
        # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    segments = np.asarray([extend(segment) for segment in segments])
    lc = mcoll.LineCollection(segments, array=z,
                              cmap=cmap, norm=norm,
                              linewidth=linewidth,
                              zorder=100)
    if ax is None:
        ax = plt.gca()

    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates,
    in the correct format for LineCollection: an array of the form
    numlines x (points per line) x 2 (x and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def extend(segment):
    slope = segment[-2, :] - segment[-1, :]
    segment[-1, :] = segment[-1, :] - slope * 0.5
    return segment
